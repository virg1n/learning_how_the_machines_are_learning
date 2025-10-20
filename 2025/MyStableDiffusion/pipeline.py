import torch
import torch.nn as nn
from tqdm import tqdm

from DDPM import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt, neg_promt="",
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
        prompt_max_length=77
    ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()


        if idle_device is not None:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        clip = models['clip']
        clip.to(device)

        clip.to(device)
        clip.eval()
        for p in clip.parameters():
            p.requires_grad_(False)

        if do_cfg:
            pos_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=prompt_max_length).input_ids
            neg_tokens = tokenizer.batch_encode_plus([neg_promt], padding="max_length", max_length=prompt_max_length).input_ids
            pos_tokens = torch.tensor(pos_tokens, dtype=torch.long, device=device)
            neg_tokens = torch.tensor(neg_tokens, dtype=torch.long, device=device)
        
            pos_context = clip(pos_tokens)
            neg_context = clip(neg_tokens)
        
            # put UNCOND FIRST to match the canonical formula
            context = torch.cat([neg_context, pos_context], dim=0)

        else:
            pos_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=prompt_max_length
            ).input_ids
            pos_tokens = torch.tensor(pos_tokens, dtype=torch.long ,device=device)
            context = clip(pos_tokens)

        if sampler_name=="ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_n_inference_steps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler_name")
        
        latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        if input_image is not None:
            input_image = input_image.resize((WIDTH, HEIGHT))
            input_image = torch.tensor(input_image, dtype=torch.float32, device=device)
            input_image = rescale(input_image, (0, 255), (-1, 1))
            
            encoder = models['encoder']
            encoder.to(device)

            input_image = input_image.unsqueeze(0) # (B, W, H, C)
            input_image = input_image.permute(0, 3, 1, 2) # (B, C, W, H)

            input_image_noise = torch.randn(latent_shape, generator=generator, device=device)
            encoded_image = encoder(input_image, input_image_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(encoded_image, sampler.timesteps[0])

            to_idle(encoder)

        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion.to(device)

        # timestamps = tqdm(sampler.timesteps)
        for step in tqdm(sampler.steps):
            time_embedded = get_time_embedding(step).to(device)

            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1) # 2*B, C, W, H 
            
            noise = diffusion(model_input, context ,time_embedded)
            
            # if do_cfg:
            #     uncond, cond = noise.chunk(2)
            #     delta = (cond - uncond).pow(2).mean().sqrt().item()
            #     print(f"step {int(step):4d}  ||cond - uncond|| = {delta:.6f}")

            if do_cfg:
                uncond, cond = noise.chunk(2)
                noise = uncond + cfg_scale * (cond - uncond)
                # pos_image, neg_image = noise.chunk(2)
                # noise = cfg_scale * (pos_image - neg_image) + neg_image # 1*B, C, W, H 

            latents = sampler.step(step, latents, noise)
        
        to_idle(diffusion)
        
        decoder = models['decoder']
        decoder.to(device)

        final_images = decoder(latents)

        to_idle(decoder)

        final_images = rescale(final_images, (-1, 1), (0, 255), clamp=True)
        final_images = final_images.permute(0, 2, 3, 1)
        final_images = final_images.to("cpu", torch.uint8).numpy()

        return final_images[0]


def rescale(image, init_range, new_range, clamp=False):
    init_min, init_max = init_range
    new_min, new_max = new_range

    image -= init_min
    image *= (new_max - new_min) / (init_max-init_min)
    image += new_min

    if clamp:
        image = image.clamp(new_min, new_max)
    return image

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)