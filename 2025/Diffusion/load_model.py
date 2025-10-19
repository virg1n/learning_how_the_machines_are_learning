from encoder import Encoder
from decoder import Decoder
from diffusion import Diffusion
from CLIP import CLIP

from conversion_script import load_from_standard_weights

def load_model_from_checkpoint(checkpoint_path, device):
    all_models = load_from_standard_weights(checkpoint_path, device)

    encoder = Encoder().to(device=device)
    encoder.load_state_dict(all_models['encoder'], strict=True)

    decoder = Decoder().to(device=device)
    decoder.load_state_dict(all_models['decoder'], strict=True)

    clip = CLIP(vocab_size=49408, n_dims=768, n_tokens=77, n_heads=12).to(device=device)
    clip.load_state_dict(all_models['clip'], strict=True)

    diffusion = Diffusion().to(device=device)
    diffusion.load_state_dict(all_models['diffusion'], strict=True)

    return {
        "encoder": encoder,
        "decoder": decoder,
        "clip": clip,
        "diffusion": diffusion
    }
