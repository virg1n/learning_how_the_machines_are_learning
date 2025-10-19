
import load_model
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "./data/v1-5-pruned-emaonly.ckpt"
models = load_model.load_model_from_checkpoint(model_file, DEVICE)

## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "A cat"
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 43

output_image = pipeline.generate(
    prompt=prompt,
    neg_promt=uncond_prompt,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
Image.fromarray(output_image)

from PIL import Image
from pathlib import Path

# Create output directory
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Convert the NumPy array (output_image) to a PIL Image
final_image = Image.fromarray(output_image)

# Define a filename (you can use the prompt or timestamp)
output_path = output_dir / "generated_cat.png"

# Save the image
final_image.save(output_path)

print(f"Image saved to: {output_path.resolve()}")
