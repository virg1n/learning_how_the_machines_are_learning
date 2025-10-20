from transformers import CLIPTokenizer
import torch

from PIL import Image
from pathlib import Path

from load_model import load_model_from_checkpoint
from pipeline import generate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "./data/v1-5-pruned-emaonly.ckpt"
models = load_model_from_checkpoint(model_file, DEVICE)


prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
negative_prompt = ""
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 43

output_image = generate(
    prompt=prompt,
    neg_promt=negative_prompt,
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


output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

final_image = Image.fromarray(output_image)

output_path = output_dir / "generated_cat.png"

final_image.save(output_path)

print(f"Image saved to: {output_path.resolve()}")
