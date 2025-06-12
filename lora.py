import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("D:/mandavi/lora/lora_output", weight_name="adapter_model.safetensors")  

prompt = "a  lotus in the water, surrounded by green leaves, morning light"

with torch.autocast("cuda"):
    image = pipe(prompt).images[0]


image.save("D:/mandavi/lora/generated_lotus.png")
print("Image saved.")
