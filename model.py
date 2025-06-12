import os
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig

class LotusDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        txt_path = os.path.join(self.image_dir, img_name.replace(".jpg", ".txt"))

        image = Image.open(img_path).convert("RGB").resize((512, 512))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        tokens = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.squeeze(0)

        return {"pixel_values": image, "input_ids": tokens}


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["proj_in", "proj_out", "conv1", "conv2"]  
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.train()


    dataset = LotusDataset("D:/mandavi/lora/lora_img")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)

    print("🏁 Starting training...")
    for epoch in range(10):
        for step, batch in enumerate(dataloader):
            images = batch["pixel_values"].to(device, dtype=torch.float16)
            input_ids = batch["input_ids"].to(device)

            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            encoder_hidden_states = pipe.text_encoder(input_ids)[0]

            noise_pred = pipe.unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    print("Saving LoRA weights to D:/mandavi/lora/lora_output...")
    pipe.unet.save_pretrained("D:/mandavi/lora/lora_output", safe_serialization=True)
    print("Training complete.")

if __name__ == "__main__":
    train()
