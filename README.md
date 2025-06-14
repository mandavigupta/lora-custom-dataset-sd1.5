# 🌸 LoRA Fine-Tuning with Custom Lotus Dataset (Stable Diffusion v1.5)

This project demonstrates how to fine-tune [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) using a custom dataset of lotus images and prompts with **LoRA (Low-Rank Adaptation)**.

We train a lightweight adapter using LoRA, and generate high-quality images with minimal compute cost. The project uses Hugging Face 🤗 `diffusers`, `peft`, and a simple `PyTorch` pipeline.

---

## 📁 Project Structure

---

## 🚀 Clone the Repository

```bash
git clone https://github.com/mandavigupta/lora-custom-dataset-sd1.5.git
cd lora-custom-dataset-sd1.5
```

## 🏗️ Requirements

Install required libraries:

```bash
pip install torch torchvision diffusers transformers peft

```
A CUDA-enabled GPU is strongly recommended for training and inference.
---
🧠 Training a LoRA Adapter
Use model.py to train a LoRA adapter on your custom dataset (lotus images + prompts):

```bash
python model.py
```
Dataset must be in a folder (/lora_img/) containing .jpg images and corresponding .txt prompt files.

Each prompt file should be named the same as the image (e.g., lotus1.jpg → lotus1.txt).

LoRA weights will be saved to /lora_output.
Example Prompt (lotus1.txt):
```bash
a beautiful pink lotus in water, surrounded by green leaves
```
🎨 Image Generation with LoRA
Use the following script (lora.py) to generate images with your trained LoRA weights:
```bash
python lora.py
```
⚙️ Key Features
✅ Fine-tunes Stable Diffusion v1.5 using LoRA for memory-efficient training.

✅ Custom dataset loading with image-to-prompt alignment.

✅ Supports training on consumer GPUs using float16.

✅ Outputs high-quality generative images with custom themes (e.g., lotus).

🏁 Results
After training, you can generate custom lotus-style images by prompting Stable Diffusion with your fine-tuned LoRA adapter.

<p align="center"> <img src="generated_lotus.png" width="400"/> </p>
📌 Credits
Diffusers by Hugging Face

PEFT (Parameter-Efficient Fine-Tuning)

Inspired by LoRA training guides on DreamBooth and community resources.

