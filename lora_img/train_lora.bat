@echo off
accelerate launch train_network.py ^
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ^
 --train_data_dir="D:/mandavi/lotus_lora" ^
 --output_dir="D:/mandavi/lotus_lora_output" ^
 --resolution=512 ^
 --train_batch_size=1 ^
 --learning_rate=1e-4 ^
 --network_dim=128 ^
 --network_alpha=64 ^
 --max_train_steps=1000 ^
 --save_model_as=safetensors ^
 --output_name=lotus_lora
pause
