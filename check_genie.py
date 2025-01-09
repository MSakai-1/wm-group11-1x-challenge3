import os
import subprocess
import torch

# GPUの指定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 を指定

def check_gpu_availability():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Using CPU.")

output_dir = 'data/genie_baseline_generated'
checkpoint_dir = '1x-technologies/GENIE_138M'
maskgit_steps = 2
temperature = 0

# GPU確認
check_gpu_availability()


os.makedirs(output_dir, exist_ok=True)


for i in range(0, 241, 10):
    generate_cmd = [
        "python", "genie/generate.py",
        "--checkpoint_dir", checkpoint_dir,
        "--output_dir", output_dir,
        "--example_ind", str(i),
        "--maskgit_steps", str(maskgit_steps),
        "--temperature", str(temperature)
    ]
    subprocess.run(generate_cmd, check=True)

    visualize_cmd = [
        "python", "visualize.py",
        "--token_dir", output_dir
    ]
    subprocess.run(visualize_cmd, check=True)

    generated_gif = os.path.join(output_dir, "generated_offset0.gif")
    generated_png = os.path.join(output_dir, "generated_comic_offset0.png")
    new_gif_name = os.path.join(output_dir, f"example_{i}.gif")
    new_png_name = os.path.join(output_dir, f"example_{i}.png")

    if os.path.exists(generated_gif):
        os.rename(generated_gif, new_gif_name)
    if os.path.exists(generated_png):
        os.rename(generated_png, new_png_name)

# 評価
evaluate_cmd = [
    "python", "genie/evaluate.py",
    "--checkpoint_dir", checkpoint_dir,
    "--maskgit_steps", str(maskgit_steps)
]
subprocess.run(evaluate_cmd, check=True)
