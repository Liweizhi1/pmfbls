import subprocess
import sys
import json
import re
from tqdm import tqdm
import time
import os

def run_training():
    cmd = [
        "python", "main.py",
        "--dataset", "mini_imagenet",
        "--arch", "dino_base_patch16",
        "--pretrained_weights", "pretrained_ckpts/dino_vitbase16_pretrain.pth",
        "--pretrained-checkpoint-path", ".",
        "--output_dir", "outputs/meta_vitbase_official",
        "--epochs", "100",
        "--nEpisode", "2000",
        "--nValEpisode", "200",
        "--batch-size", "1",
        "--num_workers", "8",
        "--device", "cuda"
    ]

    # Check if checkpoint exists for resuming
    checkpoint_path = "outputs/meta_vitbase_official/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, resuming...")
        cmd.extend(["--resume", checkpoint_path])
    else:
        print("No checkpoint found, starting from scratch...")

    print(f"Executing command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    total_epochs = 100
    pbar = tqdm(total=total_epochs, desc="Meta-Training", unit="epoch")
    
    # Try to determine start epoch from log file if resuming
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        # This is a rough estimate, the process output will correct it
        pass

    current_epoch = 0
    
    try:
        for line in process.stdout:
            line = line.strip()
            if not line: continue
            
            # Check for JSON log line
            if line.startswith("{") and "epoch" in line:
                try:
                    data = json.loads(line)
                    epoch = data.get("epoch")
                    if epoch is not None:
                        # Update progress bar
                        if epoch > current_epoch:
                            pbar.update(epoch - current_epoch)
                            current_epoch = epoch
                        elif epoch == -1:
                            # Validation before training
                            pass
                        else:
                            # Maybe resumed, sync pbar
                            if epoch > pbar.n:
                                pbar.update(epoch - pbar.n)
                                current_epoch = epoch
                        
                        # Update description with stats
                        acc = data.get("test_acc1", 0)
                        loss = data.get("train_loss", 0)
                        pbar.set_postfix({"Acc": f"{acc:.2f}%", "Loss": f"{loss:.4f}"})
                except json.JSONDecodeError:
                    pass
            
            # Also print interesting lines to console (optional, but maybe too noisy)
            # if "Accuracy" in line or "Error" in line:
            #     tqdm.write(line)

    except KeyboardInterrupt:
        print("\nStopping training...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(1)
    
    process.wait()
    pbar.close()
    
    if process.returncode == 0:
        print("Training completed successfully!")
    else:
        print(f"Training failed with return code {process.returncode}")

if __name__ == "__main__":
    run_training()
