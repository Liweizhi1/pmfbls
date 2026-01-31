import time
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
log_file_path = Path("outputs/meta_vitbase_official/log.txt")
total_epochs = 100

def monitor():
    print(f"Monitoring log file: {log_file_path}")
    print("Waiting for updates... (Ctrl+C to stop)")
    
    # Initialize progress bar
    # Initial state
    last_epoch = -1
    
    # Check if file exists
    if not log_file_path.exists():
        print("Log file not found yet. Waiting...")
        while not log_file_path.exists():
            time.sleep(5)
            
    # Create progress bar
    pbar = tqdm(total=total_epochs, desc="Training Progress", unit="epoch")
    
    while True:
        try:
            with open(log_file_path, "r") as f:
                lines = f.readlines()
            
            current_epoch = -1
            last_stats = {}
            
            # Parse lines to find the latest epoch
            for line in lines:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    if "epoch" in data:
                        ep = data["epoch"]
                        # Handle initial validation epoch -1
                        if ep == -1:
                            continue
                        current_epoch = ep
                        last_stats = data
                except json.JSONDecodeError:
                    continue
            
            # Update progress bar
            if current_epoch > last_epoch:
                diff = current_epoch - last_epoch
                pbar.update(diff)
                last_epoch = current_epoch
                
                # Display stats
                if last_stats:
                    stats_str = (
                        f"Epoch {current_epoch}/{total_epochs} | "
                        f"Train Loss: {last_stats.get('train_loss', 0):.4f} | "
                        f"Test Acc: {last_stats.get('test_acc1', 0):.2f}% | "
                        f"Best Acc: {last_stats.get('best_test_acc', 0):.2f}%"
                    )
                    tqdm.write(stats_str)
            
            if current_epoch >= total_epochs - 1:
                pbar.close()
                print("\nTraining Finished!")
                break
                
            time.sleep(5)
            
        except KeyboardInterrupt:
            pbar.close()
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            tqdm.write(f"Error reading log: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor()
