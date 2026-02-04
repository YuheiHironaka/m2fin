import torch
import os

checkpoint_path = r'C:\Users\y.hironaka\Downloads\0906best_ckpt.pth.tar'

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"File: {os.path.basename(checkpoint_path)}")
        if isinstance(checkpoint, dict):
            for k, v in checkpoint.items():
                if k != 'model' and k != 'optimizer':
                    print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error: {e}")
