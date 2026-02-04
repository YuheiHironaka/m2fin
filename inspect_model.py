import torch
import os

checkpoint_path = r'C:\Users\y.hironaka\Downloads\0906best_ckpt.pth.tar'

if not os.path.exists(checkpoint_path):
    print(f"File not found: {checkpoint_path}")
else:
    try:
        # Load on CPU to avoid CUDA dependency issues in basic script
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"File: {os.path.basename(checkpoint_path)}")
        print(f"Type of object: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print("\nKeys in checkpoint:")
            for key in checkpoint.keys():
                if key != 'model': # Don't print the huge model weight dict
                    val = checkpoint[key]
                    if isinstance(val, (int, float, str, list)):
                        print(f"  {key}: {val}")
                    else:
                        print(f"  {key}: {type(val)}")
            
            if 'model' in checkpoint:
                print(f"\nModel state dict found with {len(checkpoint['model'])} layers.")
        else:
            print("Checkpoint is not a dictionary (might be a direct state_dict).")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
