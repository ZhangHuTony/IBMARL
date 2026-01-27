import torch


# Load the data
path = "/home/tony-zhang/Research/R2BC/results/navigation_r2bc_decent_20260126_210952/demonstrations.pt"

demo_data = torch.load(path, map_location="cpu", weights_only=False)

import torch
import numpy as np

# Load the data as you did before
# demo_data = torch.load(path, map_location="cpu", weights_only=False)

def inspect_data(data, indent=0):
    tab = "  " * indent
    
    if isinstance(data, dict):
        print(f"{tab}Dictionary with keys: {list(data.keys())}")
        for k, v in data.items():
            print(f"{tab}Key: {k}")
            inspect_data(v, indent + 1)
            
    elif isinstance(data, list):
        print(f"{tab}List of length: {len(data)}")
        if len(data) > 0:
            first_elem = data[0]
            # --- NEW LOGIC TO PRINT DIMENSIONS ---
            if hasattr(first_elem, 'shape'):
                # Handles numpy arrays or tensors
                print(f"{tab}First element dimensions: {first_elem.shape}")
            elif isinstance(first_elem, (list, tuple)):
                # Handles nested lists
                print(f"{tab}First element dimensions: ({len(first_elem)},)")
            else:
                # Handles scalars (int, float, bool)
                print(f"{tab}First element type: {type(first_elem).__name__} (Scalar)")
            # --------------------------------------
            
            print(f"{tab}Inspecting first element:")
            inspect_data(first_elem, indent + 1)
            
    elif isinstance(data, torch.Tensor):
        print(f"{tab}Tensor | Shape: {data.shape} | Dtype: {data.dtype}")
        
    else:
        print(f"{tab}{type(data).__name__}: {str(data)[:50]}...")

print("--- Demonstration Data Structure ---")
inspect_data(demo_data)