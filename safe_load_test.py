import torch
from torch.serialization import add_safe_globals
import torch.utils.data

# ✅ allow the dataset class to be safely unpickled
add_safe_globals([torch.utils.data.dataset.TensorDataset])

# path to your file
PYTORCH_STATE = "improved_dataset_state.pth"

print("Loading checkpoint...")
state = torch.load(PYTORCH_STATE, map_location="cpu", weights_only=False)
print("✅ Successfully loaded state dictionary")
print("Keys:", list(state.keys())[:10])
