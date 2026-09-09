import torch
print("torch:", torch.__version__)
print("built CUDA:", torch.version.cuda)
print("CUDA usable:", torch.cuda.is_available())
if torch.cuda.is_available():
	print("GPU:", torch.cuda.get_device_name(0))
	x = torch.rand(1024, 1024, device="cuda")
	print("tensor device:", x.device)
