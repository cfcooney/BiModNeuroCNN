import torch

def gpu_check():
	"""
	Test script for discovering GPUs
	Expected Output: GPU and Cuda available: True
					******************************
					Number of GPUs: 1
					******************************
					Current GPU: 0
					******************************
					Current GPU location: <torch.cuda.device object at 0x000002B67ABF6940>
					******************************
					GPU device type: GeForce 940MX
					******************************

	"""
	available = torch.cuda.is_available()
	print(f"GPU and Cuda available: {available}")

	print("*"*30)

	n_gpus = torch.cuda.device_count()
	print(f"Number of GPUs: {n_gpus}")

	print("*"*30)

	device = torch.cuda.current_device()
	print(f"Current GPU: {device}")

	print("*"*30)

	location = torch.cuda.device(0)
	print(f"Current GPU location: {location}")

	print("*"*30)

	type_gpu = torch.cuda.get_device_name(0)
	print(f"GPU device type: {type_gpu}")

	print("*"*30)