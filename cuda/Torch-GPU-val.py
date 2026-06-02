import torch

print("PyTorch Version:", torch.__version__)

# 1. Check if CUDA (GPU support) is available
is_cuda = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda}")

if is_cuda:
    # 2. Get the number of available GPUs
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")

    # 3. Get details for each GPU
    for i in range(device_count):
        print(f"\n--- GPU {i} Details ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(i)}")
        # Get memory information (converted to GB)
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1e9
        print(f"Total Memory: {total_mem:.2f} GB")
        
        # Check current utilization
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"Memory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved: {reserved:.2f} GB")

    # 4. Verification: Run a simple operation on the GPU
    # Create a tensor directly on the GPU
    device = torch.device("cuda")
    x = torch.ones((3, 3), device=device)
    y = x + x
    
    print(f"\nTest operation successful!")
    print(f"Tensor is on device: {y.device}")
else:
    print("PyTorch is currently using the CPU.")
