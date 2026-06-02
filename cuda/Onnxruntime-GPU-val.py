import onnxruntime as ort

# Get the full list of supported backends and the active ones
all_providers = ort.get_all_providers()
available_providers = set(ort.get_available_providers())

print("ONNX Runtime Provider Availability:")
print("-" * 40)

# Loop through and print status indicators
for provider in all_providers:
    if provider in available_providers:
        print(f"[ ✓ ] {provider}")
    else:
        print(f"[   ] {provider}")
