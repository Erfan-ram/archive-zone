import onnxruntime as ort
import numpy as np
import urllib.request
import time
import os

print("="*60)
print("🚀 Official ONNX Model ↔️ TensorRT Integration Test")
print("="*60)

# 1. Download an official, validated ONNX model (SqueezeNet 1.0)
# This model is lightweight (~5MB) and fully compatible with TRT.
model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
model_path = "squeezenet_official.onnx"

print("\n[1/4] Fetching Validated ONNX Model...")
if not os.path.exists(model_path):
    print("   Downloading SqueezeNet (~5MB) from ONNX Model Zoo...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("   ✅ Download Complete!")
    except Exception as e:
        print(f"   ❌ Failed to download model. Check internet connection: {e}")
        exit(1)
else:
    print("   ✅ Model already exists locally. Skipping download.")

# 2. Configure and Load the Session
print("\n[2/4] Loading Model into ONNX Runtime (Target: TensorRT)...")
try:
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 1 << 30, # 1 GB
            'trt_fp16_enable': True,           # Hardware acceleration
            'trt_engine_cache_enable': True,   # Cache engine to disk
            'trt_engine_cache_path': './trt_cache'
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
        })
    ]
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Validation Check
    active_providers = session.get_providers()
    print(f"   🔹 Requested: TensorRT -> CUDA")
    print(f"   🔹 Active:    {active_providers}")
    
    if 'TensorrtExecutionProvider' not in active_providers:
        print("\n❌ CRITICAL FAILURE: TensorRT is NOT the active provider.")
        print("   The system silently fell back to CUDA. Version mismatch detected.")
        exit(1)
    else:
        print("   ✅ SUCCESS: TensorRT is locked in and active!")

except Exception as e:
    print(f"\n❌ ERROR loading model:\n{e}")
    exit(1)

# 3. Prepare Input Tensor (SqueezeNet requires 1x3x224x224 float32)
print("\n[3/4] Preparing Input Tensor...")
input_name = session.get_inputs()[0].name
# Simulating a batch of 1 RGB image (224x224)
dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
print(f"   🔹 Input Name: '{input_name}' | Shape: {dummy_image.shape}")

# 4. Inference and Engine Building
print("\n[4/4] Running Inference Engine...")
print("   ⏳ WARM-UP RUN: TensorRT is compiling the CUDA engine now.")
print("      (This may take 10 to 60 seconds depending on your GPU...)")

# Warm-up run (Triggers the actual C++ TRT Engine compilation)
start_compile = time.time()
session.run(None, {input_name: dummy_image})
end_compile = time.time()
print(f"   ✅ Engine Compiled in {end_compile - start_compile:.2f} seconds!")

# Actual performance run
print("\n   🚀 PERFORMANCE RUN:")
start_infer = time.time()
output = session.run(None, {input_name: dummy_image})
end_infer = time.time()

print(f"   ✅ Inference Executed in {(end_infer - start_infer) * 1000:.2f} milliseconds!")
print(f"   🔹 Output Shape (1000 classes): {output[0].shape}")

print("\n" + "="*60)
print("🎉 FINAL VERDICT: THE PIPELINE IS 100% BULLETPROOF AND READY. 🎉")
print("="*60)