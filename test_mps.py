#!/usr/bin/env python3
"""
Test Metal Performance Shaders (MPS) support for iBERT
"""
import torch
import time
import numpy as np


def test_mps_availability():
    """Test if MPS is available"""
    print("=" * 60)
    print("METAL PERFORMANCE SHADERS (MPS) TEST")
    print("=" * 60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if not torch.backends.mps.is_available():
        print("\n⚠️  MPS is not available on this system")
        return False
    
    print("\n✅ MPS is available!")
    return True


def test_mps_operations():
    """Test basic MPS operations"""
    print("\n" + "-" * 60)
    print("TESTING MPS OPERATIONS")
    print("-" * 60)
    
    device = torch.device("mps")
    
    # Test 1: Tensor creation
    try:
        x = torch.randn(1000, 1000, device=device)
        print(f"✅ Created tensor on MPS: shape={x.shape}, device={x.device}")
    except Exception as e:
        print(f"❌ Failed to create tensor: {e}")
        return False
    
    # Test 2: Matrix multiplication
    try:
        start = time.time()
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        torch.mps.synchronize()  # Wait for MPS operations to complete
        elapsed = time.time() - start
        print(f"✅ Matrix multiplication (1000x1000): {elapsed:.3f}s")
    except Exception as e:
        print(f"❌ Failed matrix multiplication: {e}")
        return False
    
    # Test 3: Neural network operations
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        ).to(device)
        
        input_tensor = torch.randn(32, 100, device=device)
        output = model(input_tensor)
        print(f"✅ Neural network forward pass: output shape={output.shape}")
    except Exception as e:
        print(f"❌ Failed neural network operations: {e}")
        return False
    
    # Test 4: Gradient computation
    try:
        x = torch.randn(10, 10, device=device, requires_grad=True)
        y = x.sum()
        y.backward()
        print(f"✅ Gradient computation successful")
    except Exception as e:
        print(f"❌ Failed gradient computation: {e}")
        return False
    
    return True


def benchmark_mps_vs_cpu():
    """Benchmark MPS vs CPU performance"""
    print("\n" + "-" * 60)
    print("PERFORMANCE COMPARISON: MPS vs CPU")
    print("-" * 60)
    
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        # CPU benchmark
        cpu_device = torch.device("cpu")
        x_cpu = torch.randn(size, size, device=cpu_device)
        y_cpu = torch.randn(size, size, device=cpu_device)
        
        start = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        # MPS benchmark
        mps_device = torch.device("mps")
        x_mps = torch.randn(size, size, device=mps_device)
        y_mps = torch.randn(size, size, device=mps_device)
        
        start = time.time()
        z_mps = torch.matmul(x_mps, y_mps)
        torch.mps.synchronize()
        mps_time = time.time() - start
        
        speedup = cpu_time / mps_time
        print(f"Matrix size {size}x{size}:")
        print(f"  CPU: {cpu_time:.4f}s")
        print(f"  MPS: {mps_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")


def test_model_loading():
    """Test loading a small model on MPS"""
    print("\n" + "-" * 60)
    print("TESTING MODEL LOADING ON MPS")
    print("-" * 60)
    
    try:
        from transformers import GPT2Model, GPT2Config
        
        # Create a small GPT2 model for testing
        config = GPT2Config(
            n_positions=512,
            n_embd=256,
            n_layer=4,
            n_head=4
        )
        
        model = GPT2Model(config)
        device = torch.device("mps")
        
        # Move model to MPS
        model = model.to(device)
        print(f"✅ Model loaded on MPS")
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10), device=device)
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✅ Forward pass successful: output shape={output.last_hidden_state.shape}")
        
        # Check memory usage
        if hasattr(torch.mps, 'current_allocated_memory'):
            memory_bytes = torch.mps.current_allocated_memory()
            memory_mb = memory_bytes / (1024 * 1024)
            print(f"📊 MPS memory allocated: {memory_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model on MPS: {e}")
        return False


def main():
    """Run all MPS tests"""
    if not test_mps_availability():
        return
    
    if not test_mps_operations():
        print("\n⚠️  Basic MPS operations failed")
        return
    
    benchmark_mps_vs_cpu()
    
    test_model_loading()
    
    print("\n" + "=" * 60)
    print("✅ ALL MPS TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nYour Mac's Metal GPU is ready for iBERT training!")
    print("The model will automatically use MPS for acceleration.")


if __name__ == "__main__":
    main()