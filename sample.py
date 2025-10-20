import torch
import torch.nn as nn

def sanity_test():
    """
    Comprehensive PyTorch sanity test to verify installation and basic functionality.
    """
    print("=" * 60)
    print("PyTorch Sanity Test")
    print("=" * 60)
    
    # 1. Check PyTorch version
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    # 2. Check device availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA is available - Device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ MPS (Apple Silicon) is available")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU (no GPU detected)")
    
    # 3. Test basic tensor operations
    print("\n--- Testing Basic Tensor Operations ---")
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y
    print(f"✓ Tensor addition: {x.tolist()} + {y.tolist()} = {z.tolist()}")
    
    # 4. Test tensor on device
    print("\n--- Testing Device Operations ---")
    a = torch.randn(3, 3).to(device)
    b = torch.randn(3, 3).to(device)
    c = torch.matmul(a, b)
    print(f"✓ Matrix multiplication on {device}: {c.shape}")
    
    # 5. Test basic neural network
    print("\n--- Testing Neural Network ---")
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    
    input_data = torch.randn(2, 10).to(device)
    output = model(input_data)
    print(f"✓ Simple NN forward pass: input shape {input_data.shape} -> output shape {output.shape}")
    
    # 6. Test autograd
    print("\n--- Testing Autograd ---")
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1
    y.backward()
    print(f"✓ Gradient computation: d(x²+3x+1)/dx at x=2.0 is {x.grad.item():.1f} (expected: 7.0)")
    
    # 7. Test basic training loop operations
    print("\n--- Testing Training Operations ---")
    model = nn.Linear(5, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    x_train = torch.randn(10, 5).to(device)
    y_train = torch.randn(10, 1).to(device)
    
    # One training step
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    print(f"✓ Training step completed - Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! PyTorch is working correctly. 🎉")
    print("=" * 60)

if __name__ == "__main__":
    try:
        sanity_test()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
