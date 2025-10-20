# ExecuTorch Setup Instructions

ExecuTorch is not yet available as a pip package. Follow these instructions to install it from source:

## Option 1: Install from PyTorch Nightly (Recommended)

```bash
# Install PyTorch nightly with ExecuTorch support
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install ExecuTorch from source
git clone https://github.com/pytorch/executorch.git
cd executorch
pip3 install -e .
```

## Option 2: Build from Source

```bash
# Clone ExecuTorch repository
git clone https://github.com/pytorch/executorch.git
cd executorch

# Install dependencies
pip3 install -r requirements.txt

# Build and install
python3 setup.py install
```

## Option 3: Use Docker (Alternative)

```bash
# Pull ExecuTorch Docker image
docker pull pytorch/executorch:latest

# Run container
docker run -it --rm -v $(pwd):/workspace pytorch/executorch:latest
```

## Verification

After installation, verify ExecuTorch is working:

```python
import torch
from executorch import to_edge_transform_and_lower, to_executorch

print("ExecuTorch installed successfully!")
```

## For Samsung S25 Ultra Development

Make sure to install the Android-specific dependencies:

```bash
# Install Android development tools
pip3 install android-tools-adb

# Install additional mobile optimization packages
pip3 install onnx onnxruntime
```

## Troubleshooting

If you encounter issues:

1. **Python Version**: Ensure you're using Python 3.8+
2. **PyTorch Version**: Use PyTorch 2.1+ for ExecuTorch compatibility
3. **Build Tools**: Install build essentials on Linux/Mac
4. **Android SDK**: Ensure Android SDK is installed for mobile development

For more help, see the [ExecuTorch documentation](https://docs.pytorch.org/executorch/1.0/index.html).

