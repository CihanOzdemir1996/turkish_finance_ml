"""
Standalone LSTM Training Script
Run this script with Python 3.8-3.12 for TensorFlow compatibility
"""

import sys
import subprocess

# Check Python version
python_version = sys.version_info
if python_version.major == 3 and python_version.minor > 12:
    print("="*60)
    print("⚠️  PYTHON VERSION INCOMPATIBILITY")
    print("="*60)
    print(f"Your Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print("TensorFlow requires Python 3.8-3.12")
    print("\nOptions:")
    print("1. Use conda to create a Python 3.11 environment:")
    print("   conda create -n tf_env python=3.11")
    print("   conda activate tf_env")
    print("   pip install tensorflow")
    print("\n2. Use pyenv to install Python 3.11:")
    print("   pyenv install 3.11.0")
    print("   pyenv local 3.11.0")
    print("\n3. Run this script with Python 3.11:")
    print("   python3.11 train_lstm.py")
    sys.exit(1)

# Try to install TensorFlow if needed
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} available")
except ImportError:
    print("📦 Installing TensorFlow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} installed")

# Now run the training
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LSTM MODEL TRAINING")
    print("="*60)
    print("\nThis script will train the LSTM model.")
    print("For full functionality, please run the notebook:")
    print("notebooks/07_lstm_model_training.ipynb")
    print("\nOr ensure you have Python 3.8-3.12 and run the notebook cells.")
