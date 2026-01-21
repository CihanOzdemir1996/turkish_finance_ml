# TensorFlow Setup Guide for LSTM Training

## ⚠️ Python Version Compatibility Issue

Your current Python version (3.14.2) is **not compatible** with TensorFlow. TensorFlow officially supports **Python 3.8-3.12**.

## Solution Options

### Option 1: Use Python 3.11 (Recommended)

#### Using Anaconda/Miniconda:
```bash
# Create a new environment with Python 3.11
conda create -n tf_env python=3.11

# Activate the environment
conda activate tf_env

# Install TensorFlow and other dependencies
pip install tensorflow pandas numpy scikit-learn xgboost matplotlib seaborn joblib

# Install Jupyter if needed
pip install jupyter

# Run Jupyter from this environment
jupyter notebook
```

#### Using pyenv (if installed):
```bash
# Install Python 3.11
pyenv install 3.11.0

# Set it for this project
pyenv local 3.11.0

# Install TensorFlow
pip install tensorflow
```

### Option 2: Use Python 3.12
Same steps as above, but use `python=3.12` instead of `python=3.11`.

### Option 3: Use Virtual Environment with Python 3.11
```bash
# Download Python 3.11 from python.org if not installed
# Then create virtual environment:
python3.11 -m venv tf_env
tf_env\Scripts\activate  # Windows
# or
source tf_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

After setting up the environment, verify TensorFlow works:

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {tf.__version__}")
```

## Running the LSTM Notebook

Once you have a compatible Python environment:

1. Activate your environment (e.g., `conda activate tf_env`)
2. Navigate to the project directory
3. Start Jupyter: `jupyter notebook`
4. Open `notebooks/07_lstm_model_training.ipynb`
5. Run all cells

The notebook will automatically:
- Check Python version
- Install TensorFlow if needed
- Train the LSTM model
- Compare with XGBoost

## Alternative: Use PyTorch Instead

If you prefer to stick with Python 3.14, we can modify the notebook to use PyTorch instead of TensorFlow. PyTorch has better support for newer Python versions. Let me know if you'd like this option!
