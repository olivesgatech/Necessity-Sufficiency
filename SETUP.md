# Setup Instructions

## Quick Setup

### 1. Clone or Download Repository

```bash
# If using git
git clone <repository-url>
cd necessity-sufficiency-xai

# Or extract the zip file
unzip necessity-sufficiency-xai.zip
cd necessity-sufficiency-xai
```

### 2. Create Virtual Environment

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import sklearn; import shap; import lime; print('All packages installed successfully!')"
```

## Running Your First Analysis

### Basic Example

```bash
cd src
python main.py --dataset breast_cancer --model logistic
```

This will:
1. Load the Breast Cancer dataset
2. Train a Logistic Regression model
3. Calculate global necessity and sufficiency scores
4. Evaluate LIME and SHAP robustness
5. Generate visualizations in `../results/`

### Expected Output

```
================================================================================
NECESSITY AND SUFFICIENCY ANALYSIS
================================================================================

Dataset: breast_cancer
Model: logistic
Number of samples: 100
Top-k features: 5

================================================================================
LOADING DATA
================================================================================
Dataset shape: (569, 30)
Number of features: 30
Class distribution: [212 357]

================================================================================
TRAINING MODEL
================================================================================
Training accuracy: 0.9497
Test accuracy: 0.9825

================================================================================
CALCULATING GLOBAL NECESSITY AND SUFFICIENCY SCORES
================================================================================

Feature 1/30: mean radius
Calculating necessity for feature 0: 100%|██████████| 100/100
  Necessity: 0.4321
  Sufficiency: 0.2145

...

================================================================================
XAI ROBUSTNESS EVALUATION
================================================================================

Evaluating LIME necessity robustness...
100%|██████████████████████████████████████| 50/50

Evaluating LIME sufficiency robustness...
100%|██████████████████████████████████████| 50/50

...

================================================================================
ANALYSIS COMPLETE!
================================================================================

All results saved to: ../results
```

## Using Jupyter Notebooks

### Start Jupyter

```bash
jupyter notebook
```

### Open Notebooks

1. `notebooks/01_introduction.ipynb` - Introduction and concepts
2. `notebooks/02_toy_example.ipynb` - Validation with synthetic data
3. `notebooks/03_full_analysis.ipynb` - Complete analysis walkthrough

## Troubleshooting

### Issue: SHAP installation fails

**Solution**: Install specific version
```bash
pip install shap==0.41.0 --no-cache-dir
```

### Issue: LIME import error

**Solution**: Reinstall lime
```bash
pip uninstall lime
pip install lime==0.2.0.1
```

### Issue: Matplotlib/Seaborn display issues

**Solution**: Install backend
```bash
pip install PyQt5  # Or: pip install tk
```

### Issue: Memory error on large datasets

**Solution**: Reduce sample size
```bash
python main.py --dataset breast_cancer --model logistic --n_samples 50
```

### Issue: SHAP is very slow

**Solution**: This is normal for KernelExplainer. Wait or reduce test samples:
```bash
python main.py --dataset breast_cancer --model logistic --n_samples 30
```

## Advanced Setup

### Using Conda

```bash
conda create -n necessity-xai python=3.8
conda activate necessity-xai
pip install -r requirements.txt
```

### Development Installation

```bash
# Install in editable mode with dev dependencies
pip install -e .
pip install pytest black flake8 mypy
```

### GPU Support (Optional)

If using models that benefit from GPU:
```bash
pip install torch torchvision  # For PyTorch models
# or
pip install tensorflow-gpu     # For TensorFlow models
```

## Verification Tests

### Test 1: Import Verification

```python
python -c "
from src.counterfactual_generator import ForwardCounterfactualGenerator
from src.xai_evaluator import XAIRobustnessEvaluator
from src.visualization import NecessitySufficiencyVisualizer
print('All modules imported successfully!')
"
```

### Test 2: Quick Analysis

```bash
cd src
python main.py --dataset iris --model logistic --n_samples 20 --top_k 3
```

Should complete in under 5 minutes.

### Test 3: Toy Example Validation

```bash
cd notebooks
jupyter nbconvert --to notebook --execute 02_toy_example.ipynb
```

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4 GB RAM
- 1 GB disk space

### Recommended Requirements
- Python 3.9+
- 8 GB RAM
- 2 GB disk space
- Multi-core CPU for parallelization

### Tested Platforms
- Ubuntu 20.04, 22.04
- macOS 11+
- Windows 10, 11
- Google Colab
- Jupyter Lab

## Next Steps

After setup, explore:

1. **README.md** - Overview and quick start
2. **docs/methodology.md** - Detailed methodology
3. **notebooks/** - Interactive tutorials
4. **src/main.py** - Run complete analysis

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the error message carefully
3. Ensure all dependencies are installed: `pip list`
4. Check Python version: `python --version` (should be 3.8+)
5. Open an issue on GitHub with:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

## Contact

For questions or support:
- Email: pchowdhury6@gatech.edu
- GitHub Issues: [repository]/issues
