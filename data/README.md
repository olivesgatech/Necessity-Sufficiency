# Data Directory

This directory is for storing custom datasets.

## Built-in Datasets

The framework includes two built-in datasets from scikit-learn:

### 1. Breast Cancer Wisconsin (Diagnostic)

- **Source**: UCI Machine Learning Repository
- **Samples**: 569
- **Features**: 30 (all numerical)
- **Target**: Binary (Malignant vs Benign)
- **Usage**: `--dataset breast_cancer`

**Features include:**
- Mean radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Concave points, symmetry, fractal dimension
- Standard error and worst values for each

### 2. Iris Dataset (Binary)

- **Source**: UCI Machine Learning Repository  
- **Samples**: 150
- **Features**: 4 (sepal/petal measurements)
- **Target**: Binary (Virginica vs Others)
- **Usage**: `--dataset iris`

## Adding Custom Datasets

To use your own data:

### Option 1: Modify main.py

Add your dataset to the `load_data()` function:

```python
def load_data(dataset_name: str):
    if dataset_name == 'my_data':
        X = pd.read_csv('data/my_features.csv').values
        y = pd.read_csv('data/my_labels.csv').values.ravel()
        feature_names = ['feature1', 'feature2', ...]
        return X, y, feature_names
    # ... existing code
```

### Option 2: Use API Directly

```python
import pandas as pd
from counterfactual_generator import ForwardCounterfactualGenerator

# Load your data
df = pd.read_csv('data/my_dataset.csv')
X = df.drop('target', axis=1).values
y = df['target'].values
feature_names = df.drop('target', axis=1).columns.tolist()

# Continue with analysis...
```

## Data Format Requirements

Your custom dataset should:

1. **Features (X)**: 
   - NumPy array or pandas DataFrame
   - Shape: (n_samples, n_features)
   - Numerical values only

2. **Labels (y)**:
   - NumPy array
   - Shape: (n_samples,)
   - Binary classification: 0/1

3. **Feature Names**:
   - List of strings
   - Length: n_features

## Example Custom Dataset

```python
# my_dataset.csv
feature1,feature2,feature3,target
1.2,3.4,0.5,0
2.1,1.8,0.9,1
0.8,4.2,0.3,0
...
```

## Large Datasets

For datasets > 10,000 samples:

1. Use sampling in analysis:
   ```bash
   python main.py --dataset large_data --n_samples 500
   ```

2. Consider feature selection first:
   ```python
   from sklearn.feature_selection import SelectKBest
   selector = SelectKBest(k=20)
   X_selected = selector.fit_transform(X, y)
   ```

3. Use stratified sampling:
   ```python
   from sklearn.model_selection import train_test_split
   X_sample, _, y_sample, _ = train_test_split(
       X, y, train_size=0.1, stratify=y
   )
   ```

## Data Privacy

⚠️ **Important**: Never commit sensitive or proprietary data to version control!

Add to `.gitignore`:
```
data/*.csv
data/*.txt
data/*.xlsx
!data/README.md
```
