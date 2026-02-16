# Detailed Methodology

This document provides a comprehensive explanation of the necessity and sufficiency framework for evaluating XAI robustness.

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Forward Counterfactual Generation](#forward-counterfactual-generation)
3. [Necessity Calculation](#necessity-calculation)
4. [Sufficiency Calculation](#sufficiency-calculation)
5. [Global Score Aggregation](#global-score-aggregation)
6. [XAI Robustness Evaluation](#xai-robustness-evaluation)
7. [Mathematical Formulation](#mathematical-formulation)

---

## Theoretical Foundation

### Philosophical Background

The concepts of necessity and sufficiency come from philosophical logic:

**Necessary Condition**: A condition A is necessary for B if the absence of A guarantees the absence of B.
- Example: "Oxygen is necessary for human life"
- In logic: B → A (if B then A must be true)

**Sufficient Condition**: A condition A is sufficient for B if the presence of A guarantees the presence of B.
- Example: "Being a mammal is sufficient to have a spine"
- In logic: A → B (if A then B must be true)

### Application to Machine Learning

In the context of ML model explanations:

**Necessity asks**: "Is this feature necessary for the model to make its current prediction?"
- If changing the feature changes the prediction, the feature is necessary

**Sufficiency asks**: "Is this feature sufficient to drive the model toward a specific prediction?"
- If setting the feature to a target value produces the target prediction, the feature is sufficient

---

## Forward Counterfactual Generation

### Traditional Counterfactual Approach

Traditional methods (e.g., DiCE, Wachter) optimize toward a target:

```
Find x' = argmin ||x' - x||
         subject to f(x') = y_target
```

**Problems**:
- Requires target prediction
- Computationally expensive optimization
- Struggles with high-dimensional sparse data
- May produce unrealistic instances

### Forward Approach

Our forward approach perturbs features and observes outcomes:

```python
def generate_forward_counterfactuals(instance, feature_idx):
    counterfactuals = []
    for _ in range(n_perturbations):
        cf = instance.copy()
        cf[feature_idx] = sample_new_value()
        counterfactuals.append(cf)
    return counterfactuals
```

**Advantages**:
- No target needed
- Simple and fast
- Works with any data type
- No optimization required

---

## Necessity Calculation

### Local Necessity Score

For a single instance k with feature vector x and prediction y*:

```
Local_Necessity(x_j, k) = (Number of CFs where prediction changed) / (Total CFs generated)
```

**Algorithm**:
1. Generate counterfactuals by perturbing feature x_j
2. Get model predictions for all counterfactuals
3. Count how many predictions differ from original y*
4. Divide by total number of counterfactuals

**Interpretation**:
- Score = 1.0: Feature is completely necessary (always changes prediction)
- Score = 0.5: Feature changes prediction half the time
- Score = 0.0: Feature is not necessary (never changes prediction)

### Example

```
Original instance: [x1=0.5, x2=1, x3=0] → prediction = 1

Counterfactuals (perturbing x2):
- [0.5, 0, 0] → prediction = 0  ✓ Changed
- [0.5, 1, 0] → prediction = 1  ✗ Same
- [0.5, 0, 0] → prediction = 0  ✓ Changed
- [0.5, 1, 0] → prediction = 1  ✗ Same

Necessity(x2) = 2/4 = 0.50
```

---

## Sufficiency Calculation

### Local Sufficiency Score

For a reference instance r with target prediction y*, evaluate on opposite class instances:

```
Local_Sufficiency(x_j, r) = (Number achieving target) / (Total opposite class instances)
```

**Algorithm**:
1. Identify reference instance r with feature value x_j = a and prediction y*
2. Select instances K with opposite prediction (y ≠ y*)
3. For each k in K, intervene: k' = k with k'[j] = a
4. Get model predictions for all k'
5. Count how many now predict y*
6. Divide by |K|

**Interpretation**:
- Score = 1.0: Feature is completely sufficient (always produces target)
- Score = 0.5: Feature produces target half the time
- Score = 0.0: Feature is not sufficient (never produces target)

### Example

```
Reference instance: [x1=0.8, x2=1, x3=1] → prediction = 1

Opposite class instances:
- k1: [0.2, 0, 0] → prediction = 0
- k2: [0.3, 0, 1] → prediction = 0  
- k3: [0.1, 1, 0] → prediction = 0

Intervene x2 := 1:
- k1': [0.2, 1, 0] → prediction = 0  ✗ Still 0
- k2': [0.3, 1, 1] → prediction = 1  ✓ Now 1!
- k3': [0.1, 1, 0] → prediction = 0  ✗ Still 0

Sufficiency(x2) = 1/3 = 0.33
```

---

## Global Score Aggregation

### Motivation

Local scores vary across instances. Global scores provide dataset-level feature importance.

### Aggregation Method

```
Global_Necessity(x_j) = (1/N) Σ_{i=1}^N Local_Necessity(x_j, instance_i)

Global_Sufficiency(x_j) = (1/R) Σ_{r=1}^R Local_Sufficiency(x_j, reference_r)
```

Where:
- N = number of sampled instances
- R = number of reference instances (from each class)

### Sampling Strategy

**For Necessity**:
- Randomly sample N instances from test set
- Calculate local necessity for each
- Average across all instances

**For Sufficiency**:
- Sample R references from each predicted class
- For each reference, sample opposite class instances
- Average across all references and opposite instances

### Statistical Considerations

- Use stratified sampling to ensure class balance
- Sample size N should be large enough for stable estimates (typically 100-200)
- Use confidence intervals for small sample sizes
- Consider variance across instances

---

## XAI Robustness Evaluation

### Hypothesis

**If a feature is truly important, it should be:**
1. **Necessary**: Changing it affects the model's decision
2. **Sufficient**: It can drive the model toward specific outcomes

**Therefore**: Features ranked as "important" by LIME/SHAP should have high necessity and sufficiency scores.

### Evaluation Protocol

For each test instance:

1. **Get XAI Rankings**
   ```python
   lime_rankings = get_lime_importance(instance)
   # e.g., [feature_3, feature_1, feature_5, ...]
   ```

2. **Map to Global Scores**
   ```python
   for rank, feature in enumerate(lime_rankings):
       necessity_at_rank[rank].append(global_necessity[feature])
       sufficiency_at_rank[rank].append(global_sufficiency[feature])
   ```

3. **Aggregate Across Instances**
   ```python
   mean_necessity_by_rank = {
       rank_1: mean(necessity_at_rank[1]),
       rank_2: mean(necessity_at_rank[2]),
       ...
   }
   ```

### Robustness Criteria

An XAI method is **robust** if:

✓ Scores decrease monotonically with rank
```
Score[Rank 1] > Score[Rank 2] > Score[Rank 3] > ...
```

✓ Top-ranked features have substantially higher scores
```
Score[Rank 1] >> Score[Rank 5]
```

✓ Pattern is consistent across necessity AND sufficiency

### Non-Robust Patterns

**Pattern 1: Non-Monotonic**
```
Rank 1: 0.45
Rank 2: 0.72  ← Higher than Rank 1!
Rank 3: 0.58
```
*Interpretation*: XAI method is not identifying the most necessary/sufficient features

**Pattern 2: Flat Distribution**
```
Rank 1: 0.51
Rank 2: 0.49
Rank 3: 0.50
Rank 4: 0.48
```
*Interpretation*: Rankings don't correspond to actual importance

---

## Mathematical Formulation

### Notation

- x: feature vector
- x_j: feature j
- x_{-j}: all features except j
- f: trained model
- y*: prediction of interest
- U: context space (dataset)
- CF_i(k): i-th counterfactual of instance k

### Necessity Score

**Local Necessity**:
```
γ_j^N(k) = (1/n) Σ_{i=1}^n 𝟙{f(CF_i(k)) ≠ y* | CF_i differs only in x_j}
```

**Global Necessity**:
```
Γ_j^N = (1/|U|) Σ_{k∈U} γ_j^N(k)
```

Where:
- n = number of counterfactuals per instance
- 𝟙{·} = indicator function (1 if true, 0 if false)

### Sufficiency Score

**Local Sufficiency**:
```
γ_j^S(r) = (1/|K|) Σ_{k∈K} 𝟙{f(k') = y* | k'_j = r_j, k'_{-j} = k_{-j}}
```

**Global Sufficiency**:
```
Γ_j^S = (1/|R|·|K|) Σ_{r∈R} Σ_{k∈K} 𝟙{f(k') = y*}
```

Where:
- R = set of reference instances (f(r) = y*)
- K = set of opposite class instances (f(k) ≠ y*)
- k' = instance k with feature j set to reference value

### Conditional Probabilities

**Necessity** (Probability of prediction change):
```
α = Pr(y ≠ y* | x_j ← a', x_{-j} = b, f(x) = y*)
```

**Sufficiency** (Probability of achieving target):
```
β = Pr(f(x) = y* | x_j ← a)
```

### Relationship to Actual Causality

Based on Halpern (2016), feature x_j = a is an **actual cause** of y* if:
- **AC1** (Necessity): α = 1
- **AC2** (Sufficiency): β = 1

Our framework relaxes these to probabilistic scores (0 ≤ α, β ≤ 1) to handle:
- Non-deterministic models
- Noisy data
- Feature interactions
- Approximate causality

---

## Comparison with Alternative Approaches

### vs. DiCE/Wachter Counterfactuals

| Aspect | DiCE/Wachter | Forward CF (Ours) |
|--------|--------------|-------------------|
| Target needed | ✓ Yes | ✗ No |
| Optimization | ✓ Required | ✗ Not required |
| High-dim data | ✗ Struggles | ✓ Works well |
| Sparse data | ✗ Difficult | ✓ Handles |
| Speed | Slow | Fast |
| Causal model | Not required | Not required |

### vs. Probabilistic Causal Models

| Aspect | Galhotra et al. | Our Approach |
|--------|-----------------|--------------|
| Causal model | ✓ Required | ✗ Not required |
| Domain knowledge | ✓ Needed | ✗ Optional |
| Structural equations | ✓ Required | ✗ Not required |
| Model-agnostic | ✗ Limited | ✓ Fully |

### vs. Feature Attribution Only

| Aspect | LIME/SHAP Alone | + Our Framework |
|--------|-----------------|-----------------|
| Causal grounding | ✗ No | ✓ Yes |
| Robustness check | ✗ No | ✓ Yes |
| Disagreement handling | ✗ No | ✓ Yes |
| Validation | ✗ Difficult | ✓ Principled |

---

## Computational Complexity

### Time Complexity

**Necessity Calculation**:
- Per instance: O(n_pert × n_features × t_pred)
- Global: O(N × n_pert × n_features × t_pred)

Where:
- n_pert = number of perturbations
- n_features = number of features
- t_pred = model prediction time
- N = number of sampled instances

**Sufficiency Calculation**:
- Per reference: O(|K| × t_pred)
- Global: O(R × |K| × t_pred)

Where:
- R = number of reference instances
- |K| = size of opposite class

**Total Complexity**: O(N × n_pert × n_features × t_pred)
- Linear in dataset size
- Linear in number of features
- Parallelizable across instances

### Space Complexity

- O(n_pert × n_features): store counterfactuals
- O(N × n_features): store scores
- Overall: O(max(n_pert, N) × n_features)

### Practical Considerations

**Typical Runtime** (Breast Cancer dataset, 30 features):
- Necessity: ~5 minutes (100 instances, 50 perturbations)
- Sufficiency: ~3 minutes (100 references)
- LIME robustness: ~10 minutes (50 test instances)
- SHAP robustness: ~15 minutes (50 test instances)

**Scalability**:
- Can handle thousands of features
- Parallelizable across instances
- Memory efficient (stream processing)
- Suitable for production use

---

## References

1. Chowdhury et al. (2023). "Explaining Explainers: Necessity and Sufficiency in Tabular Data"
2. Chowdhury et al. (2025). "A unified framework for evaluating the robustness of machine-learning interpretability"
3. Halpern (2016). "Actual Causality"
4. Pearl (2009). "Causality: Models, Reasoning and Inference"
5. Swartz (1997). "The Concepts of Necessary Conditions and Sufficient Conditions"
6. Wachter et al. (2017). "Counterfactual Explanations without Opening the Black Box"
7. Mothilal et al. (2020). "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"
