# Methodology

## 3.1 Problem Formulation

We formulate the energy-efficient neural architecture search as a constrained multi-objective optimization problem. Given a search space of ResNet architectures **A** and approximate multiplier configurations **M**, we seek to find the optimal configuration **(a*, m*)** that maximizes classification accuracy while minimizing energy consumption under formal constraints:

```
(a*, m*) = argmax_{a∈A, m∈M} f(a, m)
subject to:
    Accuracy(a, m) ≥ Qc    (Quality Constraint)
    Energy(a, m) ≤ Ec      (Energy Constraint)
```

where **Qc** represents the minimum acceptable accuracy threshold and **Ec** denotes the maximum permissible energy budget in microjoules (µJ). This formulation enables systematic exploration of the accuracy-energy Pareto frontier while ensuring deployability constraints are satisfied through Signal Temporal Logic (STL) monitoring.

## 3.2 Experimental Evolution and Search Space Design

Our methodology evolved through three distinct phases, progressively expanding the search space based on empirical insights:

### Phase 1: Initial Multiplier Exploration (Nov 20-24, 2025)

**Objective**: Identify safe approximate multipliers for heterogeneous approximation

We began with a constrained search space to establish baseline performance and identify approximate multipliers with acceptable error characteristics:

**Fixed Architecture**:
- ResNet-18 adapted for CIFAR-10: 4 stages, 2 blocks per stage
- Filters: [64, 128, 256, 512]
- Total parameters: ~11M

**Multiplier Pairs Explored**:
1. {1JJQ (Exact), 2V0 (MAE=0.0015%)} - Ultra-low error baseline
2. {1JJQ (Exact), LK8 (MAE=0.0046%)} - Low error evaluation
3. {1JJQ (Exact), R92 (MAE=0.0170%)} - Medium error evaluation
4. {1JJQ (Exact), 0AB (MAE=0.0570%)} - High error evaluation

**Constraints**: Qc = 0.89, Ec = 500 µJ (initial conservative bound)

This phase revealed critical findings:
- Multipliers with MAE ≤ 0.02% maintain >90% accuracy
- Multipliers with MAE > 0.02% cause catastrophic failures (>65% of trials)
- Heterogeneous assignment (mixing exact and approximate) outperforms homogeneous approximation
- Energy constraint of 500 µJ was too restrictive; relaxed to 5000 µJ

### Phase 2: Expanded Multiplier Search (Nov 25-Dec 1, 2025)

**Objective**: Explore full spectrum of approximate multipliers with relaxed energy constraints

Based on Phase 1 insights, we expanded the multiplier set while maintaining the ResNet-18 architecture:

**Full Multiplier Set (8 options)**:

| Multiplier | Power (mW) | Delay (ns) | MAE (%) | Energy/MAC (pJ) | Category |
|------------|-----------|------------|---------|-----------------|----------|
| mul8u_1JJQ | 0.391 | 1.43 | 0.0000 | 569 | Exact (baseline) |
| mul8u_2V0  | 0.386 | 1.42 | 0.0015 | 558 | Very low error |
| mul8u_LK8  | 0.370 | 1.40 | 0.0046 | 528 | Low error |
| mul8u_17C8 | 0.355 | 1.39 | 0.0090 | 548 | Low error |
| mul8u_R92  | 0.345 | 1.41 | 0.0170 | 496 | Medium error |
| mul8u_18UH | 0.330 | 1.42 | 0.0250 | 478 | Medium-high error |
| mul8u_0AB  | 0.302 | 1.44 | 0.0570 | 445 | High error |
| mul8u_197B | 0.206 | 1.50 | 0.1200 | 315 | Very high error |

All multipliers are 8×8 unsigned from EvoApproxLib, synthesized using PDK45 technology (45nm, 1V, 25°C).

**Constraints**: Qc = 0.89, Ec = 5000 µJ

This phase confirmed:
- R92 (MAE=0.017%) is the optimal high-performance multiplier
- 2V0 and LK8 provide excellent accuracy with minimal energy savings
- Heterogeneous patterns (Med→Exact→Med→Exact) achieve best accuracy

### Phase 3: Architecture Search with Asymmetric Patterns (Dec 1-2, 2025)

**Objective**: Jointly optimize architecture topology and multiplier assignment

Having identified promising multipliers, we expanded the architecture search space to explore depth distribution strategies:

**Variable Architecture Configurations**:
- **Number of stages**: 3 (CIFAR-10 standard, reduced from 4)
- **Base filters**: 16 (yielding [16, 32, 64] filters per stage)
- **Blocks per stage**: 11 distinct patterns

**Block Distribution Patterns**:
1. **Symmetric**: [3,3,3], [5,5,5], [7,7,7], [9,9,9] - uniform depth
2. **Progressive**: [3,4,5], [4,5,6], [5,7,9] - increasing depth
3. **Asymmetric**: [6,4,2] (early-heavy), [2,4,6] (late-heavy), [3,6,3], [4,6,4] (hourglass)

This yields architectures from ResNet-20 (20 layers) to ResNet-56 (56 layers) with 0.27M-0.85M parameters.

**Rationale for 3 Stages**:
- CIFAR-10 (32×32 input) requires only 3 downsampling steps: 32→16→8
- 4 stages would produce 4×4 feature maps (too small for 3×3 convolutions)
- Matches original He et al. (2016) CIFAR-10 ResNet design

**Total Search Space**: 11 architectures × 8³ = 5,632 configurations

**Rationale for Asymmetric Patterns**:
Asymmetric block distributions enable architectural diversity and may discover that:
- Early stages (high-resolution feature maps) benefit from more capacity
- Late stages (low-resolution, high-channel) may require less depth
- Hourglass patterns balance capacity where needed most

## 3.3 Bayesian Neural Architecture Search

### 3.3.1 Gaussian Process Surrogate Model

To efficiently navigate the large search space (5,632 configurations) without exhaustive evaluation, we employ Bayesian optimization with a Gaussian Process (GP) surrogate model:

```
f(x) ~ GP(μ(x), k(x, x'))
```

where:
- **x** ∈ ℝ¹⁰ is the encoded architecture representation
- **μ(x)** is the mean function (assumed zero)
- **k(x, x')** is the Matérn-2.5 covariance kernel

The Matérn kernel models smooth but non-differentiable relationships between architecture parameters and performance:

```
k(x, x') = σ² (1 + √5r + 5r²/3) exp(-√5r)
where r = ||x - x'|| / ℓ
```

Hyperparameters **σ²** (signal variance) and **ℓ** (length scale) are learned via maximum likelihood estimation using scikit-learn's `GaussianProcessRegressor` with 5 random restarts.

### 3.3.2 Architecture Encoding

Each configuration is encoded as a 10-dimensional numerical vector:

```
x = [num_stages, b₁, b₂, b₃, b₄, base_filters, m₁, m₂, m₃, m₄]
```

where:
- **b_i** = blocks in stage *i* (0 if stage doesn't exist)
- **m_i** = multiplier index (0-7) for stage *i* (0 if stage doesn't exist)
- **base_filters** = initial channel count (16 for CIFAR-10)

This representation captures both architectural topology and heterogeneous approximation strategy, enabling the GP to learn complex interactions between design choices.

### 3.3.3 Acquisition Function and Optimization

We use the Expected Improvement (EI) acquisition function to balance exploration (trying novel configurations) and exploitation (refining known good regions):

```
EI(x) = (μ(x) - f(x⁺) - ξ) Φ(Z) + σ(x) φ(Z)
```

where:
- **f(x⁺)** = current best observed value
- **μ(x)**, **σ(x)** = GP predictive mean and standard deviation
- **Z** = (μ(x) - f(x⁺) - ξ) / σ(x)
- **ξ** = exploration parameter (0.01)
- **Φ**, **φ** = standard normal CDF and PDF

At each iteration, we:
1. Generate 100 random candidate architectures
2. Evaluate EI for each candidate using the current GP model
3. Select the candidate with maximum EI for evaluation

### 3.3.4 Optimization Procedure

**Algorithm 1: Bayesian NAS with STL Constraints**
```
Input: Search space A × M, constraints Qc and Ec, trials N = 20
Output: Optimal architecture (a*, m*) and Pareto set P

1. Initialize: D₀ = ∅  (empty observation set)
2.
3. // Phase 1: Random Initialization (N_init = 5)
4. For i = 1 to 5:
5.     Sample (aᵢ, mᵢ) ~ Uniform(A × M)
6.     yᵢ = Train-and-Evaluate(aᵢ, mᵢ)
7.     D_i = D_{i-1} ∪ {(encode(aᵢ, mᵢ), yᵢ)}
8. End For
9.
10. // Phase 2: Bayesian Optimization (15 iterations)
11. For i = 6 to 20:
12.     Fit GP: p(f | D_{i-1}) using Matérn kernel
13.     Generate 100 candidates: C = {sample() for _ in range(100)}
14.     (aᵢ, mᵢ) = argmax_{c ∈ C} EI(encode(c) | D_{i-1})
15.     yᵢ = Train-and-Evaluate(aᵢ, mᵢ)
16.     D_i = D_{i-1} ∪ {(encode(aᵢ, mᵢ), yᵢ)}
17. End For
18.
19. // Phase 3: Pareto Analysis
20. P = Pareto-Optimal-Set(D_20)
21. (a*, m*) = Best-Accuracy(P, subject to STL constraints)
22. Return (a*, m*), P
```

This approach requires only **20 evaluations** compared to 5,632 possible configurations, achieving a **282× reduction** in search cost while maintaining optimality guarantees.

## 3.4 Energy Modeling

### 3.4.1 MAC-based Analytical Model

We employ a fine-grained MAC (Multiply-Accumulate) based energy model that accounts for both computational primitives and architectural parameters. Total network energy is:

```
E_total = Σ_{s=1}^{S} Σ_{b=1}^{B_s} (E_mul(m_s) + E_add) × MAC_count(s, b)
```

where:
- **S** = number of stages (3 for CIFAR-10 ResNet)
- **B_s** = number of residual blocks in stage *s* (variable)
- **m_s** = multiplier assigned to stage *s*
- **E_mul(m_s)** = energy per multiplication for multiplier *m_s* (picojoules)
- **E_add** = energy per addition (10 pJ for 8-bit adder, constant)
- **MAC_count(s, b)** = MAC operations in block *b* of stage *s*

### 3.4.2 MAC Operation Counting

Each residual block contains two 3×3 convolutions. For a block in stage *s*:

```
MAC_count(s, b) = Σ_{conv=1}^{2} (C_in × C_out × K² × H_s × W_s)
```

where:
- **C_in**, **C_out** = input and output channel counts
- **K** = kernel size (3 for all ResNet blocks)
- **H_s**, **W_s** = spatial dimensions at stage *s*

Feature map dimensions follow the ResNet downsampling schedule:
- Stage 0 (no downsampling): 32×32 = 1024 spatial locations
- Stage 1 (stride-2 downsample): 16×16 = 256 spatial locations
- Stage 2 (stride-2 downsample): 8×8 = 64 spatial locations

The first block of each stage (except stage 0) applies stride-2 convolution, halving spatial dimensions. Subsequent blocks maintain dimensions.

### 3.4.3 Per-Multiplier Energy Calculation

Energy per MAC operation is derived from synthesized power and delay characteristics:

```
E_MAC(m) = (P_mul(m) × t_mul(m)) + (P_add × t_add)
```

where power **P** is in watts and delay **t** is in seconds. Converting from (mW, ns) to picojoules:

```
E_pJ = (P_mW × 10⁻³ W/mW) × (t_ns × 10⁻⁹ s/ns) × 10¹² pJ/J
```

Example calculation for mul8u_R92:
```
E_mul = (0.345 mW × 10⁻³) × (1.41 ns × 10⁻⁹) × 10¹² = 486 pJ
E_add = (0.050 mW × 10⁻³) × (0.20 ns × 10⁻⁹) × 10¹² = 10 pJ
E_MAC = 496 pJ
```

The model enables accurate stage-level energy attribution and supports heterogeneous approximation optimization.

### 3.4.4 Total Energy Aggregation

After computing energy for all MACs, we convert from picojoules to microjoules:

```
E_total_µJ = E_total_pJ / 10⁶
```

For ResNet-20 (3 stages, [3,3,3] blocks) with all-exact multipliers:
- Stage 0: ~65 µJ (115M MACs × 569 pJ/MAC)
- Stage 1: ~150 µJ (264M MACs × 569 pJ/MAC)
- Stage 2: ~301 µJ (528M MACs × 569 pJ/MAC)
- **Total: ~1116 µJ**

Approximate multipliers reduce energy proportionally to their power savings (e.g., R92 saves ~13%).

## 3.5 Signal Temporal Logic Constraint Monitoring

### 3.5.1 STL Specification

We formalize design constraints using Signal Temporal Logic (STL), enabling formal verification of deployability requirements. For each candidate architecture, we define:

```
φ = (accuracy ≥ Qc) ∧ (energy ≤ Ec)
```

where:
- **Qc** = quality constraint (89% for CIFAR-10, 2% below exact baseline of 91%)
- **Ec** = energy constraint (5000 µJ, sufficient for approximate designs)
- **∧** = logical AND operator

This specification is evaluated at steady-state (after training convergence).

### 3.5.2 Robustness Semantics

STL robustness quantifies constraint satisfaction strength:

```
ρ(φ) = min(ρ_acc, ρ_energy)
```

where:
- **ρ_acc = accuracy - Qc** (accuracy margin above threshold)
- **ρ_energy = Ec - energy** (energy margin below limit)

The robustness value provides actionable feedback:
- **ρ(φ) > 0**: Constraints satisfied with margin **ρ**
- **ρ(φ) = 0**: Boundary case (exactly at limit)
- **ρ(φ) < 0**: Constraints violated, |ρ| indicates severity

For example, a configuration with 92% accuracy and 1000 µJ energy:
```
ρ_acc = 0.92 - 0.89 = 0.03
ρ_energy = 5000 - 1000 = 4000
ρ(φ) = min(0.03, 4000) = 0.03 (accuracy is the limiting factor)
```

This metric serves as the objective function when `objective='stl_robustness'` in Bayesian optimization, enabling direct optimization of deployability with maximum safety margin.

### 3.5.3 Implementation

We implement STL monitoring using RTAMT (Real-Time Approximate Monitoring Tool) library:

```python
from rtamt import StlDiscreteTimeSpecification

spec = StlDiscreteTimeSpecification()
spec.declare_var('accuracy', 'float')
spec.declare_var('energy', 'float')
spec.spec = f'accuracy >= {Qc} and energy <= {Ec}'
spec.parse()
spec.pastify()
robustness = spec.update(0, [('accuracy', acc), ('energy', e)])
```

This approach provides formal guarantees that selected architectures meet deployment requirements while maximizing performance margins.

## 3.6 Model Training and Evaluation

### 3.6.1 Two-Phase Evaluation Strategy

To isolate approximation effects from training dynamics, we employ a two-phase evaluation protocol:

**Phase 1: Exact Training**
1. Build architecture with standard `Conv2D` layers (exact multiplication)
2. Train on CIFAR-10 with data augmentation and learning rate schedule
3. Save trained weights to temporary file
4. Evaluate exact accuracy on 10,000-image test set

**Phase 2: Approximate Inference**
1. Build identical architecture with `FakeApproxConv2D` layers (approximate multiplication)
2. Load weights from Phase 1 (weight transfer without retraining)
3. Evaluate approximate accuracy on test set
4. Compute energy consumption using MAC-based analytical model
5. Evaluate STL robustness

This methodology ensures that accuracy degradation is **solely due to approximate multiplier errors**, not suboptimal training with noisy gradients. Training with exact multipliers and evaluating with approximate ones mimics real deployment scenarios where models are trained on high-precision GPUs and deployed on low-power approximate accelerators.

### 3.6.2 Training Configuration

We follow ResNet training best practices for CIFAR-10:

**Optimizer**: Adam with piecewise-constant learning rate schedule
- Epochs 0-39: lr = 1e-3 (initial training)
- Epochs 40-59: lr = 1e-4 (fine-tuning)
- Epochs 60+: lr = 1e-5 (convergence)

**Data Augmentation** (applied to training set only):
- Random rotation: ±15°
- Random horizontal shift: 10% of width
- Random vertical shift: 10% of height
- Random horizontal flip: 50% probability
- Implemented via `ImageDataGenerator` from Keras

**Regularization**:
- Batch normalization after each convolution
- L2 weight decay: 1e-4 (implicit in Adam)
- Dropout: Not used (BatchNorm provides sufficient regularization)

**Data Split**:
- Training: 45,000 images (90% of original training set)
- Validation: 5,000 images (10% of original training set, for learning rate tuning)
- Test: 10,000 images (held-out evaluation set)

**Hardware Configuration**:
- GPU: NVIDIA V100 (32GB VRAM)
- Batch size: 256 (optimized for V100 memory and throughput)
- Data loading: 4 parallel workers with threading (avoids "too many files open" error)
- Mixed precision: Not used (full FP32 to avoid approximation in training phase)
- Training duration: ~17 minutes per trial (60 epochs × 176 steps/epoch)

**Normalization**: Input images scaled to [0, 1] by dividing by 255

### 3.6.3 Approximate Multiplier Integration

We implement approximate convolution using a custom TensorFlow layer that emulates approximate multiplier behavior through lookup table injection:

```python
from keras.layers.fake_approx_convolutional import FakeApproxConv2D

class FakeApproxConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, mul_map_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mul_table = self.load_multiplier_table(mul_map_file)

    def call(self, inputs):
        # Perform standard convolution (exact computation)
        outputs = super().call(inputs)
        # Apply approximate multiplier error via lookup table
        outputs = self.apply_approximation(outputs, self.mul_table)
        return outputs
```

The multiplier lookup table is a 256×256 matrix (for 8×8 multipliers) mapping exact products to approximate outputs. This software emulation enables rapid architecture evaluation without FPGA synthesis or hardware implementation, while accurately modeling the error characteristics of physical approximate multipliers.

**Integration Points**:
- Residual blocks: Both 3×3 convolutions use the assigned stage multiplier
- Skip connections: Kept exact (standard `Conv2D`) for stability
- Initial convolution: Kept exact for feature extraction quality
- Final dense layer: Kept exact for classification precision

This selective approximation strategy follows hardware design best practices, approximating only the computationally dominant operations (residual block convolutions account for >95% of MACs).

## 3.7 Pareto Optimality Analysis

After NAS completion, we identify the Pareto frontier—architectures where no other design achieves **both** higher accuracy **and** lower energy. This multi-objective analysis provides designers with a set of optimal tradeoff points for deployment selection.

**Dominance Relation**: Architecture **(a₁, m₁)** dominates **(a₂, m₂)** if and only if:

```
Accuracy(a₁, m₁) ≥ Accuracy(a₂, m₂)  ∧
Energy(a₁, m₁) ≤ Energy(a₂, m₂)      ∧
[strict inequality in at least one objective]
```

**Pareto Frontier Identification** (pairwise dominance check):

```python
def check_pareto_optimal(results):
    pareto_indices = []
    for i, result_i in enumerate(results):
        is_dominated = False
        for j, result_j in enumerate(results):
            if i == j:
                continue
            # Check if j dominates i
            if (result_j['approx_accuracy'] > result_i['approx_accuracy'] and
                result_j['energy'] < result_i['energy']):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(i)
    return pareto_indices
```

This yields a Pareto set **P** ⊆ **Results** of non-dominated architectures. Designers can then select from **P** based on application-specific priorities:
- **Accuracy-critical**: Choose max(accuracy) ∈ P subject to energy budget
- **Energy-critical**: Choose min(energy) ∈ P subject to accuracy requirement
- **Balanced**: Choose architecture with best harmonic mean of normalized objectives

The Pareto analysis complements STL constraint satisfaction by offering multiple deployment-ready options along the optimal accuracy-energy tradeoff curve.

## 3.8 Evaluation Metrics

We assess each candidate architecture using seven complementary metrics:

1. **Exact Accuracy** (Acc_exact): Test set accuracy with exact multipliers (baseline)
2. **Approximate Accuracy** (Acc_approx): Test set accuracy with approximate multipliers
3. **Accuracy Drop** (ΔAcc): Acc_exact - Acc_approx (approximation overhead)
4. **Total Energy** (E_total): Network energy in microjoules (µJ)
5. **Energy per Stage**: Stage-wise energy breakdown [E₀, E₁, E₂] for heterogeneity analysis
6. **STL Robustness** (ρ): Constraint satisfaction margin (positive = satisfied)
7. **Pareto Optimality**: Binary indicator of Pareto frontier membership

Additionally, we track:
- **Training history**: Epoch-wise training/validation loss and accuracy curves
- **Convergence behavior**: Final epoch learning rate and loss values
- **Search efficiency**: EI values and GP predictive uncertainty over iterations

## 3.9 Experimental Parameters

**Dataset**: CIFAR-10
- 50,000 training images (45,000 train + 5,000 validation)
- 10,000 test images
- 32×32 RGB, 10 classes

**Search Configuration**:
- Algorithm: Bayesian optimization with Matérn-2.5 kernel
- Trials per experiment: 20 (5 random init + 15 Bayesian)
- Training epochs per trial: 60
- Total GPU hours per experiment: ~5.7 hours (20 trials × 17 min/trial)

**STL Constraints**:
- Quality constraint (Qc): 0.89 (89% minimum accuracy)
- Energy constraint (Ec): 5000 µJ (maximum energy budget)

**Baseline Comparisons**:
- Exact ResNet-20: All mul8u_1JJQ (exact) multipliers
- Homogeneous approximation: Single multiplier across all stages
- Random heterogeneous: Uniform random multiplier assignment per stage

**Reproducibility**:
- Fixed random seeds: TensorFlow (42), NumPy (42), Python (42)
- Deterministic GPU operations enabled
- Code and multiplier tables publicly available
- All experiments logged with configuration snapshots

This comprehensive methodology enables rigorous evaluation of approximate DNN design strategies while maintaining scientific reproducibility.
