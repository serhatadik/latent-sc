```
# Joint Sparse Superposition Reconstruction: Theory and Implementation

This document provides a comprehensive mathematical treatment of the joint sparse superposition reconstruction algorithm, an alternative approach to transmitter localization based on convex optimization and sparse recovery.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Linear Superposition Model](#3-linear-superposition-model)
4. [Whitening Transformation](#4-whitening-transformation)
5. [Sparse Optimization](#5-sparse-optimization)
6. [Implementation](#6-implementation)
7. [Comparison with Likelihood-Based Approach](#7-comparison-with-likelihood-based-approach)
8. [Parameter Selection](#8-parameter-selection)

---

## 1. Overview

### Problem Statement

**Given:**
- $M$ sensors with observed received power measurements
- $N$ grid points (potential transmitter locations)
- Prior knowledge of propagation environment (path loss model, shadowing statistics)

**Find:**
- Sparse transmit power field $\mathbf{t} \in \mathbb{R}^{N}$ indicating:
  - **Where** transmitter(s) are located (non-zero entries)
  - **How much** power each transmitter emits (magnitude of non-zero entries)

### Key Insight

Instead of the likelihood-based two-stage approach:
1. Optimize transmit power at each location independently
2. Compute likelihood to weight hypotheses
3. Marginalize to get signal strength

We formulate as a **single-stage joint optimization** with **explicit sparsity constraint**:

$$\hat{\mathbf{t}} = \arg\min_{\mathbf{t}\ge 0} \underbrace{\|\mathbf{W}(\log_{10}(\mathbf{A}_{\text{model}}\mathbf{t} + \epsilon) - \log_{10}(\mathbf{p} + \epsilon))\|_{2}^{2}}_{\text{Data Fidelity (Log Domain)}} + \underbrace{\lambda \|\mathbf{\Omega}\mathbf{t}\|_{1}}_{\text{Weighted Sparsity Penalty}}$$

This is a **non-convex optimization problem** due to the logarithm in the data fidelity term.

---

## 2. Mathematical Formulation

### Variables and Notation

| Symbol | Dimension | Description | Units |
|--------|-----------|-------------|-------|
| $\mathbf{t}$ | $\mathbb{R}^{N}$ | Transmit power field (decision variable) | mW (linear) |
| $\mathbf{p}$ | $\mathbb{R}^{M}$ | Observed received powers | mW (linear) |
| $\mathbf{p}^{th}$ | $\mathbb{R}^{M}$ | Theoretical received powers | mW (linear) |
| $\mathbf{A}_{\text{model}}$ | $\mathbb{R}^{M \times N}$ | Propagation matrix (path gains) | Dimensionless |
| $\mathbf{V}$ | $\mathbb{R}^{M \times M}$ | Covariance matrix | dB² |
| $\mathbf{W}$ | $\mathbb{R}^{M \times M}$ | Whitening matrix | dB⁻¹ |
| $\mathbf{\Omega}$ | $\mathbb{R}^{N \times N}$ | Diagonal weighting matrix | - |
| $\lambda$ | $\mathbb{R}_{\ge 0}$ | Regularization parameter | mW |
| $M$ | Integer | Number of sensors | - |
| $N$ | Integer | Number of grid points | - |

### Complete Problem Formulation

$$\begin{aligned}
\hat{\mathbf{t}} &= \arg\min_{\mathbf{t}} \quad \|\mathbf{W}(\log_{10}(\mathbf{A}_{\text{model}}\mathbf{t} + \epsilon) - \log_{10}(\mathbf{p} + \epsilon))\|_{2}^{2} + \lambda \|\mathbf{\Omega}\mathbf{t}\|_{1} \\
&\text{subject to} \quad \mathbf{t} \ge \mathbf{0}
\end{aligned}$$

**Objective Function Components:**

1. **Data Fidelity Term:** $\|\mathbf{W}(\log_{10}(\mathbf{A}_{\text{model}}\mathbf{t} + \epsilon) - \log_{10}(\mathbf{p} + \epsilon))\|_{2}^{2}$
   - Measures how well the model fits observations in the **logarithmic (dB) domain**
   - Matches the physical reality that path loss and shadowing are additive in dB
   - $\epsilon$ is a small constant for numerical stability

2. **Weighted Sparsity Penalty:** $\lambda \|\mathbf{\Omega}\mathbf{t}\|_{1} = \lambda \sum_{i=1}^{N} \Omega_{ii} |t_{i}|$
   - $\mathbf{\Omega}$ is a diagonal weighting matrix where $\Omega_{ii} = \|\mathbf{a}_{i}\|_{2}$ (L2-norm of the $i$-th column of $\mathbf{A}_{\text{model}}$).
   - **Motivation:** This weighting cancels out the path loss bias. Transmitters close to sensors have large column norms (large $\Omega_{ii}$), so they are penalized more heavily. This forces the optimizer to choose locations that match the spatial pattern across multiple sensors rather than just exploiting the power gain of one nearby sensor.
   - $\lambda$ controls the overall sparsity-accuracy trade-off.

3. **Non-negativity Constraint:** $\mathbf{t} \ge \mathbf{0}$
   - Physical constraint (power cannot be negative)
   - Makes problem convex and ensures meaningful solution

4. **Sensor Exclusion Zone:** $t_i = 0 \quad \forall i \in \mathcal{E}$
   - $\mathcal{E} = \{i \mid \exists j : \text{dist}(\text{loc}_i, \text{sensor}_j) < R_{\text{excl}}\}$
   - **Motivation:** The solver may find a "trivial solution" by placing a transmitter directly on top of a sensor to perfectly match its power reading, while setting others to zero. This is mathematically valid but physically unlikely (we assume we are looking for unknown transmitters, not the sensors themselves).
   - **Implementation:** We enforce a hard constraint that transmit power must be zero within a radius $R_{\text{excl}}$ (default 50m) of any sensor.

---

## 3. Linear Superposition Model

### Theoretical Foundation

The received power at sensor $j$ is the **superposition** of contributions from all potential transmitter locations:

$$p_{j}^{th} = \sum_{i=1}^{N} A_{ji} \, t_{i}$$

or in matrix form:

$$\mathbf{p}^{th} = \mathbf{A}_{\text{model}} \, \mathbf{t}$$

### Propagation Matrix Construction

Each element of $\mathbf{A}_{\text{model}}$ represents the **linear path gain** from grid point $i$ to sensor $j$:

$$A_{ji} = G(d_{ij})$$

where $G(d)$ is derived from the log-distance path loss model.

**Derivation:**

Starting from the log-distance model in dB:
$$P_{\text{rx}}[\text{dBm}] = P_{\text{tx}}[\text{dBm}] - \text{PL}[\text{dB}]$$

$$P_{\text{rx}}[\text{dBm}] = P_{\text{tx}}[\text{dBm}] - \left(p_{i0} + 10 n_{p} \log_{10}(d/d_{0})\right)$$

Converting to linear scale (mW):
$$P_{\text{rx}}[\text{mW}] = P_{\text{tx}}[\text{mW}] \cdot 10^{-\text{PL}[\text{dB}]/10}$$

Therefore, the linear path gain is:
$$\boxed{A_{ji} = 10^{-(p_{i0} + 10 n_{p} \log_{10}(d_{ij}/d_{0}))/10}}$$

**Properties:**
- $0 \le A_{ji} \le 1$ (attenuation, no amplification)
- $A_{ji} \to 0$ as $d_{ij} \to \infty$ (far-field decay)
- $A_{ji}$ depends only on distance (simplified model, no shadowing)

### Code Implementation

**Function:** `compute_propagation_matrix()`
**Location:** `src/sparse_reconstruction/propagation_matrix.py`

```python
def compute_linear_path_gain(distance, pi0=0, np_exponent=2, di0=1):
    """Linear path gain from distance."""
    path_loss_dB = pi0 + 10 * np_exponent * np.log10(distance / di0)
    return 10 ** (-path_loss_dB / 10)

def compute_propagation_matrix(sensor_locations, map_shape, scale, np_exponent):
    """Build M×N propagation matrix."""
    M = len(sensor_locations)
    N = height * width
    A = np.zeros((M, N))

    for j, sensor in enumerate(sensor_locations):
        for i in range(N):
            distance = euclidean_distance(grid_point_i, sensor, scale)
            A[j, i] = compute_linear_path_gain(distance, np_exponent=np_exponent)

    return A
```

**Key Points:**
- Vectorized implementation for efficiency
- Returns linear gains (not dB)
- Memory: $M \times N \times 8$ bytes (float64)
- For 100×100 grid, 10 sensors: ~80 KB

---

## 4. Whitening Transformation

### Motivation

Sensor measurements have **correlated errors** due to shadowing. The covariance matrix $\mathbf{V}$ models this correlation:

$$V_{kl} = \sigma^{2} \exp\left(-\frac{d_{kl}}{\delta_{c}}\right)$$

Direct least squares $\|\mathbf{A}\mathbf{t} - \mathbf{p}\|_{2}^{2}$ treats all errors equally. **Whitening** transforms to uncorrelated errors, allowing standard $\ell_2$ norm.

### Mathematical Definition

The **whitening matrix** $\mathbf{W}$ is defined as:

$$\mathbf{W} = \mathbf{V}^{-1/2}$$

**Properties:**
1. $\mathbf{W} \mathbf{V} \mathbf{W}^{T} = \mathbf{I}$ (whitened covariance is identity)
2. Whitened errors are uncorrelated: $\text{Cov}(\mathbf{W}\boldsymbol{\epsilon}) = \mathbf{I}$
3. Equivalence: $\|\mathbf{e}\|_{\mathbf{V}^{-1}}^{2} = \mathbf{e}^{T}\mathbf{V}^{-1}\mathbf{e} = \|\mathbf{W}\mathbf{e}\|_{2}^{2}$

### Computation via Cholesky Decomposition

**Step 1:** Cholesky factorization of $\mathbf{V}$:
$$\mathbf{V} = \mathbf{L} \mathbf{L}^{T}$$

where $\mathbf{L}$ is lower triangular.

**Step 2:** Whitening matrix:
$$\mathbf{W} = \mathbf{L}^{-1}$$

**Verification:**
$$\mathbf{W} \mathbf{V} \mathbf{W}^{T} = \mathbf{L}^{-1} (\mathbf{L} \mathbf{L}^{T}) (\mathbf{L}^{-1})^{T} = \mathbf{L}^{-1} \mathbf{L} \mathbf{L}^{T} (\mathbf{L}^{T})^{-1} = \mathbf{I}$$

### Alternative Methods

1. **SVD (Robust):**
   - $\mathbf{V} = \mathbf{U} \mathbf{S} \mathbf{U}^{T}$
   - $\mathbf{W} = \mathbf{U} \mathbf{S}^{-1/2} \mathbf{U}^{T}$
   - Handles near-singular matrices better

2. **Eigenvalue Decomposition:**
   - $\mathbf{V} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{T}$
   - $\mathbf{W} = \mathbf{Q} \mathbf{\Lambda}^{-1/2} \mathbf{Q}^{T}$
   - Middle ground between speed and robustness

### Code Implementation

**Function:** `compute_whitening_matrix()`
**Location:** `src/sparse_reconstruction/whitening.py`

```python
def compute_whitening_matrix(cov_matrix, method='cholesky'):
    """Compute W = V^(-1/2)."""
    if method == 'cholesky':
        L = np.linalg.cholesky(cov_matrix)
        W = np.linalg.inv(L)
    elif method == 'svd':
        U, s, _ = np.linalg.svd(cov_matrix)
        W = U @ np.diag(1.0 / np.sqrt(s)) @ U.T
    return W
```

**Computational Complexity:**
- Cholesky: $O(M^{3})$
- SVD: $O(M^{3})$
- For $M = 10$: ~microseconds

---

## 5. Sparse Optimization

### Problem Structure

After whitening, the problem becomes:

$$\min_{\mathbf{t}\ge 0} \|\tilde{\mathbf{A}}\mathbf{t} - \tilde{\mathbf{p}}\|_{2}^{2} + \lambda \|\mathbf{t}\|_{1}$$

where:
- $\tilde{\mathbf{A}} = \mathbf{W} \mathbf{A}_{\text{model}}$ (whitened propagation matrix)
- $\tilde{\mathbf{p}} = \mathbf{W} \mathbf{p}$ (whitened observations)

**This is a non-negative LASSO problem.**

### Non-Convexity

The problem is **non-convex** because:
1. The composition of the convex quadratic norm with the concave logarithm function results in a non-convex objective.
2. Specifically, $\log_{10}(\mathbf{A}\mathbf{t})$ is concave, and $\|\cdot\|^2$ is convex.

**Implication:** The solver may converge to a **local minimum** rather than the global minimum. The solution depends on the initialization.

### Optimality Conditions (KKT)

Let $f(\mathbf{t}) = \|\tilde{\mathbf{A}}\mathbf{t} - \tilde{\mathbf{p}}\|_{2}^{2} + \lambda \|\mathbf{t}\|_{1}$

**Karush-Kuhn-Tucker (KKT) conditions:**

$$\begin{aligned}
\nabla_{\mathbf{t}} f(\mathbf{t}^{*}) + \boldsymbol{\mu}^{*} &= \mathbf{0} \\
\mathbf{t}^{*} &\ge \mathbf{0} \\
\boldsymbol{\mu}^{*} &\ge \mathbf{0} \\
\mu_{i}^{*} t_{i}^{*} &= 0 \quad \forall i
\end{aligned}$$

where $\boldsymbol{\mu}^{*}$ are the Lagrange multipliers for non-negativity constraints.

### Sparsity Mechanism

The $\ell_1$ penalty $\lambda \|\mathbf{t}\|_{1}$ induces sparsity through:

1. **Soft Thresholding:** For unconstrained LASSO, solution is:
   $$t_{i}^{*} = \text{sign}(t_{i}^{\text{LS}}) \max\{|t_{i}^{\text{LS}}| - \lambda / (2\alpha), 0\}$$
   where $t_{i}^{\text{LS}}$ is the least squares solution.

2. **Geometric Interpretation:**
   - $\ell_1$ ball has sharp corners along coordinate axes
   - Level sets of quadratic term are ellipsoids
   - Intersection tends to occur at corners → sparse solution

3. **Statistical Interpretation:**
   - $\ell_1$ penalty corresponds to Laplace prior on $\mathbf{t}$
   - Bayesian MAP estimate with sparsity-inducing prior

### Iterative Reweighting (L0 Approximation)

While $\ell_1$ minimization promotes sparsity, it is a convex relaxation of the ideal $\ell_0$ minimization (minimizing the count of non-zero elements). To better approximate the $\ell_0$ norm and enhance sparsity, we implement **Iterative Reweighted $\ell_1$ Minimization**.

**Motivation:**
The $\ell_1$ norm penalizes large coefficients more than the $\ell_0$ norm (which penalizes all non-zeros equally). Reweighting attempts to correct this bias by assigning large weights to small coefficients (forcing them to zero) and small weights to large coefficients (allowing them to remain).

**Algorithm:**

1. **Initialization ($k=0$):**
   Solve the standard weighted $\ell_1$ problem with sensitivity weights $\Omega_{ii}$:
   $$\mathbf{t}^{(0)} = \arg\min_{\mathbf{t}\ge 0} \|\mathbf{W}(\log_{10}(\mathbf{A}\mathbf{t} + \epsilon) - \log_{10}(\mathbf{p} + \epsilon))\|_{2}^{2} + \lambda \sum_{i} \Omega_{ii} |t_{i}|$$

2. **Reweighting Step ($k > 0$):**
   Update the weights based on the solution from the previous iteration $\mathbf{t}^{(k-1)}$:
   $$w_{i}^{(k)} = \frac{\Omega_{ii}}{|t_{i}^{(k-1)}| + \epsilon_{\text{rw}}}$$
   
   where $\epsilon_{\text{rw}}$ is a small damping factor (e.g., $10^{-12}$) to prevent division by zero. In our implementation, $\epsilon_{\text{rw}}$ is adaptive, scaling with the signal magnitude.

3. **Optimization:**
   Solve the weighted problem:
   $$\mathbf{t}^{(k)} = \arg\min_{\mathbf{t}\ge 0} \|\mathbf{W}(\log_{10}(\mathbf{A}\mathbf{t} + \epsilon) - \log_{10}(\mathbf{p} + \epsilon))\|_{2}^{2} + \lambda \sum_{i} w_{i}^{(k)} |t_{i}|$$

4. **Termination:**
   Repeat until convergence (relative change in $\mathbf{t}$ is small) or maximum iterations reached (typically 5-10).

**Result:**
This process typically yields solutions that are much sparser than standard LASSO and closer to the true $\ell_0$ solution, effectively removing "ghost" transmitters and sharpening the localization.

### Solver Implementations

#### 1. scipy L-BFGS-B (Recommended)

**Advantages:**
- Robust general-purpose optimizer
- Handles box constraints ($\mathbf{t} \ge 0$) efficiently
- Works well with non-convex objectives
- No external dependencies

**Disadvantages:**
- May find local minima
- Slower than specialized convex solvers (if the problem were convex)

**Code:**
```python
from scipy.optimize import minimize

def objective(t):
    # Log-domain objective
    log_diff = np.log10(A @ t + eps) - np.log10(p + eps)
    return np.sum((W @ log_diff)**2) + lambda_reg * np.sum(np.abs(t))

result = minimize(objective, t0, method='L-BFGS-B', bounds=[(0, None)]*N)
t_est = result.x
```

**Complexity:** $O(MN \cdot k)$ where $k$ is number of iterations (~100-1000)

#### 2. CVXPY / scikit-learn (Not Supported)

Standard LASSO solvers (CVXPY, scikit-learn) rely on the problem being convex (specifically, quadratic data fidelity). They **cannot** solve the log-domain formulation directly.

### Code Implementation

**Function:** `solve_sparse_reconstruction()`
**Location:** `src/sparse_reconstruction/sparse_solver.py`

**Auto-selection logic:**
```python
def solve_sparse_reconstruction(A, W, p, lambda_reg, solver='auto'):
    if solver == 'auto':
        try:
            return solve_sparse_reconstruction_cvxpy(...)
        except ImportError:
            try:
                return solve_sparse_reconstruction_sklearn(...)
            except ImportError:
                return solve_sparse_reconstruction_scipy(...)
```

---

## 6. Implementation

### Complete Pipeline

**High-Level Function:** `joint_sparse_reconstruction()`
**Location:** `src/sparse_reconstruction/reconstruction.py`

**Step-by-Step Execution:**

```python
from src.sparse_reconstruction import joint_sparse_reconstruction

# Input
sensor_locations = np.array([[10, 20], [30, 40], [50, 60]])  # M=3
observed_powers_dBm = np.array([-80, -85, -90])
map_shape = (100, 100)  # N=10,000

# Reconstruct
tx_map, info = joint_sparse_reconstruction(
    sensor_locations=sensor_locations,
    observed_powers_dBm=observed_powers_dBm,
    map_shape=map_shape,
    scale=5.0,              # 5 m/pixel
    np_exponent=2.0,        # Free space
    sigma=4.5,              # 4.5 dB shadowing
    delta_c=400,            # 400 m correlation
    lambda_reg=0.01,        # Sparsity parameter
    solver='auto',          # Try cvxpy, sklearn, scipy
    verbose=True
)

# Output
# tx_map: (100, 100) array in dBm
# info: dict with solver details, sparsity, peak location
```

### Internal Pipeline Stages

```
Stage 1: Unit Conversion
  observed_powers_dBm → observed_powers_linear (mW)
  Formula: P[mW] = 10^(P[dBm]/10)

Stage 2: Covariance Matrix
  Build V ∈ ℝ^(M×M) with exponential decay
  V_kl = σ² exp(-d_kl / δ_c)

Stage 3: Whitening Matrix
  Cholesky: V = L·L^T → W = L^(-1)

Stage 4: Propagation Matrix
  Build A_model ∈ ℝ^(M×N)
  A[j,i] = 10^(-PL(d_ij)/10)

Stage 5: Whitening
  (Applied within optimization loop)
  
Stage 6: Sparse Optimization
  Solve: min_{t≥0} ‖W(log10(A·t + ε) - log10(p + ε))‖² + λ‖t‖₁
  Using: scipy L-BFGS-B

Stage 7: Reshape & Convert
  t_est (N,) → tx_map (height, width)
  tx_map_linear (mW) → tx_map_dBm (dBm)

Output: tx_map, info
```

### Computational Complexity

| Stage | Operation | Complexity | Typical Time (M=10, N=10⁴) |
|-------|-----------|------------|----------------------------|
| 1 | Unit conversion | $O(M)$ | <1 ms |
| 2 | Covariance | $O(M^{2})$ | <1 ms |
| 3 | Whitening | $O(M^{3})$ | <1 ms |
| 4 | Propagation | $O(MN)$ | ~500 ms |
| 5 | Whitening | $O(M^{2}N)$ | ~10 ms |
| 6 | Optimization | $O(MNk)$ | ~5-30 s |
| 7 | Reshape | $O(N)$ | <1 ms |
| **Total** | | | **~6-30 s** |

where $k$ = optimization iterations (10-1000 depending on solver)

### Memory Requirements

| Object | Size | Example (M=10, N=10⁴) |
|--------|------|----------------------|
| `A_model` | $M \times N \times 8$ bytes | ~800 KB |
| `W` | $M \times M \times 8$ bytes | ~800 bytes |
| `t_est` | $N \times 8$ bytes | ~80 KB |
| **Total** | | **~1 MB** |

---

## 7. Comparison with Likelihood-Based Approach

### Likelihood-Based (Existing)

**Formulation:**
- Stage 1: Optimize $\hat{t}_{i}$ independently for each grid point $i$
- Stage 2: Compute likelihood $\mathcal{L}_{i} = f(\mathbf{p} | t_{i}, \text{location}_i)$
- Stage 3: Normalize to PMF: $f(i|\mathbf{p}) = \mathcal{L}_{i} / \sum_{j} \mathcal{L}_{j}$
- Stage 4: Marginalize: $\hat{p}_{k} = \sum_{i} p_{k}^{th}(t_{i}, \text{location}_i) \cdot f(i|\mathbf{p})$

**Characteristics:**
- Two-stage (optimize, then weight)
- Bayesian interpretation (posterior over locations)
- Smooth probability distribution
- Handles uncertainty explicitly
- Computationally expensive ($N$ independent optimizations)

### Sparse Reconstruction (New)

**Formulation:**
- Single-stage joint optimization
- Explicit sparsity constraint ($\ell_1$ penalty)
- Convex problem (global optimum guaranteed)
- Direct solution for transmit power field

**Characteristics:**
- Single-stage (joint optimization)
- Convex optimization interpretation
- Sparse solution (few non-zeros)
- Implicit localization through sparsity
- Computationally efficient (one optimization)

### Side-by-Side Comparison

| Aspect | Likelihood-Based | Sparse Reconstruction |
|--------|------------------|----------------------|
| **Formulation** | Two-stage Bayesian | Single-stage non-convex |
| **Objective** | Maximize likelihood | Minimize log-domain error + $\ell_1$ |
| **Output** | PMF + marginal power | Sparse power field |
| **Localization** | Explicit (PMF peak) | Implicit (non-zero entries) |
| **Uncertainty** | Full posterior | Sparse point estimate |
| **Sparsity** | Implicit (via likelihood concentration) | Explicit ($\lambda \|\mathbf{t}\|_1$) |
| **Optimization** | $N$ independent problems | 1 joint problem |
| **Complexity** | $O(N \cdot M \cdot k)$ | $O(MN \cdot k)$ |
| **Time** | ~5-10 min | ~10-30 sec |
| **Convexity** | Non-convex (per-pixel) | Non-convex (joint) |
| **Solution** | Multiple local minima | Potential local minima |
| **Tunability** | $\sigma, \delta_c$ | $\lambda$ |

### When to Use Each

**Likelihood-Based:**
- Need full posterior distribution over locations
- Want to quantify localization uncertainty
- Interested in probability maps for decision-making
- Can afford longer computation time
- Multiple transmitters with overlapping coverage

**Sparse Reconstruction:**
- Want fast, efficient localization
- Assume sparse transmitter distribution
- Need globally optimal solution (convex)
- Interested in direct power field estimate
- Single or few well-separated transmitters
- Have prior knowledge about sparsity

---

## 8. Parameter Selection

### Sparsity Parameter $\lambda$

**Role:** Controls sparsity-accuracy trade-off

**Effect:**
- $\lambda = 0$: Standard least squares (dense solution, may overfit)
- $\lambda \to \infty$: All-zero solution (too sparse, underfit)
- Optimal $\lambda$: Balance between data fit and sparsity

**Selection Methods:**

1. **Heuristic:** $\lambda = \alpha \|\tilde{\mathbf{p}}\|_{2}$ where $\alpha \in [0.01, 0.1]$
   - Scales with observation magnitude
   - Good starting point: $\alpha = 0.01$

2. **Cross-Validation:**
   - Split sensors into training/validation sets
   - Optimize on training, evaluate on validation
   - Choose $\lambda$ minimizing validation error

3. **Information Criteria:**
   - AIC: $\text{AIC} = 2k + n\ln(\text{RSS}/n)$
   - BIC: $\text{BIC} = n\ln(\text{RSS}/n) + k\ln(n)$
   - where $k$ = number of non-zero coefficients, $n = M$

4. **L-Curve Method:**
   - Plot $\|\tilde{\mathbf{A}}\mathbf{t} - \tilde{\mathbf{p}}\|_{2}$ vs $\|\mathbf{t}\|_{1}$ for various $\lambda$
   - Choose $\lambda$ at "corner" (optimal trade-off)

**Typical Range:** $\lambda \in [10^{-4}, 10^{-1}]$ (depends on problem scale)

### Path Loss Exponent $n_p$

**Role:** Controls spatial decay rate

**Values:**
- Free space: $n_p = 2$
- Urban: $n_p = 2.7 - 3.5$
- Indoor: $n_p = 1.6 - 1.8$
- Dense urban: $n_p = 3.5 - 5$

**Impact on $\mathbf{A}_{\text{model}}$:**
- Larger $n_p$ → faster decay → more localized influence
- Smaller $n_p$ → slower decay → broader influence

### Shadowing Parameters $\sigma, \delta_c$

**Role:** Model spatial correlation in covariance matrix $\mathbf{V}$

**$\sigma$ (standard deviation):**
- Typical range: 4-8 dB
- Larger $\sigma$ → more uncertain measurements
- Affects all matrix elements equally (diagonal)

**$\delta_c$ (correlation distance):**
- Typical range: 200-600 m
- Larger $\delta_c$ → measurements correlated over longer distances
- Affects off-diagonal structure

**Impact on Whitening:**
- Larger $\sigma$ → larger $\mathbf{V}$ → smaller $\mathbf{W}$ → less downweighting
- Larger $\delta_c$ → more off-diagonal correlation → more complex $\mathbf{W}$

### Scale Parameter

**Role:** Convert pixels to meters

**Determination:**
- From map metadata: scale = meters_per_pixel
- Typical values: 1-10 m/pixel
- Affects distance calculations in $\mathbf{A}_{\text{model}}$ and $\mathbf{V}$

---

## Appendix A: Mathematical Proofs

### Non-Convexity Analysis

**Theorem:** The objective function $f(\mathbf{t}) = \|\mathbf{W}(\log_{10}(\mathbf{A}\mathbf{t} + \epsilon) - \log_{10}(\mathbf{p} + \epsilon))\|_{2}^{2} + \lambda \|\mathbf{t}\|_{1}$ is non-convex.

**Discussion:**

The function $g(x) = \log(x)$ is concave. The composition of a convex function (norm squared) with a concave function (logarithm) is not necessarily convex. In this case, the "bowl" shape of the quadratic error surface is distorted by the logarithm, creating a landscape that may have multiple basins of attraction (local minima).

However, in practice, for sparse reconstruction problems where the true solution is very sparse, the $\ell_1$ penalty term helps to guide the solver towards the correct solution, effectively convexifying the problem locally around the sparse support.

---

## Appendix B: Numerical Stability

### Condition Number

The condition number of the problem is:
$$\kappa(\tilde{\mathbf{A}}) = \frac{\sigma_{\max}(\tilde{\mathbf{A}})}{\sigma_{\min}(\tilde{\mathbf{A}})}$$

**Guidelines:**
- $\kappa < 10^{4}$: Well-conditioned
- $10^{4} < \kappa < 10^{8}$: Moderate, use careful numerics
- $\kappa > 10^{8}$: Ill-conditioned, results may be unreliable

**Improving Conditioning:**
1. Regularization: Add small $\epsilon \mathbf{I}$ to $\mathbf{V}$
2. Scaling: Normalize $\mathbf{A}_{\text{model}}$ columns
3. Preconditioning: Transform to better-scaled problem

### Numerical Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Small path gains** | $A_{ji} \sim 10^{-10}$ | Normalize by max gain |
| **Singular $\mathbf{V}$** | Cholesky fails | Use SVD whitening |
| **Zero observations** | $p_{j} = 0$ | Add small noise: $p_j + \epsilon$ |
| **Unbounded solution** | $\|\mathbf{t}\|$ very large | Increase $\lambda$ or add upper bounds |
| **All-zero solution** | $\mathbf{t} = \mathbf{0}$ | Decrease $\lambda$ |

---

## References

### Papers

1. **Compressed Sensing:**
   - Candès & Tao, "Decoding by Linear Programming" (2005)
   - Donoho, "Compressed Sensing" (2006)

2. **LASSO:**
   - Tibshirani, "Regression Shrinkage and Selection via the Lasso" (1996)
   - Efron et al., "Least Angle Regression" (2004)

3. **Localization:**
   - Patwari et al., "Relative Location Estimation in Wireless Sensor Networks" (2003)

### Software

- **CVXPY:** Diamond & Boyd, "CVXPY: A Python-Embedded Modeling Language for Convex Optimization" (2016)
- **scikit-learn:** Pedregosa et al., "Scikit-learn: Machine Learning in Python" (2011)

### Code Modules

- `src/sparse_reconstruction/propagation_matrix.py`
- `src/sparse_reconstruction/whitening.py`
- `src/sparse_reconstruction/sparse_solver.py`
- `src/sparse_reconstruction/reconstruction.py`

---

*Document created: 2025-11-19*
*Algorithm: Joint Sparse Superposition Reconstruction*
*Version: 1.0*
```
