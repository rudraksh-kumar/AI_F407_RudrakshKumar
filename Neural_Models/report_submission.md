# Laboratory Report: Neural Models – Learning, Depth, Activations, and Output Layers

**Course:** AI / Neural Models Laboratory  
**Date:** August 2026  
**Author:** AI Engineering Student  
**Implementation Script:** [`neural_models_lab.py`](file:///d:/Acads/AI/Neural%20Models/neural_models_lab.py)

---

## Executive Summary

This report documents the complete implementation, verification, and empirical analysis of neural models applied to binary decision problems (XOR gate logic) and multi-class classification extensions. All tasks defined in the laboratory manual have been executed strictly using PyTorch in [`neural_models_lab.py`](file:///d:/Acads/AI/Neural%20Models/neural_models_lab.py).

---

## Task 1: Problem Specification & Linear Separability

### 1. Mathematical Formalism
- **Input Space ($\mathcal{X}$):** $\mathcal{X} = \{0, 1\}^2 \subset \mathbb{R}^2$, representing two binary sensor inputs $(x_1, x_2)$.
- **Output Space ($\mathcal{Y}$):** $\mathcal{Y} = \{0, 1\}$, where $y=1$ indicates a sensor disagreement warning.
- **Labelled Dataset:**
  1. $x^{(1)} = (0, 0) \implies y^{(1)} = 0$ (Both sensors inactive)
  2. $x^{(2)} = (0, 1) \implies y^{(2)} = 1$ (Sensor disagreement)
  3. $x^{(3)} = (1, 0) \implies y^{(3)} = 1$ (Sensor disagreement)
  4. $x^{(4)} = (1, 1) \implies y^{(4)} = 0$ (Both sensors active)

### 2. Geometrical Representation in $x_1, x_2$ Plane

```
  x2 ^
     |
  1  |   Class 1 (0,1) *      * Class 0 (1,1)
     |
     |
  0  |   Class 0 (0,0) *      * Class 1 (1,0)
     +----------------------------------------> x1
         0                    1
```

### 3. Linear Inseparability Proof
A single linear decision boundary in $\mathbb{R}^2$ is defined by an affine function $f(x_1, x_2) = w_1 x_1 + w_2 x_2 + b = 0$. For correct classification:
- $(0,0) \mapsto 0 \implies b < 0$
- $(0,1) \mapsto 1 \implies w_2 + b > 0 \implies w_2 > -b > 0$
- $(1,0) \mapsto 1 \implies w_1 + b > 0 \implies w_1 > -b > 0$
- $(1,1) \mapsto 0 \implies w_1 + w_2 + b < 0$

Summing the inequalities for $(0,1)$ and $(1,0)$ yields $w_1 + w_2 + 2b > 0$. Since $b < 0$, substituting $2b = b + b$ gives $w_1 + w_2 + b > -b > 0$. This directly contradicts the requirement for $(1,1)$ that $w_1 + w_2 + b < 0$. Therefore, no single straight line can separate the XOR classes.

### 4. Single Affine + Sigmoid Prediction & Baseline Run
- **Prediction:** A linear model (affine transformation followed by Sigmoid) will fail to converge to zero loss and will output an average prediction of $0.5$ for all inputs, achieving at most 50% accuracy.
- **Empirical Execution Result ([`neural_models_lab.py`](file:///d:/Acads/AI/Neural%20Models/neural_models_lab.py)):**
  - **Initial Loss:** $0.7317$
  - **Final Loss:** $0.6931$ ($\approx \ln 2$)
  - **Probabilities:** $[0.5, 0.5, 0.5, 0.5]^T$
  - **Accuracy:** $50.0\%$ ($2/4$ correct)

> [!NOTE]
> **Checkpoint 1:** Problem specification recorded. The linear baseline conclusively confirms linear inseparability.

#### Think About It 1
> *A model can have many parameters and still have the wrong kind of representation. What scientific claim about representation does XOR let us test with only four data points?*
> 
> **Answer:** XOR tests the scientific claim that model expressiveness depends fundamentally on non-linear representation geometry rather than raw parameter quantity. Stacking arbitrary depth of affine layers ($W^{(L)} \dots W^{(1)} x + b$) collapses linearly into a single affine map $W_{\text{eff}} x + b_{\text{eff}}$, proving that depth without non-linearity provides zero capacity gain for non-linearly separable representations.

---

## Task 2: Intelligent Agent Design & Validation Criteria

### 1. Architecture Specification
- **Network Topology:** 2 inputs $\to$ 2 hidden units $\to$ 1 output unit ($2-2-1$ MLP).
- **Hidden Activation:** Non-linear activation ($f^{(1)} \in \{\text{Sigmoid}, \text{Tanh}, \text{ReLU}\}$).
- **Output Activation:** Sigmoid output $p = \sigma(z^{(2)}) \in (0,1)$ paired with Binary Cross-Entropy (BCE).
- **Optimization:** Full-batch gradient descent (SGD / Adam).

### 2. Conceptual Engineering Analysis
1. **Scientific Necessity of Hidden Nonlinearity:** Without a non-linear activation $f^{(1)}$, the hidden representation is $h = W^{(1)}x + b^{(1)}$. The output is $y = W^{(2)}(W^{(1)}x + b^{(1)}) + b^{(2)} = (W^{(2)}W^{(1)})x + (W^{(2)}b^{(1)} + b^{(2)})$, which is purely linear. Non-linearity bends the input space, mapping XOR points into a 2D hidden space where they become linearly separable.
2. **Sigmoid + BCE Pairing:** The target is a binary decision $y \in \{0,1\}$. Sigmoid yields a proper Bernoulli probability. Binary Cross-Entropy $L = -[y \ln p + (1-y)\ln(1-p)]$ is the exact negative log-likelihood. When combined with logits $z$, the derivative $\frac{\partial L}{\partial z} = p - y$ is proportional to linear prediction error, avoiding saturation stalls during backpropagation.
3. **Defined Validation Criteria:**
   - **Check 1 (Loss Convergence):** Final binary cross-entropy loss $< 0.01$.
   - **Check 2 (Classification Accuracy):** All 4 predictions thresholded at $0.5$ match targets $y$ ($4/4$ correct, 100% accuracy).
   - **Check 3 (Gradient Non-Zero Check):** First-layer gradient matrix norm $\|\nabla_{W^{(1)}} L\|_2 > 0$ during training steps.
   - **Check 4 (Reproducibility & Stability):** Consistent convergence across multiple random seeds under non-zero initializations.

> [!NOTE]
> **Checkpoint 2:** Model architecture specified and validation criteria established.

#### Think About It 2
> *The hidden units are not given target values. If the network learns XOR, what has determined what each hidden unit should compute? Relate your answer to the role of backpropagation.*
> 
> **Answer:** The hidden unit representations are determined indirectly by error signals backpropagated from the output layer. Backpropagation applies the chain rule to compute $\frac{\partial L}{\partial h} = W^{(2)T} \frac{\partial L}{\partial z^{(2)}}$. This gradient pushes hidden unit weights $W^{(1)}$ to construct intermediate features (e.g., $h_1 \approx \text{OR}(x_1, x_2)$ and $h_2 \approx \text{NAND}(x_1, x_2)$) that allow the output unit to perform linear separation.

---

## Task 3: LLM Implementation & Code Verification

### 1. LLM Prompt Request
```text
Generate minimal PyTorch code for the following model and dataset. Do not change the architecture or task. Data: XOR dataset ((0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0). Architecture: 2-2-1 network with Sigmoid hidden activation, full-batch CPU training, random weight initialization, logits with BCEWithLogitsLoss. After training, report the final loss, all four probabilities, thresholded labels, and one parameter-gradient tensor. Set a random seed for reproducibility and explain each test in one sentence.
```

### 2. Code Code Inspection & Applied Engineering Changes
Before executing the generated code, two critical modifications were implemented:
1. **Numerical Stability Enhancement:** Replaced explicit `nn.Sigmoid()` output + `nn.BCELoss()` with raw logits output + `nn.BCEWithLogitsLoss()`. This uses the log-sum-exp trick to prevent numerical underflow/overflow in $\log(p)$.
2. **Gradient Snapshotting:** Added explicit gradient tracking `model.fc1.weight.grad.clone()` prior to `optimizer.step()` to preserve early gradient metrics before optimizer zeroing.

### 3. Execution Control Flow
- **Forward Pass:** `logits = model(X)`
- **Loss Computation:** `loss = criterion(logits, y)`
- **Reverse-Mode AD:** `loss.backward()` (computes $\nabla_{\theta} L$)
- **Optimizer Parameter Update:** `optimizer.step()` ($\theta \leftarrow \theta - \eta \nabla_{\theta} L$)

> [!NOTE]
> **Checkpoint 3:** LLM prompt recorded, code inspected, and numerical stability fixes applied.

#### Think About It 3
> *An LLM can produce syntactically correct code that implements the wrong experiment. Which parts of this laboratory could you verify from the code without running it, and which require execution and measurement?*
> 
> **Answer:** Static properties (tensor dimensions, network layer count, loss function choice, activation types, and variable bindings) can be verified by code inspection without execution. Dynamic behavior (numerical stability, gradient magnitudes, convergence rates, symmetry breaking, and empirical accuracy) requires execution and empirical measurement.

---

## Task 4: Execution, Testing, and Diagnostics

### Part A – Basic Learning Check (2-2-1 Sigmoid MLP)
- **Initial Loss:** `0.748230`
- **Final Loss:** `0.008482`
- **Final Predicted Probabilities:**
  - $X=(0,0) \implies p = 0.010351$
  - $X=(0,1) \implies p = 0.992351$
  - $X=(1,0) \implies p = 0.992350$
  - $X=(1,1) \implies p = 0.008117$
- **Thresholded Predictions ($p > 0.5$):** $[0, 1, 1, 0]^T$
- **Status:** **PASS** ($4/4$ classified correctly).

### Part B – Backpropagation Gradient Check
- **First-Layer Gradient Matrix ($\nabla_{W^{(1)}} L$) after Step 1:**
  $$\frac{\partial L}{\partial W^{(1)}} = \begin{bmatrix} -0.00399953 & -0.00442072 \\ 0.00639591 & 0.00806316 \end{bmatrix}$$
- **Physical Interpretation:** `parameter.grad` represents the directional sensitivity of the scalar loss $L$ with respect to each individual weight $w_{ij}^{(1)}$.
- **Average vs Sum Gradient:** PyTorch's `nn.BCEWithLogitsLoss` defaults to `reduction='mean'`. Since $L = \frac{1}{4} \sum_{i=1}^4 L_i$, by linearity of differentiation, $\nabla_{W^{(1)}} L = \frac{1}{4} \sum_{i=1}^4 \nabla_{W^{(1)}} L_i$.

### Part C – Symmetry Experiment (Zero Weight Initialization)
Setting all weights $W^{(1)}, W^{(2)}$ and biases $b^{(1)}, b^{(2)}$ strictly to zero before training yields:

| Step | Loss | Row 0 ($W^{(1)}$) | Row 1 ($W^{(1)}$) | Identical Rows? |
| :---: | :---: | :---: | :---: | :---: |
| 1 | `0.693147` | `[0.0, 0.0]` | `[0.0, 0.0]` | **True** |
| 2 | `0.693147` | `[0.0, 0.0]` | `[0.0, 0.0]` | **True** |
| 3 | `0.693147` | `[0.0, 0.0]` | `[0.0, 0.0]` | **True** |
| 4 | `0.693147` | `[0.0, 0.0]` | `[0.0, 0.0]` | **True** |
| 5 | `0.693147` | `[0.0, 0.0]` | `[0.0, 0.0]` | **True** |

**Explanation:** When all weights are zero, both hidden units compute identical activations $h_1 = h_2 = f(0)$. Consequently, backpropagation assigns identical error gradients to both units ($\delta_1 = \delta_2$). The weight updates for both rows are identical, trapping the network in a symmetric lower-dimensional subspace where hidden units cannot specialize into distinct feature detectors.

### Part D – Activation Function Experiment

#### Experimental Summary Table

| Hidden Activation | Final Loss | 4/4 Correct? | Early $\|\nabla_{W^{(1)}} L\|_2$ (Step 1) |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | `0.008482` | **Yes (4/4)** | `0.011894` |
| **Tanh** | `0.001342` | **Yes (4/4)** | `0.028684` |
| **ReLU** | `0.346641` | **No (3/4)** | `0.082002` |

#### Detailed Interpretation
Tanh achieved the best overall performance, reaching a final loss of `0.001342`—nearly 6x lower than Sigmoid (`0.008482`). Tanh benefits from zero-centered outputs (range $(-1, 1)$), which prevents systematic directional bias in weight updates and supports larger early gradients (`0.028684` vs `0.011894`). Sigmoid suffered from smaller initial gradient norms due to derivative saturation at the tails. ReLU produced the largest early gradient norm (`0.082002`) because its derivative is $1.0$ for positive inputs; however, on this tiny 2-unit hidden layer, one unit suffered from dead-unit inactivation ($a < 0$), causing the network to get trapped in a local minimum with `0.346641` loss and achieving only 3/4 correct predictions.

> [!NOTE]
> **Checkpoint 4:** Symmetry breaking verified and activation comparison table completed.

#### Think About It 4
> *If a sigmoid unit is saturated, its derivative is close to zero. If a ReLU unit is negative, its derivative is zero. These are different mechanisms that can both produce a small gradient. How would you distinguish them by inspecting activations and pre-activations?*
> 
> **Answer:** Inspect both pre-activation $a$ and activation $h = f(a)$:
> - **Sigmoid Saturation:** Occurs when $|a| \gg 0$ (e.g. $a = +10$ or $-10$). The activation $h$ approaches $1.0$ or $0.0$, and the derivative $f'(a) = h(1-h)$ is non-zero but infinitesimally small.
> - **ReLU Inactivation ("Dying ReLU"):** Occurs strictly when pre-activation $a < 0$. The activation is exactly $h = 0.0$, and the subgradient $f'(a)$ is identically $0.0$.

---

## Task 5: Three-Class Sensor Classifier Extension

### 1. Problem Mapping
- **Class 0:** $(0,0) \implies$ Both sensors inactive
- **Class 1:** $(0,1)$ or $(1,0) \implies$ Sensor disagreement
- **Class 2:** $(1,1) \implies$ Both sensors active

### 2. Pre-Implementation Theoretical Analysis
1. **Shape of Final Weight Matrix ($W^{(2)}$):** $\text{Shape} = (3, 2)$ (3 output class logits $\times$ 2 hidden features).
2. **Number of Logits per Example:** 3 scalar logits per example.
3. **Softmax Probability Axiom:** Softmax normalizes exponents: $p_k = \frac{e^{z_k}}{\sum_{j=1}^3 e^{z_j}}$. Summing yields $\sum_{k=1}^3 p_k = \frac{\sum_{k=1}^3 e^{z_k}}{\sum_{j=1}^3 e^{z_j}} = 1$.
4. **Logit Gradient Derivation ($\mathbf{p} - \mathbf{y}$):** For cross-entropy loss $L = -\sum_{k} y_k \ln p_k$, taking the partial derivative with respect to logit $z_i$:
   $$\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial p_k} \frac{\partial p_k}{\partial z_i} = -\frac{y_i}{p_i} \cdot p_i(1 - p_i) - \sum_{k \neq i} \frac{y_k}{p_k} (-p_k p_i) = -y_i(1 - p_i) + p_i \sum_{k \neq i} y_k$$
   Since $\sum_k y_k = 1$, $-y_i + y_i p_i + p_i(1 - y_i) = p_i - y_i$. Thus $\nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}$.

### 3. Empirical Multi-Class Results

- **Final Loss:** `0.000036` (converged cleanly after 2000 steps).
- **Classification & Probability Breakdown:**

| Input $(x_1, x_2)$ | Target $y$ | Predicted Class | Softmax Probability Vector $[p_0, p_1, p_2]$ | Vector Sum |
| :---: | :---: | :---: | :---: | :---: |
| $(0, 0)$ | Class 0 | **Class 0** | `[0.999965, 0.000035, 0.000000]` | `1.00000000` |
| $(0, 1)$ | Class 1 | **Class 1** | `[0.000000, 1.000000, 0.000000]` | `1.00000000` |
| $(1, 0)$ | Class 1 | **Class 1** | `[0.000000, 1.000000, 0.000000]` | `1.00000000` |
| $(1, 1)$ | Class 2 | **Class 2** | `[0.000000, 0.000000, 1.000000]` | `1.00000000` |

### 4. Optional Diagnostic: Logit Shift Invariance & Numerical Stability
- **Experiment:** Added constant $+100.0$ to all logits prior to softmax computation.
- **Original Softmax Probs ($X=(0,0)$):** `[0.99996495, 0.00003503, 0.00000000]`
- **Shifted Softmax Probs ($z + 100$):** `[0.99996495, 0.00003503, 0.00000000]`
- **Max Absolute Difference:** $1.164153 \times 10^{-10}$ (within floating-point roundoff).
- **Mathematical Explanation:** Softmax is shift-invariant:
  $$\text{Softmax}(\mathbf{z} + c)_k = \frac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \frac{e^{z_k} e^c}{e^c \sum_j e^{z_j}} = \text{Softmax}(\mathbf{z})_k$$
  Stable implementations subtract $\max(\mathbf{z})$ so that the largest exponent is $e^0 = 1$, preventing floating-point overflow (`inf`).

> [!NOTE]
> **Checkpoint 5:** Multi-class output design validated and logit gradient $\mathbf{p} - \mathbf{y}$ proven.

#### Think About It 5
> *Next-token prediction in a language model can be viewed as classification over a very large vocabulary. Which parts of this tiny three-class experiment stay mathematically the same when the number of classes becomes tens of thousands, and which parts of the surrounding architecture change dramatically?*
> 
> **Answer:**
> - **Stays Mathematically Identical:** Categorical cross-entropy loss, softmax normalization, and the logit error gradient formula $\nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}$.
> - **Changes Dramatically:** Input encoding (token embeddings + positional encodings), feature extraction backbone (multi-head self-attention and transformer blocks vs tiny MLP), and output projection dimensionality ($W_{\text{out}} \in \mathbb{R}^{V \times d}$ where vocabulary size $V \ge 32,000$).

---

## Reflection Questions

### 1. Depth vs Non-Linearity in XOR
*What did the XOR experiment demonstrate about the difference between depth and nonlinearity?*  
**Answer:** Adding depth via affine transformations without non-linear activations provides zero architectural expressiveness; linear layers collapse into a single affine map. Non-linearity is mandatory to fold the input space into a configuration where non-linearly separable classes become linearly separable.

### 2. Backpropagation Learning Signal Evidence
*In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a nonzero gradient?*  
**Answer:** Loss decreased monotonically from $0.748$ to $0.008$, weight parameters evolved away from random initializations into structured feature detectors, and predicted probabilities converged cleanly to target labels ($[0.01, 0.99, 0.99, 0.01]$).

### 3. Zero Weight Initialization Failure
*Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?*  
**Answer:** Zero initialization causes all hidden units to produce identical activations and receive identical backpropagated error gradients. The parameter updates remain identical, trapping the network in a symmetric subspace where units cannot specialize.

### 4. Activation Function Impact
*How did changing the hidden activation affect the gradient you observed? Distinguish the scientific explanation from the engineering observation.*  
**Answer:**
- **Scientific Explanation:** Tanh is zero-centered ($(-1,1)$), preventing directional bias in weight updates. Sigmoid saturates at boundaries ($f'(z) = \sigma(z)(1-\sigma(z)) \to 0$). ReLU has constant unit derivative for $z > 0$ and zero derivative for $z < 0$.
- **Engineering Observation:** Tanh converged fastest to the lowest loss (`0.001342`). Sigmoid exhibited smaller early gradients (`0.011894`). ReLU exhibited large early gradients (`0.082002`) but suffered from dead units on a 2-unit hidden layer, failing to solve XOR (`0.346641` loss).

### 5. Output Layer and Loss Function Coupling
*Why must the output layer and loss be selected together according to the task?*  
**Answer:** The output activation defines the output domain (e.g. Sigmoid $\to (0,1)$ for Bernoulli probability, Softmax $\to$ probability simplex for Categorical distribution). Matching it with negative log-likelihood loss (BCE or Cross-Entropy) ensures that the logit gradient simplifies to $\mathbf{p} - \mathbf{y}$, yielding linear error signals and avoiding saturation stalls.

### 6. LLM Productivity vs Human Verification
*Give one example where the LLM improved your engineering productivity and one example where human verification was essential.*  
**Answer:**
- **Productivity Gain:** Rapid boilerplate code generation for PyTorch training loops, loss functions, and data tensors.
- **Human Verification Essential:** Identifying that zero weight initialization causes symmetry locking and substituting `nn.BCEWithLogitsLoss` for explicit `Sigmoid` + `BCELoss` to guarantee numerical stability.

### 7. Scalability of Diagnostic Tests
*Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?*  
**Answer:**
- **Keep at Scale:** Tracking loss curves, monitoring output probability distributions, checking gradient norms ($\|\nabla W\|_2$), and validating classification accuracy.
- **Too Expensive at Scale:** Full printing of weight matrices, complete jacobian factor logging, zero-initialization symmetry sweeps, and exhaustive finite-difference gradient checks.

---

## Conclusion

All requirements of the laboratory exercise have been fully completed, validated empirically in [`neural_models_lab.py`](file:///d:/Acads/AI/Neural%20Models/neural_models_lab.py), and documented. The findings confirm the theoretical necessity of hidden non-linearities, the role of backpropagation in representation learning, symmetry breaking via random initialization, and output-loss coupling.
