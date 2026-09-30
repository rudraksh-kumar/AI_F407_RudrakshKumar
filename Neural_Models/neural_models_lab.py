import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ==============================================================================
# Task 1: Linear Model Baseline
# ==============================================================================
def run_task1_linear_baseline():
    print("=" * 70)
    print("TASK 1: Linear Model Baseline (2 inputs -> 1 output, no hidden layer)")
    print("=" * 70)
    
    set_seed(42)
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    
    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    initial_loss = criterion(model(X), y).item()
    print(f"Initial Linear Model Loss: {initial_loss:.4f}")
    
    for epoch in range(2000):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    probs = model(X).detach()
    preds = (probs > 0.5).float()
    acc = (preds == y).float().mean().item()
    
    print(f"Final Linear Model Loss: {final_loss:.4f}")
    print(f"Probabilities:\n{probs.numpy()}")
    print(f"Predictions:\n{preds.numpy()}")
    print(f"Accuracy: {acc * 100:.1f}% (Correct: {int(acc * 4)}/4)")
    print("Explanation: Linear boundary cannot separate XOR; model outputs ~0.5 for all inputs.\n")
    return final_loss, probs, preds


# ==============================================================================
# Task 3 & 4 (Parts A, B): Baseline Binary XOR MLP
# ==============================================================================
def run_task3_4_binary_xor(activation_name='sigmoid', num_epochs=5000, lr=0.5, seed=42):
    set_seed(seed)
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    
    if activation_name == 'sigmoid':
        act_layer = nn.Sigmoid()
    elif activation_name == 'tanh':
        act_layer = nn.Tanh()
    elif activation_name == 'relu':
        act_layer = nn.ReLU()
    else:
        raise ValueError(f"Unknown activation: {activation_name}")
        
    class XORNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.act = act_layer
            self.fc2 = nn.Linear(2, 1)
            
        def forward(self, x):
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
            
    model = XORNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    optimizer.zero_grad()
    initial_logits = model(X)
    initial_loss = criterion(initial_logits, y)
    initial_loss_val = initial_loss.item()
    initial_loss.backward()
    
    early_grad_W1 = model.fc1.weight.grad.clone()
    early_grad_norm = torch.norm(early_grad_W1).item()
    
    optimizer.step()
    
    for epoch in range(1, num_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
    final_loss_val = loss.item()
    final_logits = model(X)
    final_probs = torch.sigmoid(final_logits).detach()
    final_preds = (final_probs > 0.5).float()
    correct_count = (final_preds == y).float().sum().item()
    all_correct = (correct_count == 4.0)
    
    return {
        'model': model,
        'initial_loss': initial_loss_val,
        'final_loss': final_loss_val,
        'probs': final_probs,
        'preds': final_preds,
        'all_correct': all_correct,
        'correct_count': int(correct_count),
        'early_grad_W1': early_grad_W1,
        'early_grad_norm': early_grad_norm,
        'final_grad_W1': model.fc1.weight.grad.clone()
    }


# ==============================================================================
# Task 4 Part C: Symmetry Experiment (Zero Initialization)
# ==============================================================================
def run_symmetry_experiment():
    print("=" * 70)
    print("TASK 4 PART C: Symmetry Experiment (Zero Initialization)")
    print("=" * 70)
    
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    
    class ZeroInitXORNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.act = nn.Sigmoid()
            self.fc2 = nn.Linear(2, 1)
            
            nn.init.constant_(self.fc1.weight, 0.0)
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.constant_(self.fc2.weight, 0.0)
            nn.init.constant_(self.fc2.bias, 0.0)
            
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))
            
    model = ZeroInitXORNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    print("Initial hidden weights (W1):\n", model.fc1.weight.detach().numpy())
    
    for step in range(1, 6):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        grad_W1 = model.fc1.weight.grad.clone()
        optimizer.step()
        
        row1_equal_row2 = torch.allclose(model.fc1.weight[0], model.fc1.weight[1])
        print(f"Step {step}: Loss = {loss.item():.4f}")
        print(f"  W1 weights:\n{model.fc1.weight.detach().numpy()}")
        print(f"  W1 grad:\n{grad_W1.numpy()}")
        print(f"  Row 0 and Row 1 identical? {row1_equal_row2}")
        
    print("Conclusion: Weights remain strictly symmetric (identical across hidden units).\n")


# ==============================================================================
# Task 4 Part D: Activation Function Experiment
# ==============================================================================
def run_activation_experiment():
    print("=" * 70)
    print("TASK 4 PART D: Activation Function Experiment (Sigmoid vs Tanh vs ReLU)")
    print("=" * 70)
    
    activations = ['sigmoid', 'tanh', 'relu']
    results = {}
    
    for act in activations:
        res = run_task3_4_binary_xor(activation_name=act, num_epochs=5000, lr=0.5, seed=42)
        results[act] = res
        
    print(f"{'Hidden Activation':<18} | {'Final Loss':<12} | {'4/4 Correct?':<12} | {'Early ||Grad W1||2':<20}")
    print("-" * 70)
    for act in activations:
        res = results[act]
        correct_str = f"Yes ({res['correct_count']}/4)" if res['all_correct'] else f"No ({res['correct_count']}/4)"
        print(f"{act.capitalize():<18} | {res['final_loss']:<12.6f} | {correct_str:<12} | {res['early_grad_norm']:<20.6f}")
        
    print("\nDetailed breakdown:")
    for act in activations:
        res = results[act]
        print(f"\n[{act.upper()}]")
        print(f"  Initial Loss: {res['initial_loss']:.6f}")
        print(f"  Final Loss:   {res['final_loss']:.6f}")
        print(f"  Probabilities: {res['probs'].squeeze().tolist()}")
        print(f"  Predictions:   {res['preds'].squeeze().tolist()}")
        print(f"  First-layer Grad Norm (Step 1): {res['early_grad_norm']:.6f}")
        
    return results


# ==============================================================================
# Task 5: Three-Class Extension
# ==============================================================================
def run_task5_three_class_extension():
    print("\n" + "=" * 70)
    print("TASK 5: Three-Class Extension (Multiclass Cross-Entropy)")
    print("=" * 70)
    
    set_seed(42)
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    
    class ThreeClassNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.act = nn.Tanh()
            self.fc2 = nn.Linear(2, 3)
            
        def forward(self, x):
            h = self.act(self.fc1(x))
            logits = self.fc2(h)
            return logits
            
    model = ThreeClassNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    print(f"Final Weight Matrix Shape (fc2.weight): {model.fc2.weight.shape} (Outputs x Inputs = 3 x 2)")
    
    for epoch in range(2000):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    final_logits = model(X).detach()
    softmax_probs = torch.softmax(final_logits, dim=1)
    preds = torch.argmax(softmax_probs, dim=1)
    
    print(f"\nTraining Complete after 2000 steps.")
    print(f"Final Loss: {final_loss:.6f}")
    print("\nInput -> Target | Predicted Class | Softmax Probabilities [Class 0, Class 1, Class 2]")
    print("-" * 75)
    inputs_str = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
    for i in range(4):
        p_vec = softmax_probs[i].numpy()
        p_sum = p_vec.sum()
        print(f"X={inputs_str[i]} -> y={y[i].item()} | Pred={preds[i].item()}          | Probs=[{p_vec[0]:.4f}, {p_vec[1]:.4f}, {p_vec[2]:.4f}] (Sum={p_sum:.6f})")
        
    ex0_probs = softmax_probs[0]
    print(f"\nNumerical Verification for Example 0 (X=(0,0)):")
    print(f"  Probability Vector: {ex0_probs.numpy()}")
    print(f"  Sum of components: {ex0_probs.sum().item():.8f}")
    
    print("\nOptional Diagnostic: Adding constant +100 to logits before softmax")
    shifted_logits = final_logits + 100.0
    shifted_probs = torch.softmax(shifted_logits, dim=1)
    max_prob_diff = (softmax_probs - shifted_probs).abs().max().item()
    print(f"  Original Softmax Probs[0]: {softmax_probs[0].numpy()}")
    print(f"  Shifted Softmax Probs[0]:  {shifted_probs[0].numpy()}")
    print(f"  Max Absolute Difference:   {max_prob_diff:.8e}")
    print("  Conclusion: Softmax is translation-invariant along the class axis (softmax(z) == softmax(z + c)).")


# ==============================================================================
# Main Driver Function
# ==============================================================================
def main():
    print("Running Complete Neural Models Lab Experiments...\n")
    
    run_task1_linear_baseline()
    
    print("=" * 70)
    print("TASK 3 & TASK 4 PARTS A & B: Baseline Binary XOR (2-2-1 Sigmoid MLP)")
    print("=" * 70)
    res_base = run_task3_4_binary_xor(activation_name='sigmoid', num_epochs=5000, lr=0.5, seed=42)
    print(f"Initial Loss: {res_base['initial_loss']:.6f}")
    print(f"Final Loss:   {res_base['final_loss']:.6f}")
    print(f"Predicted Probabilities:\n{res_base['probs'].squeeze().numpy()}")
    print(f"Thresholded Predictions:\n{res_base['preds'].squeeze().numpy()}")
    print(f"Gradients of W1 after initial backward pass:\n{res_base['early_grad_W1'].numpy()}")
    print(f"Gradients of W1 after final backward pass:\n{res_base['final_grad_W1'].numpy()}\n")
    
    run_symmetry_experiment()
    run_activation_experiment()
    run_task5_three_class_extension()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
