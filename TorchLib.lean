-- NN.Spec — Core specification layer
import TorchLib.Core
import TorchLib.IR
import TorchLib.Layers
import TorchLib.Models

-- NN.Runtime — execution/training layer
import TorchLib.Runtime.Autograd
import TorchLib.Runtime.Training
import TorchLib.Runtime.TorchLean

-- NN.Verification — verification layer
import TorchLib.Verification.IBP
import TorchLib.Verification.CROWN
import TorchLib.Verification.Certificate

/-!
# TorchLib

A Lean 4 implementation of TorchLean, a PyTorch-style neural network library
with three co-designed layers sharing a single op-tagged SSA/DAG IR:

```
                 ┌─────────────────────────────────────────┐
  NN.Spec        │  Specification Layer   (α = ℝ, proofs)  │
  ───────────    │  Core, Layers, Models                   │
                 └─────────────────────────────────────────┘
                 ┌─────────────────────────────────────────┐
  NN.Runtime     │  Runtime Layer  (α = Float/Rat)         │
  ──────────     │  Autograd, TorchLean API, Training      │
                 └─────────────────────────────────────────┘
                 ┌─────────────────────────────────────────┐
  NN.Verification│  Verification Layer  (α = Interval)     │
  ──────────────-│  IBP, CROWN/LiRPA, Certificate          │
                 └─────────────────────────────────────────┘
                 ┌─────────────────────────────────────────┐
  Shared IR      │  Op-tagged SSA/DAG computation graph    │
                 └─────────────────────────────────────────┘
```

The scalar parameter `α` ("scalar polymorphism") is the key abstraction:
- `α = Float`    → eager numeric execution
- `α = Rat`      → exact rational arithmetic / proofs
- `α = Interval` → sound interval-based verification
-/

