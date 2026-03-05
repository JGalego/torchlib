# TorchLib

A Lean 4 library for neural network modeling, training, and formal verification.

>  ⚠️ **Warning:** This package is a work in progress and may contain incomplete features, placeholders, and breaking changes. Use with caution and refer to the source code for the latest status.

> Adapted from [TorchLean](https://leandojo.org/torchlean.html) by George *et al.* (2026)

## Overview

**TorchLib** brings deep learning primitives to Lean 4, combining expressive neural network construction with formal verification techniques.

It provides a composable layer system, automatic differentiation, a training loop, and certified robustness verification via Interval Bound Propagation (IBP) and CROWN.

## Features

- **Core primitives** - tensors, shapes, and basic operations
- **Intermediate representation** - computation graph IR for model inspection and transformation
- **Layer library** - composable building blocks (linear, activation, etc.)
- **Model construction** - high-level API for defining neural networks
- **Automatic differentiation** - forward and backward pass via `Autograd`
- **Training** - loss functions, optimizers, and training loops
- **Formal verification** - certified robustness bounds via IBP and CROWN
- **Certificates** - machine-checkable proofs of network properties

## Project Structure

```
TorchLib/
├── Core.lean           # Core types and tensor primitives
├── IR.lean             # Intermediate representation
├── Layers.lean         # Layer definitions
├── Models.lean         # High-level model API
├── Runtime/
│   ├── Autograd.lean   # Automatic differentiation
│   ├── TorchLean.lean  # LibTorch/runtime bindings
│   └── Training.lean   # Training loop and optimizers
└── Verification/
    ├── Certificate.lean # Verification certificates
    ├── CROWN.lean       # CROWN bound propagation
    └── IBP.lean         # Interval Bound Propagation
```

## Requirements

- [Lean 4](https://leanprover.github.io/) (see `lean-toolchain` for the exact version)
- [Lake](https://github.com/leanprover/lake) (included with Lean 4)

## Installation

Add the following to your `lakefile.toml`:

```toml
[[require]]
name = "TorchLib"
git = "https://github.com/jgalego/torchlib"
rev = "main"
```

Then run:

```sh
lake update
lake build
```

## Building from Source

```sh
git clone https://github.com/jgalego/torchlib
cd torchlib
lake build
```

## Running Tests

```sh
lake build TorchLibTests
lake env lean scripts/RunTests.lean
```

Or run individual test modules:

```sh
lake env lean TorchLibTests/Core.lean
lake env lean TorchLibTests/Autograd.lean
lake env lean TorchLibTests/IBP.lean
```

## Usage

### Defining a Model

```lean
import TorchLib

open TorchLib

def myModel : Model := sequential [
  linear 784 128,
  relu,
  linear 128 10
]
```

### Training

```lean
import TorchLib.Runtime.Training

def main : IO Unit := do
  let model := myModel
  let opt   := sgd (lr := 0.01)
  trainLoop model opt dataset epochs := 10
```

### Verification with IBP

```lean
import TorchLib.Verification.IBP

-- Compute certified output bounds under input perturbation ε
let bounds := ibp myModel input ε
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
