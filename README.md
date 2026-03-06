# TorchLib

![mission](https://img.shields.io/badge/mission-make_it_formal-purple) ![license](https://img.shields.io/badge/license-MIT-lightgrey.svg) ![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen) ![last commit](https://img.shields.io/github/last-commit/JGalego/torchlib)

A [Lean 4](https://lean-lang.org/) library for neural network modeling, training, and formal verification.

> ⚠️ **Warning:** This module is a work in progress and may contain incomplete features, placeholders, and breaking changes. Use with caution and refer to the source code for the latest status.

> 🎖️ **Acknowledgments:** This project is inspired by the design of [PyTorch](https://pytorch.org/) via [TorchLean](https://leandojo.org/torchlean.html) (George *et al.*, 2026) and [CSLib](https://www.cslib.io/). A warm thanks to the PyTorch and Lean communities for their support, and to everyone crazy enough to live at the intersection of formal verification and machine learning!

<img src="TorchLib.png" alt="TorchLib Logo" width="50%"/>

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

## Examples

Runnable examples live in the [`examples/`](examples/) directory.

Each file is self-contained and can be checked individually:

```sh
lake env lean examples/Tensors.lean
lake env lean examples/LinearLayer.lean
lake env lean examples/MLP.lean
lake env lean examples/Autograd.lean
lake env lean examples/Training.lean
lake env lean examples/Verification.lean
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
