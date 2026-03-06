import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models
import TorchLib.Runtime.Autograd
import TorchLib.Runtime.Training

/-!
# Example: Training

Demonstrates loss functions and optimizers (SGD, Adam) from
`TorchLib.Runtime.Training`.
-/

open TorchLib TorchLib.Runtime

private def say [Repr α] (label : String) (v : α) : IO Unit :=
  IO.println s!"{label}: {reprStr v}"

-- ---------------------------------------------------------------------------
-- Loss functions
-- ---------------------------------------------------------------------------

-- MSE on two 4-element vectors
def pred4   : Tensor Float := { shape := [4], data := #[1.0, 2.0, 3.0, 4.0] }
def target4 : Tensor Float := { shape := [4], data := #[1.5, 2.5, 3.5, 4.5] }

#eval say "mseLoss"          (mseLoss pred4 target4)

-- Cross-entropy from logits: 2 samples, 3 classes
-- Targets: sample 0 → class 2, sample 1 → class 0
def logits23 : Tensor Float :=
  { shape := [2, 3], data := #[0.1, 0.2, 2.0,   -- sample 0: class 2 has high score
                                1.8, 0.1, 0.5] } -- sample 1: class 0 has high score

#eval say "crossEntropyLoss" (crossEntropyLoss logits23 #[2, 0])

-- Binary cross-entropy: predictions vs binary targets
def probs : Tensor Float := { shape := [4], data := #[0.9, 0.1, 0.8, 0.2] }
def btgt  : Tensor Float := { shape := [4], data := #[1.0, 0.0, 1.0, 0.0] }

#eval say "binaryCELoss"     (binaryCELoss probs btgt)

-- ---------------------------------------------------------------------------
-- SGD optimizer: one parameter update step
-- ---------------------------------------------------------------------------

def sgdCfg : SGDConfig := { lr := 0.1, momentum := 0.0, weightDecay := 0.0 }

-- A single "weight" tensor and its gradient
def w0   : Tensor Float := { shape := [3], data := #[1.0, 2.0, 3.0] }
def grad : Tensor Float := { shape := [3], data := #[0.1, 0.2, 0.3] }

def sgdSt0 : SGDState := SGD.initState sgdCfg ["w"]

-- Apply one update: w ← w - lr * grad = w - 0.1 * grad
def sgdStep1   := SGD.step sgdCfg sgdSt0 [("w", w0, grad)]
def sgdSt1     := sgdStep1.1
def newParams1 := sgdStep1.2

#eval say "SGD (no momentum)"  (newParams1.map (fun (n, p) => (n, p.data)))

-- ---------------------------------------------------------------------------
-- SGD with momentum
-- ---------------------------------------------------------------------------

def sgdMomCfg : SGDConfig := { lr := 0.1, momentum := 0.9 }

def sgdMomSt0 : SGDState := SGD.initState sgdMomCfg ["w"]

-- First step: velocity starts at 0, so update = lr * grad
def sgdMomStep1 := SGD.step sgdMomCfg sgdMomSt0 [("w", w0, grad)]
def sgdMomSt1   := sgdMomStep1.1

-- Second step: velocity carries over, giving a larger update
def sgdMomStep2 := SGD.step sgdMomCfg sgdMomSt1 [("w", w0, grad)]
def newParams2  := sgdMomStep2.2

#eval say "SGD (momentum=0.9)" (newParams2.map (fun (_, p) => p.data))

-- ---------------------------------------------------------------------------
-- Adam optimizer
-- ---------------------------------------------------------------------------

def adamCfg : AdamConfig := { lr := 0.001, beta1 := 0.9, beta2 := 0.999, eps := 1e-8 }

def adamSt0 : AdamState := Adam.initState adamCfg ["w"]

-- Step 1
def adamStep1   := Adam.step adamCfg adamSt0 [("w", w0, grad)]
def adamSt1     := adamStep1.1
def adamParams1 := adamStep1.2

#eval say "Adam step 1" (adamParams1.map (fun (n, p) => (n, p.data)))

-- Step 2 (bias correction changes the effective learning rate)
def adamStep2   := Adam.step adamCfg adamSt1 [("w", w0, grad)]
def adamParams2 := adamStep2.2

#eval say "Adam step 2" (adamParams2.map (fun (_, p) => p.data))

-- ---------------------------------------------------------------------------
-- Manual training loop sketch
-- ---------------------------------------------------------------------------

-- A tiny MLP and a 2-step gradient descent loop (no autograd integration,
-- to show the pure-Lean iteration pattern).

def tinyMLP : MLP Float := MLP.init 4 [8] 1

/-- Run `n` pseudo-training steps, printing the MSE loss each time.
    (Gradients are not computed here; this illustrates the loop structure.) -/
def trainLoop (n : Nat) : IO Unit := do
  let xs : Tensor Float := Tensor.ones  [16, 4]
  let ys : Tensor Float := Tensor.zeros [16, 1]
  for i in [:n] do
    let pred := MLP.forward tinyMLP xs
    let loss := mseLoss pred ys
    IO.println s!"step {i}: loss = {loss}"

#eval trainLoop 3
