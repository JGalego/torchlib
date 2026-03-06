import TorchLib.Core
import TorchLib.IR
import TorchLib.Layers

/-!
# Example: Linear Layer

Demonstrates initializing and using a `Linear` (fully-connected) layer
from `TorchLib.Layers`.
-/

open TorchLib

private def say [Repr α] (label : String) (v : α) : IO Unit :=
  IO.println s!"{label}: {reprStr v}"

-- ---------------------------------------------------------------------------
-- Initialization
-- ---------------------------------------------------------------------------

-- A fully-connected layer: 4 inputs → 2 outputs
-- Weights are initialized to 0.1, bias to 0.0
def fc : Linear Float := Linear.init 4 2

#eval say "fc.weight.shape" fc.weight.shape
#eval say "fc.bias.shape"   fc.bias.shape

-- ---------------------------------------------------------------------------
-- Forward pass
-- ---------------------------------------------------------------------------

-- Input: 1 sample of 4 features → batch shape [1, 4]
def input1 : Tensor Float := Tensor.ones [1, 4]

-- output shape: [1, 2]
-- Each output = sum(w_j * 1.0) + 0.0  = 4 * 0.1 = 0.4
def out1 : Tensor Float := Linear.forward fc input1

#eval say "out1.shape" out1.shape
#eval say "out1.data"  out1.data

-- ---------------------------------------------------------------------------
-- Batch forward pass
-- ---------------------------------------------------------------------------

-- 3 samples, 4 features each
def input3 : Tensor Float := Tensor.ones [3, 4]

def out3 : Tensor Float := Linear.forward fc input3

#eval say "out3.shape" out3.shape
#eval say "out3.data"  out3.data

-- ---------------------------------------------------------------------------
-- Custom weights: identity-like layer (in=2, out=2)
-- ---------------------------------------------------------------------------

-- Layer with explicit weights  W = [[2,0],[0,2]]  b = [1, -1]
def fc2 : Linear Float :=
  { weight := { shape := [2, 2], data := #[2.0, 0.0, 0.0, 2.0] }
    bias   := { shape := [2],    data := #[1.0, -1.0] } }

-- Input [1, 2] = [[3, 4]]
-- Expected output: [2*3 + 1, 2*4 - 1] = [7, 7]
def input2 : Tensor Float := { shape := [1, 2], data := #[3.0, 4.0] }

#eval say "fc2 [[3,4]]" (Linear.forward fc2 input2).data
