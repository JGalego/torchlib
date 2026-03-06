import TorchLib.Core
import TorchLib.IR
import TorchLib.Layers

/-!
# Example: Linear Layer

Demonstrates initializing and using a `Linear` (fully-connected) layer
from `TorchLib.Layers`.
-/

open TorchLib

-- ---------------------------------------------------------------------------
-- Initialization
-- ---------------------------------------------------------------------------

-- A fully-connected layer: 4 inputs → 2 outputs
-- Weights are initialized to 0.1, bias to 0.0
def fc : Linear Float := Linear.init 4 2

#eval fc.weight.shape   -- [2, 4]
#eval fc.bias.shape     -- [2]

-- ---------------------------------------------------------------------------
-- Forward pass
-- ---------------------------------------------------------------------------

-- Input: 1 sample of 4 features → batch shape [1, 4]
def input1 : Tensor Float := Tensor.ones [1, 4]

-- output shape: [1, 2]
-- Each output = sum(w_j * 1.0) + 0.0  = 4 * 0.1 = 0.4
def out1 : Tensor Float := Linear.forward fc input1

#eval out1.shape   -- [1, 2]
#eval out1.data    -- #[0.4, 0.4]

-- ---------------------------------------------------------------------------
-- Batch forward pass
-- ---------------------------------------------------------------------------

-- 3 samples, 4 features each
def input3 : Tensor Float := Tensor.ones [3, 4]

def out3 : Tensor Float := Linear.forward fc input3

#eval out3.shape   -- [3, 2]
#eval out3.data    -- #[0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

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

#eval (Linear.forward fc2 input2).data   -- #[7.0, 7.0]
