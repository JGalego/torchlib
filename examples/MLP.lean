import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models

/-!
# Example: Multi-Layer Perceptron (MLP)

Demonstrates building, inspecting, and running forward passes through `MLP`
and `CNN` models from `TorchLib.Models`.
-/

open TorchLib

-- ---------------------------------------------------------------------------
-- Building an MLP
-- ---------------------------------------------------------------------------

-- 784-input MLP, two hidden layers [256, 128], 10-class output, no dropout
def mlp : MLP Float := MLP.init 784 [256, 128] 10

-- Inspect structure
#eval mlp.layers.size    -- 3  (784→256, 256→128, 128→10)
#eval mlp.outputSize     -- 10

-- ---------------------------------------------------------------------------
-- Forward pass: single image (flat 28×28)
-- ---------------------------------------------------------------------------

def img : Tensor Float := Tensor.zeros [1, 784]

def logits : Tensor Float := MLP.forward mlp img

#eval logits.shape   -- [1, 10]

-- ---------------------------------------------------------------------------
-- Forward pass: mini-batch of 8 images
-- ---------------------------------------------------------------------------

def batch8 : Tensor Float := Tensor.ones [8, 784]

def batchLogits : Tensor Float := MLP.forward mlp batch8

#eval batchLogits.shape   -- [8, 10]

-- ---------------------------------------------------------------------------
-- Small MLP for inspection (2-in, 1-out, one hidden layer of 4)
-- ---------------------------------------------------------------------------

def tiny : MLP Float := MLP.init 2 [4] 1

-- Weight shapes for each layer
#eval tiny.layers.map (fun l => l.weight.shape)
-- #[[4, 2], [1, 4]]

-- Bias shapes
#eval tiny.layers.map (fun l => l.bias.shape)
-- #[[4], [1]]

-- Run a single 2-feature sample
def x2 : Tensor Float := { shape := [1, 2], data := #[1.0, -1.0] }
def y2 : Tensor Float := MLP.forward tiny x2
#eval y2.shape   -- [1, 1]

-- ---------------------------------------------------------------------------
-- MLP with dropout
-- ---------------------------------------------------------------------------

def mlpDrop : MLP Float := MLP.init 32 [64, 32] 10 (dropoutP := 0.5)
#eval mlpDrop.dropout.p   -- 0.5

-- ---------------------------------------------------------------------------
-- CNN: convolutional network
-- ---------------------------------------------------------------------------

-- Two conv blocks: (1→8, k=3) and (8→16, k=3). Flatten size 400, MLP head → 10.
def cnn : CNN Float :=
  CNN.init [(1, 8, 3), (8, 16, 3)] 400 10 [128]

#eval cnn.convBlocks.size   -- 2
#eval cnn.head.outputSize   -- 10
