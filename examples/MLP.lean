import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models

/-!
# Example: Multi-Layer Perceptron (MLP)

Demonstrates building, inspecting, and running forward passes through `MLP`
and `CNN` models from `TorchLib.Models`.
-/

open TorchLib

/-- Print a labelled value to stdout using its `Repr` instance. -/
private def say [Repr α] (label : String) (v : α) : IO Unit :=
  IO.println s!"{label}: {reprStr v}"

-- ---------------------------------------------------------------------------
-- Building an MLP
-- ---------------------------------------------------------------------------

-- 784-input MLP, two hidden layers [256, 128], 10-class output, no dropout
def mlp : MLP Float := MLP.init 784 [256, 128] 10

-- Inspect structure
#eval say "mlp.layers.size" mlp.layers.size
#eval say "mlp.outputSize"  mlp.outputSize

-- ---------------------------------------------------------------------------
-- Forward pass: single image (flat 28×28)
-- ---------------------------------------------------------------------------

def img : Tensor Float := Tensor.zeros [1, 784]

def logits : Tensor Float := MLP.forward mlp img

#eval say "logits.shape" logits.shape

-- ---------------------------------------------------------------------------
-- Forward pass: mini-batch of 8 images
-- ---------------------------------------------------------------------------

def batch8 : Tensor Float := Tensor.ones [8, 784]

def batchLogits : Tensor Float := MLP.forward mlp batch8

#eval say "batchLogits.shape" batchLogits.shape

-- ---------------------------------------------------------------------------
-- Small MLP for inspection (2-in, 1-out, one hidden layer of 4)
-- ---------------------------------------------------------------------------

def tiny : MLP Float := MLP.init 2 [4] 1

-- Weight shapes for each layer
#eval say "tiny weight shapes" (tiny.layers.map (fun l => l.weight.shape))

-- Bias shapes
#eval say "tiny bias shapes"   (tiny.layers.map (fun l => l.bias.shape))

-- Run a single 2-feature sample
def x2 : Tensor Float := { shape := [1, 2], data := #[1.0, -1.0] }
def y2 : Tensor Float := MLP.forward tiny x2
#eval say "y2.shape" y2.shape

-- ---------------------------------------------------------------------------
-- MLP with dropout
-- ---------------------------------------------------------------------------

def mlpDrop : MLP Float := MLP.init 32 [64, 32] 10 (dropoutP := 0.5)
#eval say "mlpDrop.dropout.p" mlpDrop.dropout.p

-- Forward pass with dropout: some activations are zeroed during training
def dropInput : Tensor Float := Tensor.ones [1, 32]
def dropOut   : Tensor Float := MLP.forward mlpDrop dropInput

#eval say "dropOut.shape" dropOut.shape
#eval say "dropOut.data"  dropOut.data

-- ---------------------------------------------------------------------------
-- CNN: convolutional network
-- ---------------------------------------------------------------------------

-- Two conv blocks: (1→8, k=3) and (8→16, k=3). Flatten size 400, MLP head → 10.
def cnn : CNN Float :=
  CNN.init [(1, 8, 3), (8, 16, 3)] 400 10 [128]

#eval say "cnn.convBlocks.size" cnn.convBlocks.size
#eval say "cnn.head.outputSize" cnn.head.outputSize

-- Forward pass: input [batch, C_in=1, H=9, W=9].
-- After two k=3 convolutions (stride 1, no padding): 9→7→5,
-- so the flat feature size is 16×5×5 = 400, matching `flatSize`.
def cnnInput  : Tensor Float := Tensor.full [1, 1, 9, 9] 0.5
def cnnLogits : Tensor Float := CNN.forward cnn cnnInput

#eval say "cnnLogits.shape" cnnLogits.shape
#eval say "cnnLogits.data"  cnnLogits.data

-- ---------------------------------------------------------------------------
-- Main
-- ---------------------------------------------------------------------------

def main : IO Unit := do
  IO.println "=== MLP ==="
  say "mlp.layers.size" mlp.layers.size
  say "mlp.outputSize"  mlp.outputSize
  say "logits.shape" logits.shape
  say "batchLogits.shape" batchLogits.shape

  IO.println "\n=== Tiny MLP ==="
  say "tiny weight shapes" (tiny.layers.map (fun l => l.weight.shape))
  say "tiny bias shapes"   (tiny.layers.map (fun l => l.bias.shape))
  say "y2.shape" y2.shape

  IO.println "\n=== MLP with dropout ==="
  say "mlpDrop.dropout.p" mlpDrop.dropout.p
  say "dropOut.shape" dropOut.shape
  say "dropOut.data"  dropOut.data

  IO.println "\n=== CNN ==="
  say "cnn.convBlocks.size" cnn.convBlocks.size
  say "cnn.head.outputSize" cnn.head.outputSize
  say "cnnLogits.shape" cnnLogits.shape
  say "cnnLogits.data"  cnnLogits.data
