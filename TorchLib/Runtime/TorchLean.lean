import TorchLib.Core
import TorchLib.IR
import TorchLib.Layers
import TorchLib.Models
import TorchLib.Runtime.Autograd
import TorchLib.Runtime.Training

/-!
# TorchLib.Runtime.TorchLean

PyTorch-style high-level API — eager and compiled execution.

This module re-exports the core building blocks under a clean, flat namespace
that mirrors the PyTorch API as closely as possible in Lean 4.

## Eager mode

Functions called directly on `Tensor Float` values; gradients tracked via
`AutogradEngine`.

## Compiled mode

A model is first lowered to the shared `IR.Graph` via `toGraph`, then the
`IR.evalGraph` interpreter (or a future native-code backend) is invoked.

The distinction between eager and compiled is surfaced through the
`ExecutionMode` configuration.
-/

namespace TorchLib.Runtime.TorchLean

open TorchLib
open TorchLib.Runtime

-- ---------------------------------------------------------------------------
-- Execution mode
-- ---------------------------------------------------------------------------

/-- Controls how a model is executed. -/
inductive ExecutionMode
  | eager     -- Direct Lean evaluation (default)
  | compiled  -- Lower to IR then interpret
  deriving Repr, BEq

-- ---------------------------------------------------------------------------
-- nn.Module wrapper
-- ---------------------------------------------------------------------------

/-- A `Module` bundles:
    - `params`    — the current weight `StateDict`
    - `forward`   — takes params + input, returns output tensor
    - `training`  — flag enabling dropout / batch-norm training behaviour -/
structure Module where
  /-- Current weight state dictionary. -/
  params   : StateDict Float
  /-- Forward function: takes parameters and an input tensor. -/
  forwardFn : StateDict Float → Tensor Float → Tensor Float
  /-- Whether the module is in training mode. -/
  training : Bool := true
  /-- Human-readable module name. -/
  name     : String := "module"

namespace Module

/-- Evaluate the module on an input. -/
def forward (m : Module) (x : Tensor Float) : Tensor Float :=
  m.forwardFn m.params x

/-- Switch to eval mode (disables dropout/batchnorm training). -/
def eval (m : Module) : Module := { m with training := false }

/-- Switch to train mode. -/
def train (m : Module) : Module := { m with training := true }

/-- Return all parameter tensors as a flat list `(name, tensor)`. -/
def parameters (m : Module) : List (String × Tensor Float) := m.params.params

/-- Count total trainable parameters. -/
def numParams (m : Module) : Nat :=
  m.params.params.foldl (fun acc (_, t) => acc + t.numel) 0

/-- Update parameters from a list of `(name, newTensor)` pairs. -/
def updateParams (m : Module) (updates : List (String × Tensor Float)) : Module :=
  { m with params := updates.foldl (fun sd (k, v) => sd.insert k v) m.params }

end Module

-- ---------------------------------------------------------------------------
-- Functional API  (torch.nn.functional)
-- ---------------------------------------------------------------------------

namespace F

/-- ReLU activation. -/
def relu (x : Tensor Float) : Tensor Float := x.apply (fun v => if v > 0.0 then v else 0.0)

/-- Leaky ReLU. -/
def leakyRelu (negSlope : Float := 0.01) (x : Tensor Float) : Tensor Float :=
  x.apply (fun v => if v > 0.0 then v else negSlope * v)

/-- Sigmoid. -/
def sigmoid (x : Tensor Float) : Tensor Float :=
  x.apply (fun v => 1.0 / (1.0 + Float.exp (-v)))

/-- Tanh. -/
def tanh (x : Tensor Float) : Tensor Float := x.apply Float.tanh

/-- Softmax along the last axis of a 2-D tensor. -/
def softmax (x : Tensor Float) : Tensor Float := Tensor.softmax x

/-- Log-softmax (numerically stable). -/
def logSoftmax (x : Tensor Float) : Tensor Float :=
  match x.shape with
  | [m, n] =>
    let data := Id.run do
      let mut d := x.data
      for i in [:m] do
        let mut maxV : Float := d.get! (i * n)
        for j in [:n] do
          if d.get! (i * n + j) > maxV then maxV := d.get! (i * n + j)
        let mut s : Float := 0.0
        for j in [:n] do
          s := s + Float.exp (d.get! (i * n + j) - maxV)
        let logZ := Float.log s + maxV
        for j in [:n] do
          d := d.set! (i * n + j) (d.get! (i * n + j) - logZ)
      return d
    { shape := x.shape, data }
  | _ => x

/-- GELU activation. -/
def gelu (x : Tensor Float) : Tensor Float :=
  x.apply (fun v =>
    0.5 * v * (1.0 + Float.tanh (0.7978845608 * (v + 0.044715 * v * v * v))))

/-- SiLU / Swish activation. -/
def silu (x : Tensor Float) : Tensor Float :=
  x.apply (fun v => v / (1.0 + Float.exp (-v)))

/-- Linear: `y = x W^T + b`. -/
def linear (x w b : Tensor Float) : Tensor Float :=
  let l : Linear Float := { weight := w, bias := b }
  Linear.forward l x

/-- Layer normalisation. -/
def layerNorm (x : Tensor Float) (eps : Float := 1e-5) : Tensor Float :=
  Tensor.layerNorm x eps

/-- Dropout (inference: identity; training: Bernoulli masking). -/
def dropout (x : Tensor Float) (p : Float := 0.5) (training : Bool := false) : Tensor Float :=
  Dropout.forward { p, training } x

/-- MSE loss. -/
def mseLoss (pred target : Tensor Float) : Float := TorchLib.Runtime.mseLoss pred target

/-- Cross-entropy from logits. -/
def crossEntropyLoss (logits : Tensor Float) (targets : Array Nat) : Float :=
  TorchLib.Runtime.crossEntropyLoss logits targets

/-- Flatten a tensor from dimension `startDim` onwards to a 1-D slice. -/
def flatten (x : Tensor Float) : Tensor Float := x.flatten

/-- Concatenate along axis 0. -/
def cat (tensors : List (Tensor Float)) : Tensor Float :=
  tensors.foldl Tensor.cat (Tensor.zeros [0])

end F

-- ---------------------------------------------------------------------------
-- Training loop utilities
-- ---------------------------------------------------------------------------

/-- One training step: forward → loss → backward → optimizer step.
    Returns `(updatedStateDict, loss)`. -/
def trainStep
    (model    : Module)
    (optimizer : AdamConfig)
    (optimState : AdamState)
    (x       : Tensor Float)
    (targets : Array Nat)
    : IO (Module × AdamState × Float) := do
  let eng ← AutogradEngine.init
  -- Build variables for all parameters
  let paramVars ← model.params.params.mapM (fun (name, t) => do
    let v ← eng.mkVar t
    return (name, v))
  -- Forward pass (using the variables' data)
  let inputVar ← eng.mkVar x
  let pred := model.forward inputVar.data
  -- Loss
  let loss := F.crossEntropyLoss pred targets
  let lossTensor : Tensor Float := { shape := [1], data := #[loss] }
  let lossVar ← eng.mkVar lossTensor
  -- Backward
  let tape ← eng.backward lossVar
  -- Collect gradients
  let grads := paramVars.filterMap (fun (name, v) =>
    tape.getGrad v.id |>.map (fun g => (name, v.data, g)))
  -- Optimiser step
  let (newOptState, newParams) := Adam.step optimizer optimState grads
  let newModel := model.updateParams newParams
  return (newModel, newOptState, loss)

-- ---------------------------------------------------------------------------
-- Model factories (mirrors torch.nn.Sequential / named constructors)
-- ---------------------------------------------------------------------------

/-- Build an `MLP` `Module`. -/
def mlpModule (inputSize : Nat) (hiddenSizes : List Nat) (outputSize : Nat)
    (dropoutP : Float := 0.0) : Module :=
  let mlp := MLP.init inputSize hiddenSizes outputSize dropoutP (α := Float)
  let paramList : List (String × Tensor Float) :=
    let layerList := mlp.layers.toList
    (List.range layerList.length).zip layerList |>.flatMap (fun (i, l) =>
      [("layer" ++ toString i ++ ".weight", l.weight),
       ("layer" ++ toString i ++ ".bias",   l.bias)])
  { params    := { params := paramList }
    forwardFn := fun _sd inp => MLP.forward mlp inp
    name      := "MLP" }

/-- Build a `Transformer` `Module`. -/
def transformerModule
    (vocabSize embedDim numHeads ffnDim numLayers maxSeqLen outputDim : Nat)
    (eps : Float := 1e-5) (dropoutP : Float := 0.0) : Module :=
  let tr := Transformer.init vocabSize embedDim numHeads ffnDim numLayers
              maxSeqLen outputDim eps dropoutP (α := Float)
  { params    := StateDict.empty  -- weights embedded in closures
    forwardFn := fun _sd inp =>
      -- For demonstration: treat the input as a 1-D index tensor
      let tokenIds := inp.data.toList.map (fun v => v.toUInt64.toNat)
      Transformer.forward tr tokenIds
    name := "Transformer" }

-- ---------------------------------------------------------------------------
-- DataLoader stub
-- ---------------------------------------------------------------------------

/-- A minimal `DataLoader` that wraps a flat list of `(input, label)` pairs and
    yields minibatches.  In a production implementation this would use
    `IO.Channel` or lazy streams. -/
structure DataLoader where
  /-- Array of `(input, labels)` pairs forming the dataset. -/
  dataset   : Array (Tensor Float × Array Nat)
  /-- Number of samples per minibatch. -/
  batchSize : Nat
  /-- Whether to shuffle data each epoch. -/
  shuffle   : Bool := true

instance : Repr DataLoader where
  reprPrec dl prec := Repr.addAppParen
    f!"DataLoader \{ dataset.size := {reprPrec dl.dataset.size 0}, batchSize := {reprPrec dl.batchSize 0}, shuffle := {reprPrec dl.shuffle 0} }"
    prec

namespace DataLoader

/-- Iterate over all minibatches in epoch order (no shuffling for purity). -/
def batches (dl : DataLoader)
    : List (Array (Tensor Float) × Array (Array Nat)) :=
  let n := dl.dataset.size
  let numBatches := (n + dl.batchSize - 1) / dl.batchSize
  (List.range numBatches).map (fun bi =>
    let start := bi * dl.batchSize
    let end_  := (start + dl.batchSize).min n
    let batch := dl.dataset.extract start end_
    (batch.map (·.1), batch.map (·.2)))

end DataLoader

-- ---------------------------------------------------------------------------
-- Summary utility
-- ---------------------------------------------------------------------------

/-- Print a brief model summary. -/
def summary (m : Module) : String :=
  let header := "Model: " ++ m.name ++ "\n"
  let rows := m.params.params.map (fun (name, t) =>
    "  " ++ name ++ "\t" ++ toString t.shape ++ "\t" ++ toString t.numel)
  let footer := "Total parameters: " ++ toString (m.numParams)
  header ++ rows.foldl (· ++ "\n" ++ ·) "" ++ "\n" ++ footer

end TorchLib.Runtime.TorchLean
