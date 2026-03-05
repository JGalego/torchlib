import TorchLib.Core
import TorchLib.IR

/-!
# TorchLib.Runtime.Autograd

Reverse-mode automatic differentiation (backpropagation) using a dynamic
**tape** (Wengert list) and a static **computation graph**.

## Design

Every `Variable` wraps a `Tensor Float` and optionally carries a gradient
accumulator.  During the forward pass, each primitive operation records itself
on a `Tape` as a `TapeEntry`:

```
TapeEntry := { grad_fn : List (Tensor Float) → List (Tensor Float), output_ids, input_ids }
```

Calling `backward` on a scalar loss replays the tape in reverse — the same
pattern as PyTorch's autograd engine.

## Relation to the IR

The same computation graph that the interpreter (`IR.evalGraph`) walks forward
is here walked backward to accumulate gradients.  For compiled execution the
tape would be derived from the static `Graph` rather than built dynamically.
-/

namespace TorchLib.Runtime

-- ---------------------------------------------------------------------------
-- Variable (a tensor with gradient tracking)
-- ---------------------------------------------------------------------------

/-- A unique identifier for each `Variable` created during a forward pass. -/
abbrev VarId := Nat

/-- A `Variable` wraps a `Tensor Float` with optional gradient storage.
    `requiresGrad` controls whether gradients are accumulated for this node. -/
structure Variable where
  id           : VarId
  data         : Tensor Float
  requiresGrad : Bool := true
  deriving Repr

-- ---------------------------------------------------------------------------
-- Tape (Wengert list)
-- ---------------------------------------------------------------------------

/-- A `GradFn` takes the gradient(s) of the outputs and produces the
    gradient(s) for each input. -/
abbrev GradFn := List (Tensor Float) → List (Tensor Float)

/-- One entry on the tape. -/
structure TapeEntry where
  outputIds : List VarId
  inputIds  : List VarId
  gradFn    : GradFn

/-- The `Tape` is a mutable list of tape entries plus a gradient accumulator
    (one `Tensor Float` per `VarId`, initialised to zero on first use). -/
structure Tape where
  entries  : Array TapeEntry
  grads    : Array (VarId × Tensor Float)  -- sparse accumulator

namespace Tape

def empty : Tape := { entries := #[], grads := #[] }

/-- Append an entry to the tape (called during the forward pass). -/
def push (t : Tape) (e : TapeEntry) : Tape :=
  { t with entries := t.entries.push e }

/-- Accumulate `grad` into the slot for `id`. -/
def accumulate (t : Tape) (id : VarId) (grad : Tensor Float) : Tape :=
  match t.grads.findIdx? (·.1 = id) with
  | some i =>
    let (_, existing) := t.grads.get! i
    { t with grads := t.grads.set! i (id, existing + grad) }
  | none =>
    { t with grads := t.grads.push (id, grad) }

/-- Look up accumulated gradient for `id`. -/
def getGrad (t : Tape) (id : VarId) : Option (Tensor Float) :=
  t.grads.find? (·.1 = id) |>.map (·.2)

end Tape

-- ---------------------------------------------------------------------------
-- Differentiation rules (VJPs for each op)
-- ---------------------------------------------------------------------------

/-- Vector-Jacobian product (VJP) rules for primitive ops.

    `vjp op inputs output_grad` returns `[∂L/∂input_i]` given `∂L/∂output`. -/
def vjp (op : IR.OpCode) (inputs : List (Tensor Float))
    (outGrad : Tensor Float) : List (Tensor Float) :=
  match op, inputs with
  -- Unary ops
  | .neg,     [_]    => [outGrad.map (· * -1.0)]
  | .abs,     [x]    => [x.zipWith (fun xi gi => if xi >= 0.0 then gi else -gi) outGrad]
  | .sqrt,    [x]    => [x.zipWith (fun xi gi => gi / (2.0 * Float.sqrt xi)) outGrad]
  | .exp,     [x]    => [x.zipWith (fun xi gi => gi * Float.exp xi) outGrad]
  | .log,     [x]    => [x.zipWith (fun xi gi => gi / xi) outGrad]
  | .relu,    [x]    => [x.zipWith (fun xi gi => if xi > 0.0 then gi else 0.0) outGrad]
  | .sigmoid, [x]    =>
      let s := x.map (fun xi => 1.0 / (1.0 + Float.exp (-xi)))
      [s.zipWith (fun si gi => gi * si * (1.0 - si)) outGrad]
  | .tanh,    [x]    =>
      let t := x.map Float.tanh
      [t.zipWith (fun ti gi => gi * (1.0 - ti * ti)) outGrad]
  | .softplus,[x]    =>
      let s := x.map (fun xi => 1.0 / (1.0 + Float.exp (-xi)))
      [s.zipWith (· * ·) outGrad]
  | .gelu,    [_]    => [outGrad]  -- approximate: identity VJP (placeholder)
  | .silu,    [x]    =>
      -- d/dx (x σ(x)) = σ(x) + x σ(x)(1-σ(x))
      let s := x.map (fun xi => 1.0 / (1.0 + Float.exp (-xi)))
      let ds := s.zipWith (fun si xi => si + xi * si * (1.0 - si)) x
      [ds.zipWith (· * ·) outGrad]
  -- Binary elementwise
  | .add,     [_, _] => [outGrad, outGrad]
  | .sub,     [_, _] => [outGrad, outGrad.map (· * -1.0)]
  | .mul,     [a, b] => [b.zipWith (· * ·) outGrad, a.zipWith (· * ·) outGrad]
  | .div,     [a, b] =>
      let ga := b.map (1.0 / ·) |>.zipWith (· * ·) outGrad
      let gb := a.zipWith (· / ·) b |>.zipWith (· / ·) b |>.map (· * -1.0)
                |>.zipWith (· * ·) outGrad
      [ga, gb]
  -- Reductions
  | .sumAll,  [x]    =>
      -- Broadcast scalar gradient to full tensor shape
      let s := if outGrad.data.isEmpty then 0.0 else outGrad.data.get! 0
      [x.map (fun _ => s)]
  | .sumAxis _, [x]  => [x.map (fun _ =>
      if outGrad.data.isEmpty then 0.0 else outGrad.data.getD 0 default)]
  -- Linear algebra
  | .matmul,  [a, b] =>
      -- ∂L/∂A = ∂L/∂C · B^T,  ∂L/∂B = A^T · ∂L/∂C
      let ga := Tensor.matmul outGrad b.transpose
      let gb := Tensor.matmul a.transpose outGrad
      [ga, gb]
  | .transpose, [x]  =>
      let _ := x
      [outGrad.transpose]
  -- Flatten/reshape
  | .flatten, [x]    =>
      match outGrad.reshape x.shape with
      | some g => [g]
      | none   => [outGrad]
  | .reshape s, [x]  =>
      match outGrad.reshape x.shape with
      | some g => [g]
      | none   =>
        let _ := s
        [outGrad]
  | _, ins => ins.map (fun _ => outGrad)  -- default: pass-through

-- ---------------------------------------------------------------------------
-- AutogradEngine
-- ---------------------------------------------------------------------------

/-- The `AutogradEngine` manages a global `Tape` and `VarId` counter during a
    forward pass.  It is threaded through computations using `IO.Ref`. -/
structure AutogradEngine where
  tape      : IO.Ref Tape
  nextVarId : IO.Ref Nat

namespace AutogradEngine

/-- Create a fresh engine. -/
def init : IO AutogradEngine := do
  let tape      ← IO.mkRef Tape.empty
  let nextVarId ← IO.mkRef 0
  return { tape, nextVarId }

/-- Allocate a fresh `VarId`. -/
def freshId (eng : AutogradEngine) : IO VarId := do
  let id ← eng.nextVarId.get
  eng.nextVarId.set (id + 1)
  return id

/-- Make a `Variable`, registering it on the engine. -/
def mkVar (eng : AutogradEngine) (data : Tensor Float)
    (requiresGrad : Bool := true) : IO Variable := do
  let id ← eng.freshId
  return { id, data, requiresGrad }

/-- Record a primitive operation on the tape. -/
def record (eng : AutogradEngine) (op : IR.OpCode)
    (inputs : List Variable) (outputs : List Variable) : IO Unit := do
  if inputs.any (·.requiresGrad) then
    let entry : TapeEntry :=
      { outputIds := outputs.map (·.id)
        inputIds  := inputs.map (·.id)
        gradFn    := fun outGrads =>
          let outGrad := outGrads.headD (Tensor.zeros [1])
          vjp op (inputs.map (·.data)) outGrad }
    eng.tape.modify (·.push entry)

/-- `apply op inputs` runs the op, allocates output variables, and records the
    entry on the tape. -/
def apply (eng : AutogradEngine) (op : IR.OpCode) (inputs : List Variable)
    : IO (List Variable) := do
  let outTensors := IR.applyOp op (inputs.map (·.data))
  let outputs ← outTensors.mapM (fun t => eng.mkVar t)
  eng.record op inputs outputs
  return outputs

-- ---------------------------------------------------------------------------
-- Backward pass
-- ---------------------------------------------------------------------------

/-- Run backpropagation from `lossVar` (a scalar variable).
    Returns the tape with accumulated gradients. -/
def backward (eng : AutogradEngine) (lossVar : Variable) : IO Tape := do
  -- Seed the loss gradient with 1.0
  let seedGrad : Tensor Float := { shape := [1], data := #[1.0] }
  eng.tape.modify (·.accumulate lossVar.id seedGrad)
  -- Replay tape in reverse
  let t ← eng.tape.get
  let entries := t.entries.reverse
  let tape ← IO.mkRef t
  for entry in entries do
    let curTape ← tape.get
    -- Gather output gradients
    let outGrads := entry.outputIds.map (fun id =>
      curTape.getGrad id |>.getD (Tensor.zeros [1]))
    -- Compute input gradients
    let inGrads := entry.gradFn outGrads
    -- Accumulate into input slots
    let updatedTape := inGrads.zip entry.inputIds |>.foldl
      (fun acc (g, id) => acc.accumulate id g) curTape
    tape.set updatedTape
  return ← tape.get

end AutogradEngine

-- ---------------------------------------------------------------------------
-- Convenience: zero-gradient sweep
-- ---------------------------------------------------------------------------

/-- Set all gradients in `tape` to zero for the given variable ids.
    Used before each training step. -/
def zeroGrads (tape : Tape) (ids : List VarId) : Tape :=
  ids.foldl (fun t id =>
    match t.grads.findIdx? (·.1 = id) with
    | some i =>
      let (_, g) := t.grads.get! i
      { t with grads := t.grads.set! i (id, Tensor.zeros g.shape) }
    | none => t) tape

end TorchLib.Runtime
