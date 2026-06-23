import TorchLib.Core
import TorchLib.IR
import TorchLib.Layers
import TorchLib.Models

/-!
# TorchLib.Runtime.Float32

**IEEE-754 binary32 kernel (`IEEE32Exec`)**

TorchLean distinguishes its *trusted* numeric semantics (binary32, the format
real hardware uses) from the convenient binary64 `Float` runtime.  This module
provides the same split:

- a `Scalar Float32` instance, so the *entire* scalar-polymorphic stack
  (`Tensor`, `Layers`, `Models`) can be instantiated at `α = Float32` and run
  in genuine single precision; and
- `IEEE32Exec` — an interpreter that evaluates the shared SSA/DAG `IR.Graph` in
  binary32, mirroring the `Float`-valued `IR.applyOp` op-for-op.

`Float32` is Lean's runtime-backed binary32 type: every operation rounds to 24
significant bits exactly as the hardware would.  For example `(0.1 : Float)`
and `(0.1 : Float).toFloat32` are *not* equal — the rounding is real, not
simulated.
-/

namespace TorchLib

-- ---------------------------------------------------------------------------
-- Scalar instance for binary32
-- ---------------------------------------------------------------------------

/-- IEEE-754 binary32 as a `Scalar`.  Arithmetic and transcendentals round to
    single precision after every operation, matching `IEEE32Exec`. -/
instance : Scalar Float32 where
  ofNat n   := Float32.ofNat n
  ofRat r   := (Float.ofInt r.num / Float.ofNat r.den).toFloat32
  sqrt      := Float32.sqrt
  exp       := Float32.exp
  log       := Float32.log
  sigmoid x := 1.0 / (1.0 + Float32.exp (-x))
  relu x    := if x > 0.0 then x else 0.0
  tanh      := Float32.tanh
  inv x     := 1.0 / x
  abs       := Float32.abs

namespace Tensor

/-- Round a `Float` (binary64) tensor to binary32. -/
def toF32 (t : Tensor Float) : Tensor Float32 := t.map Float.toFloat32

/-- Widen a binary32 tensor back to `Float` (binary64). -/
def toF64 (t : Tensor Float32) : Tensor Float := t.map Float32.toFloat

end Tensor

namespace Runtime

-- ---------------------------------------------------------------------------
-- IEEE32Exec — binary32 interpreter for the shared IR
-- ---------------------------------------------------------------------------

/-- Apply a single `IR.OpCode` to binary32 input tensors.  This is the exact
    binary32 analogue of `IR.applyOp`; every op rounds to single precision. -/
def applyOp32 (op : IR.OpCode) (inputs : List (Tensor Float32))
    : List (Tensor Float32) :=
  match op, inputs with
  -- Unary elementwise
  | .neg,     [t] => [t.map (-·)]
  | .abs,     [t] => [t.map Float32.abs]
  | .sqrt,    [t] => [t.map Float32.sqrt]
  | .exp,     [t] => [t.map Float32.exp]
  | .log,     [t] => [t.map Float32.log]
  | .relu,    [t] => [t.map (fun x => if x > 0.0 then x else 0.0)]
  | .sigmoid, [t] => [t.map (fun x => 1.0 / (1.0 + Float32.exp (-x)))]
  | .tanh,    [t] => [t.map Float32.tanh]
  | .softplus,[t] => [t.map (fun x => Float32.log (1.0 + Float32.exp x))]
  | .gelu,    [t] => [t.map (fun x =>
      0.5 * x * (1.0 + Float32.tanh (0.7978845608 * (x + 0.044715 * x * x * x))))]
  | .silu,    [t] => [t.map (fun x => x / (1.0 + Float32.exp (-x)))]
  -- Binary elementwise
  | .add,     [a, b] => [a + b]
  | .sub,     [a, b] => [a - b]
  | .mul,     [a, b] => [a * b]
  | .div,     [a, b] => [a.zipWith (· / ·) b]
  | .maximum, [a, b] => [a.zipWith (fun x y => if x > y then x else y) b]
  | .minimum, [a, b] => [a.zipWith (fun x y => if x < y then x else y) b]
  -- Reductions
  | .sumAll,  [t] => [{ shape := [1], data := #[t.sum] }]
  -- Shape
  | .flatten, [t] => [t.flatten]
  | .transpose, [t] => [t.transpose]
  -- Linear algebra
  | .matmul,  [a, b] => [Tensor.matmul a b]
  | .bmm,     [a, b] => [Tensor.bmm a b]
  -- Normalisation
  | .layerNorm _ eps, [t] => [Tensor.layerNorm t eps.toFloat32]
  -- Concatenation
  | .cat _ _, ts => [ts.foldl Tensor.cat (Tensor.zeros [])]
  | _, _ => inputs  -- unhandled: pass through

/-- Interpreter environment binding SSA `ValueId`s to binary32 tensors. -/
structure InterpEnv32 where
  /-- Mapping from `ValueId` to its binary32 tensor value. -/
  values : List (IR.ValueId × Tensor Float32)

namespace InterpEnv32

/-- Empty environment. -/
def empty : InterpEnv32 := { values := [] }

/-- Bind a `ValueId` to a tensor, replacing any existing binding. -/
def insert (env : InterpEnv32) (id : IR.ValueId) (t : Tensor Float32) : InterpEnv32 :=
  { values := (id, t) :: env.values.filter (·.1 ≠ id) }

/-- Look up the tensor bound to a `ValueId`. -/
def lookup (env : InterpEnv32) (id : IR.ValueId) : Option (Tensor Float32) :=
  env.values.find? (·.1 = id) |>.map (·.2)

/-- Look up several `ValueId`s, dropping any that are missing. -/
def lookupMany (env : InterpEnv32) (ids : List IR.ValueId) : List (Tensor Float32) :=
  ids.filterMap (lookup env)

end InterpEnv32

/-- `IEEE32Exec`: evaluate an `IR.Graph` in binary32 under initial bindings. -/
def evalGraph32 (g : IR.Graph) (env : InterpEnv32) : InterpEnv32 :=
  g.nodes.foldl (fun e n =>
    let args := e.lookupMany n.inputs
    let results := applyOp32 n.op args
    results.zip n.outputs |>.foldl (fun e' (t, v) => e'.insert v.id t) e) env

-- ---------------------------------------------------------------------------
-- Casting a trained Float model down to binary32
-- ---------------------------------------------------------------------------

/-- Cast a `Linear Float` layer to binary32. -/
def Linear.toF32 (l : Linear Float) : Linear Float32 :=
  { weight := l.weight.toF32, bias := l.bias.toF32 }

/-- Cast an `MLP Float` to binary32 so it can be executed in single precision. -/
def MLP.toF32 (m : MLP Float) : MLP Float32 :=
  { layers := m.layers.map Linear.toF32
    dropout := { p := m.dropout.p, training := m.dropout.training }
    outputSize := m.outputSize }

/-- Run an `MLP Float` forward in binary32, returning a binary32 tensor.
    The eager-mode counterpart to `evalGraph32`. -/
def mlpForward32 (m : MLP Float) (x : Tensor Float) : Tensor Float32 :=
  MLP.forward (MLP.toF32 m) x.toF32

/-- Maximum absolute element-wise gap between the binary64 and binary32 forward
    passes — a direct measure of the finite-precision rounding error. -/
def precisionGap (m : MLP Float) (x : Tensor Float) : Float :=
  let y64 := MLP.forward m x
  let y32 := (mlpForward32 m x).toF64
  (y64 - y32).map Float.abs |>.data.foldl Float.max 0.0

end Runtime
end TorchLib
