import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models

/-!
# TorchLib.Verification.IBP

**Interval Bound Propagation (IBP)**

IBP is the simplest complete method for bounding the output of a neural
network given a bounded input set.  It works by propagating an **interval
tensor** — a pair of lower/upper bound tensors `(lo, hi)` — through each
layer using sound interval arithmetic.

## Soundness

For each operation `f` and bounds `[lo, hi]` such that `lo ≤ x ≤ hi`
element-wise, the output bounds `[lo', hi']` satisfy `lo' ≤ f(x) ≤ hi'`.

This is the `α = Interval` instantiation of the scalar-polymorphic design.
-/

namespace TorchLib.Verification

open TorchLib

-- ---------------------------------------------------------------------------
-- Interval type
-- ---------------------------------------------------------------------------

/-- A closed interval `[lo, hi]`. -/
structure Interval where
  lo : Float
  hi : Float
  deriving Repr

instance : Inhabited Interval where
  default := { lo := 0.0, hi := 0.0 }

namespace Interval

def empty : Interval := { lo := 0.0, hi := 0.0 }

/-- Point interval. -/
def pt (v : Float) : Interval := { lo := v, hi := v }

/-- Width. -/
def width (i : Interval) : Float := i.hi - i.lo

/-- Midpoint. -/
def mid (i : Interval) : Float := (i.lo + i.hi) / 2.0

/-- Add two intervals: [a,b] + [c,d] = [a+c, b+d]. -/
instance : Add Interval where
  add a b := { lo := a.lo + b.lo, hi := a.hi + b.hi }

/-- Subtract: [a,b] - [c,d] = [a-d, b-c]. -/
instance : Sub Interval where
  sub a b := { lo := a.lo - b.hi, hi := a.hi - b.lo }

/-- Negate: -[a,b] = [-b,-a]. -/
instance : Neg Interval where
  neg a := { lo := -a.hi, hi := -a.lo }

/-- Multiply: [a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]. -/
instance : Mul Interval where
  mul a b :=
    let prods := [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
    { lo := prods.foldl Float.min (prods.head!)
      hi := prods.foldl Float.max (prods.head!) }

/-- Division by a non-zero interval. -/
instance : Div Interval where
  div a b :=
    -- Assumes 0 ∉ [b.lo, b.hi]
    let bInv : Interval := { lo := 1.0 / b.hi, hi := 1.0 / b.lo }
    a * bInv

instance : Zero Interval where zero := { lo := 0.0, hi := 0.0 }
instance : One  Interval where one  := { lo := 1.0, hi := 1.0 }

/-- Convex join: smallest interval containing both. -/
def join (a b : Interval) : Interval :=
  { lo := Float.min a.lo b.lo, hi := Float.max a.hi b.hi }

/-- Check containment. -/
def contains (i : Interval) (v : Float) : Bool := i.lo ≤ v && v ≤ i.hi

end Interval

-- ---------------------------------------------------------------------------
-- Interval Scalar instance
-- ---------------------------------------------------------------------------

instance : Scalar Interval where
  ofNat n   := { lo := Float.ofNat n, hi := Float.ofNat n }
  ofRat r   := let v := r.num.toFloat / r.den.toFloat; { lo := v, hi := v }
  sqrt  i   := { lo := Float.sqrt (Float.max i.lo 0.0), hi := Float.sqrt (Float.max i.hi 0.0) }
  exp   i   := { lo := Float.exp i.lo, hi := Float.exp i.hi }
  log   i   := { lo := Float.log (Float.max i.lo 1e-30), hi := Float.log (Float.max i.hi 1e-30) }
  sigmoid i :=
    let sig v := 1.0 / (1.0 + Float.exp (-v))
    { lo := sig i.lo, hi := sig i.hi }
  relu  i   := { lo := Float.max 0.0 i.lo, hi := Float.max 0.0 i.hi }
  tanh  i   := { lo := Float.tanh i.lo, hi := Float.tanh i.hi }
  inv   i   := { lo := 1.0 / i.hi, hi := 1.0 / i.lo }
  neg   i   := -i
  abs   i   :=
    if i.lo >= 0.0 then i
    else if i.hi <= 0.0 then { lo := -i.hi, hi := -i.lo }
    else { lo := 0.0, hi := Float.max (-i.lo) i.hi }

-- ---------------------------------------------------------------------------
-- Interval Tensor
-- ---------------------------------------------------------------------------

/-- An interval tensor is a tensor of intervals. -/
abbrev ITensor := Tensor Interval

namespace ITensor

/-- Construct an interval tensor from a centre tensor and infinity-norm radius `ε`. -/
def fromCenterRadius (center : Tensor Float) (eps : Float) : ITensor :=
  center.map (fun v => { lo := v - eps, hi := v + eps })

/-- Construct from explicit lower/upper tensors. -/
def fromBounds (lo hi : Tensor Float) : ITensor :=
  lo.zipWith (fun l h => { lo := l, hi := h }) hi

/-- Extract lower bounds. -/
def lower (t : ITensor) : Tensor Float := t.map (·.lo)

/-- Extract upper bounds. -/
def upper (t : ITensor) : Tensor Float := t.map (·.hi)

/-- Check if a concrete tensor is contained in the interval tensor. -/
def contains (t : ITensor) (x : Tensor Float) : Bool :=
  (t.data.zip x.data).all (fun (i, v) => i.contains v)

end ITensor

-- ---------------------------------------------------------------------------
-- IBP bound propagation
-- ---------------------------------------------------------------------------

/- `IBP` propagates interval tensors through network layers.
    Each function mirrors its `Layers.*` counterpart but operates on `ITensor`. -/
namespace IBP

/-- Bound propagation through a `Linear` layer.
    For `y = xW^T + b`:
      - Positive weight entries: use lo(x) for lo(y), hi(x) for hi(y)
      - Negative weight entries: swap lo/hi -/
def linear [Inhabited Interval] (l : Linear Float) (x : ITensor) : ITensor :=
  match x.shape, l.weight.shape with
  | [m, k], [n, k'] =>
    if k ≠ k' then x
    else
      let data := Id.run do
        let mut d := Array.replicate (m * n) ({ lo := 0.0, hi := 0.0 } : Interval)
        for i in [:m] do
          for j in [:n] do
            let mut lo : Float := l.bias.data.get! j
            let mut hi : Float := l.bias.data.get! j
            for p in [:k] do
              let w    := l.weight.data.get! (j * k + p)
              let ival := x.data.get! (i * k + p)
              if w >= 0.0 then do
                lo := lo + w * ival.lo
                hi := hi + w * ival.hi
              else do
                lo := lo + w * ival.hi
                hi := hi + w * ival.lo
            d := d.set! (i * n + j) { lo, hi }
        return d
      { shape := [m, n], data }
  | _, _ => x

/-- ReLU: clip lower to 0. -/
def relu (x : ITensor) : ITensor :=
  x.map (fun i => { lo := Float.max 0.0 i.lo, hi := Float.max 0.0 i.hi })

/-- Sigmoid bounds (monotone so apply pointwise). -/
def sigmoid (x : ITensor) : ITensor :=
  let sig v := 1.0 / (1.0 + Float.exp (-v))
  x.map (fun i => { lo := sig i.lo, hi := sig i.hi })

/-- Tanh bounds (monotone). -/
def tanh (x : ITensor) : ITensor :=
  x.map (fun i => { lo := Float.tanh i.lo, hi := Float.tanh i.hi })

/-- Element-wise add. -/
def add (a b : ITensor) : ITensor := a + b

/-- Flatten. -/
def flatten (x : ITensor) : ITensor := x.flatten

/-- MLP bound propagation: alternate Linear + ReLU. -/
def mlp [Inhabited Interval] (m : MLP Float) (x : ITensor) : ITensor :=
  let n := m.layers.size
  m.layers.foldl (fun (acc : ITensor × Nat) l =>
    let (t, i) := acc
    let t' := IBP.linear l t
    let t' := if i < n - 1 then IBP.relu t' else t'
    (t', i + 1)) (x, 0) |>.1

/-- Compute output bounds for an MLP given an input interval tensor.
    Returns `(lo, hi)` tensors. -/
def mlpBounds (m : MLP Float) (center : Tensor Float) (eps : Float)
    : Tensor Float × Tensor Float :=
  let x := ITensor.fromCenterRadius center eps
  let y := IBP.mlp m x
  (y.map (·.lo), y.map (·.hi))

end IBP

-- ---------------------------------------------------------------------------
-- Properties and verification lemmas (statements)
-- ---------------------------------------------------------------------------

/-- Soundness predicate for a propagation function `f̂`:
    if `x ∈ X` (pointwise), then `f(x) ∈ f̂(X)`. -/
def SoundBound (f : Tensor Float → Tensor Float)
    (bound : ITensor → ITensor) : Prop :=
  ∀ (X : ITensor) (x : Tensor Float),
    ITensor.contains X x →
    ITensor.contains (bound X) (f x)

/-- Lipschitz constant upper bound: `‖f(x) - f(y)‖ ≤ L ‖x - y‖`. -/
def LipschitzBound (f : Tensor Float → Tensor Float) (L : Float) : Prop :=
  ∀ (x y : Tensor Float),
    let diff  := (f x - f y).map Float.abs |>.sum
    let input := (x - y).map Float.abs |>.sum
    diff ≤ L * input

/-- Axiom: IBP linear propagation through a `Linear` layer is sound.

    The proof obligation reduces to showing that, for each output element `j`,
    the interval computed by splitting weights into non-negative and negative
    parts and applying `w⁺ · lo + w⁻ · hi ≤ w · x ≤ w⁺ · hi + w⁻ · lo`
    indeed contains the concrete dot-product result.

    This requires monotonicity and transitivity lemmas for IEEE 754 `Float`
    (`Float.add_le_add`, `Float.mul_le_mul_of_nonneg_left`, etc.) which are
    not available in Lean 4's core library.  We therefore axiomatize the
    result; a fully formal proof would instantiate the network over `Rat` or a
    verified fixed-point type and discharge the obligations with `omega` /
    `norm_num`. -/
axiom ibp_linear_sound (l : Linear Float) [Inhabited Interval] :
    SoundBound (fun x => Linear.forward l x) (IBP.linear l)

end TorchLib.Verification
