/-!
# TorchLib.Core

Foundational types for TorchLib: scalar polymorphism, shapes, and tensors.

The key design principle is **scalar polymorphism**: definitions are parameterized
over a scalar type `α` so that the same network definition can be:
- Proven correct over `ℝ` (NN.Spec)
- Executed efficiently over `Float` / `Rat` (NN.Runtime)
- Bounded over `Interval` (NN.Verification)
-/

-- ---------------------------------------------------------------------------
-- Compatibility shims for APIs missing in Lean 4.28
-- ---------------------------------------------------------------------------

/-- `Array.get!` with panic-free fallback to `default`. -/
@[inline] def Array.get! [Inhabited α] (a : Array α) (i : Nat) : α :=
  a.getD i default

/-- `Float.max` — not built-in in Lean 4.28. -/
@[inline] def Float.max (a b : Float) : Float := if a > b then a else b

/-- `Float.min` — not built-in in Lean 4.28. -/
@[inline] def Float.min (a b : Float) : Float := if a < b then a else b

/-- `Int.toFloat` — not built-in in Lean 4.28. -/
@[inline] def Int.toFloat (n : Int) : Float := Float.ofInt n

namespace TorchLib

-- ---------------------------------------------------------------------------
-- Shape
-- ---------------------------------------------------------------------------

/-- A tensor shape is a list of dimension sizes. -/
abbrev Shape := List Nat

/-- Total number of elements given a shape. -/
def Shape.numel : Shape → Nat
  | []      => 1
  | d :: ds => d * Shape.numel ds

/-- Number of dimensions (rank). -/
def Shape.rank (s : Shape) : Nat := s.length

/-- A shape is valid if all dimensions are positive. -/
def Shape.valid (s : Shape) : Prop := s.all (· > 0) = true

-- ---------------------------------------------------------------------------
-- Scalar typeclass
-- ---------------------------------------------------------------------------

/-- `Scalar α` captures the algebraic structure required to define neural network
    operations.  It extends the basic numeric hierarchy with transcendental and
    activation functions so that the same layer code can be evaluated over
    `Float`, `Rat`, or symbolic `ℝ`. -/
class Scalar (α : Type) extends
    Add α, Sub α, Mul α, Div α, Neg α,
    Zero α, One α, Repr α where
  /-- Scalar from a natural number literal. -/
  ofNat   : Nat → α
  /-- Scalar from a rational literal (used for weight initialisation). -/
  ofRat   : Rat → α
  /-- Square root. -/
  sqrt    : α → α
  /-- Natural exponential. -/
  exp     : α → α
  /-- Natural logarithm (partial; undefined at ≤ 0 for reals). -/
  log     : α → α
  /-- Sigmoid activation: σ(x) = 1/(1+e^{-x}). -/
  sigmoid : α → α
  /-- Rectified linear unit. -/
  relu    : α → α
  /-- Hyperbolic tangent. -/
  tanh    : α → α
  /-- Multiplicative inverse. -/
  inv     : α → α
  /-- Absolute value. -/
  abs     : α → α

/-- Convenience: the scalar zero. -/
def Scalar.szero [Scalar α] : α := Zero.zero

/-- Convenience: the scalar one. -/
def Scalar.sone [Scalar α] : α := One.one

instance : Scalar Float where
  ofNat n   := Float.ofNat n
  ofRat r   := Float.ofInt r.num / Float.ofNat r.den
  sqrt      := Float.sqrt
  exp       := Float.exp
  log       := Float.log
  sigmoid x := 1.0 / (1.0 + Float.exp (-x))
  relu x    := if x > 0.0 then x else 0.0
  tanh      := Float.tanh
  inv x     := 1.0 / x
  abs x     := Float.abs x

/-! ## Rat scalar instance -/
instance : Scalar Rat where
  ofNat n   := (n : Rat)
  ofRat r   := r
  sqrt _    := 0  -- not computable for Rat; placeholder
  exp _     := 0
  log _     := 0
  sigmoid _ := 0
  relu x    := if x > 0 then x else 0
  tanh _    := 0
  inv x     := x⁻¹
  abs x     := if x ≥ 0 then x else -x

-- ---------------------------------------------------------------------------
-- Tensor
-- ---------------------------------------------------------------------------

/-- A multidimensional array with a statically-tracked shape stored at runtime.

    The flat `data` array stores elements in row-major (C-contiguous) order.
    `shape` is carried at runtime for dynamic shape checks. -/
structure Tensor (α : Type) where
  shape  : Shape
  data   : Array α
  deriving Repr

instance [Inhabited α] : Inhabited (Tensor α) where
  default := { shape := [], data := #[] }

namespace Tensor

/-- Construct a tensor filled with a constant value. -/
def full (s : Shape) (v : α) : Tensor α :=
  { shape := s, data := Array.replicate s.numel v }

/-- Zero-filled tensor. -/
def zeros [Zero α] (s : Shape) : Tensor α := full s 0

/-- One-filled tensor. -/
def ones [One α] (s : Shape) : Tensor α := full s 1

/-- Number of elements. -/
def numel (t : Tensor α) : Nat := t.shape.numel

/-- Rank (number of dimensions). -/
def rank (t : Tensor α) : Nat := t.shape.rank

/-- Get element at a flat index with a default. -/
def getD [Inhabited α] (t : Tensor α) (i : Nat) : α := t.data.getD i default

/-- Set element at a flat index. -/
def set (t : Tensor α) (i : Nat) (v : α) : Tensor α :=
  { t with data := t.data.set! i v }

/-- Map a function over every element. -/
def map (f : α → β) (t : Tensor α) : Tensor β :=
  { shape := t.shape, data := t.data.map f }

/-- Element-wise binary operation (can change output type). -/
def zipWith (f : α → β → γ) (a : Tensor α) (b : Tensor β) : Tensor γ :=
  { shape := a.shape, data := Array.zipWith f a.data b.data }

/-- Flatten to a 1-D tensor. -/
def flatten (t : Tensor α) : Tensor α :=
  { shape := [t.numel], data := t.data }

/-- Reshape (total elements must agree). -/
def reshape (t : Tensor α) (s : Shape) : Option (Tensor α) :=
  if s.numel = t.numel then some { shape := s, data := t.data } else none

/-- Transpose a 2-D tensor. -/
def transpose [Inhabited α] (t : Tensor α) : Tensor α :=
  match t.shape with
  | [m, n] =>
    let init := t.data.getD 0 default
    let data := Id.run do
      let mut d := Array.replicate (m * n) init
      for i in [:m] do
        for j in [:n] do
          d := d.set! (j * m + i) (t.data.getD (i * n + j) default)
      return d
    { shape := [n, m], data }
  | _ => t  -- only defined for 2-D

-- ---------------------------------------------------------------------------
-- Arithmetic operations
-- ---------------------------------------------------------------------------

instance [Add α] : Add (Tensor α) where
  add a b := zipWith (· + ·) a b

instance [Sub α] : Sub (Tensor α) where
  sub a b := zipWith (· - ·) a b

instance [Mul α] : Mul (Tensor α) where
  mul a b := zipWith (· * ·) a b

/-- Scalar multiply. -/
def smul [Mul α] (s : α) (t : Tensor α) : Tensor α := t.map (s * ·)

/-- Matrix multiplication of two 2-D tensors: [m,k] × [k,n] → [m,n]. -/
def matmul [Inhabited α] [Add α] [Mul α] [Zero α] (a b : Tensor α) : Tensor α :=
  match a.shape, b.shape with
  | [m, k], [k', n] =>
    if k ≠ k' then a
    else
      let data := Id.run do
        let mut d := Array.replicate (m * n) (Zero.zero : α)
        for i in [:m] do
          for j in [:n] do
            let mut acc : α := Zero.zero
            for p in [:k] do
              acc := acc + a.data.getD (i * k + p) default * b.data.getD (p * n + j) default
            d := d.set! (i * n + j) acc
        return d
      { shape := [m, n], data }
  | _, _ => a

/-- Batch matrix multiplication [b,m,k] × [b,k,n] → [b,m,n]. -/
def bmm [Inhabited α] [Add α] [Mul α] [Zero α] (a b : Tensor α) : Tensor α :=
  match a.shape, b.shape with
  | [bs, m, k], [bs', k', n] =>
    if bs ≠ bs' || k ≠ k' then a
    else
      let data := Id.run do
        let mut d := Array.replicate (bs * m * n) (Zero.zero : α)
        for bi in [:bs] do
          for i in [:m] do
            for j in [:n] do
              let mut acc : α := Zero.zero
              for p in [:k] do
                let ai := bi * m * k + i * k + p
                let bi_ := bi * k * n + p * n + j
                acc := acc + a.data.getD ai default * b.data.getD bi_ default
              d := d.set! (bi * m * n + i * n + j) acc
        return d
      { shape := [bs, m, n], data }
  | _, _ => a

/-- Sum all elements. -/
def sum [Add α] [Zero α] (t : Tensor α) : α :=
  t.data.foldl (· + ·) 0

/-- Sum along the last axis of a 2-D tensor, producing a 1-D tensor. -/
def sumLastAxis [Inhabited α] [Add α] [Zero α] (t : Tensor α) : Tensor α :=
  match t.shape with
  | [m, n] =>
    let data := Id.run do
      let mut d := Array.replicate m (Zero.zero : α)
      for i in [:m] do
        let mut acc : α := Zero.zero
        for j in [:n] do
          acc := acc + t.data.getD (i * n + j) default
        d := d.set! i acc
      return d
    { shape := [m], data }
  | _ => t

/-- Apply a scalar function element-wise. -/
def apply (f : α → α) (t : Tensor α) : Tensor α := t.map f

/-- Softmax along last dimension of a 2-D tensor. -/
def softmax [Inhabited α] [Scalar α] (t : Tensor α) : Tensor α :=
  match t.shape with
  | [m, n] =>
    let data := Id.run do
      let mut d := t.data
      for i in [:m] do
        -- Compute exp and sum
        let mut s : α := Scalar.szero
        for j in [:n] do
          let v := Scalar.exp (d.getD (i * n + j) default)
          d := d.set! (i * n + j) v
          s := s + v
        -- Normalize
        let invS := Scalar.inv s
        for j in [:n] do
          d := d.set! (i * n + j) (invS * d.getD (i * n + j) default)
      return d
    { shape := t.shape, data }
  | _ => t

/-- Layer normalisation over the last dimension. -/
def layerNorm [Inhabited α] [Scalar α] (t : Tensor α) (eps : α) : Tensor α :=
  match t.shape with
  | [m, n] =>
    let nf := Scalar.ofNat (α := α) n
    let data := Id.run do
      let mut d := t.data
      for i in [:m] do
        -- mean
        let mut mean : α := Scalar.szero
        for j in [:n] do mean := mean + d.getD (i * n + j) default
        mean := Scalar.inv nf * mean
        -- variance
        let mut var : α := Scalar.szero
        for j in [:n] do
          let diff := d.getD (i * n + j) default - mean
          var := var + diff * diff
        var := Scalar.inv nf * var
        let std := Scalar.sqrt (var + eps)
        let invStd := Scalar.inv std
        -- normalise
        for j in [:n] do
          d := d.set! (i * n + j) (invStd * (d.getD (i * n + j) default - mean))
      return d
    { shape := t.shape, data }
  | _ => t

/-- Concatenate two tensors along dimension 0. -/
def cat (a b : Tensor α) : Tensor α :=
  match a.shape, b.shape with
  | d :: ds, d' :: ds' =>
    if ds = ds' then
      { shape := (d + d') :: ds, data := a.data ++ b.data }
    else a
  | _, _ => a

end Tensor

-- ---------------------------------------------------------------------------
-- Named-parameter convenience record
-- ---------------------------------------------------------------------------

/-- A named collection of tensors (analogous to `state_dict` in PyTorch). -/
structure StateDict (α : Type) where
  params : List (String × Tensor α)
  deriving Repr

namespace StateDict

def empty : StateDict α := { params := [] }

def insert (sd : StateDict α) (k : String) (v : Tensor α) : StateDict α :=
  { params := (k, v) :: sd.params.filter (·.1 ≠ k) }

def lookup [Inhabited α] (sd : StateDict α) (k : String) : Option (Tensor α) :=
  sd.params.find? (·.1 = k) |>.map (·.2)

end StateDict

end TorchLib
