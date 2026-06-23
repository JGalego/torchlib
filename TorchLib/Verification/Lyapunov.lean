import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models
import TorchLib.Verification.IBP

/-!
# TorchLib.Verification.Lyapunov

**Lyapunov-function verification for learned controllers**

For a discrete-time closed-loop system `xₖ₊₁ = f(xₖ)` (where `f` bundles the
plant dynamics and a learned controller into one map), a candidate Lyapunov
function `V` certifies asymptotic stability of the origin on a region `R` when

1. **Positivity**  `V(x) ≥ 0` for all `x ∈ R`, and
2. **Decrease**    `ΔV(x) = V(f(x)) − V(x) < 0` for all `x ∈ R` away from the
   origin.

Both conditions are discharged *soundly* on a grid of boxes covering `R`.

## Why a mean-value form

A naive `sup V(f(box)) − inf V(box)` bound throws away the correlation between
`f(x)` and `x` and is too loose to certify even a linear contraction.  Instead
we use the **mean-value form**: on each cell with centre `c` and radius `r`,

```
ΔV(x) = ΔV(c) + ∇ΔV(ξ)·(x − c)        ⇒    ΔV(x) ≤ ΔV(c) + Σⱼ |Gⱼ|·rⱼ
```

where `ΔV(c)` is computed *exactly* and `G ⊇ ∇ΔV(·)` is a sound interval
enclosure of the gradient over the cell.  The gradient

```
∇ΔV(x) = Jf(x)ᵀ · ∇V(f(x)) − ∇V(x)
```

needs an interval enclosure of the network Jacobian `Jf`, obtained by
propagating ReLU derivative intervals (`{0}`, `{1}` or `[0,1]` per neuron)
through the layer weights.  This keeps the certificate sound while being tight
enough to verify real contractions.
-/

namespace TorchLib.Verification

open TorchLib

/-- A candidate Lyapunov function with the data the mean-value certificate needs:
    a sound enclosure of `V` over a box, the exact value, and a sound enclosure
    of `∇V` over a box. -/
structure LyapV where
  /-- Sound interval enclosure of `V` over a box (one `Interval` per state dim). -/
  bound    : Array Interval → Interval
  /-- Concrete evaluation of `V` at a point. -/
  eval     : Array Float → Float
  /-- Sound enclosure of `∇V` over a box, one `Interval` per coordinate. -/
  gradBound : Array Interval → Array Interval

namespace LyapV

/-- Tight interval enclosure of `x²` for `x ∈ [l, u]`. -/
private def sq (i : Interval) : Interval :=
  if i.lo ≥ 0.0 then { lo := i.lo * i.lo, hi := i.hi * i.hi }
  else if i.hi ≤ 0.0 then { lo := i.hi * i.hi, hi := i.lo * i.lo }
  else { lo := 0.0, hi := Float.max (i.lo * i.lo) (i.hi * i.hi) }

/-- Weighted quadratic `V(x) = Σ qᵢ xᵢ²` (positive definite when every `qᵢ > 0`).
    Gradient `∂V/∂xᵢ = 2 qᵢ xᵢ`. -/
def quadratic (q : Array Float) : LyapV where
  bound box := Id.run do
    let mut acc : Interval := { lo := 0.0, hi := 0.0 }
    for i in [:q.size] do
      let qi := q.get! i
      acc := acc + { lo := qi, hi := qi } * sq (box.getD i default)
    return acc
  eval x := Id.run do
    let mut acc := 0.0
    for i in [:q.size] do
      let xi := x.getD i 0.0
      acc := acc + q.get! i * xi * xi
    return acc
  gradBound box := Id.run do
    let mut g : Array Interval := #[]
    for i in [:q.size] do
      let qi := q.get! i
      g := g.push ({ lo := 2.0 * qi, hi := 2.0 * qi } * box.getD i default)
    return g

end LyapV

namespace Lyapunov

-- ---------------------------------------------------------------------------
-- Small interval linear-algebra helpers
-- ---------------------------------------------------------------------------

/-- Interval matrix product: `[m,k] · [k,n] → [m,n]` (row-major). -/
private def imatmul (a : Array Interval) (m k : Nat) (b : Array Interval) (n : Nat)
    : Array Interval := Id.run do
  let mut d := Array.replicate (m * n) (default : Interval)
  for i in [:m] do
    for j in [:n] do
      let mut acc : Interval := { lo := 0.0, hi := 0.0 }
      for p in [:k] do
        acc := acc + a.getD (i * k + p) default * b.getD (p * n + j) default
      d := d.set! (i * n + j) acc
  return d

/-- Embed a `Float` weight matrix as a matrix of point intervals. -/
private def pointMat (W : Tensor Float) : Array Interval :=
  W.data.map (fun w => { lo := w, hi := w })

/-- Left-multiply by a diagonal matrix `D` (given as a vector of diagonal
    intervals): `(D·J)[i,j] = Dᵢ · J[i,j]`. -/
private def scaleRows (D : Array Interval) (J : Array Interval) (rows cols : Nat)
    : Array Interval := Id.run do
  let mut d := J
  for i in [:rows] do
    let di := D.getD i default
    for j in [:cols] do
      d := d.set! (i * cols + j) (di * J.getD (i * cols + j) default)
  return d

/-- ReLU derivative enclosure for a pre-activation interval: `{1}` if surely
    positive, `{0}` if surely non-positive, `[0,1]` if unstable. -/
private def reluDeriv (i : Interval) : Interval :=
  if i.lo ≥ 0.0 then { lo := 1.0, hi := 1.0 }
  else if i.hi ≤ 0.0 then { lo := 0.0, hi := 0.0 }
  else { lo := 0.0, hi := 1.0 }

-- ---------------------------------------------------------------------------
-- Interval Jacobian of an MLP over a cell
-- ---------------------------------------------------------------------------

/-- Pre-activation interval bounds feeding each ReLU (all layers but the last). -/
private def preActs (f : MLP Float) (cell : ITensor) : List ITensor :=
  let L := f.layers.size
  let (_, _, pres) := f.layers.toList.foldl
    (fun (acc : ITensor × Nat × List ITensor) l =>
      let (x, i, pres) := acc
      let pre := IBP.linear l x
      if i < L - 1 then (IBP.relu pre, i + 1, pres ++ [pre])
      else (pre, i + 1, pres)) (cell, 0, [])
  pres

/-- Sound interval enclosure of the MLP Jacobian `Jf(x)` (shape `[out, in]`,
    flattened row-major) over the input cell, using ReLU derivative intervals. -/
def jacobian (f : MLP Float) (cell : ITensor) (inDim : Nat) : Array Interval :=
  let layers := f.layers.toList
  let pres := preActs f cell
  Id.run do
    let dflt : Linear Float := { weight := Tensor.zeros [], bias := Tensor.zeros [] }
    let l0 := layers.headD dflt
    let (mut0, _) := match l0.weight.shape with | [o, _] => (o, 0) | _ => (0, 0)
    let mut J := pointMat l0.weight        -- [out₀, inDim]
    let mut rows := mut0
    for i in [1:layers.length] do
      let l := layers.getD i dflt
      let (oi, ki) := match l.weight.shape with | [o, k] => (o, k) | _ => (0, 0)
      -- diagonal ReLU' of the previous layer's pre-activation
      let D := (pres.getD (i - 1) cell).data.map reluDeriv
      let DJ := scaleRows D J rows inDim
      J := imatmul (pointMat l.weight) oi ki DJ inDim
      rows := oi
    return J

-- ---------------------------------------------------------------------------
-- Grid + certification
-- ---------------------------------------------------------------------------

/-- Per-dimension subdivision of `[lo, hi]` into `splits` equal intervals. -/
private def subIntervals (lo hi : Float) (splits : Nat) : List Interval :=
  let n := Nat.max 1 splits
  let w := (hi - lo) / Float.ofNat n
  (List.range n).map (fun k =>
    { lo := lo + Float.ofNat k * w, hi := lo + Float.ofNat (k + 1) * w })

/-- Grid of state boxes covering the region `∏ [loᵢ, hiᵢ]`, `splits` per axis. -/
def gridCells (los his : Array Float) (splits : Nat) : List (Array Interval) :=
  (List.range los.size).foldl (fun (cells : List (Array Interval)) d =>
    let subs := subIntervals (los.getD d 0.0) (his.getD d 0.0) splits
    cells.flatMap (fun cell => subs.map (fun s => cell.push s))) [#[]]

/-- Convert a state box to an `[1, n]` interval tensor for IBP. -/
private def boxToITensor (cell : Array Interval) : ITensor :=
  { shape := [1, cell.size], data := cell }

/-- Minimum `‖x‖∞` over a box (`0` if the box straddles the origin). -/
private def cellInfNorm (cell : Array Interval) : Float :=
  cell.foldl (fun acc i =>
    let d := if i.contains 0.0 then 0.0 else Float.min (Float.abs i.lo) (Float.abs i.hi)
    Float.max acc d) 0.0

/-- Outcome of a Lyapunov certification sweep. -/
structure Result where
  /-- Total grid cells inspected. -/
  cells          : Nat
  /-- Cells that satisfied both conditions (or were inside the exclusion ball). -/
  certifiedCells : Nat
  /-- Worst (largest) sound upper bound on `ΔV` across checked cells. -/
  worstDeltaV    : Float
  /-- Smallest certified value of `V` (lower bound) across cells. -/
  minV           : Float
  /-- Whether stability is certified on the whole region. -/
  certified      : Bool

instance : Repr Result where
  reprPrec r prec := Repr.addAppParen
    f!"Result \{ cells := {reprPrec r.cells 0}, certifiedCells := {reprPrec r.certifiedCells 0}, worstDeltaV := {reprPrec r.worstDeltaV 0}, minV := {reprPrec r.minV 0}, certified := {reprPrec r.certified 0} }"
    prec

/-- Certify a Lyapunov function for the closed-loop map `f` (an `MLP` mapping a
    state to the next state) over the box region `∏ [losᵢ, hisᵢ]`, using the
    mean-value form for the decrease condition.

    `epsilon` is the required strict-decrease margin; `excludeRadius` skips the
    `‖x‖∞ ≤ excludeRadius` ball around the origin where `ΔV → 0` is expected. -/
def certify (f : MLP Float) (V : LyapV) (los his : Array Float)
    (epsilon : Float := 1e-4) (excludeRadius : Float := 1e-2) (splits : Nat := 8)
    : Result :=
  let n := los.size
  let cells := gridCells los his splits
  Id.run do
    let mut nCert := 0
    let mut worst := -1e30
    let mut minV  := 1e30
    let mut allOk := true
    for cell in cells do
      -- positivity from a sound enclosure of V on the cell
      let vBoxLo := (V.bound cell).lo
      minV := Float.min minV vBoxLo
      if cellInfNorm cell ≤ excludeRadius then
        nCert := nCert + 1
        if vBoxLo < 0.0 then allOk := false
      else
        -- centre values (exact)
        let centre := cell.map (fun i => (i.lo + i.hi) / 2.0)
        let centreT : Tensor Float := { shape := [1, n], data := centre }
        let fc := (MLP.forward f centreT).data
        let dVc := V.eval fc - V.eval centre
        -- interval gradient of ΔV over the cell:  Jfᵀ·∇V(f(box)) − ∇V(box)
        let Jf := jacobian f (boxToITensor cell) n
        let fBox := (IBP.mlp f (boxToITensor cell)).data
        let gradVf := V.gradBound fBox          -- length n
        let gradV  := V.gradBound cell           -- length n
        let radius := cell.map (fun i => (i.hi - i.lo) / 2.0)
        -- remainder Σⱼ max|Gⱼ|·rⱼ where Gⱼ = (Jfᵀ ∇V(f))ⱼ − ∇Vⱼ
        let remainder := Id.run do
          let mut rem := 0.0
          for j in [:n] do
            let mut acc : Interval := { lo := 0.0, hi := 0.0 }
            for i in [:n] do
              acc := acc + Jf.getD (i * n + j) default * gradVf.getD i default
            let gj := acc - gradV.getD j default
            let absG := Float.max (Float.abs gj.lo) (Float.abs gj.hi)
            rem := rem + absG * radius.getD j 0.0
          return rem
        let dVUpper := dVc + remainder
        worst := Float.max worst dVUpper
        let ok := vBoxLo ≥ 0.0 && dVUpper ≤ -epsilon
        if ok then nCert := nCert + 1 else allOk := false
    return { cells := cells.length, certifiedCells := nCert
             worstDeltaV := worst, minV := minV, certified := allOk }

end Lyapunov

end TorchLib.Verification
