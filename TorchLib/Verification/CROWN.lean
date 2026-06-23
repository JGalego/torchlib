import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models
import TorchLib.Verification.IBP

/-!
# TorchLib.Verification.CROWN

**CROWN / LiRPA — Affine Relaxations and Dual Certificates**

CROWN ([Zhang et al. 2018]) tightens IBP bounds by propagating **affine
relaxations** *backward* through the network.  Each nonlinear activation is
replaced by a linear upper/lower envelope; back-substituting these envelopes
expresses the network output as a linear function of the original input, which
is then *concretised* over the input box.

The output is bounded as

```
A_lo · x₀ + d_lo  ≤  f(x₀)  ≤  A_up · x₀ + d_up   for all x₀ ∈ [x₀-ε, x₀+ε]
```

where `A_lo, A_up` are `[out_dim, in_dim]` matrices obtained by composing the
per-layer relaxations.  Concretisation is sign-aware: a positive coefficient
pairs with the input lower bound for the output lower bound, etc.

## α-CROWN

For an *unstable* ReLU neuron (pre-activation interval straddles 0) the lower
envelope `α·x` has a free slope `α ∈ [0,1]`.  α-CROWN treats these slopes as
optimisable parameters and tightens the certified bound by projected gradient
ascent (here via SPSA — simultaneous perturbation stochastic approximation,
with a deterministic LCG so the optimisation is reproducible).

## Reference

"Efficient Neural Network Robustness Certification with General Activation
Functions", Zhang et al., NeurIPS 2018.
-/

namespace TorchLib.Verification

open IBP

-- ---------------------------------------------------------------------------
-- Small dense-linear-algebra helpers (row-major Tensor Float)
-- ---------------------------------------------------------------------------

/-- Identity matrix of dimension `d`, shape `[d, d]`. -/
def Tensor.eye (d : Nat) : Tensor Float := Id.run do
  let mut data := Array.replicate (d * d) (0.0 : Float)
  for i in [:d] do
    data := data.set! (i * d + i) 1.0
  return { shape := [d, d], data }

/-- Matrix–vector product: `[m, n] · [n] → [m]`. -/
def Tensor.matVec (a : Tensor Float) (v : Tensor Float) : Tensor Float :=
  match a.shape with
  | [m, n] => Id.run do
    let mut d := Array.replicate m (0.0 : Float)
    for i in [:m] do
      let mut acc := 0.0
      for j in [:n] do
        acc := acc + a.data.get! (i * n + j) * v.data.getD j 0.0
      d := d.set! i acc
    return { shape := [m], data := d }
  | _ => v

-- ---------------------------------------------------------------------------
-- Linear bound representation
-- ---------------------------------------------------------------------------

/-- A **linear bound** expresses the network output as a linear function of the
    original input `x₀`:  `loW · x₀ + loB ≤ f(x₀) ≤ upW · x₀ + upB`.

    Shapes: `loW`, `upW` are `[output_dim, current_dim]`; `loB`, `upB` are
    `[output_dim]`. -/
structure LinearBound where
  /-- Lower bounding weight matrix, shape `[output_dim, current_dim]`. -/
  loW : Tensor Float
  /-- Lower bounding bias, shape `[output_dim]`. -/
  loB : Tensor Float
  /-- Upper bounding weight matrix, shape `[output_dim, current_dim]`. -/
  upW : Tensor Float
  /-- Upper bounding bias, shape `[output_dim]`. -/
  upB : Tensor Float

instance : Repr LinearBound where
  reprPrec lb prec := Repr.addAppParen
    f!"LinearBound \{ loW := {reprPrec lb.loW 0}, loB := {reprPrec lb.loB 0}, upW := {reprPrec lb.upW 0}, upB := {reprPrec lb.upB 0} }"
    prec

namespace LinearBound

/-- The identity linear bound for dimension `d`: `x ↦ x` with no slack. -/
def identity (d : Nat) : LinearBound :=
  { loW := Tensor.eye d, loB := Tensor.zeros [d]
    upW := Tensor.eye d, upB := Tensor.zeros [d] }

/-- Compose with a linear layer `v = W·x + b` (exact substitution).

    Given bounds in the variable `v`, returns bounds in `x`:
    `A·v + d = A·(W·x + b) + d = (A·W)·x + (A·b + d)`. -/
def composeLinear (lb : LinearBound) (W b : Tensor Float) : LinearBound :=
  { loW := Tensor.matmul lb.loW W
    upW := Tensor.matmul lb.upW W
    loB := Tensor.matVec lb.loW b + lb.loB
    upB := Tensor.matVec lb.upW b + lb.upB }

/-- Sign-aware concretisation over the input box `[inLo, inHi]`. -/
def concretize (lb : LinearBound) (inLo inHi : Tensor Float)
    : Tensor Float × Tensor Float :=
  match lb.loW.shape with
  | [m, n] =>
    let (loData, hiData) := Id.run do
      let mut lo := Array.replicate m (0.0 : Float)
      let mut hi := Array.replicate m (0.0 : Float)
      for i in [:m] do
        let mut lacc := lb.loB.data.getD i 0.0
        let mut hacc := lb.upB.data.getD i 0.0
        for j in [:n] do
          let l := inLo.data.getD j 0.0
          let h := inHi.data.getD j 0.0
          let wl := lb.loW.data.get! (i * n + j)
          let wu := lb.upW.data.get! (i * n + j)
          -- lower bound: positive weight × input-lo, negative weight × input-hi
          lacc := lacc + (if wl ≥ 0.0 then wl * l else wl * h)
          -- upper bound: positive weight × input-hi, negative weight × input-lo
          hacc := hacc + (if wu ≥ 0.0 then wu * h else wu * l)
        lo := lo.set! i lacc
        hi := hi.set! i hacc
      return (lo, hi)
    ({ shape := [m], data := loData }, { shape := [m], data := hiData })
  | _ => (lb.loB, lb.upB)

end LinearBound

-- ---------------------------------------------------------------------------
-- Affine relaxation of activation functions
-- ---------------------------------------------------------------------------

/-- Per-neuron affine relaxation `loSlope·x + loIntercept ≤ σ(x) ≤ upSlope·x + upIntercept`. -/
structure AffineRelax where
  /-- Lower-bound slope. -/
  loSlope : Float
  /-- Lower-bound intercept. -/
  loIntercept : Float
  /-- Upper-bound slope. -/
  upSlope : Float
  /-- Upper-bound intercept. -/
  upIntercept : Float
  deriving Inhabited

instance : Repr AffineRelax where
  reprPrec ar prec := Repr.addAppParen
    f!"AffineRelax \{ loSlope := {reprPrec ar.loSlope 0}, loIntercept := {reprPrec ar.loIntercept 0}, upSlope := {reprPrec ar.upSlope 0}, upIntercept := {reprPrec ar.upIntercept 0} }"
    prec

namespace AffineRelax

/-- ReLU relaxation for a neuron with pre-activation bounds `[l, u]`.

    The optional `alpha` overrides the lower-envelope slope for *unstable*
    neurons (`l < 0 < u`); the standard CROWN default is the adaptive choice
    `α = 1` if `u ≥ -l` else `0`, which minimises the relaxation area.
    - `u ≤ 0`:  relu ≡ 0   (both envelopes 0)
    - `l ≥ 0`:  relu ≡ x   (both envelopes identity)
    - else:     upper = chord `(u/(u-l))·x − l·u/(u-l)`, lower = `α·x` -/
def relu (l u : Float) (alpha : Option Float := none) : AffineRelax :=
  if u ≤ 0.0 then
    { loSlope := 0.0, loIntercept := 0.0, upSlope := 0.0, upIntercept := 0.0 }
  else if l ≥ 0.0 then
    { loSlope := 1.0, loIntercept := 0.0, upSlope := 1.0, upIntercept := 0.0 }
  else
    let upSlope := u / (u - l)
    let upIntercept := -(l * u) / (u - l)
    let loSlope := match alpha with
      | some a => Float.max 0.0 (Float.min 1.0 a)
      | none   => if u ≥ -l then 1.0 else 0.0
    { loSlope, loIntercept := 0.0, upSlope, upIntercept }

/-- Whether a neuron is *unstable* (its relaxation has a free lower slope). -/
def unstable (l u : Float) : Bool := l < 0.0 && u > 0.0

/-- Sigmoid relaxation on `[l, u]`: secant upper bound, midpoint-tangent lower bound. -/
def sigmoid (l u : Float) : AffineRelax :=
  let sig v := 1.0 / (1.0 + Float.exp (-v))
  let dsig v := let s := sig v; s * (1.0 - s)
  let sl := sig l
  let upSlope := if u != l then (sig u - sl) / (u - l) else dsig l
  let upIntercept := sl - upSlope * l
  let mid := (l + u) / 2.0
  let loSlope := dsig mid
  let loIntercept := sig mid - loSlope * mid
  { loSlope, loIntercept, upSlope, upIntercept }

/-- Tanh relaxation. -/
def tanh (l u : Float) : AffineRelax :=
  let th v := Float.tanh v
  let dth v := 1.0 - th v * th v
  let tl := th l
  let upSlope := if u != l then (th u - tl) / (u - l) else dth l
  let upIntercept := tl - upSlope * l
  let mid := (l + u) / 2.0
  let loSlope := dth mid
  let loIntercept := th mid - loSlope * mid
  { loSlope, loIntercept, upSlope, upIntercept }

end AffineRelax

-- ---------------------------------------------------------------------------
-- CROWN back-substitution
-- ---------------------------------------------------------------------------

namespace CROWN

/-- Back-substitute a `LinearBound` through a ReLU layer whose pre-activations
    have interval bounds `preBounds` (one per neuron), using the per-neuron
    lower slopes `alphas` (defaulting to CROWN's adaptive choice).

    For each entry `A[i,j]` of the running bound, the sign of the coefficient
    selects which envelope (upper chord or lower `α`-line) is substituted — this
    is what makes back-substitution sound. -/
def composeRelu (lb : LinearBound) (preBounds : ITensor)
    (alphas : Array Float) : LinearBound :=
  match lb.loW.shape with
  | [m, n] =>
    let relax := Id.run do
      let mut rs : Array AffineRelax := Array.replicate n default
      for j in [:n] do
        let iv := preBounds.data.getD j default
        rs := rs.set! j (AffineRelax.relu iv.lo iv.hi (alphas[j]?))
      return rs
    let (loW, loB, upW, upB) := Id.run do
      let mut loW := lb.loW.data
      let mut upW := lb.upW.data
      let mut loB := lb.loB.data
      let mut upB := lb.upB.data
      for i in [:m] do
        let mut loBias := loB.getD i 0.0
        let mut upBias := upB.getD i 0.0
        for j in [:n] do
          let r := relax.get! j
          let idx := i * n + j
          -- lower bound  out ≥ loW · relu(y)
          let cl := loW.get! idx
          if cl ≥ 0.0 then
            loW := loW.set! idx (cl * r.loSlope)
            loBias := loBias + cl * r.loIntercept
          else
            loW := loW.set! idx (cl * r.upSlope)
            loBias := loBias + cl * r.upIntercept
          -- upper bound  out ≤ upW · relu(y)
          let cu := upW.get! idx
          if cu ≥ 0.0 then
            upW := upW.set! idx (cu * r.upSlope)
            upBias := upBias + cu * r.upIntercept
          else
            upW := upW.set! idx (cu * r.loSlope)
            upBias := upBias + cu * r.loIntercept
        loB := loB.set! i loBias
        upB := upB.set! i upBias
      return (loW, loB, upW, upB)
    { loW := { shape := [m, n], data := loW }
      upW := { shape := [m, n], data := upW }
      loB := { shape := [m], data := loB }
      upB := { shape := [m], data := upB } }
  | _ => lb

/-- Forward IBP pass recording the pre-activation interval bounds feeding each
    ReLU (all layers except the last). -/
def preActivations (m : MLP Float) (center : Tensor Float) (eps : Float)
    : List ITensor :=
  let x0 := ITensor.fromCenterRadius center eps
  let L := m.layers.size
  let (_, _, pres) := m.layers.toList.foldl
    (fun (acc : ITensor × Nat × List ITensor) l =>
      let (x, i, pres) := acc
      let pre := IBP.linear l x
      if i < L - 1 then (IBP.relu pre, i + 1, pres ++ [pre])
      else (pre, i + 1, pres)) (x0, 0, [])
  pres

/-- CROWN output bounds with explicit per-ReLU lower slopes `alphasPerLayer`
    (one `Array Float` per hidden layer, indexed by neuron). -/
def boundsWithAlpha (m : MLP Float) (center : Tensor Float) (eps : Float)
    (alphasPerLayer : List (Array Float)) : Tensor Float × Tensor Float :=
  let pres := preActivations m center eps
  let layers := m.layers.toList
  let L := layers.length
  let outDim := m.outputSize
  -- Back-substitute from the output layer to the input.
  let lb := Id.run do
    let mut lb := LinearBound.identity outDim
    for k in [:L] do
      let i := L - 1 - k                       -- layer index, last → first
      let l := layers.getD i { weight := Tensor.zeros [], bias := Tensor.zeros [] }
      lb := lb.composeLinear l.weight l.bias    -- now linear in xᵢ
      if i ≠ 0 then
        let pre := pres.getD (i - 1) (ITensor.fromCenterRadius center eps)
        let alphas := alphasPerLayer.getD (i - 1) #[]
        lb := CROWN.composeRelu lb pre alphas    -- now linear in ŷ_{i-1}
    return lb
  let inLo := center.map (· - eps)
  let inHi := center.map (· + eps)
  lb.concretize inLo inHi

/-- CROWN's adaptive default lower slopes (`α = 1` if `u ≥ -l` else `0`). -/
def defaultAlphas (m : MLP Float) (center : Tensor Float) (eps : Float)
    : List (Array Float) :=
  (preActivations m center eps).map (fun pre =>
    pre.data.map (fun iv => if iv.hi ≥ -iv.lo then 1.0 else 0.0))

/-- Run CROWN through an MLP using the adaptive default relaxation. -/
def mlpBounds (m : MLP Float) (center : Tensor Float) (eps : Float)
    : Tensor Float × Tensor Float :=
  boundsWithAlpha m center eps (defaultAlphas m center eps)

end CROWN

-- ---------------------------------------------------------------------------
-- α-CROWN — optimise the unstable-ReLU lower slopes
-- ---------------------------------------------------------------------------

namespace AlphaCROWN

/-- Total certified width `Σ (upper − lower)` — the objective α-CROWN minimises. -/
def totalWidth (lo hi : Tensor Float) : Float :=
  (hi - lo).data.foldl (· + ·) 0.0

/-- Clamp every slope to `[0, 1]`. -/
def project (a : List (Array Float)) : List (Array Float) :=
  a.map (·.map (fun x => Float.max 0.0 (Float.min 1.0 x)))

/-- Perturb `alphas` by `±delta` per coordinate using the LCG `rng` for the
    signs; returns the two perturbed slope sets and the next RNG state. -/
def perturb (alphas : List (Array Float)) (delta : Float) (rng : UInt64)
    : List (Array Float) × List (Array Float) × UInt64 := Id.run do
  let mut r := rng
  let mut plus  : List (Array Float) := []
  let mut minus : List (Array Float) := []
  for layer in alphas do
    let mut lp : Array Float := #[]
    let mut lm : Array Float := #[]
    for a in layer do
      r := r * 6364136223846793005 + 1442695040888963407
      let sign : Float := if (r >>> 63) == 1 then 1.0 else -1.0
      lp := lp.push (a + delta * sign)
      lm := lm.push (a - delta * sign)
    plus  := plus  ++ [lp]
    minus := minus ++ [lm]
  return (project plus, project minus, r)

/-- α-CROWN: minimise the certified output width by SPSA projected gradient
    descent over the unstable-ReLU lower slopes.  Returns the tightest bounds
    found across `steps` iterations (monotone — never worse than plain CROWN). -/
def bounds (m : MLP Float) (center : Tensor Float) (eps : Float)
    (steps : Nat := 20) (delta : Float := 0.1) (lr : Float := 0.5)
    : Tensor Float × Tensor Float := Id.run do
  let mut alphas := CROWN.defaultAlphas m center eps
  let mut best   := CROWN.boundsWithAlpha m center eps alphas
  let mut bestW  := totalWidth best.1 best.2
  let mut rng : UInt64 := 0x2545F4914F6CDD1D
  for _ in [:steps] do
    let (ap, am, rng') := perturb alphas delta rng
    rng := rng'
    let bp := CROWN.boundsWithAlpha m center eps ap
    let bm := CROWN.boundsWithAlpha m center eps am
    let wp := totalWidth bp.1 bp.2
    let wm := totalWidth bm.1 bm.2
    -- SPSA gradient estimate g ≈ (f(α+Δ) − f(α−Δ)) / (2Δ) along the sign vector;
    -- step opposite the better-improving perturbation direction.
    let dir : Float := if wp ≤ wm then 1.0 else -1.0
    let g := (wp - wm) / (2.0 * delta)
    alphas := project (alphas.map (·.map (fun a => a - lr * g * dir * delta)))
    -- track the best concrete bounds seen
    for cand in [(bp.1, bp.2, wp), (bm.1, bm.2, wm)] do
      let (lo, hi, w) := cand
      if w < bestW then
        best := (lo, hi); bestW := w
    let cur := CROWN.boundsWithAlpha m center eps alphas
    let wc := totalWidth cur.1 cur.2
    if wc < bestW then best := cur; bestW := wc
  return best

end AlphaCROWN

-- ---------------------------------------------------------------------------
-- LiRPA — general perturbation analysis
-- ---------------------------------------------------------------------------

/-- Configuration for the LiRPA bound-tightening method. -/
inductive BoundMethod
  | ibp                  -- pure interval bound propagation
  | crown                -- CROWN back-substitution (adaptive slopes)
  | crownIBP             -- IBP lower + CROWN upper (CROWN-IBP hybrid)
  | alphaCrown           -- α-CROWN (optimised slopes)
  deriving Repr, BEq

/-- Compute output bounds using the selected method. -/
def computeBounds (method : BoundMethod)
    (m : MLP Float) (center : Tensor Float) (eps : Float)
    : Tensor Float × Tensor Float :=
  match method with
  | .ibp        => IBP.mlpBounds m center eps
  | .crown      => CROWN.mlpBounds m center eps
  | .crownIBP   =>
    -- Hybrid: intersect IBP and CROWN bounds (take the tighter of each side).
    let (ibpLo, ibpHi) := IBP.mlpBounds m center eps
    let (crLo, crHi)   := CROWN.mlpBounds m center eps
    (ibpLo.zipWith Float.max crLo, ibpHi.zipWith Float.min crHi)
  | .alphaCrown => AlphaCROWN.bounds m center eps

end TorchLib.Verification
