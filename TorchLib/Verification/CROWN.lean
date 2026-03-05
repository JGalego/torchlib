import TorchLib.Core
import TorchLib.Layers
import TorchLib.Verification.IBP

/-!
# TorchLib.Verification.CROWN

**CROWN / LiRPA — Affine Relaxations and Dual Certificates**

CROWN ([Zhang et al. 2018]) tightens IBP bounds by propagating **affine
relaxations** backward through the network.  Each nonlinear activation is
replaced by a linear upper/lower envelope; the composition of these envelopes
gives a linear bound on the network output as a function of the input.

LiRPA (Linear/Nonlinear/Relaxation-based Perturbation Analysis) is the
general framework of which CROWN is one instance.

## Key data structures

- `LinearBound` — `lo_W x + lo_b ≤ f(x) ≤ up_W x + up_b`
- `AffineRelax`  — per-layer linear relaxation of a nonlinear op
- `CrownState`   — intermediate bounds accumulated during back-substitution

## Reference

"Efficient Neural Network Robustness Certification with General Activation
Functions", Zhang et al., NeurIPS 2018.
-/

namespace TorchLib.Verification

open IBP

-- ---------------------------------------------------------------------------
-- Linear bound representation
-- ---------------------------------------------------------------------------

/-- A **linear bound** expresses the output of a layer as a linear function
    of the original input `x₀`:

    `lo_W · x₀ + lo_b ≤ f(x₀) ≤ up_W · x₀ + up_b`

    Shapes:
    - `loW`, `upW`: `[output_dim, input_dim]`
    - `loB`, `upB`: `[output_dim]`
-/
structure LinearBound where
  loW : Tensor Float   -- lower bounding weight matrix
  loB : Tensor Float   -- lower bounding bias
  upW : Tensor Float   -- upper bounding weight matrix
  upB : Tensor Float   -- upper bounding bias
  deriving Repr

namespace LinearBound

/-- Initialise a trivial (identity) linear bound for dimension `d`. -/
def identity (d : Nat) : LinearBound :=
  { loW := Tensor.zeros [d, d]  -- will be set to I
    loB := Tensor.zeros [d]
    upW := Tensor.zeros [d, d]
    upB := Tensor.zeros [d] }

/-- Compose a linear bound `lb` (for f) with a weight matrix `W` (for g = W f):
    `g_lo = W_+ · f_lo + W_- · f_up`  (≤ Wf ≤ `W_+ · f_up + W_- · f_lo`) -/
def compose [Inhabited Float] (lb : LinearBound) (W b : Tensor Float) : LinearBound :=
  -- Decompose W into positive and negative parts
  let Wpos := W.map (fun v => if v >= 0.0 then v else 0.0)
  let Wneg := W.map (fun v => if v < 0.0  then v else 0.0)
  let loW' := Tensor.matmul Wpos lb.loW + Tensor.matmul Wneg lb.upW
  let upW' := Tensor.matmul Wpos lb.upW + Tensor.matmul Wneg lb.loW
  let loB' := Tensor.matmul Wpos lb.loB + Tensor.matmul Wneg lb.upB + b
  let upB' := Tensor.matmul Wpos lb.upB + Tensor.matmul Wneg lb.loB + b
  -- loB' / upB' are 2-D after matmul; flatten to 1-D
  { loW := loW', upW := upW'
    loB := loB'.flatten, upB := upB'.flatten }

/-- Compute concrete lower/upper bounds from a `LinearBound` and input interval. -/
def concretize [Inhabited Float] (lb : LinearBound) (inputLo inputHi : Tensor Float)
    : Tensor Float × Tensor Float :=
  let loOut := Tensor.matmul lb.loW inputLo + lb.loB
  let hiOut := Tensor.matmul lb.upW inputHi + lb.upB
  (loOut, hiOut)

end LinearBound

-- ---------------------------------------------------------------------------
-- Affine relaxation of activation functions
-- ---------------------------------------------------------------------------

/-- Per-neuron affine relaxation:
    `α · x + β_lo ≤ σ(x) ≤ α · x + β_up`
    where `α` is the lower-bound slope, `β_lo`/`β_up` are intercepts. -/
structure AffineRelax where
  loSlope : Float
  loIntercept : Float
  upSlope : Float
  upIntercept : Float
  deriving Repr

namespace AffineRelax

/-- ReLU relaxation for a neuron with pre-activation bounds `[l, u]`.
    - If `u ≤ 0`:  relu = 0   → both bounds are 0
    - If `l ≥ 0`:  relu = x   → both bounds are identity
    - Else (0 ∈ [l,u]): upper: chord from (l,0) to (u,u); lower: 0 -/
def relu (l u : Float) : AffineRelax :=
  if u ≤ 0.0 then
    { loSlope := 0.0, loIntercept := 0.0, upSlope := 0.0, upIntercept := 0.0 }
  else if l ≥ 0.0 then
    { loSlope := 1.0, loIntercept := 0.0, upSlope := 1.0, upIntercept := 0.0 }
  else
    -- upper bound: chord slope u/(u-l), intercept -lu/(u-l)
    let upSlope := u / (u - l)
    let upIntercept := -(l * u) / (u - l)
    -- lower bound: 0 (or alternatively x if |u|>|l|)  — use zero for safety
    { loSlope := 0.0, loIntercept := 0.0, upSlope, upIntercept }

/-- Sigmoid relaxation on `[l, u]` using secant upper bound and tangent lower bound. -/
def sigmoid (l u : Float) : AffineRelax :=
  let sig v := 1.0 / (1.0 + Float.exp (-v))
  let dsig v := let s := sig v; s * (1.0 - s)
  let sl := sig l
  let su := sig u
  -- upper: secant (chord)
  let upSlope := if u != l then (su - sl) / (u - l) else dsig l
  let upIntercept := sl - upSlope * l
  -- lower: tangent at midpoint
  let mid := (l + u) / 2.0
  let loSlope := dsig mid
  let loIntercept := sig mid - loSlope * mid
  { loSlope, loIntercept, upSlope, upIntercept }

/-- Tanh relaxation. -/
def tanh (l u : Float) : AffineRelax :=
  let th v := Float.tanh v
  let dth v := 1.0 - th v * th v
  let tl := th l
  let tu := th u
  let upSlope := if u != l then (tu - tl) / (u - l) else dth l
  let upIntercept := tl - upSlope * l
  let mid := (l + u) / 2.0
  let loSlope := dth mid
  let loIntercept := th mid - loSlope * mid
  { loSlope, loIntercept, upSlope, upIntercept }

end AffineRelax

-- ---------------------------------------------------------------------------
-- CROWN back-substitution state
-- ---------------------------------------------------------------------------

/-- Accumulated state during CROWN backward pass through a linear stack. -/
structure CrownState where
  /-- Current linear bound (back-substituted so far). -/
  bound : LinearBound
  /-- Per-neuron pre-activation bounds (from IBP forward pass). -/
  preActBounds : List (ITensor)
  deriving Repr

-- ---------------------------------------------------------------------------
-- CROWN propagation through a linear stack
-- ---------------------------------------------------------------------------

namespace CROWN

/-- Apply one linear layer during CROWN back-substitution. -/
def propLinear [Inhabited Float] (s : CrownState) (l : Linear Float) : CrownState :=
  let newBound := LinearBound.compose s.bound l.weight l.bias
  { s with bound := newBound }

/-- Apply one ReLU layer: tighten bounds using `AffineRelax.relu`. -/
def propRelu [Inhabited Float] (s : CrownState) (preBounds : ITensor) : CrownState :=
  let n := s.bound.loW.shape.getLast!
  -- Compute per-neuron relaxation matrices (diagonal)
  let loSlopes := preBounds.data.map (fun i => (AffineRelax.relu i.lo i.hi).loSlope)
  let upSlopes := preBounds.data.map (fun i => (AffineRelax.relu i.lo i.hi).upSlope)
  let loInts   := preBounds.data.map (fun i => (AffineRelax.relu i.lo i.hi).loIntercept)
  let upInts   := preBounds.data.map (fun i => (AffineRelax.relu i.lo i.hi).upIntercept)
  -- Diagonal matrices
  let diagLo := Tensor.zeros [n, n]
  let diagUp := Tensor.zeros [n, n]
  let diagLo := Id.run do
    let mut d := diagLo.data
    for i in [:n] do d := d.set! (i * n + i) (loSlopes.getD i default)
    return { diagLo with data := d }
  let diagUp := Id.run do
    let mut d := diagUp.data
    for i in [:n] do d := d.set! (i * n + i) (upSlopes.getD i default)
    return { diagUp with data := d }
  -- Updated weight matrices
  let loW' := Tensor.matmul diagLo s.bound.loW
  let upW' := Tensor.matmul diagUp s.bound.upW
  -- Updated biases
  let loB' := Id.run do
    let mut d := s.bound.loB.data
    for i in [:n] do
      d := d.set! i (loSlopes.get! i * d.get! i + loInts.get! i)
    return { s.bound.loB with data := d }
  let upB' := Id.run do
    let mut d := s.bound.upB.data
    for i in [:n] do
      d := d.set! i (upSlopes.get! i * d.get! i + upInts.get! i)
    return { s.bound.upB with data := d }
  { s with bound := { loW := loW', upW := upW', loB := loB', upB := upB' } }

/-- Run CROWN through an MLP, using IBP pre-activation bounds for relaxation.
    Returns `(lo, hi)` output bound tensors. -/
def mlpBounds [Inhabited Float] (m : MLP Float) (center : Tensor Float) (eps : Float)
    : Tensor Float × Tensor Float :=
  let inputDim := center.numel
  let inputLo := center.map (· - eps)
  let inputHi := center.map (· + eps)
  -- IBP forward to get pre-activation bounds per layer
  let x0 := ITensor.fromCenterRadius center eps
  let (_, layerBounds) := m.layers.toList.foldl
    (fun (acc : ITensor × List ITensor) l =>
      let (t, bs) := acc
      let t' := IBP.linear l t
      let t' := IBP.relu t'
      (t', bs ++ [t'])) (x0, [])
  -- Initialise back-substitution at output layer
  let outDim := m.outputSize
  let initBound : LinearBound :=
    { loW := Tensor.zeros [outDim, outDim]  -- identity placeholder
      loB := Tensor.zeros [outDim]
      upW := Tensor.zeros [outDim, outDim]
      upB := Tensor.zeros [outDim] }
  let initState : CrownState := { bound := initBound, preActBounds := layerBounds }
  -- Back-substitute
  let finalState := m.layers.toList.reverse.foldl
    (fun (s : CrownState) l =>
      let s' := CROWN.propLinear s l
      CROWN.propRelu s' (s.preActBounds.headD x0)) initState
  -- Concretize with input bounds
  let _ := inputDim
  LinearBound.concretize finalState.bound inputLo inputHi

end CROWN

-- ---------------------------------------------------------------------------
-- LiRPA — general perturbation analysis
-- ---------------------------------------------------------------------------

/-- Configuration for LiRPA bound tightening method. -/
inductive BoundMethod
  | ibp                  -- pure interval bound propagation
  | crown                -- CROWN back-substitution
  | crownIBP             -- IBP + CROWN hybrid (CROWN-IBP)
  | alpha_crown          -- α-CROWN (learnable slopes)
  deriving Repr, BEq

/-- Compute bounds using the selected method. -/
def computeBounds [Inhabited Float] (method : BoundMethod)
    (m : MLP Float) (center : Tensor Float) (eps : Float)
    : Tensor Float × Tensor Float :=
  match method with
  | .ibp        => IBP.mlpBounds m center eps
  | .crown      => CROWN.mlpBounds m center eps
  | .crownIBP   =>
    -- Hybrid: IBP for lower, CROWN for upper
    let (ibpLo, _)    := IBP.mlpBounds m center eps
    let (_, crownHi)  := CROWN.mlpBounds m center eps
    (ibpLo, crownHi)
  | .alpha_crown => CROWN.mlpBounds m center eps  -- placeholder

end TorchLib.Verification
