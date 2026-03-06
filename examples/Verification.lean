import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models
import TorchLib.Runtime.Training
import TorchLib.Verification.IBP

/-!
# Example: Formal Verification with IBP

**Problem:** given a trained network and an input perturbation radius ε,
can we guarantee the predicted class never changes for *any* input within
the ℓ∞ ball of radius ε around a test point?

**Method:** Interval Bound Propagation (IBP) propagates a *pair* of
tensors `(lo, hi)` — the lower and upper bounds on every activation —
through each layer using sound interval arithmetic.  The final `(lo, hi)`
pair provably brackets all possible output logits under that perturbation.

**Certification condition** (binary classifier, class 0 vs 1):

    lo[0] > hi[1]

i.e. the worst-case logit for class 0 still beats the best-case logit
for class 1.  If this holds the predicted class cannot flip.

## Why default-initialized weights fail

`MLP.init` sets every weight to 0.1 and every bias to 0.  A uniform-input
uniform-weight network produces identical logits for every class → no
decision margin → IBP can never certify anything.  We must train first.
-/

open TorchLib TorchLib.Runtime TorchLib.Verification

-- ---------------------------------------------------------------------------
-- Dataset:  class 0 = [+1,+1,+1,+1],  class 1 = [−1,−1,−1,−1]
-- ---------------------------------------------------------------------------

def trainXs : Tensor Float :=
  { shape := [16, 4]
    data  := Array.replicate 32 1.0 ++ Array.replicate 32 (-1.0) }

def trainYs : Tensor Float :=
  -- 8 rows [1, 0] then 8 rows [0, 1]
  let r0 := (Array.replicate 8 #[1.0, 0.0]).foldl (· ++ ·) #[]
  let r1 := (Array.replicate 8 #[0.0, 1.0]).foldl (· ++ ·) #[]
  { shape := [16, 2], data := r0 ++ r1 }

-- Nominal test point: a class-0 sample
def x0 : Tensor Float := Tensor.full [1, 4] 1.0

-- ---------------------------------------------------------------------------
-- Training: Linear(4 → 2), analytic MSE gradients, SGD
-- ---------------------------------------------------------------------------
-- For  pred = xs @ W^T + b  and  loss = MSE(pred, ys):
--   dL/dW = (2/batch) * (pred − ys)^T @ xs    shape [out, in]
--   dL/db = (2/batch) * (pred − ys)^T @ 1s    shape [out]

def trainStep (cfg : SGDConfig) (st : SGDState) (l : Linear Float)
    (xs ys : Tensor Float) : SGDState × Linear Float × Float :=
  let batch  := (xs.shape.headD 1).toFloat
  let batchN := xs.shape.headD 1
  let pred   := Linear.forward l xs
  let loss   := mseLoss pred ys
  let diff   := pred - ys                              -- [batch, out]
  let gradW  := Tensor.matmul diff.transpose xs        -- [out, in]
                  |>.map (· * (2.0 / batch))
  -- sum diff over batch: diff^T @ ones_col → [out, 1] → flatten → [out]
  let ones   : Tensor Float := Tensor.ones [batchN, 1]
  let gradB  := Tensor.matmul diff.transpose ones
                  |>.flatten |>.map (· * (2.0 / batch))
  let r  := SGD.step cfg st [("w", l.weight, gradW), ("b", l.bias, gradB)]
  let st' := r.1
  let ps  := r.2
  let w'  := ps.find? (fun p => p.1 == "w") |>.map (·.2) |>.getD l.weight
  let b'  := ps.find? (fun p => p.1 == "b") |>.map (·.2) |>.getD l.bias
  (st', ({ weight := w', bias := b' } : Linear Float), loss)

def trainLinear (n : Nat) (lr : Float := 0.05) : IO (Linear Float) := do
  let cfg : SGDConfig := { lr }
  let mut l  := Linear.init 4 2
  let mut st := SGD.initState cfg ["w", "b"]
  for i in [:n] do
    let (st', l', loss) := trainStep cfg st l trainXs trainYs
    st := st'
    l  := l'
    if i % 10 == 0 then
      IO.println s!"  step {i}: loss = {loss}"
  return l

-- ---------------------------------------------------------------------------
-- IBP certification
-- ---------------------------------------------------------------------------

def certify (l : Linear Float) (x : Tensor Float) (eps : Float)
    : Tensor Float × Tensor Float :=
  let bounds := ITensor.fromCenterRadius x eps
  let out    := IBP.linear l bounds
  (out.map (·.lo), out.map (·.hi))

-- ---------------------------------------------------------------------------
-- Main
-- ---------------------------------------------------------------------------

def main : IO Unit := do
  -- Train
  IO.println "=== Training Linear(4→2) ==="
  let trained ← trainLinear 50

  -- Show what the network learned
  IO.println ""
  IO.println s!"W row 0 (should be ≈ +0.125): {reprStr (trained.weight.data.extract 0 4)}"
  IO.println s!"W row 1 (should be ≈ −0.125): {reprStr (trained.weight.data.extract 4 8)}"
  IO.println s!"bias:                         {reprStr trained.bias.data}"

  -- Nominal forward pass
  let nomOut := Linear.forward trained x0
  IO.println ""
  IO.println "=== Nominal pass at x0 = [1,1,1,1] ==="
  IO.println s!"logits:       {reprStr nomOut.data}"
  let c0wins := if nomOut.data.getD 0 0.0 > nomOut.data.getD 1 0.0 then "true" else "false"
  IO.println s!"class 0 wins: {c0wins}"

  -- IBP sweep
  IO.println ""
  IO.println "=== IBP certification sweep ==="
  for eps in ([0.01, 0.05, 0.1, 0.2, 0.5] : List Float) do
    let (lo, hi) := certify trained x0 eps
    let lo0    := lo.data.getD 0 0.0
    let hi1    := hi.data.getD 1 0.0
    let margin := lo0 - hi1
    let status := if lo0 > hi1 then "CERTIFIED ✓" else "not certified"
    IO.println s!"  ε = {eps}: lo[0] = {lo0}  hi[1] = {hi1}  margin = {margin}  → {status}"

#eval main
