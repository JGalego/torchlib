import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models
import TorchLib.Verification.IBP

/-!
# Example: Formal Verification with IBP

Demonstrates Interval Bound Propagation (IBP) from
`TorchLib.Verification.IBP`.

Given an `MLP` and an ℓ∞ perturbation radius ε, IBP computes provably
sound lower and upper bounds on the network outputs for all inputs within
the ball.  These bounds can be used to certify adversarial robustness.
-/

open TorchLib TorchLib.Verification

-- ---------------------------------------------------------------------------
-- Network under analysis
-- ---------------------------------------------------------------------------

-- Small 4-input, 2-class classifier: 4 → 8 → 8 → 2
def net : MLP Float := MLP.init 4 [8, 8] 2

-- ---------------------------------------------------------------------------
-- Concrete forward pass at the nominal point
-- ---------------------------------------------------------------------------

def x0 : Tensor Float := Tensor.full [1, 4] 0.5

def nominalOutput : Tensor Float := MLP.forward net x0

#eval nominalOutput.shape   -- [1, 2]
#eval nominalOutput.data

-- ---------------------------------------------------------------------------
-- IBP: propagate interval bounds with ε = 0.1
-- ---------------------------------------------------------------------------

def eps : Float := 0.1

-- Build the interval tensor  [x0 - ε, x0 + ε]  element-wise
def inputBounds : ITensor := ITensor.fromCenterRadius x0 eps

#eval inputBounds.shape        -- [1, 4]
#eval (inputBounds.map (·.lo)).data   -- all 0.4
#eval (inputBounds.map (·.hi)).data   -- all 0.6

-- Propagate through the MLP
def outputBounds : ITensor := IBP.mlp net inputBounds

def loBounds : Tensor Float := outputBounds.map (·.lo)
def hiBounds : Tensor Float := outputBounds.map (·.hi)

#eval loBounds.data   -- lower bound on each output logit
#eval hiBounds.data   -- upper bound on each output logit

-- ---------------------------------------------------------------------------
-- Certified robustness check
-- ---------------------------------------------------------------------------

-- The nominal output is within the certified bounds
#eval ITensor.contains outputBounds nominalOutput   -- true

-- Convenience wrapper: center + ε → (lo, hi) tensors
def mlpBoundsResult := IBP.mlpBounds net x0 eps
def lo := mlpBoundsResult.1
def hi := mlpBoundsResult.2

#eval lo.data
#eval hi.data

-- ---------------------------------------------------------------------------
-- Margin and robustness certificate
-- ---------------------------------------------------------------------------

-- For binary classification (2 classes), the network is certifiably robust
-- for class 0 if the lower bound of logit 0 exceeds the upper bound of logit 1.
--
--   robust := lo[0] > hi[1]
--
-- (A negative margin means robustness cannot be certified with IBP at this ε.)

def certifyClass0 (lo hi : Tensor Float) : String :=
  let lo0 := lo.data.getD 0 0.0
  let hi1 := hi.data.getD 1 0.0
  if lo0 > hi1 then
    s!"CERTIFIED robust for class 0  (margin = {lo0 - hi1})"
  else
    s!"NOT certified  (gap = {lo0 - hi1}, try smaller ε)"

#eval certifyClass0 lo hi

-- ---------------------------------------------------------------------------
-- Sweep over radii to find the largest certifiable ε
-- ---------------------------------------------------------------------------

def certificationSweep : IO Unit := do
  let radii : List Float := [0.01, 0.05, 0.1, 0.2, 0.5]
  for ε in radii do
    let (lo, hi) := IBP.mlpBounds net x0 ε
    let lo0 := lo.data.getD 0 0.0
    let hi1 := hi.data.getD 1 0.0
    let certified := if lo0 > hi1 then "yes" else "no"
    IO.println s!"ε = {ε}: certified = {certified}"

#eval certificationSweep

-- ---------------------------------------------------------------------------
-- IBP soundness axiom (from the library)
-- ---------------------------------------------------------------------------

-- The `ibp_linear_sound` axiom asserts that for any `Linear Float` layer `l`:
--
--   ∀ X x, x ∈ X → Linear.forward l x ∈ IBP.linear l X
--
-- This can be brought into scope as:

#check @ibp_linear_sound
