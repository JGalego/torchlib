import TorchLib.Core
import TorchLib.IR
import TorchLib.Runtime.Autograd

/-!
# Example: Automatic Differentiation

Demonstrates the reverse-mode autograd engine from
`TorchLib.Runtime.Autograd`.

The `AutogradEngine` builds a tape during the forward pass and replays it
in reverse during `backward` to accumulate gradients.
-/

open TorchLib TorchLib.Runtime

/-- Print a labelled value to stdout using its `Repr` instance. -/
private def say [Repr α] (label : String) (v : α) : IO Unit :=
  IO.println s!"{label}: {reprStr v}"

/-- Default `Variable` used as a fallback for `Array.headD` when the
    autograd engine returns an empty output list. -/
private def defaultVar (shape : Shape) : Variable :=
  { id := 0, data := Tensor.zeros shape, requiresGrad := false }

-- ---------------------------------------------------------------------------
-- Helper: create engine, variables, run backward, read gradient
-- ---------------------------------------------------------------------------

/-- Simple scalar loss: `loss = relu(x)`, `x = 3.0`.
    Expected: `d(loss)/dx = 1.0`  (since x > 0). -/
def exReluGrad : IO Unit := do
  IO.println "\n--- relu grad: loss = relu(x), x = 3.0 ---"
  let eng ← AutogradEngine.init

  -- Create leaf variable x = [[3.0]]
  let x ← eng.mkVar { shape := [1, 1], data := #[3.0] }

  -- Forward: y = relu(x)
  let ys ← eng.apply .relu [x]
  let y := ys.headD (defaultVar [1, 1])

  -- Backward from y (seed gradient = 1.0)
  let tape ← eng.backward y
  let grad := tape.getGrad x.id

  IO.println s!"x.data   = {reprStr x.data}"
  IO.println s!"y.data   = {reprStr y.data}"
  IO.println s!"grad x   = {reprStr grad}"

#eval exReluGrad

-- ---------------------------------------------------------------------------
-- Squared loss: `loss = (x - target)^2`
-- Gradient: `d/dx = 2*(x - target)`
-- ---------------------------------------------------------------------------

/-- `loss = (x - 2.0)^2`, x = 5.0.
    Expected: grad = 2*(5-2) = 6.0. -/
def exSquaredLoss : IO Unit := do
  IO.println "\n--- squared loss: loss = (x - 2)², x = 5.0 ---"
  let eng ← AutogradEngine.init

  let x   ← eng.mkVar { shape := [1], data := #[5.0] }
  let tgt ← eng.mkVar { shape := [1], data := #[2.0] } (requiresGrad := false)

  -- diff = x - target
  let diffs ← eng.apply .sub [x, tgt]
  let diff  := diffs.headD (defaultVar [1])

  -- loss = diff * diff  (element-wise)
  let losses ← eng.apply .mul [diff, diff]
  let loss   := losses.headD (defaultVar [1])

  -- Backward
  let tape ← eng.backward loss
  let grad := tape.getGrad x.id

  IO.println s!"loss     = {reprStr loss.data}"
  IO.println s!"grad x   = {reprStr grad}"    -- expected: some {shape := [1], data := #[6.0]}

#eval exSquaredLoss

-- ---------------------------------------------------------------------------
-- Chain rule: sigmoid of a linear output
-- `y = sigmoid(w * x + b)`, compute dy/dw and dy/dx
-- ---------------------------------------------------------------------------

def exSigmoidChain : IO Unit := do
  IO.println "\n--- sigmoid chain: y = sigmoid(w·x + b), w=0.5, x=1.0 ---"
  let eng ← AutogradEngine.init

  -- scalar inputs
  let w ← eng.mkVar { shape := [1], data := #[0.5] }
  let x ← eng.mkVar { shape := [1], data := #[1.0] }
  let b ← eng.mkVar { shape := [1], data := #[0.1] } (requiresGrad := false)

  -- pre-activation: z = w * x
  let wxs ← eng.apply .mul [w, x]
  let wx  := wxs.headD (defaultVar [1])

  -- z = wx + b
  let zs ← eng.apply .add [wx, b]
  let z  := zs.headD (defaultVar [1])

  -- y = sigmoid(z)
  let ys ← eng.apply .sigmoid [z]
  let y  := ys.headD (defaultVar [1])

  -- Backward
  let tape ← eng.backward y
  IO.println s!"y        = {reprStr y.data}"
  IO.println s!"grad w   = {reprStr (tape.getGrad w.id)}"
  IO.println s!"grad x   = {reprStr (tape.getGrad x.id)}"

#eval exSigmoidChain

-- ---------------------------------------------------------------------------
-- VJP rules reference via `vjp`
-- ---------------------------------------------------------------------------

-- Unlike the examples above, `vjp` is a pure (non-IO) function — no engine
-- or tape is needed.  It computes one vector-Jacobian product directly.

-- Inspect the exp VJP: d/dx exp(x) = exp(x), evaluated at x=0 → grad=1
def exExpVjp : List (Tensor Float) :=
  let x   : Tensor Float := { shape := [1], data := #[0.0] }
  let dL  : Tensor Float := { shape := [1], data := #[1.0] }
  vjp .exp [x] dL

#eval say "exp VJP at x=0" exExpVjp

-- ---------------------------------------------------------------------------
-- Main
-- ---------------------------------------------------------------------------

def main : IO Unit := do
  exReluGrad
  exSquaredLoss
  exSigmoidChain
  IO.println ""
  say "exp VJP at x=0" exExpVjp
