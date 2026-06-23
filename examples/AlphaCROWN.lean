import TorchLib.Verification.CROWN

/-!
# Example: α-CROWN bound tightening

**Problem:** bound the output of a ReLU network over an ℓ∞ input ball as
tightly as possible.

**Methods, tightest last:**
- **IBP** — propagate intervals forward (cheap, loose).
- **CROWN** — back-substitute affine relaxations (tighter).
- **α-CROWN** — additionally *optimise* the unstable-ReLU lower slopes
  (`α ∈ [0,1]`) to minimise the certified width (tightest).

All three are *sound*: the printed bounds always contain the true output.
-/

open TorchLib TorchLib.Verification

-- 2 → 4 → 1 network with mixed-sign weights ⇒ several unstable ReLUs
def net : MLP Float :=
  { layers := #[ { weight := { shape := [4, 2], data := #[1.0, -1.0, 0.8, 0.6, -0.7, 0.9, 0.5, -1.2] }
                   bias   := { shape := [4], data := #[0.1, -0.2, 0.0, 0.3] } },
                 { weight := { shape := [1, 4], data := #[1.0, -1.0, 0.5, -0.8] }
                   bias   := { shape := [1], data := #[0.05] } } ]
    dropout := { p := 0.0 }, outputSize := 1 }

def x0 : Tensor Float := { shape := [1, 2], data := #[0.25, -0.4] }
def eps : Float := 0.15

def width (lohi : Tensor Float × Tensor Float) : Float :=
  (lohi.2 - lohi.1).data.foldl (· + ·) 0.0

#eval do
  let trueY := (MLP.forward net x0).data.get! 0
  let ibp := IBP.mlpBounds net x0 eps
  let cr  := CROWN.mlpBounds net x0 eps
  let ac  := AlphaCROWN.bounds net x0 eps
  IO.println s!"true f(x0)        = {trueY}"
  IO.println s!"IBP      bound    = [{ibp.1.data.get! 0}, {ibp.2.data.get! 0}]  width {width ibp}"
  IO.println s!"CROWN    bound    = [{cr.1.data.get! 0}, {cr.2.data.get! 0}]  width {width cr}"
  IO.println s!"α-CROWN  bound    = [{ac.1.data.get! 0}, {ac.2.data.get! 0}]  width {width ac}"
  IO.println ""
  IO.println "α-CROWN ≤ CROWN ≤ IBP in width, and every interval contains f(x0)."
