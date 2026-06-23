import TorchLib.Verification.CROWN

/-!
# Example: α-CROWN bound tightening

**Problem:** bound the output of a ReLU network over an ℓ∞ input ball as
tightly as possible.

**Methods:**
- **IBP** — propagate intervals forward (cheap).
- **CROWN** — back-substitute affine relaxations through the network.
- **α-CROWN** — additionally *optimise* the unstable-ReLU lower slopes
  (`α ∈ [0,1]`) to minimise the certified width.

All three are *sound*: the printed bounds always contain the true output.

**What is and isn't guaranteed:**
- α-CROWN ≤ CROWN in width *always* — it starts from CROWN's adaptive slopes
  and keeps the tightest result it finds, so it can only improve.
- CROWN is *usually* tighter than IBP, but not always; for deep networks the
  back-substitution relaxations can lose to plain interval arithmetic.  The
  `crownIBP` hybrid takes the better of the two on each side, so it is ≤ both.

α-CROWN only helps when there are ≥ 2 hidden ReLU layers, so that one layer's
relaxation feeds another and the jointly-optimal slope is interior to `[0,1]`.
This network has two hidden layers, so the optimisation has something to do.
-/

open TorchLib TorchLib.Verification

-- 2 → 3 → 3 → 1 network (two hidden ReLU layers) with mixed-sign weights
def net : MLP Float :=
  { layers := #[ { weight := { shape := [3, 2], data := #[1.0, -1.0, 0.7, 0.8, -0.9, 0.5] }
                   bias   := { shape := [3], data := #[0.1, -0.1, 0.2] } },
                 { weight := { shape := [3, 3], data := #[0.6, -0.7, 0.5, -0.8, 0.9, -0.4, 0.3, -0.6, 0.7] }
                   bias   := { shape := [3], data := #[0.0, 0.1, -0.1] } },
                 { weight := { shape := [1, 3], data := #[1.0, -1.0, 0.8] }
                   bias   := { shape := [1], data := #[0.0] } } ]
    dropout := { p := 0.0 }, outputSize := 1 }

def x0 : Tensor Float := { shape := [1, 2], data := #[0.3, -0.2] }
def eps : Float := 0.15

def width (lohi : Tensor Float × Tensor Float) : Float :=
  (lohi.2 - lohi.1).data.foldl (· + ·) 0.0

#eval do
  let trueY := (MLP.forward net x0).data.get! 0
  let ibp := IBP.mlpBounds net x0 eps
  let cr  := CROWN.mlpBounds net x0 eps
  let ac  := AlphaCROWN.bounds net x0 eps
  let hy  := computeBounds .crownIBP net x0 eps
  IO.println s!"true f(x0)        = {trueY}"
  IO.println s!"IBP      bound    = [{ibp.1.data.get! 0}, {ibp.2.data.get! 0}]  width {width ibp}"
  IO.println s!"CROWN    bound    = [{cr.1.data.get! 0}, {cr.2.data.get! 0}]  width {width cr}"
  IO.println s!"α-CROWN  bound    = [{ac.1.data.get! 0}, {ac.2.data.get! 0}]  width {width ac}"
  IO.println s!"CROWN-IBP bound   = [{hy.1.data.get! 0}, {hy.2.data.get! 0}]  width {width hy}"
  IO.println ""
  IO.println "α-CROWN tightened CROWN by optimising the ReLU slopes; every"
  IO.println "interval still contains f(x0) (all methods are sound)."
