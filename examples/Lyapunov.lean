import TorchLib.Verification.Lyapunov

/-!
# Example: Lyapunov stability of a learned controller

**Problem:** given a discrete-time closed-loop map `xₖ₊₁ = f(xₖ)` (plant +
controller, here an MLP) and a candidate Lyapunov function `V`, certify that
the origin is asymptotically stable on a region.

**Method:** cover the region with a grid of boxes; on each box use the
mean-value form (exact `ΔV` at the centre + interval-Jacobian remainder) to get
a *sound* upper bound on `ΔV(x) = V(f(x)) − V(x)`.  If `V ≥ 0` and `ΔV < 0` on
every box outside a small ball around the origin, stability is certified.

Strict decrease cannot hold *at* the origin (`ΔV = 0` there), so an exclusion
ball is mathematically necessary; a finer grid shrinks the ball it needs.
-/

open TorchLib TorchLib.Verification

-- closed-loop map f(x) = c·x  (a contraction when |c| < 1)
def closedLoop (c : Float) : MLP Float :=
  { layers := #[ { weight := { shape := [2, 2], data := #[c, 0.0, 0.0, c] }
                   bias   := { shape := [2], data := #[0.0, 0.0] } } ]
    dropout := { p := 0.0 }, outputSize := 2 }

-- candidate Lyapunov function V(x) = x₁² + x₂²
def V : LyapV := LyapV.quadratic #[1.0, 1.0]

#eval do
  let region := (#[-1.0, -1.0], #[1.0, 1.0])
  let stable   := Lyapunov.certify (closedLoop 0.5) V region.1 region.2 (excludeRadius := 0.3) (splits := 8)
  let unstable := Lyapunov.certify (closedLoop 1.5) V region.1 region.2 (excludeRadius := 0.3) (splits := 8)
  IO.println "Region [-1,1]², V(x) = ‖x‖², exclusion ball ‖x‖∞ ≤ 0.3"
  IO.println ""
  IO.println s!"f(x) = 0.5x (contraction): certified={stable.certified}, worst ΔV={stable.worstDeltaV}"
  IO.println s!"  ({stable.certifiedCells}/{stable.cells} cells)"
  IO.println s!"f(x) = 1.5x (expansion):   certified={unstable.certified}, worst ΔV={unstable.worstDeltaV}"
  IO.println s!"  ({unstable.certifiedCells}/{unstable.cells} cells)"
