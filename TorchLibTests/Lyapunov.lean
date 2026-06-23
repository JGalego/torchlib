import TorchLib.Verification.Lyapunov

/-!
# TorchLibTests.Lyapunov

Tests for Lyapunov verification: a contraction is certified (on an annulus
around the origin) and an expansion is rejected.
-/

namespace TorchLibTests

open TorchLib TorchLib.Verification

-- linear map f(x) = c · x  (2-D, single linear layer)
private def linMap (c : Float) : MLP Float :=
  { layers := #[ { weight := { shape := [2, 2], data := #[c, 0.0, 0.0, c] }
                   bias   := { shape := [2], data := #[0.0, 0.0] } } ]
    dropout := { p := 0.0 }, outputSize := 2 }

private def V : LyapV := LyapV.quadratic #[1.0, 1.0]

#eval do
  -- contraction f(x) = 0.5x: V = ‖x‖² decreases ⇒ certified on the annulus
  let r := Lyapunov.certify (linMap 0.5) V #[-1.0, -1.0] #[1.0, 1.0]
            (excludeRadius := 0.3) (splits := 8)
  assert! r.certified
  assert! r.worstDeltaV < 0.0
  IO.println s!"✓ Lyapunov certifies contraction (worstΔV={r.worstDeltaV})"

#eval do
  -- expansion f(x) = 1.5x: ΔV > 0 ⇒ must NOT certify
  let r := Lyapunov.certify (linMap 1.5) V #[-1.0, -1.0] #[1.0, 1.0]
            (excludeRadius := 0.3) (splits := 8)
  assert! !r.certified
  assert! r.worstDeltaV > 0.0
  IO.println s!"✓ Lyapunov rejects expansion (worstΔV={r.worstDeltaV})"

#eval do
  -- finer grid shrinks the certifiable inner exclusion radius
  let r := Lyapunov.certify (linMap 0.5) V #[-1.0, -1.0] #[1.0, 1.0]
            (excludeRadius := 0.16) (splits := 16)
  assert! r.certified
  IO.println "✓ Lyapunov: finer grid certifies with smaller exclusion ball"

#eval do
  -- the interval Jacobian of a linear map is exactly the (constant) weight
  let J := Lyapunov.jacobian (linMap 0.5) (ITensor.fromCenterRadius { shape := [1,2], data := #[0.5, 0.5] } 0.1) 2
  assert! (J.get! 0).lo == 0.5 && (J.get! 0).hi == 0.5
  assert! (J.get! 1).lo == 0.0 && (J.get! 1).hi == 0.0
  IO.println "✓ interval Jacobian of f(x)=0.5x is 0.5·I"

end TorchLibTests
