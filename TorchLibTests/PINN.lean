import TorchLib.Verification.PINN

/-!
# TorchLibTests.PINN

Tests for certified PINN residual bounds: the interval derivative propagation
is checked against the closed form `u(x) = tanh(x)`.
-/

namespace TorchLibTests

open TorchLib TorchLib.Verification

-- 1 → 1 → 1 network with unit weights ⇒ u(x) = tanh(x)
private def uTanh : MLP Float :=
  { layers := #[ { weight := { shape := [1, 1], data := #[1.0] }, bias := { shape := [1], data := #[0.0] } },
                 { weight := { shape := [1, 1], data := #[1.0] }, bias := { shape := [1], data := #[0.0] } } ]
    dropout := { p := 0.0 }, outputSize := 1 }

#eval do
  -- at a point, enclosures match the analytic derivatives of tanh
  let (u, u', u'') := PINN.solutionDerivs uTanh SmoothAct.tanh { lo := 0.5, hi := 0.5 }
  let t := Float.tanh 0.5
  assert! (u.lo - t).abs < 1e-6
  assert! (u'.lo - (1.0 - t * t)).abs < 1e-6
  assert! (u''.lo - (-2.0 * t * (1.0 - t * t))).abs < 1e-6
  IO.println "✓ PINN derivative enclosures match tanh closed form"

#eval do
  -- enclosure over a real interval brackets the midpoint analytic value (sound)
  let (u, u', u'') := PINN.solutionDerivs uTanh SmoothAct.tanh { lo := 0.0, hi := 1.0 }
  let t := Float.tanh 0.5
  assert! u.lo ≤ t && t ≤ u.hi
  assert! u'.lo ≤ (1.0 - t * t) && (1.0 - t * t) ≤ u'.hi
  assert! u''.lo ≤ u''.hi
  IO.println "✓ PINN enclosure over [0,1] is sound"

#eval do
  -- residual certification API returns a sound enclosure and a verdict
  let c := PINN.certifyResidual uTanh SmoothAct.tanh
            (fun _ u _ u'' => u'' + u) 0.0 1.0 (tol := 1.0) (splits := 8)
  assert! c.enclosure.lo ≤ c.enclosure.hi
  assert! c.maxAbs ≥ 0.0
  assert! c.certified    -- residual stays within ±1.0 here
  IO.println s!"✓ PINN.certifyResidual: maxAbs={c.maxAbs}, certified={c.certified}"

end TorchLibTests
