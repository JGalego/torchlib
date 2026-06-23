import TorchLib.Verification.PINN

/-!
# Example: certified PINN residual bound

**Problem:** a Physics-Informed Neural Network `uθ(x)` is trained to satisfy a
differential equation.  Can we *certify* that its residual stays small over the
whole domain, not just at sampled points?

**Method:** propagate interval enclosures of `uθ`, `uθ'`, `uθ''` through the
network with forward-mode (Taylor) interval arithmetic, then bound the PDE
residual on a subdivided domain.  The enclosure is sound: it contains the
residual for *every* `x` in the domain.

Here the network computes `uθ(x) = tanh(x)` exactly (unit weights), so we can
read off the residual of, say, `u'' + u` and compare with the analytic value.
-/

open TorchLib TorchLib.Verification

-- 1 → 1 → 1 tanh network ⇒ uθ(x) = tanh(x)
def uNet : MLP Float :=
  { layers := #[ { weight := { shape := [1, 1], data := #[1.0] }, bias := { shape := [1], data := #[0.0] } },
                 { weight := { shape := [1, 1], data := #[1.0] }, bias := { shape := [1], data := #[0.0] } } ]
    dropout := { p := 0.0 }, outputSize := 1 }

#eval do
  -- pointwise enclosures vs the analytic derivatives of tanh at x = 0.5
  let (u, u', u'') := PINN.solutionDerivs uNet SmoothAct.tanh { lo := 0.5, hi := 0.5 }
  let t := Float.tanh 0.5
  IO.println s!"u(0.5)   enclosure [{u.lo}, {u.hi}]   analytic tanh(0.5)        = {t}"
  IO.println s!"u'(0.5)  enclosure [{u'.lo}, {u'.hi}]  analytic 1 - tanh²        = {1.0 - t*t}"
  IO.println s!"u''(0.5) enclosure [{u''.lo}, {u''.hi}]  analytic -2 tanh(1-tanh²) = {-2.0*t*(1.0-t*t)}"
  IO.println ""
  -- certify the residual of  u'' + u  over [0, 1]
  let c := PINN.certifyResidual uNet SmoothAct.tanh (fun _ u _ u'' => u'' + u)
            0.0 1.0 (tol := 0.5) (splits := 32)
  IO.println s!"residual (u'' + u) over [0,1]: enclosure [{c.enclosure.lo}, {c.enclosure.hi}]"
  IO.println s!"  max |residual| ≤ {c.maxAbs},  certified within ±0.5 : {c.certified}"
