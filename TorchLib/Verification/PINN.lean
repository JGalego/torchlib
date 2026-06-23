import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models
import TorchLib.Verification.IBP

/-!
# TorchLib.Verification.PINN

**Certified residual bounds for Physics-Informed Neural Networks**

A PINN approximates the solution `u(x)` of a differential equation by a neural
network `uθ`.  The *residual* is what is left when `uθ` is substituted into the
equation, e.g. for `u'' = f(x)` the residual is `R(x) = uθ''(x) − f(x)`.  A
small residual everywhere on the domain witnesses that `uθ` nearly solves the
equation.

This module produces a **sound** enclosure of the residual over an input box by
propagating *interval* enclosures of `uθ`, `uθ'` and `uθ''` through the network
using forward-mode (Taylor-arithmetic) differentiation:

```
Linear     v ↦ W·v + b      v' ↦ W·v'        v'' ↦ W·v''
σ (smooth) v ↦ σ(v)         v' ↦ σ'(v)·v'    v'' ↦ σ''(v)·v'² + σ'(v)·v''
```

All operations are interval operations from `TorchLib.Verification.IBP`, so the
resulting enclosures are guaranteed to contain the true values for every input
in the box.  Subdividing the domain tightens the (otherwise conservative)
interval arithmetic.

The network is taken to be a scalar-in / scalar-out MLP with a smooth
activation (`tanh` or `sigmoid`); ReLU is excluded because it is not twice
differentiable.
-/

namespace TorchLib.Verification

open TorchLib

/-- A smooth activation supplies interval enclosures of `(σ(z), σ'(z), σ''(z))`. -/
structure SmoothAct where
  /-- `(value, first derivative, second derivative)` enclosures at argument `z`. -/
  derivs : Interval → Interval × Interval × Interval

namespace SmoothAct

private def iconst (v : Float) : Interval := { lo := v, hi := v }

/-- `tanh`: `σ' = 1 − σ²`, `σ'' = −2σ(1 − σ²)`. -/
def tanh : SmoothAct where
  derivs z :=
    let s  := Scalar.tanh z
    let d1 := iconst 1.0 - s * s
    let d2 := iconst (-2.0) * s * d1
    (s, d1, d2)

/-- `sigmoid`: `σ' = σ(1 − σ)`, `σ'' = σ(1 − σ)(1 − 2σ)`. -/
def sigmoid : SmoothAct where
  derivs z :=
    let s  := Scalar.sigmoid z
    let d1 := s * (iconst 1.0 - s)
    let d2 := d1 * (iconst 1.0 - iconst 2.0 * s)
    (s, d1, d2)

end SmoothAct

namespace PINN

/-- Apply an affine map `W·v (+ b)` to an interval vector `v` (length = in-dim),
    with constant `Float` weights.  `b?` is added only when supplied (the bias
    drops out for derivative channels). -/
def affine (W : Tensor Float) (b? : Option (Tensor Float)) (v : Array Interval)
    : Array Interval :=
  match W.shape with
  | [out, inDim] => Id.run do
    let mut res := Array.replicate out (default : Interval)
    for i in [:out] do
      let mut acc : Interval :=
        match b? with
        | some b => { lo := b.data.getD i 0.0, hi := b.data.getD i 0.0 }
        | none   => { lo := 0.0, hi := 0.0 }
      for j in [:inDim] do
        let w := W.data.get! (i * inDim + j)
        let wv : Interval := { lo := w, hi := w }
        acc := acc + wv * v.getD j default
      res := res.set! i acc
    return res
  | _ => v

/-- Propagate `(value, u', u'')` interval enclosures through the MLP for a
    scalar input enclosed by `X`.  Activations are applied after every layer
    except the last (matching `MLP.forward`'s structure).  Returns the scalar
    enclosures `(u, u', u'')` of the network output and its first two
    derivatives over `X`. -/
def solutionDerivs (m : MLP Float) (act : SmoothAct) (X : Interval)
    : Interval × Interval × Interval :=
  let layers := m.layers.toList
  let L := layers.length
  -- input channel: value = X, du/dx = 1, d²u/dx² = 0
  let init : Array Interval × Array Interval × Array Interval :=
    (#[X], #[{ lo := 1.0, hi := 1.0 }], #[{ lo := 0.0, hi := 0.0 }])
  let (v, d1, d2) := Id.run do
    let mut st := init
    for i in [:L] do
      let l := layers.getD i { weight := Tensor.zeros [], bias := Tensor.zeros [] }
      let (v, d1, d2) := st
      let v'  := affine l.weight (some l.bias) v
      let d1' := affine l.weight none d1
      let d2' := affine l.weight none d2
      if i < L - 1 then
        -- elementwise chain rule through the smooth activation
        let n := v'.size
        let mut vo  := Array.replicate n (default : Interval)
        let mut d1o := Array.replicate n (default : Interval)
        let mut d2o := Array.replicate n (default : Interval)
        for k in [:n] do
          let (s, sp, spp) := act.derivs (v'.getD k default)
          let g1 := d1'.getD k default
          let g2 := d2'.getD k default
          vo  := vo.set!  k s
          d1o := d1o.set! k (sp * g1)
          d2o := d2o.set! k (spp * (g1 * g1) + sp * g2)
        st := (vo, d1o, d2o)
      else
        st := (v', d1', d2')
    return st
  (v.getD 0 default, d1.getD 0 default, d2.getD 0 default)

/-- A PDE residual operator: given enclosures of `(x, u, u', u'')` it returns an
    interval enclosure of the residual.  For example `u'' + u = 0` (harmonic
    oscillator) is `fun _ u _ u'' => u'' + u`. -/
abbrev Residual := (x u u' u'' : Interval) → Interval

/-- Result of a PINN certification attempt. -/
structure CertifiedResidual where
  /-- Sound enclosure of the residual over the whole domain. -/
  enclosure : Interval
  /-- Maximum absolute residual implied by the enclosure. -/
  maxAbs    : Float
  /-- Whether `maxAbs ≤ tol`, i.e. the residual is certified small. -/
  certified : Bool

instance : Repr CertifiedResidual where
  reprPrec c prec := Repr.addAppParen
    f!"CertifiedResidual \{ enclosure := {reprPrec c.enclosure 0}, maxAbs := {reprPrec c.maxAbs 0}, certified := {reprPrec c.certified 0} }"
    prec

/-- Certify that a PINN's PDE residual stays within `±tol` over `[domLo, domHi]`.

    The domain is split into `splits` equal sub-intervals; the residual is
    enclosed on each (forward-mode interval differentiation) and the results are
    joined.  More splits ⇒ tighter (sound) enclosure. -/
def certifyResidual (m : MLP Float) (act : SmoothAct) (res : Residual)
    (domLo domHi : Float) (tol : Float := 1e-2) (splits : Nat := 16)
    : CertifiedResidual :=
  let n := Nat.max 1 splits
  let w := (domHi - domLo) / Float.ofNat n
  let enclosure := Id.run do
    let mut acc : Option Interval := none
    for k in [:n] do
      let a := domLo + Float.ofNat k * w
      let b := a + w
      let X : Interval := { lo := a, hi := b }
      let (u, u', u'') := solutionDerivs m act X
      let r := res X u u' u''
      acc := some (match acc with | none => r | some p => p.join r)
    return acc.getD { lo := 0.0, hi := 0.0 }
  let maxAbs := Float.max (Float.abs enclosure.lo) (Float.abs enclosure.hi)
  { enclosure, maxAbs, certified := maxAbs ≤ tol }

end PINN

end TorchLib.Verification
