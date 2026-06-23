import TorchLib.Core

/-!
# TorchLibTests.RatScalar

Tests for the `Scalar Rat` transcendental approximations (`RatApprox`).
These used to be `:= 0` placeholders; here we check they agree with `Float`.
-/

namespace TorchLibTests

open TorchLib

private def ratToFloat (r : Rat) : Float := Float.ofInt r.num / Float.ofNat r.den

private def closeTo (r : Rat) (f : Float) (tol : Float := 1e-5) : Bool :=
  (ratToFloat r - f).abs < tol

#eval do
  assert! closeTo (Scalar.sqrt (2 : Rat)) (Float.sqrt 2.0)
  assert! closeTo (Scalar.sqrt (1 / 100 : Rat)) 0.1
  IO.println "✓ Rat sqrt ≈ Float sqrt"

#eval do
  assert! closeTo (Scalar.exp (1 : Rat)) (Float.exp 1.0)
  assert! closeTo (Scalar.exp (5 : Rat)) (Float.exp 5.0)
  assert! closeTo (Scalar.exp (-3 : Rat)) (Float.exp (-3.0))
  IO.println "✓ Rat exp ≈ Float exp (with range reduction)"

#eval do
  assert! closeTo (Scalar.log (10 : Rat)) (Float.log 10.0)
  assert! closeTo (Scalar.log (1 / 2 : Rat)) (Float.log 0.5)
  IO.println "✓ Rat log ≈ Float log"

#eval do
  assert! closeTo (Scalar.tanh (1 : Rat)) (Float.tanh 1.0)
  assert! closeTo (Scalar.sigmoid (2 : Rat)) (1.0 / (1.0 + Float.exp (-2.0)))
  IO.println "✓ Rat tanh / sigmoid ≈ Float"

#eval do
  -- relu and abs are exact on Rat
  assert! Scalar.relu (-3 : Rat) == 0
  assert! Scalar.relu (3 : Rat) == 3
  assert! Scalar.abs (-7 / 2 : Rat) == 7 / 2
  IO.println "✓ Rat relu / abs exact"

end TorchLibTests
