import TorchLib.Runtime.Float32

/-!
# TorchLibTests.Float32

Tests for the IEEE-754 binary32 kernel (`Scalar Float32`, `IEEE32Exec`).
-/

namespace TorchLibTests

open TorchLib TorchLib.Runtime TorchLib.IR

#eval do
  -- binary32 rounding is real: 0.1 is not representable, so the round-trip differs
  assert! (0.1 : Float).toFloat32.toFloat != (0.1 : Float)
  IO.println "✓ binary32 rounding is real (0.1 round-trip differs)"

#eval do
  -- the Scalar Float32 instance computes the usual functions
  let r : Float32 := Scalar.relu (-2.0 : Float32)
  assert! r == 0.0
  let s : Float32 := Scalar.sqrt (4.0 : Float32)
  assert! (s.toFloat - 2.0).abs < 1e-6
  IO.println "✓ Scalar Float32 relu / sqrt"

#eval do
  -- IEEE32Exec evaluates an IR graph in binary32, matching applyOp32
  let a : Tensor Float32 := { shape := [2], data := #[1.5, -2.0] }
  let b : Tensor Float32 := { shape := [2], data := #[0.5, 4.0] }
  let out := applyOp32 .add [a, b]
  let r := out.headD (Tensor.zeros [2])
  assert! r.data.get! 0 == 2.0 && r.data.get! 1 == 2.0
  IO.println "✓ applyOp32 .add"

#eval do
  -- a full MLP forward agrees between binary64 and binary32 to single precision
  let m : MLP Float := MLP.init 3 [4] 2
  let x : Tensor Float := { shape := [1, 3], data := #[0.5, -0.25, 0.75] }
  let gap := precisionGap m x
  assert! gap < 1e-4    -- small but generally nonzero rounding gap
  IO.println s!"✓ binary64 vs binary32 MLP forward gap = {gap}"

#eval do
  -- casting a Float model to binary32 preserves shapes
  let m : MLP Float := MLP.init 2 [3] 1
  let m32 := MLP.toF32 m
  assert! m32.layers.size == m.layers.size
  assert! m32.outputSize == 1
  IO.println "✓ MLP.toF32 preserves structure"

end TorchLibTests
