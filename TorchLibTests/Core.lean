import TorchLib.Core

/-!
# TorchLibTests.Core

Unit tests for `TorchLib.Core`: shapes, tensors, and scalar instances.
-/

namespace TorchLibTests

open TorchLib

-- ---------------------------------------------------------------------------
-- Shape tests
-- ---------------------------------------------------------------------------

#eval do  -- shape.numel
  let s : Shape := [2, 3, 4]
  assert! s.numel = 24
  IO.println "✓ Shape.numel [2,3,4] = 24"

#eval do
  let s : Shape := []
  assert! s.numel = 1
  IO.println "✓ Shape.numel [] = 1 (scalar)"

#eval do
  let s : Shape := [5]
  assert! s.rank = 1
  IO.println "✓ Shape.rank [5] = 1"

-- ---------------------------------------------------------------------------
-- Tensor construction
-- ---------------------------------------------------------------------------

#eval do
  let t := Tensor.zeros (α := Float) [3, 4]
  assert! t.shape = [3, 4]
  assert! t.numel = 12
  assert! t.data.all (· == 0.0)
  IO.println "✓ Tensor.zeros [3,4]: shape and data correct"

#eval do
  let t := Tensor.ones (α := Float) [2, 5]
  assert! t.data.all (· == 1.0)
  IO.println "✓ Tensor.ones [2,5]: all elements are 1.0"

#eval do
  let t := Tensor.full [4] (42.0 : Float)
  assert! t.data.all (· == 42.0)
  IO.println "✓ Tensor.full [4] 42.0"

-- ---------------------------------------------------------------------------
-- Tensor map / zipWith
-- ---------------------------------------------------------------------------

#eval do
  let t := Tensor.full [3] (2.0 : Float)
  let t2 := t.map (· * 3.0)
  assert! t2.data.all (· == 6.0)
  IO.println "✓ Tensor.map (* 3.0)"

#eval do
  let a := Tensor.full [4] (1.0 : Float)
  let b := Tensor.full [4] (2.0 : Float)
  let c := Tensor.zipWith (· + ·) a b
  assert! c.data.all (· == 3.0)
  IO.println "✓ Tensor.zipWith (+)"

-- ---------------------------------------------------------------------------
-- Tensor arithmetic instances
-- ---------------------------------------------------------------------------

#eval do
  let a := Tensor.full [3] (5.0 : Float)
  let b := Tensor.full [3] (3.0 : Float)
  let s := a + b
  let d := a - b
  let p := a * b
  assert! s.data.all (· == 8.0)
  assert! d.data.all (· == 2.0)
  assert! p.data.all (· == 15.0)
  IO.println "✓ Tensor (+), (-), (*) instances"

-- ---------------------------------------------------------------------------
-- Reshape / flatten
-- ---------------------------------------------------------------------------

#eval do
  let t := Tensor.zeros (α := Float) [2, 6]
  match t.reshape [3, 4] with
  | some t' => assert! t'.shape = [3, 4]; IO.println "✓ Tensor.reshape [2,6] → [3,4]"
  | none    => assert! false

#eval do
  let t := Tensor.zeros (α := Float) [2, 6]
  let f := t.flatten
  assert! f.shape = [12]
  IO.println "✓ Tensor.flatten"

-- ---------------------------------------------------------------------------
-- Matmul
-- ---------------------------------------------------------------------------

#eval do
  -- [1,2] @ [2,1] = [[a*c + b*d]]
  let a : Tensor Float := { shape := [1, 2], data := #[3.0, 4.0] }
  let b : Tensor Float := { shape := [2, 1], data := #[2.0, 1.0] }
  let c := Tensor.matmul a b
  assert! c.shape = [1, 1]
  assert! c.data.get! 0 == 10.0   -- 3*2 + 4*1
  IO.println "✓ Tensor.matmul [1,2]×[2,1] = 10.0"

-- ---------------------------------------------------------------------------
-- Transpose
-- ---------------------------------------------------------------------------

#eval do
  let t : Tensor Float := { shape := [2, 3], data := #[1,2,3,4,5,6] }
  let tT := t.transpose
  assert! tT.shape = [3, 2]
  assert! tT.data.get! 0 == 1.0
  assert! tT.data.get! 1 == 4.0
  IO.println "✓ Tensor.transpose [2,3]"

-- ---------------------------------------------------------------------------
-- StateDict
-- ---------------------------------------------------------------------------

#eval do
  let sd := StateDict.empty (α := Float)
  let sd := sd.insert "weight" (Tensor.ones [3, 3])
  let sd := sd.insert "bias"   (Tensor.zeros [3])
  assert! (sd.lookup "weight").isSome
  assert! (sd.lookup "bias").isSome
  assert! (sd.lookup "missing").isNone
  IO.println "✓ StateDict insert/lookup"

-- ---------------------------------------------------------------------------
-- Scalar Float instance
-- ---------------------------------------------------------------------------

#eval do
  let x : Float := Scalar.exp 0.0
  assert! (x - 1.0).abs < 1e-6
  IO.println "✓ Scalar Float: exp(0) ≈ 1.0"

#eval do
  let x : Float := Scalar.relu (-2.0)
  assert! x == 0.0
  let y : Float := Scalar.relu 3.0
  assert! y == 3.0
  IO.println "✓ Scalar Float: relu"

-- ---------------------------------------------------------------------------
-- Sum
-- ---------------------------------------------------------------------------

#eval do
  let t : Tensor Float := { shape := [4], data := #[1, 2, 3, 4] }
  assert! t.sum == 10.0
  IO.println "✓ Tensor.sum"

end TorchLibTests
