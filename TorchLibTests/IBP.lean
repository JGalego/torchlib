import TorchLib.Verification.IBP

/-!
# TorchLibTests.IBP

Tests for `TorchLib.Verification.IBP`: interval bound propagation.
-/

namespace TorchLibTests

open TorchLib TorchLib.Verification

-- ---------------------------------------------------------------------------
-- Interval arithmetic
-- ---------------------------------------------------------------------------

#eval do
  let a : Interval := { lo := 1.0, hi := 3.0 }
  let b : Interval := { lo := 2.0, hi := 4.0 }
  let s := a + b
  assert! s.lo == 3.0
  assert! s.hi == 7.0
  IO.println "✓ Interval add: [1,3]+[2,4] = [3,7]"

#eval do
  let a : Interval := { lo := 3.0, hi := 5.0 }
  let b : Interval := { lo := 1.0, hi := 2.0 }
  let d := a - b
  assert! d.lo == 1.0    -- 3-2
  assert! d.hi == 4.0    -- 5-1
  IO.println "✓ Interval sub: [3,5]-[1,2] = [1,4]"

#eval do
  let a : Interval := { lo := 2.0, hi := 3.0 }
  let b : Interval := { lo := 4.0, hi := 5.0 }
  let p := a * b
  assert! p.lo == 8.0    -- 2*4
  assert! p.hi == 15.0   -- 3*5
  IO.println "✓ Interval mul (both positive): [2,3]*[4,5] = [8,15]"

#eval do
  -- Negative × positive: [−3,−1] * [2,4]
  let a : Interval := { lo := -3.0, hi := -1.0 }
  let b : Interval := { lo := 2.0, hi := 4.0 }
  let p := a * b
  assert! p.lo == -12.0   -- (−3)*(4)
  assert! p.hi == -2.0    -- (−1)*(2)
  IO.println "✓ Interval mul (mixed signs): [−3,−1]*[2,4] = [−12,−2]"

#eval do
  let i : Interval := { lo := 1.0, hi := 2.0 }
  assert! i.width == 1.0
  assert! i.mid == 1.5
  IO.println "✓ Interval.width / mid"

#eval do
  let i : Interval := { lo := 1.0, hi := 3.0 }
  assert! i.contains 2.0
  assert! !i.contains 4.0
  IO.println "✓ Interval.contains"

-- ---------------------------------------------------------------------------
-- Interval join
-- ---------------------------------------------------------------------------

#eval do
  let a : Interval := { lo := 1.0, hi := 3.0 }
  let b : Interval := { lo := 2.0, hi := 5.0 }
  let j := a.join b
  assert! j.lo == 1.0
  assert! j.hi == 5.0
  IO.println "✓ Interval.join"

-- ---------------------------------------------------------------------------
-- Scalar Interval instance
-- ---------------------------------------------------------------------------

#eval do
  let i : Interval := Scalar.relu { lo := -1.0, hi := 2.0 }
  assert! i.lo == 0.0
  assert! i.hi == 2.0
  IO.println "✓ Scalar Interval relu: [−1,2] → [0,2]"

#eval do
  let i : Interval := Scalar.exp { lo := 0.0, hi := 1.0 }
  assert! (i.lo - 1.0).abs < 1e-6   -- exp(0) = 1
  assert! (i.hi - Float.exp 1.0).abs < 1e-6
  IO.println "✓ Scalar Interval exp: [0,1] → [e^0, e^1]"

-- ---------------------------------------------------------------------------
-- ITensor
-- ---------------------------------------------------------------------------

#eval do
  let center : Tensor Float := { shape := [3], data := #[0.0, 1.0, 2.0] }
  let it := ITensor.fromCenterRadius center 0.5
  assert! it.shape = [3]
  let i0 := it.data.get! 0
  let i1 := it.data.get! 1
  assert! i0.lo == -0.5 && i0.hi == 0.5
  assert! i1.lo == 0.5 && i1.hi == 1.5
  IO.println "✓ ITensor.fromCenterRadius"

#eval do
  let lo : Tensor Float := { shape := [2], data := #[1.0, 2.0] }
  let hi : Tensor Float := { shape := [2], data := #[3.0, 4.0] }
  let it := ITensor.fromBounds lo hi
  assert! (it.lower).data.get! 0 == 1.0
  assert! (it.upper).data.get! 1 == 4.0
  IO.println "✓ ITensor.fromBounds / lower / upper"

#eval do
  let center : Tensor Float := { shape := [2], data := #[0.0, 1.0] }
  let it := ITensor.fromCenterRadius center 0.1
  let x : Tensor Float := { shape := [2], data := #[0.05, 0.95] }
  assert! it.contains x
  let y : Tensor Float := { shape := [2], data := #[0.5, 1.0] }
  assert! !it.contains y
  IO.println "✓ ITensor.contains"

-- ---------------------------------------------------------------------------
-- IBP.relu
-- ---------------------------------------------------------------------------

#eval do
  let it := ITensor.fromCenterRadius
    { shape := [3], data := #[-1.0, 0.0, 2.0] } 0.5
  let out := IBP.relu it
  -- [-1.5,-0.5] → relu → [0,0]; [-0.5,0.5] → [0,0.5]; [1.5,2.5] → [1.5,2.5]
  let i0 := out.data.get! 0
  let i1 := out.data.get! 1
  let i2 := out.data.get! 2
  assert! i0.lo == 0.0 && i0.hi == 0.0
  assert! i1.lo == 0.0 && i1.hi == 0.5
  assert! i2.lo == 1.5 && i2.hi == 2.5
  IO.println "✓ IBP.relu"

-- ---------------------------------------------------------------------------
-- IBP.linear soundness spot-check
-- ---------------------------------------------------------------------------

#eval do
  -- 1×2 → 1×1 linear: weight [[1, -1]], bias [0]
  -- x ∈ [0.9,1.1] × [-0.1,0.1]
  -- y = x[0]*1 + x[1]*(-1) + 0
  -- lo(y) = lo(x[0])*1 + hi(x[1])*(-1) = 0.9 - 0.1 = 0.8
  -- hi(y) = hi(x[0])*1 + lo(x[1])*(-1) = 1.1 - (-0.1) = 1.2
  let l : Linear Float :=
    { weight := { shape := [1, 2], data := #[1.0, -1.0] }
      bias   := { shape := [1],   data := #[0.0] } }
  let center : Tensor Float := { shape := [1, 2], data := #[1.0, 0.0] }
  let it := ITensor.fromCenterRadius center 0.1
  let out := IBP.linear l it
  assert! out.shape = [1, 1]
  let i := out.data.get! 0
  IO.println s!"  IBP.linear out: [{i.lo}, {i.hi}]"
  assert! (i.lo - 0.8).abs < 1e-6
  assert! (i.hi - 1.2).abs < 1e-6
  IO.println "✓ IBP.linear soundness spot-check: [1,-1]·[1,0]±0.1 = 1.0±0.2"

-- ---------------------------------------------------------------------------
-- IBP.mlpBounds: output interval contains concrete output
-- ---------------------------------------------------------------------------

#eval do
  let mlp : MLP Float := MLP.init 2 [4] 2
  let center : Tensor Float := { shape := [1, 2], data := #[0.5, -0.5] }
  let eps := 0.01
  let (lo, hi) := IBP.mlpBounds mlp center eps
  -- Verify lo ≤ hi element-wise
  let ok := lo.data.zip hi.data |>.all (fun (l, h) => l ≤ h)
  assert! ok
  IO.println "✓ IBP.mlpBounds: lo ≤ hi for all output dims"

end TorchLibTests
