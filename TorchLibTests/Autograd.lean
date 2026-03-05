import TorchLib.Runtime.Autograd

/-!
# TorchLibTests.Autograd

Tests for `TorchLib.Runtime.Autograd`: tape-based reverse-mode AD.
-/

namespace TorchLibTests

open TorchLib TorchLib.Runtime TorchLib.IR

-- ---------------------------------------------------------------------------
-- Tape basics
-- ---------------------------------------------------------------------------

#eval do
  let t := Tape.empty
  assert! t.entries.isEmpty
  assert! t.grads.isEmpty
  IO.println "✓ Tape.empty"

#eval do
  let t := Tape.empty
  let g : Tensor Float := { shape := [2], data := #[1.0, 2.0] }
  let t := t.accumulate 0 g
  match t.getGrad 0 with
  | some grad =>
    assert! grad.data.get! 0 == 1.0
    assert! grad.data.get! 1 == 2.0
    IO.println "✓ Tape.accumulate / getGrad"
  | none => assert! false

#eval do
  -- Accumulate twice: should sum
  let t := Tape.empty
  let g1 : Tensor Float := { shape := [2], data := #[1.0, 2.0] }
  let g2 : Tensor Float := { shape := [2], data := #[3.0, 4.0] }
  let t := (t.accumulate 0 g1).accumulate 0 g2
  match t.getGrad 0 with
  | some grad =>
    assert! grad.data.get! 0 == 4.0
    assert! grad.data.get! 1 == 6.0
    IO.println "✓ Tape.accumulate sums gradients"
  | none => assert! false

-- ---------------------------------------------------------------------------
-- VJP rules
-- ---------------------------------------------------------------------------

#eval do
  -- relu: positive input → gradient passes through
  let x : Tensor Float := { shape := [3], data := #[2.0, -1.0, 0.5] }
  let dout : Tensor Float := { shape := [3], data := #[1.0, 1.0, 1.0] }
  let [dx] := vjp .relu [x] dout | assert! false
  assert! dx.data.get! 0 == 1.0   -- 2.0 > 0 → pass through
  assert! dx.data.get! 1 == 0.0   -- -1.0 ≤ 0 → zero
  assert! dx.data.get! 2 == 1.0   -- 0.5 > 0 → pass through
  IO.println "✓ vjp relu"

#eval do
  -- neg: gradient negated
  let x : Tensor Float := { shape := [2], data := #[1.0, 2.0] }
  let dout : Tensor Float := { shape := [2], data := #[3.0, 4.0] }
  let [dx] := vjp .neg [x] dout | assert! false
  assert! dx.data.get! 0 == -3.0
  assert! dx.data.get! 1 == -4.0
  IO.println "✓ vjp neg"

#eval do
  -- add: gradients pass to both inputs unchanged
  let a : Tensor Float := { shape := [2], data := #[1.0, 2.0] }
  let b : Tensor Float := { shape := [2], data := #[3.0, 4.0] }
  let dout : Tensor Float := { shape := [2], data := #[1.0, 1.0] }
  let [da, db] := vjp .add [a, b] dout | assert! false
  assert! da.data.get! 0 == 1.0
  assert! db.data.get! 0 == 1.0
  IO.println "✓ vjp add"

#eval do
  -- mul: da = b * dout, db = a * dout
  let a : Tensor Float := { shape := [2], data := #[2.0, 3.0] }
  let b : Tensor Float := { shape := [2], data := #[4.0, 5.0] }
  let dout : Tensor Float := { shape := [2], data := #[1.0, 1.0] }
  let [da, db] := vjp .mul [a, b] dout | assert! false
  assert! da.data.get! 0 == 4.0   -- b[0] * dout[0]
  assert! da.data.get! 1 == 5.0
  assert! db.data.get! 0 == 2.0   -- a[0] * dout[0]
  assert! db.data.get! 1 == 3.0
  IO.println "✓ vjp mul"

#eval do
  -- sumAll: gradient broadcast back to input shape
  let x : Tensor Float := { shape := [3], data := #[1.0, 2.0, 3.0] }
  let dout : Tensor Float := { shape := [1], data := #[5.0] }
  let [dx] := vjp .sumAll [x] dout | assert! false
  assert! dx.data.all (· == 5.0)
  IO.println "✓ vjp sumAll: broadcasts scalar gradient"

-- ---------------------------------------------------------------------------
-- AutogradEngine: forward + backward
-- ---------------------------------------------------------------------------

#eval do
  -- Compute y = relu(neg(x)) for x = [1.0, -2.0]
  -- Forward: neg → [-1.0, 2.0]; relu → [0.0, 2.0]
  -- dL/d(relu_out) = [1,1]; dL/d(neg_out) = [0,1]; dL/dx = [0,-1]
  let eng ← AutogradEngine.init
  let x ← eng.mkVar { shape := [2], data := #[1.0, -2.0] }
  let [ny] ← eng.apply .neg [x] | assert! false
  let [ry] ← eng.apply .relu [ny] | assert! false
  -- loss = sum(relu_out)
  let [loss] ← eng.apply .sumAll [ry] | assert! false
  let tape ← eng.backward loss
  match tape.getGrad x.id with
  | some dx =>
    -- x[0]=1 → neg=-1 → relu=0 → grad_relu=0 → grad_neg=0 → dx=0
    -- x[1]=-2 → neg=2 → relu=2 → grad_relu=1 → grad_neg=1 → dx=-1
    assert! dx.data.get! 0 == 0.0
    assert! ((dx.data.get! 1) - (-1.0)).abs < 1e-6
    IO.println "✓ AutogradEngine: backward through neg→relu→sumAll"
  | none => assert! false

#eval do
  -- y = a + b; loss = sum(y); check dL/da = dL/db = [1,1]
  let eng ← AutogradEngine.init
  let a ← eng.mkVar { shape := [2], data := #[1.0, 2.0] }
  let b ← eng.mkVar { shape := [2], data := #[3.0, 4.0] }
  let [y] ← eng.apply .add [a, b] | assert! false
  let [loss] ← eng.apply .sumAll [y] | assert! false
  let tape ← eng.backward loss
  match tape.getGrad a.id, tape.getGrad b.id with
  | some da, some db =>
    assert! da.data.all (· == 1.0)
    assert! db.data.all (· == 1.0)
    IO.println "✓ AutogradEngine: backward through add→sumAll"
  | _, _ => assert! false

-- ---------------------------------------------------------------------------
-- zeroGrads
-- ---------------------------------------------------------------------------

#eval do
  let t := Tape.empty
  let t := t.accumulate 0 ({ shape := [2], data := #[3.0, 4.0] } : Tensor Float)
  let t := zeroGrads t [0]
  match t.getGrad 0 with
  | some g => assert! g.data.all (· == 0.0); IO.println "✓ zeroGrads clears gradient"
  | none   => assert! false

end TorchLibTests
