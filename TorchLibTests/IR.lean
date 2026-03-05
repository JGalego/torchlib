import TorchLib.IR

/-!
# TorchLibTests.IR

Tests for `TorchLib.IR`: op-tagged SSA/DAG computation graph, builder, and
reference interpreter.
-/

namespace TorchLibTests

open TorchLib TorchLib.IR

-- ---------------------------------------------------------------------------
-- Builder basics
-- ---------------------------------------------------------------------------

#eval do
  let b := Builder.init "test"
  assert! b.graph.size = 0
  assert! b.nextId = 0
  IO.println "✓ Builder.init produces empty graph"

#eval do
  let b := Builder.init "add_graph"
  let (x, b) := b.addInput [2, 3] .float32
  let (y, b) := b.addInput [2, 3] .float32
  let (outs, b) := b.emit .add [x, y] [([2, 3], .float32)]
  let z := outs.head!
  let b := b.setOutputs [z]
  let g := b.build
  assert! g.inputs.length = 2
  assert! g.outputs.length = 1
  assert! g.size = 1
  IO.println "✓ Builder: add graph has 1 node, 2 inputs, 1 output"

-- ---------------------------------------------------------------------------
-- Graph well-formedness
-- ---------------------------------------------------------------------------

#eval do
  let b := Builder.init "wf_test"
  let (x, b) := b.addInput [4] .float32
  let (outs, b) := b.emit .relu [x] [([4], .float32)]
  let y := outs.head!
  let b := b.setOutputs [y]
  let g := b.build
  assert! g.wellFormed
  IO.println "✓ Graph.wellFormed: single relu node is well-formed"

-- ---------------------------------------------------------------------------
-- Interpreter: unary elementwise ops
-- ---------------------------------------------------------------------------

#eval do
  let b := Builder.init "relu_graph"
  let (x, b) := b.addInput [4] .float32
  let (outs, b) := b.emit .relu [x] [([4], .float32)]
  let y := outs.head!
  let b := b.setOutputs [y]
  let g := b.build
  let inp : Tensor Float := { shape := [4], data := #[-1.0, -0.5, 0.0, 2.0] }
  let env := InterpEnv.empty.insert x inp
  let env' := evalGraph g env
  match env'.lookup y with
  | some t =>
    assert! t.data.get! 0 == 0.0
    assert! t.data.get! 1 == 0.0
    assert! t.data.get! 2 == 0.0
    assert! t.data.get! 3 == 2.0
    IO.println "✓ Interpreter: relu([-1,-0.5,0,2]) = [0,0,0,2]"
  | none => assert! false

#eval do
  let b := Builder.init "neg_graph"
  let (x, b) := b.addInput [3] .float32
  let (outs, b) := b.emit .neg [x] [([3], .float32)]
  let y := outs.head!
  let b := b.setOutputs [y]
  let g := b.build
  let inp : Tensor Float := { shape := [3], data := #[1.0, -2.0, 0.0] }
  let env := InterpEnv.empty.insert x inp
  let env' := evalGraph g env
  match env'.lookup y with
  | some t =>
    assert! t.data.get! 0 == -1.0
    assert! t.data.get! 1 == 2.0
    assert! t.data.get! 2 == 0.0
    IO.println "✓ Interpreter: neg([1,-2,0]) = [-1,2,0]"
  | none => assert! false

-- ---------------------------------------------------------------------------
-- Interpreter: binary ops
-- ---------------------------------------------------------------------------

#eval do
  let b := Builder.init "add_interp"
  let (x, b) := b.addInput [3] .float32
  let (y, b) := b.addInput [3] .float32
  let (outs, b) := b.emit .add [x, y] [([3], .float32)]
  let z := outs.head!
  let b := b.setOutputs [z]
  let g := b.build
  let tx : Tensor Float := { shape := [3], data := #[1.0, 2.0, 3.0] }
  let ty : Tensor Float := { shape := [3], data := #[4.0, 5.0, 6.0] }
  let env := (InterpEnv.empty.insert x tx).insert y ty
  let env' := evalGraph g env
  match env'.lookup z with
  | some t =>
    assert! t.data.get! 0 == 5.0
    assert! t.data.get! 1 == 7.0
    assert! t.data.get! 2 == 9.0
    IO.println "✓ Interpreter: add([1,2,3],[4,5,6]) = [5,7,9]"
  | none => assert! false

-- ---------------------------------------------------------------------------
-- Interpreter: matmul
-- ---------------------------------------------------------------------------

#eval do
  let b := Builder.init "matmul_graph"
  let (x, b) := b.addInput [1, 2] .float32
  let (y, b) := b.addInput [2, 1] .float32
  let (outs, b) := b.emit .matmul [x, y] [([1, 1], .float32)]
  let z := outs.head!
  let b := b.setOutputs [z]
  let g := b.build
  let tx : Tensor Float := { shape := [1, 2], data := #[3.0, 4.0] }
  let ty : Tensor Float := { shape := [2, 1], data := #[2.0, 1.0] }
  let env := (InterpEnv.empty.insert x tx).insert y ty
  let env' := evalGraph g env
  match env'.lookup z with
  | some t =>
    assert! t.data.get! 0 == 10.0
    IO.println "✓ Interpreter: matmul [1,2]×[2,1] = 10.0"
  | none => assert! false

-- ---------------------------------------------------------------------------
-- Interpreter: chained nodes (neg → relu)
-- ---------------------------------------------------------------------------

#eval do
  let b := Builder.init "chain_graph"
  let (x, b) := b.addInput [3] .float32
  let (outs1, b) := b.emit .neg [x] [([3], .float32)]
  let y := outs1.head!
  let (outs2, b) := b.emit .relu [y] [([3], .float32)]
  let z := outs2.head!
  let b := b.setOutputs [z]
  let g := b.build
  assert! g.size = 2
  let inp : Tensor Float := { shape := [3], data := #[1.0, -2.0, 0.5] }
  let env := InterpEnv.empty.insert x inp
  let env' := evalGraph g env
  match env'.lookup z with
  | some t =>
    assert! t.data.get! 0 == 0.0
    assert! t.data.get! 1 == 2.0
    assert! t.data.get! 2 == 0.0
    IO.println "✓ Interpreter: chained neg→relu"
  | none => assert! false

-- ---------------------------------------------------------------------------
-- Graph.findNode
-- ---------------------------------------------------------------------------

#eval do
  let b := Builder.init "find_test"
  let (x, b) := b.addInput [2] .float32
  let (outs, b) := b.emit .abs [x] [([2], .float32)]
  let _ := outs  -- just to suppress unused warning
  let g := b.build
  assert! (g.findNode 2).isSome
  IO.println "✓ Graph.findNode"

end TorchLibTests
