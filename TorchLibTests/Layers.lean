import TorchLib.Layers
import TorchLib.Models

/-!
# TorchLibTests.Layers

Tests for `TorchLib.Layers`: Linear, Conv2d, Embedding, LayerNorm, Dropout, MLP.
-/

namespace TorchLibTests

open TorchLib

-- ---------------------------------------------------------------------------
-- Linear
-- ---------------------------------------------------------------------------

#eval do
  -- Linear 2→3: weight [3,2] filled with 0.1, bias [3] filled with 0
  let l : Linear Float := Linear.init 2 3
  assert! l.weight.shape = [3, 2]
  assert! l.bias.shape = [3]
  IO.println "✓ Linear.init 2→3 shapes"

#eval do
  let l : Linear Float := Linear.init 2 3
  -- Input [1,2]: row [1.0, 1.0]
  let x : Tensor Float := { shape := [1, 2], data := #[1.0, 1.0] }
  let y := Linear.forward l x
  assert! y.shape = [1, 3]
  -- each output = 0.1*1 + 0.1*1 + 0 = 0.2
  assert! ((y.data.get! 0) - 0.2).abs < 1e-6
  IO.println "✓ Linear.forward [1,2]→[1,3]: output ≈ 0.2"

-- ---------------------------------------------------------------------------
-- Embedding
-- ---------------------------------------------------------------------------

#eval do
  let e : Embedding Float := Embedding.init 10 4
  assert! e.weight.shape = [10, 4]
  IO.println "✓ Embedding.init 10×4 shape"

#eval do
  let e : Embedding Float := Embedding.init 5 3
  let out := Embedding.forward e [0, 2, 4]
  assert! out.shape = [3, 3]
  IO.println "✓ Embedding.forward [0,2,4] → shape [3,3]"

-- ---------------------------------------------------------------------------
-- LayerNorm
-- ---------------------------------------------------------------------------

#eval do
  let ln : LayerNorm Float := LayerNorm.init [4] 1e-5
  assert! ln.weight.shape = [4]
  assert! ln.bias.shape = [4]
  IO.println "✓ LayerNorm.init [4] shapes"

#eval do
  let ln : LayerNorm Float := LayerNorm.init [4] 1e-5
  -- Input: row of 4 identical values → normalized output should be ≈ 0
  let x : Tensor Float := { shape := [1, 4], data := #[3.0, 3.0, 3.0, 3.0] }
  let y := LayerNorm.forward ln x
  assert! y.shape = [1, 4]
  assert! y.data.all (fun v => v.abs < 1e-4)
  IO.println "✓ LayerNorm.forward: constant input → normalized ≈ 0"

-- ---------------------------------------------------------------------------
-- Dropout (inference mode: p=0 → identity)
-- ---------------------------------------------------------------------------

#eval do
  let d : Dropout Float := { p := 0.0 }
  let x : Tensor Float := { shape := [3], data := #[1.0, 2.0, 3.0] }
  let y := Dropout.forward d x
  assert! y.data == x.data
  IO.println "✓ Dropout p=0.0 is identity"

-- ---------------------------------------------------------------------------
-- MLP
-- ---------------------------------------------------------------------------

#eval do
  let mlp : MLP Float := MLP.init 4 [8, 4] 2
  assert! mlp.layers.size = 3   -- 4→8, 8→4, 4→2
  assert! mlp.outputSize = 2
  IO.println "✓ MLP.init 4→[8,4]→2: 3 layers"

#eval do
  let mlp : MLP Float := MLP.init 2 [4] 3
  let x : Tensor Float := { shape := [1, 2], data := #[0.5, -0.5] }
  let y := MLP.forward mlp x
  assert! y.shape = [1, 3]
  IO.println "✓ MLP.forward [1,2] → [1,3]"

-- ---------------------------------------------------------------------------
-- Conv2d shapes
-- ---------------------------------------------------------------------------

#eval do
  let c : Conv2d Float := Conv2d.init 1 4 3 3
  assert! c.weight.shape = [4, 1, 3, 3]
  assert! c.bias.shape = [4]
  IO.println "✓ Conv2d.init 1→4, 3×3 weight shape"

#eval do
  let c : Conv2d Float := Conv2d.init 1 2 3 3
  -- Input [1, 1, 5, 5]: batch=1, channels=1, 5×5 spatial
  let x := Tensor.zeros (α := Float) [1, 1, 5, 5]
  let y := Conv2d.forward c x
  -- hOut = (5 - 3)/1 + 1 = 3
  assert! y.shape = [1, 2, 3, 3]
  IO.println "✓ Conv2d.forward [1,1,5,5] → [1,2,3,3]"

end TorchLibTests
