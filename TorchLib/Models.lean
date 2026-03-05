import TorchLib.Core
import TorchLib.Layers

/-!
# TorchLib.Models

High-level model compositions built from `TorchLib.Layers`.

Models provided:
- `MLP`         — multi-layer perceptron
- `CNN`         — convolutional network with pooling & head
- `TransformerBlock` / `Transformer`  — encoder-only transformer
- `RNN` / `LSTM` / `GRU` — sequence models
-/

namespace TorchLib

-- ---------------------------------------------------------------------------
-- MLP (Multi-Layer Perceptron)
-- ---------------------------------------------------------------------------

/-- A fully-connected feed-forward network with configurable depth.

    Architecture: `Linear → Activation → ... → Linear`

    `hiddenSizes` lists the sizes of all hidden layers; the final linear maps
    to `outputSize`. -/
structure MLP (α : Type) where
  layers   : Array (Linear α)
  dropout  : Dropout α
  outputSize : Nat
  deriving Repr

namespace MLP

/-- Construct an MLP from an input size, list of hidden sizes, and output size. -/
def init [Scalar α] (inputSize : Nat) (hiddenSizes : List Nat) (outputSize : Nat)
    (dropoutP : Float := 0.0) : MLP α :=
  let allSizes := inputSize :: hiddenSizes ++ [outputSize]
  let layers := (List.range (allSizes.length - 1)).toArray.map (fun i =>
    Linear.init (allSizes.getD i default) (allSizes.getD (i + 1) default))
  { layers, dropout := { p := dropoutP }, outputSize }

/-- Forward: applies `ReLU` between hidden layers, no activation at output. -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (m : MLP α) (x : Tensor α) : Tensor α :=
  let n := m.layers.size
  m.layers.foldl (fun (acc : Tensor α × Nat) l =>
    let (t, i) := acc
    let t' := Linear.forward l t
    let t' := if i < n - 1 then t'.apply Scalar.relu else t'
    let t' := if i < n - 1 then Dropout.forward m.dropout t' else t'
    (t', i + 1)) (x, 0) |>.1

end MLP

-- ---------------------------------------------------------------------------
-- CNN (Convolutional Neural Network)
-- ---------------------------------------------------------------------------

/-- A simple CNN backbone with configurable conv blocks followed by an MLP head.

    Architecture:
      `[Conv2d → BatchNorm2d → ReLU → MaxPool2d]*` followed by `Flatten → MLP` -/
structure CNN (α : Type) where
  convBlocks : Array (Conv2d α × BatchNorm2d α)
  head       : MLP α
  deriving Repr

namespace CNN

/-- Build a CNN with `convSpec = [(C_in, C_out, kSize)]` blocks and an MLP head. -/
def init [Scalar α]
    (convSpecs : List (Nat × Nat × Nat))  -- (cIn, cOut, kernelSize)
    (flatSize outputSize : Nat)
    (hiddenSizes : List Nat := [256])
    (dropoutP : Float := 0.0) : CNN α :=
  let convBlocks := convSpecs.toArray.map (fun (cIn, cOut, k) =>
    (Conv2d.init cIn cOut k k, BatchNorm2d.init cOut (Scalar.ofRat 1e-5) (Scalar.ofRat 0.1)))
  { convBlocks
    head := MLP.init flatSize hiddenSizes outputSize dropoutP }

/-- Forward pass.  Input: `[batch, C, H, W]`.  Output: `[batch, outputSize]`. -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (m : CNN α) (x : Tensor α) : Tensor α :=
  -- Conv blocks
  let afterConv := m.convBlocks.foldl (fun t (conv, bn) =>
    let t := Conv2d.forward conv t
    let t := BatchNorm2d.forward bn t
    t.apply Scalar.relu) x
  -- Global average pool (reduce spatial dims to 1×1) then flatten
  let flat := afterConv.flatten
  MLP.forward m.head flat

end CNN

-- ---------------------------------------------------------------------------
-- Transformer Block
-- ---------------------------------------------------------------------------

/-- A single transformer encoder block.

    Architecture: `LN → MHA → residual → LN → FFN → residual` -/
structure TransformerBlock (α : Type) where
  attn   : MultiheadAttention α
  ln1    : LayerNorm α
  ln2    : LayerNorm α
  ffn1   : Linear α   -- expand: [embed_dim → ffn_dim]
  ffn2   : Linear α   -- project: [ffn_dim → embed_dim]
  dropout : Dropout α
  deriving Repr

namespace TransformerBlock

def init [Scalar α] (embedDim numHeads ffnDim : Nat)
    (eps : α) (dropoutP : Float := 0.0) : TransformerBlock α :=
  { attn    := MultiheadAttention.init embedDim numHeads dropoutP
    ln1     := LayerNorm.init [embedDim] eps
    ln2     := LayerNorm.init [embedDim] eps
    ffn1    := Linear.init embedDim ffnDim
    ffn2    := Linear.init ffnDim embedDim
    dropout := { p := dropoutP } }

/-- Forward pass.  `x` has shape `[seq_len, embed_dim]`. -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (blk : TransformerBlock α) (x : Tensor α) : Tensor α :=
  -- Self-attention with pre-LN and residual
  let x1 := LayerNorm.forward blk.ln1 x
  let a  := MultiheadAttention.forward blk.attn x1 x1 x1
  let a  := Dropout.forward blk.dropout a
  let x  := x + a
  -- FFN with pre-LN and residual
  let x2 := LayerNorm.forward blk.ln2 x
  let f  := Linear.forward blk.ffn1 x2 |>.apply Scalar.relu
  let f  := Linear.forward blk.ffn2 f
  let f  := Dropout.forward blk.dropout f
  x + f

end TransformerBlock

-- ---------------------------------------------------------------------------
-- Transformer (encoder-only, e.g. BERT-style)
-- ---------------------------------------------------------------------------

/-- A stack of transformer encoder blocks with token and positional embeddings.

    Architecture:
      `TokenEmbed + PosEmbed → [TransformerBlock]*N → LayerNorm → head` -/
structure Transformer (α : Type) where
  tokenEmbed : Embedding α      -- [vocab_size, embed_dim]
  posEmbed   : Tensor α          -- [max_seq_len, embed_dim]
  blocks     : Array (TransformerBlock α)
  norm       : LayerNorm α
  head       : Linear α          -- classification/LM head
  deriving Repr

namespace Transformer

def init [Scalar α]
    (vocabSize embedDim numHeads ffnDim numLayers maxSeqLen outputDim : Nat)
    (eps : α) (dropoutP : Float := 0.0) : Transformer α :=
  { tokenEmbed := Embedding.init vocabSize embedDim
    posEmbed   := Tensor.zeros [maxSeqLen, embedDim]
    blocks     := Array.replicate numLayers
                    (TransformerBlock.init embedDim numHeads ffnDim eps dropoutP)
    norm       := LayerNorm.init [embedDim] eps
    head       := Linear.init embedDim outputDim }

/-- Forward pass.  `tokenIds` is a list of token indices (one sequence). -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (m : Transformer α) (tokenIds : List Nat) : Tensor α :=
  -- Token + positional embeddings
  let tok := Embedding.forward m.tokenEmbed tokenIds
  let seqLen := tokenIds.length
  let posSlice : Tensor α :=
    match m.posEmbed.shape with
    | [_, d] =>
      { shape := [seqLen, d]
        data  := m.posEmbed.data.extract 0 (seqLen * d) }
    | _ => tok
  let x := tok + posSlice
  -- Transformer blocks
  let x := m.blocks.foldl (fun t blk => TransformerBlock.forward blk t) x
  -- Final norm
  let x := LayerNorm.forward m.norm x
  -- Classification head (first token, CLS-style)
  let cls : Tensor α :=
    match x.shape with
    | [_, d] => { shape := [1, d], data := x.data.extract 0 d }
    | _      => x
  Linear.forward m.head cls

end Transformer

-- ---------------------------------------------------------------------------
-- RNN (Vanilla)
-- ---------------------------------------------------------------------------

structure RNN (α : Type) where
  cell : RNNCell α
  deriving Repr

namespace RNN

def init [Scalar α] (inputSize hiddenSize : Nat) : RNN α :=
  { cell := RNNCell.init inputSize hiddenSize }

/-- Run over a sequence: returns all hidden states `[seq_len, hidden]`. -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (m : RNN α) (xs : List (Tensor α)) (h0 : Tensor α) : List (Tensor α) :=
  let (_, hs) := xs.foldl (fun (hPrev, acc) x =>
    let h := RNNCell.forward m.cell x hPrev
    (h, acc ++ [h])) (h0, [])
  hs

end RNN

-- ---------------------------------------------------------------------------
-- LSTM
-- ---------------------------------------------------------------------------

structure LSTM (α : Type) where
  cell : LSTMCell α
  deriving Repr

namespace LSTM

def init [Scalar α] (inputSize hiddenSize : Nat) : LSTM α :=
  { cell := LSTMCell.init inputSize hiddenSize }

def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (m : LSTM α) (xs : List (Tensor α)) (h0 c0 : Tensor α)
    : List (Tensor α) × Tensor α :=
  let (hFinal, _cFinal, hs) := xs.foldl (fun (h, c, acc) x =>
    let (h', c') := LSTMCell.forward m.cell x h c
    (h', c', acc ++ [h'])) (h0, c0, [])
  (hs, hFinal)

end LSTM

-- ---------------------------------------------------------------------------
-- GRU
-- ---------------------------------------------------------------------------

structure GRU (α : Type) where
  cell : GRUCell α
  deriving Repr

namespace GRU

def init [Scalar α] (inputSize hiddenSize : Nat) : GRU α :=
  { cell := GRUCell.init inputSize hiddenSize }

def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (m : GRU α) (xs : List (Tensor α)) (h0 : Tensor α) : List (Tensor α) :=
  let (_, hs) := xs.foldl (fun (hPrev, acc) x =>
    let h := GRUCell.forward m.cell x hPrev
    (h, acc ++ [h])) (h0, [])
  hs

end GRU

end TorchLib
