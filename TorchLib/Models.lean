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
  /-- Array of fully-connected layers. -/
  layers   : Array (Linear α)
  /-- Dropout applied between hidden layers. -/
  dropout  : Dropout α
  /-- Dimension of the final output. -/
  outputSize : Nat

instance [Repr α] : Repr (MLP α) where
  reprPrec x prec := Repr.addAppParen
    f!"MLP \{ layers := {reprPrec x.layers 0}, dropout := {reprPrec x.dropout 0}, outputSize := {reprPrec x.outputSize 0} }"
    prec

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
  /-- Pairs of `(Conv2d, BatchNorm2d)` blocks. -/
  convBlocks : Array (Conv2d α × BatchNorm2d α)
  /-- Classification/MLP head applied after flattening. -/
  head       : MLP α

instance [Repr α] : Repr (CNN α) where
  reprPrec x prec := Repr.addAppParen
    f!"CNN \{ convBlocks := {reprPrec x.convBlocks 0}, head := {reprPrec x.head 0} }"
    prec

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
  /-- Multi-head self-attention sub-layer. -/
  attn   : MultiheadAttention α
  /-- Pre-attention layer normalisation. -/
  ln1    : LayerNorm α
  /-- Pre-FFN layer normalisation. -/
  ln2    : LayerNorm α
  /-- FFN expansion linear layer. -/
  ffn1   : Linear α
  /-- FFN projection linear layer. -/
  ffn2   : Linear α
  /-- Residual dropout. -/
  dropout : Dropout α

instance [Repr α] : Repr (TransformerBlock α) where
  reprPrec x prec := Repr.addAppParen
    f!"TransformerBlock \{ attn := {reprPrec x.attn 0}, ln1 := {reprPrec x.ln1 0}, ln2 := {reprPrec x.ln2 0}, ffn1 := {reprPrec x.ffn1 0}, ffn2 := {reprPrec x.ffn2 0}, dropout := {reprPrec x.dropout 0} }"
    prec

namespace TransformerBlock

/-- Initialise a transformer block with the given dimensions. -/
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
  /-- Token embedding table. -/
  tokenEmbed : Embedding α
  /-- Positional embedding tensor. -/
  posEmbed   : Tensor α
  /-- Stack of transformer encoder blocks. -/
  blocks     : Array (TransformerBlock α)
  /-- Final layer normalisation. -/
  norm       : LayerNorm α
  /-- Classification or language-model head. -/
  head       : Linear α

instance [Repr α] : Repr (Transformer α) where
  reprPrec x prec := Repr.addAppParen
    f!"Transformer \{ tokenEmbed := {reprPrec x.tokenEmbed 0}, posEmbed := {reprPrec x.posEmbed 0}, blocks := {reprPrec x.blocks 0}, norm := {reprPrec x.norm 0}, head := {reprPrec x.head 0} }"
    prec

namespace Transformer

/-- Initialise the full transformer model. -/
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

/-- Vanilla recurrent neural network (wraps `RNNCell`). -/
structure RNN (α : Type) where
  /-- The underlying RNN cell. -/
  cell : RNNCell α

instance [Repr α] : Repr (RNN α) where
  reprPrec x prec := Repr.addAppParen
    f!"RNN \{ cell := {reprPrec x.cell 0} }"
    prec

namespace RNN

/-- Initialise an RNN with the given input and hidden sizes. -/
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

/-- Long Short-Term Memory network (wraps `LSTMCell`). -/
structure LSTM (α : Type) where
  /-- The underlying LSTM cell. -/
  cell : LSTMCell α

instance [Repr α] : Repr (LSTM α) where
  reprPrec x prec := Repr.addAppParen
    f!"LSTM \{ cell := {reprPrec x.cell 0} }"
    prec

namespace LSTM

/-- Initialise an LSTM. -/
def init [Scalar α] (inputSize hiddenSize : Nat) : LSTM α :=
  { cell := LSTMCell.init inputSize hiddenSize }

/-- Run the LSTM over a sequence, returning hidden states and final hidden. -/
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

/-- Gated Recurrent Unit network (wraps `GRUCell`). -/
structure GRU (α : Type) where
  /-- The underlying GRU cell. -/
  cell : GRUCell α

instance [Repr α] : Repr (GRU α) where
  reprPrec x prec := Repr.addAppParen
    f!"GRU \{ cell := {reprPrec x.cell 0} }"
    prec

namespace GRU

/-- Initialise a GRU. -/
def init [Scalar α] (inputSize hiddenSize : Nat) : GRU α :=
  { cell := GRUCell.init inputSize hiddenSize }

/-- Run the GRU over a sequence, returning all hidden states. -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (m : GRU α) (xs : List (Tensor α)) (h0 : Tensor α) : List (Tensor α) :=
  let (_, hs) := xs.foldl (fun (hPrev, acc) x =>
    let h := GRUCell.forward m.cell x hPrev
    (h, acc ++ [h])) (h0, [])
  hs

end GRU

end TorchLib
