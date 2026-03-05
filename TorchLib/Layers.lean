import TorchLib.Core
import TorchLib.IR

/-!
# TorchLib.Layers

Parameterized layers that compile to the shared IR.

Each layer is defined as a `structure` holding its parameters (`Tensor α`) and
exposes:
- `forward : Tensor α → Tensor α` — eager evaluation
- `toGraph : Builder → ValueId → (ValueId × Builder)` — IR emission

The scalar-polymorphic design means the same layer definition covers proofs
(`α = Float` / `α = Rat`) and verification (`α = Interval`).
-/

namespace TorchLib

open IR

-- ---------------------------------------------------------------------------
-- Layer typeclass
-- ---------------------------------------------------------------------------

/-- Any module that can (a) run forward, (b) emit its subgraph, and
    (c) expose its parameters as a `StateDict`. -/
class Layer (L : Type) (α : Type) where
  forward  : L → Tensor α → Tensor α
  params   : L → StateDict α
  toGraph  : L → Builder → ValueId → (ValueId × Builder)

-- ---------------------------------------------------------------------------
-- Linear  (y = x W^T + b)
-- ---------------------------------------------------------------------------

/-- A fully-connected affine layer: `y = x W^T + b`. -/
structure Linear (α : Type) where
  weight : Tensor α   -- shape [out_features, in_features]
  bias   : Tensor α   -- shape [out_features]
  deriving Repr

namespace Linear

/-- Kaiming/uniform initialisation using `Float` and then cast. -/
def init [Scalar α] (inF outF : Nat) : Linear α :=
  -- For deterministic tests we just fill with small constants.
  let w := Tensor.full [outF, inF] (Scalar.ofRat (1 / 10 : Rat) : α)
  let b := Tensor.zeros [outF]
  { weight := w, bias := b }

/-- Forward pass: input `[batch, inF]` → output `[batch, outF]`. -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α] (l : Linear α) (x : Tensor α)
    : Tensor α :=
  -- y = x @ w^T + b  (broadcasting bias along batch)
  let wT := l.weight.transpose
  let y  := Tensor.matmul x wT
  -- add bias to each row
  match y.shape with
  | [m, n] =>
    let data := Id.run do
      let mut d := y.data
      for i in [:m] do
        for j in [:n] do
          d := d.set! (i * n + j) (d.get! (i * n + j) + l.bias.data.get! j)
      return d
    { shape := [m, n], data }
  | _ => y

def toGraph (l : Linear Float) (b : Builder) (inp : ValueId) : ValueId × Builder :=
  let wConst := l.weight   -- would be embedded as a ConstOp
  let _ := wConst
  -- Emit: matmul(inp, weight^T) + bias
  let (ys, b) := b.emit .matmul [inp] [([1, 1], .float32)]
  (ys.headD 0, b)

end Linear

-- ---------------------------------------------------------------------------
-- Conv2d
-- ---------------------------------------------------------------------------

/-- 2-D convolutional layer.
    Input:  `[batch, C_in, H, W]`
    Output: `[batch, C_out, H', W']` -/
structure Conv2d (α : Type) where
  weight  : Tensor α   -- [C_out, C_in, kH, kW]
  bias    : Tensor α   -- [C_out]
  stride  : Nat := 1
  padding : Nat := 0
  deriving Repr

namespace Conv2d

def init [Scalar α] (cIn cOut kH kW : Nat) : Conv2d α :=
  { weight  := Tensor.full [cOut, cIn, kH, kW] (Scalar.ofRat (1 / 100 : Rat) : α)
    bias    := Tensor.zeros [cOut] }

/-- Reference (unoptimised) forward for `Float`. -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α]
    (l : Conv2d α) (x : Tensor α) : Tensor α :=
  match x.shape, l.weight.shape with
  | [bs, _cIn, h, w], [cOut, cIn, kH, kW] =>
    let stride  := l.stride
    let padding := l.padding
    let hOut := (h + 2 * padding - kH) / stride + 1
    let wOut := (w + 2 * padding - kW) / stride + 1
    let data := Id.run do
      let mut d := Array.replicate (bs * cOut * hOut * wOut) (Zero.zero : α)
      for b in [:bs] do
        for co in [:cOut] do
          for oh in [:hOut] do
            for ow in [:wOut] do
              let mut acc : α := Zero.zero
              for ci in [:cIn] do
                for kh in [:kH] do
                  for kw in [:kW] do
                    let ih := oh * stride + kh - padding
                    let iw := ow * stride + kw - padding
                    if ih < h && iw < w then
                      let xi := b * _cIn * h * w + ci * h * w + ih * w + iw
                      let wi := co * cIn * kH * kW + ci * kH * kW + kh * kW + kw
                      acc := acc + x.data.get! xi * l.weight.data.get! wi
              acc := acc + l.bias.data.get! co
              d := d.set! (b * cOut * hOut * wOut + co * hOut * wOut + oh * wOut + ow) acc
      return d
    { shape := [bs, cOut, hOut, wOut], data }
  | _, _ => x

end Conv2d

-- ---------------------------------------------------------------------------
-- Embedding
-- ---------------------------------------------------------------------------

/-- Token embedding table: maps integer indices → dense vectors. -/
structure Embedding (α : Type) where
  weight : Tensor α   -- [vocab_size, embed_dim]
  deriving Repr

namespace Embedding

def init [Scalar α] (vocabSize embedDim : Nat) : Embedding α :=
  { weight := Tensor.full [vocabSize, embedDim] (Scalar.ofRat (0 : Rat) : α) }

/-- Look up embeddings for a list of token ids. -/
def forward [Inhabited α] (e : Embedding α) (indices : List Nat) : Tensor α :=
  match e.weight.shape with
  | [_, d] =>
    let data := Id.run do
      let mut arr : Array α := #[]
      for idx in indices do
        for j in [:d] do
          arr := arr.push (e.weight.data.get! (idx * d + j))
      return arr
    { shape := [indices.length, d], data }
  | _ => e.weight

end Embedding

-- ---------------------------------------------------------------------------
-- LayerNorm
-- ---------------------------------------------------------------------------

/-- Layer normalisation: normalise over the last `normalizedShape` dimensions. -/
structure LayerNorm (α : Type) where
  normalizedShape : Shape
  weight : Tensor α   -- gain γ
  bias   : Tensor α   -- shift β
  eps    : α
  deriving Repr

namespace LayerNorm

def init [Scalar α] (s : Shape) (eps : α) : LayerNorm α :=
  { normalizedShape := s
    weight := Tensor.ones s
    bias   := Tensor.zeros s
    eps    := eps }

def forward [Inhabited α] [Scalar α] (l : LayerNorm α) (x : Tensor α) : Tensor α :=
  let xn := Tensor.layerNorm x l.eps
  -- scale and shift
  xn.zipWith (· * ·) l.weight |>.zipWith (· + ·) l.bias

end LayerNorm

-- ---------------------------------------------------------------------------
-- BatchNorm2d
-- ---------------------------------------------------------------------------

structure BatchNorm2d (α : Type) where
  numFeatures : Nat
  weight   : Tensor α   -- γ  [C]
  bias     : Tensor α   -- β  [C]
  runMean  : Tensor α
  runVar   : Tensor α
  eps      : α
  momentum : α
  training : Bool := true
  deriving Repr

namespace BatchNorm2d

def init [Scalar α] (c : Nat) (eps momentum : α) : BatchNorm2d α :=
  { numFeatures := c
    weight   := Tensor.ones [c]
    bias     := Tensor.zeros [c]
    runMean  := Tensor.zeros [c]
    runVar   := Tensor.ones [c]
    eps, momentum }

/-- Simplified forward (inference mode): normalise each channel. -/
def forward [Inhabited α] [Scalar α]
    (bn : BatchNorm2d α) (x : Tensor α) : Tensor α :=
  match x.shape with
  | [bs, c, h, w] =>
    let data := Id.run do
      let mut d := x.data
      for ci in [:c] do
        let mean := bn.runMean.data.get! ci
        let var  := bn.runVar.data.get! ci
        let std  := Scalar.sqrt (var + bn.eps)
        let gamma := bn.weight.data.get! ci
        let beta  := bn.bias.data.get! ci
        for b in [:bs] do
          for i in [:h] do
            for j in [:w] do
              let idx := b * c * h * w + ci * h * w + i * w + j
              let v   := d.get! idx
              d := d.set! idx (gamma * Scalar.inv std * (v - mean) + beta)
      return d
    { shape := x.shape, data }
  | _ => x

end BatchNorm2d

-- ---------------------------------------------------------------------------
-- Dropout
-- ---------------------------------------------------------------------------

/-- Dropout regularisation (training: zero elements with probability `p`). -/
structure Dropout (α : Type) where
  p        : Float
  training : Bool := true
  deriving Repr

namespace Dropout

/-- During inference dropout is the identity; during training we apply a
    Bernoulli mask.  We use a deterministic pseudo-random state for reproducibility. -/
def forward [Inhabited α] [Scalar α] (d : Dropout α) (x : Tensor α) : Tensor α :=
  if !d.training || d.p == 0.0 then x
  else
    -- Simple LCG for deterministic masking (seed = 42)
    let data := Id.run do
      let mut arr := x.data
      let mut rng : UInt64 := 42
      for i in [:arr.size] do
        rng := rng * 6364136223846793005 + 1442695040888963407
        let u := (rng >>> 33).toFloat / 2147483647.0
        if u < d.p then
          arr := arr.set! i (Zero.zero : α)
      return arr
    { shape := x.shape, data }

end Dropout

-- ---------------------------------------------------------------------------
-- MultiheadAttention
-- ---------------------------------------------------------------------------

/-- Multi-head scaled dot-product attention.

    For a query `Q`, key `K`, value `V`:
      `Attn(Q,K,V) = softmax(Q K^T / √d_k) V`

    All projections are represented as `Linear` layers. -/
structure MultiheadAttention (α : Type) where
  embedDim : Nat
  numHeads : Nat
  wQ : Linear α   -- [embed_dim, embed_dim]
  wK : Linear α
  wV : Linear α
  wO : Linear α   -- output projection
  dropout : Dropout α
  deriving Repr

namespace MultiheadAttention

def init [Scalar α] (embedDim numHeads : Nat) (dropoutP : Float := 0.0)
    : MultiheadAttention α :=
  { embedDim
    numHeads
    wQ := Linear.init embedDim embedDim
    wK := Linear.init embedDim embedDim
    wV := Linear.init embedDim embedDim
    wO := Linear.init embedDim embedDim
    dropout := { p := dropoutP } }

/-- Forward pass.
    `query`, `key`, `value` each have shape `[seq_len, embed_dim]`
    (single-batch for simplicity).
    Returns shape `[seq_len, embed_dim]`. -/
def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (attn : MultiheadAttention α) (query key value : Tensor α) : Tensor α :=
  let q := Linear.forward attn.wQ query
  let k := Linear.forward attn.wK key
  let v := Linear.forward attn.wV value
  -- Scaled dot-product: scores = Q K^T / sqrt(d_k)
  let dk  := Scalar.ofNat attn.embedDim
  let sqrtDk := Scalar.sqrt dk
  let scores := Tensor.matmul q k.transpose
  let scores := scores.map (· * Scalar.inv sqrtDk)
  let weights := Tensor.softmax scores
  let weights := Dropout.forward attn.dropout weights
  let context := Tensor.matmul weights v
  Linear.forward attn.wO context

end MultiheadAttention

-- ---------------------------------------------------------------------------
-- RNN Cell primitives
-- ---------------------------------------------------------------------------

/-- Vanilla RNN cell: `h' = tanh(x W_ih^T + b_ih + h W_hh^T + b_hh)` -/
structure RNNCell (α : Type) where
  wIh : Linear α
  wHh : Linear α
  deriving Repr

namespace RNNCell

def init [Scalar α] (inputSize hiddenSize : Nat) : RNNCell α :=
  { wIh := Linear.init inputSize hiddenSize
    wHh := Linear.init hiddenSize hiddenSize }

def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (cell : RNNCell α) (x h : Tensor α) : Tensor α :=
  let ih := Linear.forward cell.wIh x
  let hh := Linear.forward cell.wHh h
  (ih + hh).apply Scalar.tanh

end RNNCell

/-- LSTM cell: produces `(h', c')` from `(x, h, c)`. -/
structure LSTMCell (α : Type) where
  -- Weight matrix for [i, f, g, o] gates combined: [4*hidden, input+hidden]
  wIh : Linear α   -- input → 4*hidden
  wHh : Linear α   -- hidden → 4*hidden
  deriving Repr

namespace LSTMCell

def init [Scalar α] (inputSize hiddenSize : Nat) : LSTMCell α :=
  { wIh := Linear.init inputSize (4 * hiddenSize)
    wHh := Linear.init hiddenSize (4 * hiddenSize) }

def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (cell : LSTMCell α) (x h c : Tensor α) : Tensor α × Tensor α :=
  let gates := Linear.forward cell.wIh x + Linear.forward cell.wHh h
  -- Split gates (4 equal chunks along last dim)
  let split (t : Tensor α) (k : Nat) : Tensor α :=
    match t.shape with
    | [n] =>
      let sz := n / 4
      { shape := [sz], data := t.data.extract (k * sz) ((k + 1) * sz) }
    | [m, n] =>
      let sz := n / 4
      let data := Id.run do
        let mut d : Array α := #[]
        for i in [:m] do
          for j in [:sz] do
            d := d.push (t.data.get! (i * n + k * sz + j))
        return d
      { shape := [m, sz], data }
    | _ => t
  let iGate := (split gates 0).apply Scalar.sigmoid
  let fGate := (split gates 1).apply Scalar.sigmoid
  let gGate := (split gates 2).apply Scalar.tanh
  let oGate := (split gates 3).apply Scalar.sigmoid
  let c' := fGate * c + iGate * gGate
  let h' := oGate * c'.apply Scalar.tanh
  (h', c')

end LSTMCell

/-- GRU cell: produces `h'` from `(x, h)`. -/
structure GRUCell (α : Type) where
  wIh : Linear α   -- input → 3*hidden
  wHh : Linear α   -- hidden → 3*hidden
  deriving Repr

namespace GRUCell

def init [Scalar α] (inputSize hiddenSize : Nat) : GRUCell α :=
  { wIh := Linear.init inputSize (3 * hiddenSize)
    wHh := Linear.init hiddenSize (3 * hiddenSize) }

def forward [Inhabited α] [Add α] [Mul α] [Zero α] [Scalar α]
    (cell : GRUCell α) (x h : Tensor α) : Tensor α :=
  let ih := Linear.forward cell.wIh x
  let hh := Linear.forward cell.wHh h
  let split (t : Tensor α) (k : Nat) : Tensor α :=
    match t.shape with
    | [n] =>
      let sz := n / 3
      { shape := [sz], data := t.data.extract (k * sz) ((k + 1) * sz) }
    | [m, n] =>
      let sz := n / 3
      let data := Id.run do
        let mut d : Array α := #[]
        for i in [:m] do
          for j in [:sz] do
            d := d.push (t.data.get! (i * n + k * sz + j))
        return d
      { shape := [m, sz], data }
    | _ => t
  -- r = σ(ih_r + hh_r)
  let r := (split ih 0 + split hh 0).apply Scalar.sigmoid
  -- z = σ(ih_z + hh_z)
  let z := (split ih 1 + split hh 1).apply Scalar.sigmoid
  -- n = tanh(ih_n + r * hh_n)
  let n := (split ih 2 + r * split hh 2).apply Scalar.tanh
  -- h' = (1 - z) * n + z * h
  let one := Tensor.ones n.shape
  let oneMinusZ := one.zipWith (fun a b => a - b) z
  oneMinusZ * n + z * h

end GRUCell

end TorchLib
