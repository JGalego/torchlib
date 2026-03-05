import TorchLib.Core

/-!
# TorchLib.IR

Op-tagged SSA/DAG computation-graph IR — the single semantic target shared by
NN.Spec (`α = ℝ`), NN.Runtime (`α = Float`), and NN.Verification (`α = Interval`).

The IR uses Static Single Assignment (SSA) form: every value is defined exactly
once.  The graph is acyclic (DAG) — cycles arise only via explicit loop/recur
ops (used for RNNs).
-/

namespace TorchLib.IR

-- ---------------------------------------------------------------------------
-- Tensor dtype
-- ---------------------------------------------------------------------------

inductive DType
  | float32
  | float64
  | bfloat16
  | int32
  | int64
  | bool
  deriving Repr, BEq

-- ---------------------------------------------------------------------------
-- Op codes
-- ---------------------------------------------------------------------------

/-- All primitive operations understood by the IR.
    Each op corresponds to exactly one node in the computation DAG. -/
inductive OpCode
  -- Tensor creation
  | const                 -- constant tensor (embedded in node)
  | zeros                 -- zero tensor of given shape
  | ones                  -- ones tensor
  | randn                 -- standard normal sample
  -- Elementwise unary
  | neg
  | abs
  | sqrt
  | exp
  | log
  | relu
  | sigmoid
  | tanh
  | softplus
  -- Elementwise binary
  | add
  | sub
  | mul
  | div
  | pow
  | maximum
  | minimum
  -- Reduction
  | sumAll                -- sum over all elements
  | sumAxis (axis : Int)  -- sum over one axis
  | meanAxis (axis : Int)
  | maxAxis  (axis : Int)
  | minAxis  (axis : Int)
  -- Shape manipulation
  | reshape (newShape : Shape)
  | flatten
  | transpose
  | permute (dims : List Nat)
  | unsqueeze (dim : Int)
  | squeeze (dim : Int)
  | cat (axis : Nat) (n : Nat)          -- concatenate n tensors along axis
  -- Linear algebra
  | matmul                -- 2-D matrix multiply
  | bmm                   -- batched matrix multiply
  | linear                -- y = x W^T + b  (fused)
  -- Convolution
  | conv2d (stride padding dilation : Nat)
  | maxpool2d (kernel stride : Nat)
  | avgpool2d (kernel stride : Nat)
  -- Normalisation
  | batchNorm (eps momentum : Float)
  | layerNorm (normalizedShape : Shape) (eps : Float)
  | groupNorm (numGroups : Nat) (eps : Float)
  -- Attention
  | scaledDotProduct      -- softmax(Q K^T / √d_k) V
  -- Activation (additional)
  | gelu
  | silu
  | leakyRelu (negSlope : Float)
  -- Dropout / regularisation
  | dropout (p : Float)
  -- Loss
  | mseLoss
  | crossEntropyLoss
  | binaryCrossEntropy
  -- Control flow / recurrence
  | loop (body : List Nat)  -- index list of body nodes
  | select                   -- conditional select
  deriving Repr

-- ---------------------------------------------------------------------------
-- SSA values
-- ---------------------------------------------------------------------------

/-- A `ValueId` names an SSA value (the output of a node). -/
abbrev ValueId := Nat

/-- A `Value` carries type information about an SSA-defined tensor. -/
structure Value where
  id      : ValueId
  shape   : Shape
  dtype   : DType
  deriving Repr

-- ---------------------------------------------------------------------------
-- Graph nodes
-- ---------------------------------------------------------------------------

/-- A `Node` in the computation DAG.
    - `inputs` are `ValueId`s of predecessor nodes.
    - `outputs` are `Value`s produced by this node.
    - `attrs` is an opaque bag of string attributes (for ops with parameters). -/
structure Node where
  id      : Nat
  op      : OpCode
  inputs  : List ValueId
  outputs : List Value
  attrs   : List (String × String)   -- key-value string attributes
  deriving Repr

-- ---------------------------------------------------------------------------
-- Graph
-- ---------------------------------------------------------------------------

/-- A `Graph` is a topologically-ordered list of `Node`s.
    `inputs` are the graph's free variables; `outputs` are the return values. -/
structure Graph where
  name    : String
  nodes   : List Node
  inputs  : List Value    -- placeholder/parameter values
  outputs : List ValueId  -- final result value ids
  deriving Repr

namespace Graph

def empty (name : String) : Graph :=
  { name, nodes := [], inputs := [], outputs := [] }

/-- Look up a node by its id. -/
def findNode (g : Graph) (id : Nat) : Option Node :=
  g.nodes.find? (·.id = id)

/-- Add a node to the graph (appended at the end). -/
def addNode (g : Graph) (n : Node) : Graph :=
  { g with nodes := g.nodes ++ [n] }

/-- Topological order check: each node's inputs must be defined before it. -/
def wellFormed (g : Graph) : Bool :=
  let defined := g.inputs.map (·.id) |>.toArray
  let defined := Id.run do
    let mut d := defined
    let mut ok := true
    for n in g.nodes do
      for inp in n.inputs do
        if !d.contains inp then
          ok := false
      for out in n.outputs do
        d := d.push out.id
    return (ok, d) |>.1
  defined

/-- Number of nodes. -/
def size (g : Graph) : Nat := g.nodes.length

end Graph

-- ---------------------------------------------------------------------------
-- Graph Builder (imperative-style)
-- ---------------------------------------------------------------------------

/-- `Builder` provides an imperative interface for constructing a `Graph`,
    maintaining a counter for fresh `ValueId`s. -/
structure Builder where
  graph      : Graph
  nextId     : Nat
  deriving Repr

namespace Builder

def init (name : String) : Builder :=
  { graph := Graph.empty name, nextId := 0 }

/-- Allocate a fresh value id. -/
def freshId (b : Builder) : ValueId × Builder :=
  (b.nextId, { b with nextId := b.nextId + 1 })

/-- Emit a node and return the output value ids. -/
def emit (b : Builder) (op : OpCode) (inputs : List ValueId)
    (outShapes : List (Shape × DType)) (attrs : List (String × String) := [])
    : List ValueId × Builder :=
  let (ids, b') := outShapes.foldl (fun (acc : List ValueId × Builder) _sd =>
    let (vs, bld) := acc
    let (fid, bld') := bld.freshId
    (vs ++ [fid], bld')) ([], b)
  let outputs := ids.zip outShapes |>.map (fun (id, s, dt) =>
    { id, shape := s, dtype := dt : Value })
  let nodeId := b'.nextId
  let n : Node := { id := nodeId, op, inputs, outputs, attrs }
  (ids, { b' with graph := b'.graph.addNode n, nextId := b'.nextId + 1 })

/-- Add a graph input (placeholder). -/
def addInput (b : Builder) (shape : Shape) (dtype : DType) : ValueId × Builder :=
  let (fid, b') := b.freshId
  let v : Value := { id := fid, shape, dtype }
  (fid, { b' with graph := { b'.graph with inputs := b'.graph.inputs ++ [v] } })

/-- Set graph outputs. -/
def setOutputs (b : Builder) (ids : List ValueId) : Builder :=
  { b with graph := { b.graph with outputs := ids } }

/-- Finalise and return the graph. -/
def build (b : Builder) : Graph := b.graph

end Builder

-- ---------------------------------------------------------------------------
-- Interpreter (reference semantics over Float)
-- ---------------------------------------------------------------------------

/-- A simple reference interpreter that evaluates a `Graph` over `Float`.
    This is the "eager" execution mode — for compiled execution, the graph
    would be lowered to native code. -/
structure InterpEnv where
  values : List (ValueId × Tensor Float)

namespace InterpEnv

def empty : InterpEnv := { values := [] }

def insert (env : InterpEnv) (id : ValueId) (t : Tensor Float) : InterpEnv :=
  { values := (id, t) :: env.values.filter (·.1 ≠ id) }

def lookup (env : InterpEnv) (id : ValueId) : Option (Tensor Float) :=
  env.values.find? (·.1 = id) |>.map (·.2)

def lookupMany (env : InterpEnv) (ids : List ValueId) : List (Tensor Float) :=
  ids.filterMap (lookup env)

end InterpEnv

/-- Apply a single `OpCode` to a list of input tensors. -/
def applyOp (op : OpCode) (inputs : List (Tensor Float)) : List (Tensor Float) :=
  match op, inputs with
  -- Unary elementwise
  | .neg,     [t] => [t.map (· * -1.0)]
  | .abs,     [t] => [t.map Float.abs]
  | .sqrt,    [t] => [t.map Float.sqrt]
  | .exp,     [t] => [t.map Float.exp]
  | .log,     [t] => [t.map Float.log]
  | .relu,    [t] => [t.map (fun x => if x > 0.0 then x else 0.0)]
  | .sigmoid, [t] => [t.map (fun x => 1.0 / (1.0 + Float.exp (-x)))]
  | .tanh,    [t] => [t.map Float.tanh]
  | .softplus,[t] => [t.map (fun x => Float.log (1.0 + Float.exp x))]
  | .gelu,    [t] => [t.map (fun x =>
      0.5 * x * (1.0 + Float.tanh (0.7978845608 * (x + 0.044715 * x * x * x))))]
  | .silu,    [t] => [t.map (fun x => x / (1.0 + Float.exp (-x)))]
  -- Binary elementwise
  | .add,     [a, b] => [a + b]
  | .sub,     [a, b] => [a - b]
  | .mul,     [a, b] => [a * b]
  | .div,     [a, b] => [a.zipWith (· / ·) b]
  | .maximum, [a, b] => [a.zipWith (fun x y => if x > y then x else y) b]
  | .minimum, [a, b] => [a.zipWith (fun x y => if x < y then x else y) b]
  -- Reductions
  | .sumAll,  [t] => [{ shape := [1], data := #[t.sum] }]
  -- Shape
  | .flatten, [t] => [t.flatten]
  | .transpose, [t] => [t.transpose]
  -- Linear algebra
  | .matmul,  [a, b] => [Tensor.matmul a b]
  | .bmm,     [a, b] => [Tensor.bmm a b]
  -- Normalization
  | .layerNorm _ eps, [t] => [Tensor.layerNorm t eps]
  -- Activation cat
  | .cat _ _, ts => [ts.foldl Tensor.cat (Tensor.zeros [])]
  | _, _ => inputs  -- unhandled: pass through

/-- Evaluate a `Graph` under initial bindings `env`. -/
def evalGraph (g : Graph) (env : InterpEnv) : InterpEnv :=
  g.nodes.foldl (fun e n =>
    let args := e.lookupMany n.inputs
    let results := applyOp n.op args
    results.zip n.outputs |>.foldl (fun e' (t, v) => e'.insert v.id t) e
  ) env

end TorchLib.IR
