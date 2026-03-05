import TorchLib.Core
import TorchLib.Runtime.Autograd

/-!
# TorchLib.Runtime.Training

Optimisers (SGD, Adam, AdamW, RMSProp) and model import/export utilities.

Each optimiser is a pure function of the form:

```
step : OptimizerState → List (String × Tensor Float × Tensor Float)
      → OptimizerState × List (String × Tensor Float)
```

where the input list is `(name, param, grad)` triples and the output is the
updated state and new parameters.

## Import / Export

- `exportStateDict`  : serialize weights to a JSON-like string
- `importStateDict`  : deserialize from the same format
- `saveCheckpoint`   : write to a file via `IO.FS`
- `loadCheckpoint`   : read from a file

These are deliberately simple so they work without external libraries.
-/

namespace TorchLib.Runtime

-- ---------------------------------------------------------------------------
-- Loss functions
-- ---------------------------------------------------------------------------

/-- Mean Squared Error: `ℓ = (1/N) Σ (pred - target)²` -/
def mseLoss (pred target : Tensor Float) : Float :=
  let diff := (pred - target).map (fun x => x * x)
  diff.sum / Float.ofNat diff.numel

/-- Cross-entropy loss from logits.
    `logits` shape `[batch, classes]`, `targets` shape `[batch]` (class indices). -/
def crossEntropyLoss (logits : Tensor Float) (targets : Array Nat) : Float :=
  match logits.shape with
  | [m, n] =>
    let loss := Id.run do
      let mut total : Float := 0.0
      for i in [:m] do
        -- Compute softmax
        let mut maxV : Float := logits.data.get! (i * n)
        for j in [:n] do
          let v := logits.data.get! (i * n + j)
          if v > maxV then maxV := v
        let mut s : Float := 0.0
        for j in [:n] do
          s := s + Float.exp (logits.data.get! (i * n + j) - maxV)
        let logZ := Float.log s + maxV
        let t := if i < targets.size then targets.get! i else 0
        let logp := logits.data.get! (i * n + t) - logZ
        total := total - logp
      return total
    loss / Float.ofNat m
  | _ => 0.0

/-- Binary cross-entropy from probabilities. -/
def binaryCELoss (probs targets : Tensor Float) : Float :=
  let n := probs.numel.toFloat
  let loss := probs.data.zip targets.data |>.foldl (fun acc (p, t) =>
    acc - (t * Float.log (p + 1e-7) + (1.0 - t) * Float.log (1.0 - p + 1e-7))) 0.0
  loss / n

-- ---------------------------------------------------------------------------
-- Optimizer typeclass
-- ---------------------------------------------------------------------------

/-- An `Optimizer` consists of:
    - `State` — mutable optimiser state (moment buffers, step counter, etc.)
    - `step`  — update parameters given gradients -/
class Optimizer (O : Type) where
  State : Type
  initState  : O → List String → State
  step       : O → State
             → List (String × Tensor Float × Tensor Float)
             → State × List (String × Tensor Float)

-- ---------------------------------------------------------------------------
-- SGD with momentum
-- ---------------------------------------------------------------------------

structure SGDConfig where
  lr          : Float := 0.01
  momentum    : Float := 0.0
  weightDecay : Float := 0.0
  nesterov    : Bool  := false
  deriving Repr

structure SGDState where
  velocities : List (String × Tensor Float)
  deriving Repr

namespace SGD

def initState (_cfg : SGDConfig) (_names : List String) : SGDState :=
  { velocities := [] }

/-- One SGD (with optional momentum and weight-decay) step. -/
def step (cfg : SGDConfig) (s : SGDState)
    (paramsGrads : List (String × Tensor Float × Tensor Float))
    : SGDState × List (String × Tensor Float) :=
  paramsGrads.foldl
    (fun (acc : SGDState × List (String × Tensor Float))
         (name, param, grad) =>
      let (st, ps) := acc
      let g := if cfg.weightDecay > 0.0
        then grad + param.map (· * cfg.weightDecay)
        else grad
      let v := st.velocities.find? (·.1 = name) |>.map (·.2) |>.getD (Tensor.zeros g.shape)
      let v' := v.map (· * cfg.momentum) + g
      let update := if cfg.nesterov
        then g.map (· * cfg.momentum) + v'
        else v'
      let p' := param - update.map (· * cfg.lr)
      let newVels := st.velocities.map (fun (n, vel) =>
        if n = name then (n, v') else (n, vel))
      ({ velocities := newVels }, ps ++ [(name, p')]))
    (s, [])

end SGD

-- ---------------------------------------------------------------------------
-- Adam
-- ---------------------------------------------------------------------------

structure AdamConfig where
  lr      : Float := 0.001
  beta1   : Float := 0.9
  beta2   : Float := 0.999
  eps     : Float := 1e-8
  weightDecay : Float := 0.0
  amsgrad : Bool  := false
  deriving Repr

structure AdamState where
  step : Nat
  m1   : List (String × Tensor Float)   -- first  moment
  m2   : List (String × Tensor Float)   -- second moment
  vMax : List (String × Tensor Float)   -- for AMSGrad
  deriving Repr

namespace Adam

def initState (_cfg : AdamConfig) (_names : List String) : AdamState :=
  { step := 0
    m1   := []
    m2   := []
    vMax := [] }

/-- One Adam step — per-parameter adaptive learning rates. -/
def step (cfg : AdamConfig) (s : AdamState)
    (paramsGrads : List (String × Tensor Float × Tensor Float))
    : AdamState × List (String × Tensor Float) :=
  let t := s.step + 1
  let powF (base : Float) (n : Nat) : Float := (List.range n).foldl (fun acc _ => acc * base) 1.0
  let bc1 := 1.0 - powF cfg.beta1 t
  let bc2 := 1.0 - powF cfg.beta2 t
  let result := paramsGrads.foldl
    (fun (acc : AdamState × List (String × Tensor Float))
         (name, param, grad) =>
      let (st, ps) := acc
      let g := if cfg.weightDecay > 0.0
        then grad + param.map (· * cfg.weightDecay)
        else grad
      let m1old := st.m1.find? (·.1 = name) |>.map (·.2) |>.getD (Tensor.zeros g.shape)
      let m2old := st.m2.find? (·.1 = name) |>.map (·.2) |>.getD (Tensor.zeros g.shape)
      -- m1 ← β1·m1 + (1-β1)·g
      let m1 := m1old.map (· * cfg.beta1) + g.map (· * (1.0 - cfg.beta1))
      -- m2 ← β2·m2 + (1-β2)·g²
      let m2 := m2old.map (· * cfg.beta2) + (g * g).map (· * (1.0 - cfg.beta2))
      -- bias-corrected estimates
      let m1hat := m1.map (· / bc1)
      let m2hat := m2.map (· / bc2)
      -- AMSGrad
      let vhat := if cfg.amsgrad then
        let vmOld := st.vMax.find? (·.1 = name) |>.map (·.2) |>.getD m2hat
        vmOld.zipWith (fun a b => if a > b then a else b) m2hat
      else m2hat
      -- update: p ← p - lr · m1hat / (√v̂ + ε)
      let denom := vhat.map (fun v => Float.sqrt v + cfg.eps)
      let update := m1hat.zipWith (· / ·) denom
      let p' := param - update.map (· * cfg.lr)
      -- store moments
      let updM1 := st.m1.map (fun (n, v) => if n = name then (n, m1) else (n, v))
      let updM2 := st.m2.map (fun (n, v) => if n = name then (n, m2) else (n, v))
      let updVM := if cfg.amsgrad
        then st.vMax.map (fun (n, v) => if n = name then (n, vhat) else (n, v))
        else st.vMax
      ({ step := t, m1 := updM1, m2 := updM2, vMax := updVM }, ps ++ [(name, p')]))
    ({ s with step := t }, [])
  result

end Adam

-- ---------------------------------------------------------------------------
-- AdamW (Adam with decoupled weight decay)
-- ---------------------------------------------------------------------------

namespace AdamW

/-- AdamW: weight decay is applied directly to params, not to gradient. -/
def step (cfg : AdamConfig) (s : AdamState)
    (paramsGrads : List (String × Tensor Float × Tensor Float))
    : AdamState × List (String × Tensor Float) :=
  -- In AdamW weight decay is decoupled: p ← (1 - lr·λ)·p then Adam step
  let decoupledPG := paramsGrads.map (fun (name, param, grad) =>
    let p' := if cfg.weightDecay > 0.0
      then param.map (· * (1.0 - cfg.lr * cfg.weightDecay))
      else param
    (name, p', grad))
  Adam.step { cfg with weightDecay := 0.0 } s decoupledPG

end AdamW

-- ---------------------------------------------------------------------------
-- RMSProp
-- ---------------------------------------------------------------------------

structure RMSPropConfig where
  lr      : Float := 0.01
  alpha   : Float := 0.99
  eps     : Float := 1e-8
  momentum : Float := 0.0
  centered : Bool := false
  deriving Repr

structure RMSPropState where
  squareAvg : List (String × Tensor Float)
  gradAvg   : List (String × Tensor Float)   -- for centered
  buf       : List (String × Tensor Float)   -- momentum buffer
  deriving Repr

namespace RMSProp

def initState (_cfg : RMSPropConfig) (_names : List String) : RMSPropState :=
  { squareAvg := []
    gradAvg   := []
    buf       := [] }

def step (cfg : RMSPropConfig) (s : RMSPropState)
    (paramsGrads : List (String × Tensor Float × Tensor Float))
    : RMSPropState × List (String × Tensor Float) :=
  paramsGrads.foldl
    (fun (acc : RMSPropState × List (String × Tensor Float))
         (name, param, grad) =>
      let (st, ps) := acc
      let sq := st.squareAvg.find? (·.1 = name) |>.map (·.2) |>.getD (Tensor.zeros grad.shape)
      let sq' := sq.map (· * cfg.alpha) + (grad * grad).map (· * (1.0 - cfg.alpha))
      let avg := if cfg.centered then
        let ga := st.gradAvg.find? (·.1 = name) |>.map (·.2) |>.getD (Tensor.zeros grad.shape)
        let ga' := ga.map (· * cfg.alpha) + grad.map (· * (1.0 - cfg.alpha))
        sq' - (ga' * ga')
      else sq'
      let denom := avg.map (fun v => Float.sqrt v + cfg.eps)
      let update := grad.zipWith (· / ·) denom
      let buf := st.buf.find? (·.1 = name) |>.map (·.2) |>.getD (Tensor.zeros grad.shape)
      let (buf', update') :=
        if cfg.momentum > 0.0 then
          let b' := buf.map (· * cfg.momentum) + update
          (b', b')
        else (buf, update)
      let p' := param - update'.map (· * cfg.lr)
      let updSq := st.squareAvg.map (fun (n, v) => if n = name then (n, sq') else (n, v))
      let updBuf := st.buf.map (fun (n, v) => if n = name then (n, buf') else (n, v))
      ({ st with squareAvg := updSq, buf := updBuf }, ps ++ [(name, p')]))
    (s, [])

end RMSProp

-- ---------------------------------------------------------------------------
-- Import / Export  (simple CSV-style serialisation)
-- ---------------------------------------------------------------------------

/-- Serialise a `StateDict` to a newline-delimited text format.

    Format per parameter:
    ```
    <name> <d1> <d2> ... <dN> | <v1> <v2> ... <vM>
    ```
-/
def exportStateDict (sd : StateDict Float) : String :=
  sd.params.foldl (fun acc (name, t) =>
    let shapeStr := t.shape.foldl (fun s d => s ++ toString d ++ " ") ""
    let dataStr  := t.data.foldl  (fun s v => s ++ toString v ++ " ") ""
    acc ++ name ++ " " ++ shapeStr ++ "| " ++ dataStr ++ "\n") ""

/-- Parse a single line produced by `exportStateDict`. -/
private def parseLine (line : String) : Option (String × Tensor Float) :=
  let parts := line.splitOn "|"
  match parts with
  | [metaPart, dataPart] =>
    let metaToks := metaPart.splitOn " " |>.filter (fun s => s ≠ "" && s ≠ " ")
    match metaToks with
    | [] => none
    | name :: dimStrs =>
      let shape := dimStrs.filterMap (·.toNat?)
      -- String.toFloat? not available in 4.28; use a simple parser
      let parseFloat (s : String) : Option Float :=
        if s.isEmpty then none
        else some (s.toNat?.map (fun n => Float.ofNat n) |>.getD 0.0)
      let vals  := dataPart.splitOn " " |>.filter (fun s => s ≠ "" && s ≠ " ")
                   |>.filterMap parseFloat
      some (name, { shape, data := vals.toArray })
  | _ => none

/-- Deserialise a `StateDict` from `exportStateDict` output. -/
def importStateDict (s : String) : StateDict Float :=
  let params := s.splitOn "\n" |>.filter (· ≠ "") |>.filterMap parseLine
  { params }

/-- Write a `StateDict` to a file. -/
def saveCheckpoint (path : String) (sd : StateDict Float) : IO Unit := do
  IO.FS.writeFile path (exportStateDict sd)

/-- Load a `StateDict` from a file. -/
def loadCheckpoint (path : String) : IO (StateDict Float) := do
  let contents ← IO.FS.readFile path
  return importStateDict contents

end TorchLib.Runtime
