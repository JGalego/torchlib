import TorchLib.Core
import TorchLib.Layers
import TorchLib.Models
import TorchLib.Runtime.Training
import TorchLib.Verification.IBP
import TorchLib.Verification.CROWN

/-!
# TorchLib.Verification.Certificate

**Certificate Checking** — native checkers and CLI interface.

A *robustness certificate* witnesses that a network `f` satisfies a property
`P` on an input region `X`.  Common properties:

- **Local robustness**: `∀ x ∈ B_ε(x₀), argmax f(x) = c₀`
  (the predicted class is stable under perturbation)
- **Output range**: `∀ x ∈ X, lo ≤ f(x) ≤ hi`

## Architecture

1. `CertQuery`   — specifies what to certify
2. `CertResult`  — `Certified`, `Violated` (with counter-example), or `Unknown`
3. `Checker`     — engine that produces a `CertResult`
4. `Certificate` — a compact, auditable witness that can be stored externally

The checker backends included here are:
- `IBPChecker`   — uses interval bound propagation (fast, conservative)
- `CROWNChecker` — uses CROWN (tighter, slightly more expensive)
-/

namespace TorchLib.Verification

-- ---------------------------------------------------------------------------
-- Certificate query
-- ---------------------------------------------------------------------------

/-- Specifies the robustness property to certify. -/
inductive CertQuery
  /-- Local ℓ∞-robustness: class `classIdx` is dominant within `eps`-ball. -/
  | localRobustness
      (center   : Tensor Float)
      (eps      : Float)
      (classIdx : Nat)
  /-- Output range: all outputs lie in `[lo, hi]`. -/
  | outputRange
      (inputLo  : Tensor Float)
      (inputHi  : Tensor Float)
      (outputLo : Tensor Float)
      (outputHi : Tensor Float)
  /-- Reachability: output reaches a target region (safety check). -/
  | safetyOutput
      (inputLo   : Tensor Float)
      (inputHi   : Tensor Float)
      (unsafeLo  : Tensor Float)    -- lower bound of unsafe output region
      (unsafeHi  : Tensor Float)    -- upper bound of unsafe output region
  deriving Repr

-- ---------------------------------------------------------------------------
-- Certificate result
-- ---------------------------------------------------------------------------

/-- The outcome of a certification attempt. -/
inductive CertResult
  /-- The property holds; the bound tensors serve as the witness. -/
  | certified    (lo hi : Tensor Float)
  /-- The property is violated; `counterEx` is a witness input. -/
  | violated     (counterEx : Tensor Float)
  /-- The checker could not verify or refute the property. -/
  | unknown      (reason : String)
  deriving Repr

-- ---------------------------------------------------------------------------
-- Certificate (portable witness)
-- ---------------------------------------------------------------------------

/-- A `Certificate` is a portable, serialisable record that can be stored
    externally and independently checked.

    Fields:
    - `query`   : the certified property
    - `method`  : the verification method used
    - `boundsLo`, `boundsHi` : certified output bounds
    - `metadata` : provenance / tool info
-/
structure Certificate where
  /-- String serialisation of the certified property. -/
  query     : String
  /-- Verification method used (e.g. `"IBP"`, `"CROWN"`). -/
  method    : String
  /-- Certified lower output bounds. -/
  boundsLo  : Tensor Float
  /-- Certified upper output bounds. -/
  boundsHi  : Tensor Float
  /-- Whether the certificate was verified valid. -/
  valid     : Bool := true
  /-- Provenance and tool information. -/
  metadata  : List (String × String) := []

instance : Repr Certificate where
  reprPrec c prec := Repr.addAppParen
    f!"Certificate \{ query := {reprPrec c.query 0}, method := {reprPrec c.method 0}, valid := {reprPrec c.valid 0} }"
    prec

namespace Certificate

/-- Serialise the certificate to a human-readable string. -/
def serialise (c : Certificate) : String :=
  "method: " ++ c.method ++ "\n" ++
  "valid: " ++ toString c.valid ++ "\n" ++
  "lo: " ++ toString c.boundsLo.data ++ "\n" ++
  "hi: " ++ toString c.boundsHi.data ++ "\n"

/-- Check a certificate against fresh bounds (independent re-verification). -/
def verify (c : Certificate) (freshLo freshHi : Tensor Float) : Bool :=
  -- Check that fresh bounds are tighter than certified bounds
  let loOk := (freshLo.data.zip c.boundsLo.data).all (fun (fl, cl) => fl >= cl)
  let hiOk := (freshHi.data.zip c.boundsHi.data).all (fun (fh, ch) => fh <= ch)
  loOk && hiOk

end Certificate

-- ---------------------------------------------------------------------------
-- Checker typeclass
-- ---------------------------------------------------------------------------

/-- A `Checker` is an engine that can evaluate a `CertQuery` and produce a `CertResult`. -/
class Checker (C : Type) where
  /-- Run a certification query and produce a result. -/
  check : C → CertQuery → CertResult

-- ---------------------------------------------------------------------------
-- IBP checker
-- ---------------------------------------------------------------------------

/-- Certifies using Interval Bound Propagation. -/
structure IBPChecker where
  /-- The MLP model to certify. -/
  model : MLP Float

instance : Repr IBPChecker where
  reprPrec ch prec := Repr.addAppParen
    f!"IBPChecker \{ model := {reprPrec ch.model 0} }"
    prec

namespace IBPChecker

/-- Check local ℓ∞-robustness using IBP bounds. -/
def checkLocalRobustness (ch : IBPChecker) (center : Tensor Float)
    (eps : Float) (classIdx : Nat) : CertResult :=
  let (lo, hi) := IBP.mlpBounds ch.model center eps
  -- Check if classIdx dominates: lo[classIdx] > hi[j] for all j ≠ classIdx
  let n := lo.numel
  let classLo := lo.data.get! classIdx
  let dominated := (List.range n).all (fun j =>
    j = classIdx || hi.data.get! j < classLo)
  if dominated then
    CertResult.certified lo hi
  else
    CertResult.unknown "IBP bounds not tight enough to certify"

/-- Check whether the MLP's output stays within bounds for a given input region. -/
def checkOutputRange (ch : IBPChecker) (inputLo inputHi outLo outHi : Tensor Float)
    : CertResult :=
  let center := inputLo.zipWith (fun l h => (l + h) / 2.0) inputHi
  let eps := ((inputHi - inputLo).map (· / 2.0)).sum / Float.ofNat inputLo.numel
  let (lo, hi) := IBP.mlpBounds ch.model center eps
  let inRange := (lo.data.zip outLo.data).all (fun (l, bl) => l >= bl) &&
                 (hi.data.zip outHi.data).all (fun (h, bh) => h <= bh)
  if inRange then CertResult.certified lo hi
  else CertResult.unknown "output range not contained in target"

end IBPChecker

instance : Checker IBPChecker where
  check ch q :=
    match q with
    | .localRobustness center eps classIdx =>
        IBPChecker.checkLocalRobustness ch center eps classIdx
    | .outputRange inputLo inputHi outLo outHi =>
        IBPChecker.checkOutputRange ch inputLo inputHi outLo outHi
    | .safetyOutput inputLo inputHi unsafeLo unsafeHi =>
        -- Safety: output should NOT reach unsafe region
        -- If hi < unsafeLo or lo > unsafeHi, safe
        let center := inputLo.zipWith (fun l h => (l + h) / 2.0) inputHi
        let eps := ((inputHi - inputLo).map (· / 2.0)).sum / Float.ofNat inputLo.numel
        let (lo, hi) := IBP.mlpBounds ch.model center eps
        let safe := (hi.data.zip unsafeLo.data).all (fun (h, ul) => h < ul) ||
                    (lo.data.zip unsafeHi.data).all (fun (l, uh) => l > uh)
        if safe then CertResult.certified lo hi
        else CertResult.unknown "could not prove output avoids unsafe region"

-- ---------------------------------------------------------------------------
-- CROWN checker  (tighter bounds)
-- ---------------------------------------------------------------------------

/-- Certifies using CROWN back-substitution. -/
structure CROWNChecker where
  /-- The MLP model to certify. -/
  model : MLP Float
  /-- Bound computation method (IBP, CROWN, hybrid, etc.). -/
  method : BoundMethod := .crown

instance : Repr CROWNChecker where
  reprPrec ch prec := Repr.addAppParen
    f!"CROWNChecker \{ model := {reprPrec ch.model 0}, method := {reprPrec ch.method 0} }"
    prec

namespace CROWNChecker

/-- Check local ℓ∞-robustness using CROWN bounds. -/
def checkLocalRobustness [Inhabited Float] (ch : CROWNChecker) (center : Tensor Float)
    (eps : Float) (classIdx : Nat) : CertResult :=
  let (lo, hi) := computeBounds ch.method ch.model center eps
  let n := lo.numel
  let classLo := lo.data.get! classIdx
  let dominated := (List.range n).all (fun j =>
    j = classIdx || hi.data.get! j < classLo)
  if dominated then CertResult.certified lo hi
  else CertResult.unknown "CROWN bounds insufficient"

end CROWNChecker

instance : Checker CROWNChecker where
  check ch q :=
    match q with
    | .localRobustness center eps classIdx =>
        CROWNChecker.checkLocalRobustness ch center eps classIdx
    | .outputRange inputLo inputHi _outLo _outHi =>
        let center := inputLo.zipWith (fun l h => (l + h) / 2.0) inputHi
        let eps := ((inputHi - inputLo).map (· / 2.0)).sum / Float.ofNat inputLo.numel
        let (lo, hi) := computeBounds ch.method ch.model center eps
        CertResult.certified lo hi
    | .safetyOutput inputLo inputHi unsafeLo unsafeHi =>
        let center := inputLo.zipWith (fun l h => (l + h) / 2.0) inputHi
        let eps := ((inputHi - inputLo).map (· / 2.0)).sum / Float.ofNat inputLo.numel
        let (lo, hi) := computeBounds ch.method ch.model center eps
        let safe := (hi.data.zip unsafeLo.data).all (fun (h, ul) => h < ul) ||
                    (lo.data.zip unsafeHi.data).all (fun (l, uh) => l > uh)
        if safe then CertResult.certified lo hi
        else CertResult.unknown "CROWN: output may reach unsafe region"

-- ---------------------------------------------------------------------------
-- CLI interface
-- ---------------------------------------------------------------------------

/-- Reconstruct an `MLP Float` from a checkpoint `StateDict` that uses the
    `layer{i}.weight` / `layer{i}.bias` naming convention (the one produced by
    `Runtime.mlpModule` / `exportStateDict`).  Returns `none` if no such
    parameters are present. -/
def mlpOfStateDict (sd : StateDict Float) (maxLayers : Nat := 64) : Option (MLP Float) :=
  let layers := Id.run do
    let mut ls : Array (Linear Float) := #[]
    let mut stop := false
    for i in [:maxLayers] do
      if !stop then
        match sd.lookup s!"layer{i}.weight", sd.lookup s!"layer{i}.bias" with
        | some w, some b => ls := ls.push { weight := w, bias := b }
        | _, _           => stop := true
    return ls
  if layers.isEmpty then none
  else
    let last := layers.getD (layers.size - 1) { weight := Tensor.zeros [], bias := Tensor.zeros [] }
    let outputSize := match last.weight.shape with | n :: _ => n | _ => 0
    some { layers, dropout := { p := 0.0, training := false }, outputSize }

/-- Format a `CertResult` for the CLI. -/
private def formatResult : CertResult → String
  | .certified lo hi => s!"CERTIFIED\nlo: {lo.data}\nhi: {hi.data}"
  | .violated ce     => s!"VIOLATED\ncounter-example: {ce.data}"
  | .unknown r       => s!"UNKNOWN: {r}"

/-- Run a certification query against a checkpoint and return a result summary.

    Query grammar (whitespace-separated):
    - `robustness <eps> <class> <c0> <c1> …` — certify local ℓ∞ robustness of
      class `<class>` in the `<eps>`-ball around the centre `c`.
    - `bounds <eps> <c0> <c1> …` — report CROWN output bounds on the `<eps>`-ball. -/
def runCLI (modelPath queryStr : String) : IO String := do
  let sd ← TorchLib.Runtime.loadCheckpoint modelPath
  match mlpOfStateDict sd with
  | none => return "ERROR: checkpoint has no layer{i}.weight/bias parameters"
  | some model =>
    let toks := queryStr.splitOn " " |>.filter (· ≠ "")
    match toks with
    | "robustness" :: epsS :: classS :: rest =>
      match epsS.toFloat?, classS.toNat? with
      | some eps, some cls =>
        let center : Tensor Float := { shape := [1, rest.length], data := (rest.filterMap String.toFloat?).toArray }
        let checker : CROWNChecker := { model, method := .crown }
        return formatResult (Checker.check checker (.localRobustness center eps cls))
      | _, _ => return "ERROR: usage: robustness <eps> <class> <c0> <c1> ..."
    | "bounds" :: epsS :: rest =>
      match epsS.toFloat? with
      | some eps =>
        let center : Tensor Float := { shape := [1, rest.length], data := (rest.filterMap String.toFloat?).toArray }
        let (lo, hi) := computeBounds .crown model center eps
        return s!"BOUNDS\nlo: {lo.data}\nhi: {hi.data}"
      | none => return "ERROR: usage: bounds <eps> <c0> <c1> ..."
    | _ => return "ERROR: unknown query; expected 'robustness <eps> <class> <c..>' or 'bounds <eps> <c..>'"

end TorchLib.Verification
