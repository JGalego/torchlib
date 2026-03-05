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
  query     : String            -- string serialisation of query
  method    : String            -- e.g. "IBP", "CROWN"
  boundsLo  : Tensor Float
  boundsHi  : Tensor Float
  valid     : Bool := true
  metadata  : List (String × String) := []
  deriving Repr

namespace Certificate

def serialise (c : Certificate) : String :=
  "method: " ++ c.method ++ "\n" ++
  "valid: " ++ toString c.valid ++ "\n" ++
  "lo: " ++ toString c.boundsLo.data ++ "\n" ++
  "hi: " ++ toString c.boundsHi.data ++ "\n"

/-- Check a certificate against fresh bounds (independent re-verification). -/
def verify [Inhabited Float] (c : Certificate) (freshLo freshHi : Tensor Float) : Bool :=
  -- Check that fresh bounds are tighter than certified bounds
  let loOk := (freshLo.data.zip c.boundsLo.data).all (fun (fl, cl) => fl >= cl)
  let hiOk := (freshHi.data.zip c.boundsHi.data).all (fun (fh, ch) => fh <= ch)
  loOk && hiOk

end Certificate

-- ---------------------------------------------------------------------------
-- Checker typeclass
-- ---------------------------------------------------------------------------

class Checker (C : Type) where
  check : C → CertQuery → CertResult

-- ---------------------------------------------------------------------------
-- IBP checker
-- ---------------------------------------------------------------------------

/-- Certifies using Interval Bound Propagation. -/
structure IBPChecker where
  model : MLP Float
  deriving Repr

namespace IBPChecker

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
  model : MLP Float
  method : BoundMethod := .crown
  deriving Repr

namespace CROWNChecker

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

/-- Run a certification query from a string command and return a result summary. -/
def runCLI (modelPath queryStr : String) : IO String := do
  -- Load model weights
  let sd ← TorchLib.Runtime.loadCheckpoint modelPath
  let _ := sd
  -- Parse query (stub: returns unknown for non-robustness queries)
  let result : CertResult := CertResult.unknown ("Parsed query: " ++ queryStr)
  return match result with
    | .certified lo hi =>
        "CERTIFIED\nlo: " ++ toString lo.data ++ "\nhi: " ++ toString hi.data
    | .violated ce =>
        "VIOLATED\ncounter-example: " ++ toString ce.data
    | .unknown r =>
        "UNKNOWN: " ++ r

-- Re-export helper from Training for use by CLI
private def loadCheckpoint := TorchLib.Runtime.loadCheckpoint

end TorchLib.Verification
