import TorchLib.Runtime.Float32

/-!
# Example: IEEE-754 binary32 execution (`IEEE32Exec`)

**Problem:** the convenient runtime type is binary64 `Float`, but real
accelerators compute in binary32.  How much does single precision change the
result, and can we run the model in genuine binary32?

**Method:** `Float32` is Lean's runtime-backed binary32 type.  TorchLib provides
a `Scalar Float32` instance (so the whole layer/model stack runs in single
precision) and `applyOp32` / `evalGraph32` (the binary32 interpreter for the
shared IR).  `precisionGap` reports the max element-wise difference between the
two precisions.
-/

open TorchLib TorchLib.Runtime

-- explicit positive weights so the signal survives ReLU (default 0.1 weights collapse)
def model : MLP Float :=
  { layers := #[ { weight := { shape := [2, 2], data := #[0.3, 0.7, 0.6, 0.2] }
                   bias   := { shape := [2], data := #[0.1, 0.1] } },
                 { weight := { shape := [1, 2], data := #[0.9, 0.5] }
                   bias   := { shape := [1], data := #[0.05] } } ]
    dropout := { p := 0.0 }, outputSize := 1 }

def x : Tensor Float := { shape := [1, 2], data := #[0.1234567, 0.9876543] }

#eval do
  -- binary32 rounding is real: most decimals are not representable
  IO.println s!"0.1 as binary64 == binary32 round-trip?  {(0.1 : Float).toFloat32.toFloat == (0.1 : Float)}"
  IO.println ""
  let y64 := MLP.forward model x
  let y32 := mlpForward32 model x
  IO.println s!"binary64 forward : {y64.data}"
  IO.println s!"binary32 forward : {y32.data}"
  IO.println s!"max precision gap : {precisionGap model x}  (≈ {precisionGap model x * 1e9} ×10⁻⁹)"
  IO.println ""
  IO.println "The eager Float32 path and the IEEE32Exec IR interpreter share these"
  IO.println "binary32 semantics — the trusted numeric format, distinct from Float."
