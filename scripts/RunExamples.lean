/-!
# RunExamples

Example runner for TorchLib — builds the library and runs all examples.

Usage:
  lake exe runExamples
-/

def examples : Array String :=
  #[ "Tensors"
   , "LinearLayer"
   , "MLP"
   , "Autograd"
   , "Training"
   , "Verification"
   ]

def main (_ : List String) : IO UInt32 := do
  IO.println "=== TorchLib Example Runner ==="
  IO.println ""

  -- Build the main library first
  IO.println "Building TorchLib..."
  let libResult ← IO.Process.spawn {
    cmd  := "lake"
    args := #["build", "TorchLib"]
  } >>= (·.wait)

  if libResult != 0 then
    IO.eprintln "✗ TorchLib build failed\n"
    return libResult

  IO.println "✓ TorchLib build succeeded\n"

  -- Run each example
  let mut failures : Array String := #[]

  for ex in examples do
    let path := s!"examples/{ex}.lean"
    IO.print s!"\nRunning {path}...\n\n"

    let result ← IO.Process.spawn {
      cmd  := "lake"
      args := #["env", "lean", path]
    } >>= (·.wait)

    if result != 0 then
      IO.println "✗"
      failures := failures.push ex

  IO.println ""

  if failures.isEmpty then
    IO.println s!"All {examples.size} examples passed ✓"
    return 0
  else
    IO.eprintln s!"{failures.size}/{examples.size} example(s) failed: {failures}"
    return 1
