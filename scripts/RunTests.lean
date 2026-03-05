/-!
# RunTests

Test runner for TorchLib — builds the test library and reports results.

Usage:
  lake exe runTests
-/

def main (_ : List String) : IO UInt32 := do
  IO.println "=== TorchLib Test Runner ==="
  IO.println ""

  -- Build the main library first
  IO.println "Building TorchLib..."
  let libResult ← IO.Process.spawn {
    cmd  := "lake"
    args := #["build", "TorchLib"]
  } >>= (·.wait)

  if libResult != 0 then
    IO.eprintln "\n✗ TorchLib build failed"
    return libResult

  IO.println "✓ TorchLib build succeeded"
  IO.println ""

  -- Build the test library
  IO.println "Building TorchLibTests..."
  let testResult ← IO.Process.spawn {
    cmd  := "lake"
    args := #["build", "TorchLibTests"]
  } >>= (·.wait)

  if testResult != 0 then
    IO.eprintln "\n✗ TorchLibTests build failed"
    return testResult

  IO.println "✓ TorchLibTests build succeeded"
  IO.println ""
  IO.println "All tests passed ✓"
  return 0
