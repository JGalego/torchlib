import TorchLib.Verification.CROWN

/-!
# TorchLibTests.CROWN

Tests for CROWN / α-CROWN: soundness, tightness vs IBP, and that α-CROWN is
never looser than plain CROWN.
-/

namespace TorchLibTests

open TorchLib TorchLib.Verification

-- 2 → 3 → 1 MLP with mixed-sign weights ⇒ unstable ReLUs
private def m : MLP Float :=
  { layers := #[ { weight := { shape := [3, 2], data := #[1.0, -1.0, 0.5, 0.5, -1.0, 1.0] }
                   bias   := { shape := [3], data := #[0.0, 0.1, -0.2] } },
                 { weight := { shape := [1, 3], data := #[1.0, -1.0, 0.5] }
                   bias   := { shape := [1], data := #[0.0] } } ]
    dropout := { p := 0.0 }, outputSize := 1 }

private def center : Tensor Float := { shape := [1, 2], data := #[0.3, -0.2] }
private def eps : Float := 0.1

private def width (lohi : Tensor Float × Tensor Float) : Float :=
  (lohi.2 - lohi.1).data.foldl (· + ·) 0.0

#eval do
  -- identity init is a real identity matrix
  let id3 := (LinearBound.identity 3).loW
  assert! id3.data.get! 0 == 1.0 && id3.data.get! 4 == 1.0 && id3.data.get! 1 == 0.0
  IO.println "✓ LinearBound.identity is the identity matrix"

#eval do
  -- soundness: the concrete forward value lies inside every bound
  let y := (MLP.forward m center).data.get! 0
  let inside (lohi : Tensor Float × Tensor Float) : Bool :=
    lohi.1.data.get! 0 ≤ y && y ≤ lohi.2.data.get! 0
  assert! inside (IBP.mlpBounds m center eps)
  assert! inside (CROWN.mlpBounds m center eps)
  assert! inside (AlphaCROWN.bounds m center eps)
  IO.println "✓ IBP / CROWN / α-CROWN all bound the true output (sound)"

#eval do
  -- Invariant: α-CROWN is never looser than CROWN (it starts from CROWN's
  -- adaptive slopes and keeps the tightest result).  Note CROWN ≤ IBP is NOT
  -- a general invariant, so we do not assert it.
  let wIBP := width (IBP.mlpBounds m center eps)
  let wCR  := width (CROWN.mlpBounds m center eps)
  let wA   := width (AlphaCROWN.bounds m center eps)
  assert! wA ≤ wCR + 1e-9
  IO.println s!"✓ α-CROWN ≤ CROWN in width (IBP={wIBP}, CROWN={wCR}, αCROWN={wA})"

-- 2 → 3 → 3 → 1 (two hidden ReLU layers): here optimising the slopes actually
-- helps, so α-CROWN should be *strictly* tighter than CROWN.
private def deep : MLP Float :=
  { layers := #[ { weight := { shape := [3, 2], data := #[1.0, -1.0, 0.7, 0.8, -0.9, 0.5] }
                   bias   := { shape := [3], data := #[0.1, -0.1, 0.2] } },
                 { weight := { shape := [3, 3], data := #[0.6, -0.7, 0.5, -0.8, 0.9, -0.4, 0.3, -0.6, 0.7] }
                   bias   := { shape := [3], data := #[0.0, 0.1, -0.1] } },
                 { weight := { shape := [1, 3], data := #[1.0, -1.0, 0.8] }
                   bias   := { shape := [1], data := #[0.0] } } ]
    dropout := { p := 0.0 }, outputSize := 1 }

#eval do
  let wCR := width (CROWN.mlpBounds deep center 0.15)
  let wA  := width (AlphaCROWN.bounds deep center 0.15)
  assert! wA < wCR - 1e-6    -- α-CROWN genuinely optimises (not a no-op)
  IO.println s!"✓ α-CROWN strictly tightens CROWN on a 2-hidden-layer net ({wCR} → {wA})"

#eval do
  -- crownIBP hybrid is sound and (per element) no looser than either parent,
  -- so its width is ≤ both — a genuine invariant, even when CROWN > IBP.
  let (l, h) := computeBounds .crownIBP deep center 0.15
  let y := (MLP.forward deep center).data.get! 0
  assert! l.data.get! 0 ≤ y && y ≤ h.data.get! 0
  let wHy := width (l, h)
  assert! wHy ≤ width (IBP.mlpBounds deep center 0.15) + 1e-9
  assert! wHy ≤ width (CROWN.mlpBounds deep center 0.15) + 1e-9
  IO.println "✓ CROWN-IBP hybrid is sound and ≤ both IBP and CROWN in width"

end TorchLibTests
