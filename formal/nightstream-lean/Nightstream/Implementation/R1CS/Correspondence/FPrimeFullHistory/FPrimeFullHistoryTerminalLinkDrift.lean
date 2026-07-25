import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalLinkArtifact
import Nightstream.Implementation.R1CS.Ownership.FPrime.FPrimeTerminalLinkArtifact

/-!
Contract: precise row-count obstruction between the checked-in full-history
terminal-link snapshot and the current plain-carrier owner.

Owns:
- the exact 257-versus-270 metadata mismatch;
- the induced inequality of the two physical row programs.

Does not own: regeneration, current full-history column placement, semantic
authority for either artifact, or permission to select an encoding.

Emits constraints: no.

The generated snapshot owns only the affine-one row and 256 logical bit links.
The current production helper additionally owns thirteen verifier-fixed zero
padding rows. This module does not repair or certify the stale snapshot; it
records the mismatch that a future selected full-history artifact must close.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift

/-- The checked-in full-history snapshot contains only the logical prefix. -/
theorem generatedSnapshot_rowCount_eq_logicalWidth :
    FPrimeFullHistoryTerminalLink.rowCount = 1 + 256 := by
  rfl

/-- The current isolated production owner includes plain-carrier padding. -/
theorem currentPlainOwner_rowCount_eq_logicalPlusPadding :
    FPrimeTerminalLink.rowCount = 1 + 256 + 13 := by
  rfl

/-- Kernel-checked obstruction to treating the generated snapshot as the
current production terminal-link owner. -/
theorem generatedSnapshot_ne_currentPlainOwner :
    FPrimeFullHistoryTerminalLink.rowCount ≠ FPrimeTerminalLink.rowCount := by
  decide

/-- The exact row-count deficit is the thirteen plain-carrier zero pins. -/
theorem generatedSnapshot_missingPlainPaddingRows :
    FPrimeTerminalLink.rowCount -
        FPrimeFullHistoryTerminalLink.rowCount = 13 := by
  decide

/-- The mismatch is present in the physical row lists, not only metadata. -/
theorem generatedSnapshotRows_ne_currentPlainOwnerRows :
    FPrimeFullHistoryTerminalLink.rows ≠ FPrimeTerminalLink.rows := by
  intro rowsEqual
  have lengthsEqual := congrArg List.length rowsEqual
  rw [FPrimeFullHistoryTerminalLink.rows_length,
    FPrimeTerminalLink.rows_length] at lengthsEqual
  change (257 : Nat) = 270 at lengthsEqual
  omega

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift
