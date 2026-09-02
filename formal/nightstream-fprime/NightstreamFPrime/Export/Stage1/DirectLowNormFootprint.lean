import NightstreamFPrime.Export.Stage1.DirectPiRLCProductFootprint
import NightstreamFPrime.Export.Stage1.PiRLCFirst54RetainedBlocks
import NightstreamFPrime.Export.Stage1.PiRLCProductSourceBlocks
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock

/-!
Owns the current combined low-norm coordinate ledger for direct Poseidon2
and direct PiRLC product plans. Later compiler families must extend this
ledger before the complete Stage 1 fit can close.
-/

namespace NightstreamFPrime.Export.Stage1.DirectLowNormFootprint

def poseidonAndProductCoordinates : Nat :=
  PoseidonRetainedBlock.retainedCoordinateCount +
    DirectPiRLCProductFootprint.retainedCoordinateCount

@[simp] theorem poseidonAndProductCoordinates_eq :
    poseidonAndProductCoordinates = 185240460 := by
  unfold poseidonAndProductCoordinates
  rw [PoseidonRetainedBlock.retainedCoordinateCount_eq,
    DirectPiRLCProductFootprint.retainedCoordinateCount_eq]

theorem poseidonAndProductCoordinates_le_cube :
    poseidonAndProductCoordinates ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [poseidonAndProductCoordinates_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def throughFirst54Coordinates
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  poseidonAndProductCoordinates +
    PiRLCFirst54RetainedBlocks.retainedCoordinateCount program

@[simp] theorem throughFirst54Coordinates_eq
    (program : Lifecycle.Stage1.Application.Program) :
    throughFirst54Coordinates program = 187799436 := by
  simp [throughFirst54Coordinates, poseidonAndProductCoordinates_eq]

theorem throughFirst54Coordinates_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    throughFirst54Coordinates program ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [throughFirst54Coordinates_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def throughPiRLCProductSourcesCoordinates
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  throughFirst54Coordinates program +
    PiRLCProductSourceBlocks.retainedCoordinateCount program

@[simp] theorem throughPiRLCProductSourcesCoordinates_eq
    (program : Lifecycle.Stage1.Application.Program) :
    throughPiRLCProductSourcesCoordinates program = 192090168 := by
  simp [throughPiRLCProductSourcesCoordinates]

theorem throughPiRLCProductSourcesCoordinates_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    throughPiRLCProductSourcesCoordinates program ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [throughPiRLCProductSourcesCoordinates_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

end NightstreamFPrime.Export.Stage1.DirectLowNormFootprint
