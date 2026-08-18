import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX
import Nightstream.SuperNeo.Concrete.Parameters

/-!
Independent compiler model for the Nightstream strict-PiDEC canonical-X rows.

Assurance tier: model-level for the row compiler and artifact-checked for the
compact profile geometry.

Owns: an independent Lean compiler for the binary public-X recomposition,
shared-sign, and child-digit rows, plus the selected profile binding.

Does not own: equality with the generated row list, the remaining strict-PiDEC
rows, final selective lowering, whole-recursive conformance, witness
satisfaction, or cryptographic soundness.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX
open Nightstream.SuperNeo.Concrete

private abbrev logicalCoordinates : Nat := Generated.Metadata.logicalCoordinates
private abbrev childCount : Nat := Generated.Metadata.childCount
private abbrev rowsPerCoordinate : Nat := childCount + 2

/-- Exact little-endian binary recomposition emitted by the Rust compiler. -/
def recompositionRow (coordinate : CoordinateColumns) : Row :=
  let powers := (List.range coordinate.children.length).map (fun exponent => 2 ^ exponent)
  ⟨(coordinate.parent, 1) ::
      (coordinate.children.zip powers).map
        (fun pair => (pair.1, goldilocksP - pair.2)),
    [(0, 1)], []⟩

/-- The two rows that constrain one shared sign to `{-1, 0, 1}`. -/
def signProductRow (coordinate : CoordinateColumns) : Row :=
  ⟨[(0, 1), (coordinate.sign, 1)],
    [(coordinate.sign, 1)], [(coordinate.product, 1)]⟩

def signZeroRow (coordinate : CoordinateColumns) : Row :=
  ⟨[(coordinate.product, 1)],
    [(0, goldilocksP - 1), (coordinate.sign, 1)], []⟩

/-- One binary child digit is zero or the shared sign. -/
def childDigitRow (coordinate : CoordinateColumns) (child : Nat) : Option Row := do
  let digit ← coordinate.children[child]?
  pure ⟨[(digit, 1)],
    [(digit, 1), (coordinate.sign, goldilocksP - 1)], []⟩

def expectedRow : RowOwner → Option Row
  | .recomposition activeIndex =>
      (coordinates[activeIndex]?).map recompositionRow
  | .signProduct activeIndex =>
      (coordinates[activeIndex]?).map signProductRow
  | .signZero activeIndex =>
      (coordinates[activeIndex]?).map signZeroRow
  | .childDigit activeIndex child => do
      let coordinate ← coordinates[activeIndex]?
      childDigitRow coordinate child
  | .radixFourLimb .. | .radixFourReconstruction .. => none

def expectedRelativeIndex : RowOwner → Nat
  | .recomposition activeIndex => activeIndex
  | .signProduct activeIndex =>
      logicalCoordinates + activeIndex * rowsPerCoordinate
  | .signZero activeIndex =>
      logicalCoordinates + activeIndex * rowsPerCoordinate + 1
  | .childDigit activeIndex child =>
      logicalCoordinates + activeIndex * rowsPerCoordinate + 2 + child
  | .radixFourLimb .. | .radixFourReconstruction .. => 0

def expectedPhysicalIndex (owner : RowOwner) : Nat :=
  match owner with
  | .recomposition activeIndex =>
      Generated.Metadata.recompositionRowStart + activeIndex
  | _ =>
      Generated.Metadata.canonicalityRowStart +
        (expectedRelativeIndex owner - logicalCoordinates)

/-- The compact generated metadata selects the exact Nightstream binary
decomposition profile. This theorem does not inspect the generated row list. -/
theorem generated_profile_matches_nightstream :
    Generated.Metadata.radix = productionGlobalParams.b ∧
    Generated.Metadata.childCount = productionGlobalParams.k ∧
    Generated.Metadata.logicalCoordinates = 270 ∧
    Generated.Metadata.canonicalColumnCount =
      Generated.Metadata.rowCount + 1 ∧
    Generated.Metadata.rowCount =
      Generated.Metadata.logicalCoordinates *
        (Generated.Metadata.childCount + 3) := by
  decide

end Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX
