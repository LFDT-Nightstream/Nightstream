import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourCenteredDomainRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.CenteredDomain
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CenteredDomainPacking

/-!
Contract: exact production-row binding for centered-domain packing.

Owns: fail-closed decoding and coefficient classification of one production
pair row and the production odd-tail row, equality with the independent Lean
row points, and the active-selector zero-residual equivalences.

Does not own: source-column meaning, every production centered row, selector
one-hot dispatch, row multiplicity, constraint necessity, or row removal.

Emits constraints: no.

Assurance tier: artifact-checked and Rust-conformant for the two exact final
matrix rows. Their zero-residual equivalences are also security-reduced through
the production Goldilocks nonresidue theorem. This is not a family-wide
Rust-conformance theorem.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourCenteredDomainRows
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.CenteredDomain
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components

def decodedPairRow : DecodedRow :=
  (decodeRow rawPairRow).get (by decide)

def pairSelectorColumn : Fin decodedPairRow.columns := ⟨2431, by decide⟩
def pairLeftColumn : Fin decodedPairRow.columns := ⟨366187, by decide⟩
def pairRightColumn : Fin decodedPairRow.columns := ⟨366188, by decide⟩

theorem generated_pair_shape :
    decodedPairRow.rows = 8102331 ∧
    decodedPairRow.columns = 12288726 ∧
    decodedPairRow.emittedRow.val = 45768 ∧
    decodedPairRow.runIndex = 3 ∧
    decodedPairRow.family = .armDomain ∧
    decodedPairRow.arm = some 1 ∧
    IsCenteredDomainAt decodedPairRow pairSelectorColumn pairLeftColumn
      (some pairRightColumn) := by
  decide

def validatedPairRow : ValidatedCenteredDomainRow decodedPairRow where
  selectorColumn := pairSelectorColumn
  leftColumn := pairLeftColumn
  rightColumn := some pairRightColumn
  shape := generated_pair_shape.2.2.2.2.2.2

theorem generated_pair_zero_iff
    (assignment : Fin decodedPairRow.columns → F)
    (selectorOne : assignment pairSelectorColumn = 1) :
    residual decodedPairRow assignment = 0 ↔
      centeredUnitResidual (assignment pairLeftColumn) = 0 ∧
      centeredUnitResidual (assignment pairRightColumn) = 0 := by
  change
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
      (rowPoint decodedPairRow assignment) = 0 ↔ _
  rw [rowPoint_eq_centeredPairPoint decodedPairRow validatedPairRow assignment]
  change
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
      (Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.centeredPairPoint
        (assignment pairSelectorColumn) (assignment pairLeftColumn)
        (assignment pairRightColumn)) = 0 ↔ _
  rw [selectorOne]
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking.production_centeredPair_zero_iff
      (assignment pairLeftColumn) (assignment pairRightColumn)

def decodedTailRow : DecodedRow :=
  (decodeRow rawTailRow).get (by decide)

def tailSelectorColumn : Fin decodedTailRow.columns := ⟨2431, by decide⟩
def tailLeftColumn : Fin decodedTailRow.columns := ⟨10672171, by decide⟩

theorem generated_tail_shape :
    decodedTailRow.rows = 8102331 ∧
    decodedTailRow.columns = 12288726 ∧
    decodedTailRow.emittedRow.val = 4982068 ∧
    decodedTailRow.runIndex = 3 ∧
    decodedTailRow.family = .armDomain ∧
    decodedTailRow.arm = some 1 ∧
    IsCenteredDomainAt decodedTailRow tailSelectorColumn tailLeftColumn none := by
  decide

def validatedTailRow : ValidatedCenteredDomainRow decodedTailRow where
  selectorColumn := tailSelectorColumn
  leftColumn := tailLeftColumn
  rightColumn := none
  shape := generated_tail_shape.2.2.2.2.2.2

theorem generated_tail_zero_iff
    (assignment : Fin decodedTailRow.columns → F)
    (selectorOne : assignment tailSelectorColumn = 1) :
    residual decodedTailRow assignment = 0 ↔
      centeredUnitResidual (assignment tailLeftColumn) = 0 := by
  change
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
      (rowPoint decodedTailRow assignment) = 0 ↔ _
  rw [rowPoint_eq_centeredPairPoint decodedTailRow validatedTailRow assignment]
  change
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
      (Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.centeredPairPoint
        (assignment tailSelectorColumn) (assignment tailLeftColumn) 0) = 0 ↔ _
  rw [selectorOne]
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking.production_centeredTail_zero_iff
      (assignment tailLeftColumn)

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact
