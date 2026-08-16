import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedOpeningRows
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryFreshCcsAuthority

/-!
Contract: same-assignment transfer of the verifier-owned SuperNeo `b = 4`
norm to every normalized PiRLC borrow coordinate.

Assurance tier: model-level.

Owns the exact typed identity map from the normalized body assignment to a
Phi81 assignment of the same width. Proves that the outer assignment norm,
including the norm extracted from one fresh CCS opening, supplies
`BorrowCoordinatesNormFour` for all 32,400 retained borrow coordinates.

Does not own the generated proof that the complete recursive or terminal
artifact has this exact carrier width, the construction of its assignment,
selector or constant placement, stored matrices, or CCS satisfaction.

Emits constraints: no. The verifier-owned outer opening remains the norm
authority.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOuterNorm

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

namespace Normalized

abbrev BodyFinalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.BodyFinalColumns

/-- The normalized body coordinate in an outer Phi81 assignment whose exact
carrier width is the normalized body width. -/
def bodyColumn {shape : Shape}
    (widthExact : BodyFinalColumns = shape.carrierWidth)
    (column : Fin BodyFinalColumns) : Fin shape.carrierWidth :=
  ⟨column.val, by
    rw [← widthExact]
    exact column.isLt⟩

@[simp] theorem bodyColumn_val {shape : Shape}
    (widthExact : BodyFinalColumns = shape.carrierWidth)
    (column : Fin BodyFinalColumns) :
    (bodyColumn widthExact column).val = column.val := by
  rfl

/-- Exact body view of the same typed assignment opened and norm-checked by
the outer SuperNeo relation. -/
def bodyAssignment {shape : Shape}
    (widthExact : BodyFinalColumns = shape.carrierWidth)
    (outerAssignment : Assignment shape) : Fin BodyFinalColumns → F :=
  fun column => outerAssignment (bodyColumn widthExact column)

@[simp] theorem bodyAssignment_apply {shape : Shape}
    (widthExact : BodyFinalColumns = shape.carrierWidth)
    (outerAssignment : Assignment shape)
    (column : Fin BodyFinalColumns) :
    bodyAssignment widthExact outerAssignment column =
      outerAssignment (bodyColumn widthExact column) := by
  rfl

/-- The strict radix-four norm on the complete typed assignment gives the
exact natural-number norm predicate consumed by every retained PiRLC
canonical opening. -/
theorem borrowCoordinatesNormFour_of_outerNorm
    {shape : Shape}
    (widthExact : BodyFinalColumns = shape.carrierWidth)
    (outerAssignment : Assignment shape)
    (outerNorm : assignmentNormBounded 4 outerAssignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.BorrowCoordinatesNormFour
      (bodyAssignment widthExact outerAssignment) := by
  intro source lane borrow
  apply
    Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.normBoundFour_iff_centeredResidue.mpr
  apply
    (Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.concrete_norm_four_iff_centeredResidue
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.bodyBorrowValue
        (bodyAssignment widthExact outerAssignment) source lane borrow)).mp
  exact outerNorm _

universe uCommitment

/-- One verifier-owned fresh CCS opening with `b = 4` supplies all normalized
PiRLC borrow-coordinate norm facts on the same typed assignment. -/
theorem borrowCoordinatesNormFour_of_freshCcsHolds
    {shape : Shape}
    (widthExact : BodyFinalColumns = shape.carrierWidth)
    {Commitment : Type uCommitment}
    (commit : Assignment shape → Commitment)
    (params : GlobalParams)
    (baseFour : params.b = 4)
    (statement : CCSStatement shape Commitment)
    (fresh : statement.stage = .fresh)
    (outerAssignment : Assignment shape)
    (holds : CCS.Holds (relationSemantics commit) params statement
      outerAssignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.BorrowCoordinatesNormFour
      (bodyAssignment widthExact outerAssignment) := by
  apply borrowCoordinatesNormFour_of_outerNorm widthExact outerAssignment
  exact
    Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority.norm_four_of_fresh_ccsHolds
      commit params baseFour statement fresh outerAssignment holds

/-- Supported-profile specialization. The caller cannot select a different
opening bound. -/
theorem radixFourCandidate_borrowCoordinatesNormFour
    {shape : Shape}
    (widthExact : BodyFinalColumns = shape.carrierWidth)
    {Commitment : Type uCommitment}
    (commit : Assignment shape → Commitment)
    (statement : CCSStatement shape Commitment)
    (fresh : statement.stage = .fresh)
    (outerAssignment : Assignment shape)
    (holds : CCS.Holds (relationSemantics commit)
      Radix4Candidate.globalParams statement outerAssignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized.BorrowCoordinatesNormFour
      (bodyAssignment widthExact outerAssignment) := by
  exact borrowCoordinatesNormFour_of_freshCcsHolds widthExact commit
    Radix4Candidate.globalParams rfl statement fresh outerAssignment holds

end Normalized

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOuterNorm
