import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedResidualRows

/-!
Contract: same-assignment composition of the retained normalized PiRLC
algebra, input-residual, and challenge-carry rows.

Assurance tier: model-level.

Owns the exact equality of the two decoded challenge assignments and the
joint implication from retained algebra, residual, and carry acceptance to
one concrete family phase.

Does not own normalized Poseidon2 replay rows, family-overlay output
authority, state-column placement, selector authority, the stored Rust
matrices, the Rust witness encoder, recursive orchestration, or cryptographic
security assumptions.

Emits constraints: no. It composes existing normalized row meanings on one
final assignment.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows

open Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Concrete

namespace Normalized

private abbrev sourceLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout

private abbrev carryLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
    sourceLayout

abbrev Arm :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.Arm

abbrev finalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.finalColumns

theorem finalColumns_positive : 0 < finalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.finalColumns_positive

/-- Numeric source assignment used by the retained algebra rows. -/
def algebraAssignment (assignment : Fin finalColumns → F) : Nat → Nat :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.numericAssignment
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
      assignment)

/-- Numeric source assignment used by the retained carry rows. -/
def carryAssignment (assignment : Fin finalColumns → F) : Nat → Nat :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.numericAssignment
    assignment

/-- Both retained blocks decode every shared challenge-symbol column from the
same radix-seven slot in the final assignment. -/
theorem challengeAssignment_eq
    (assignment : Fin finalColumns → F)
    (source : Source) (lane : Fin ringDegree) :
    carryAssignment assignment
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
          source lane) =
      algebraAssignment assignment
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
          source lane) := by
  let column :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
      source lane
  have columnPositive : column ≠ 0 := by
    simp [column]
  have columnLtChallenge : column < 811 := by
    simp only [column,
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout_challengeSymbol]
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 15 at sourceLt
    change lane.val < 54 at laneLt
    omega
  have columnLtSource : column <
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumns := by
    unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumns
    omega
  have columnLtLocal : column <
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumns := by
    unfold Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumns
    omega
  unfold carryAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.numericAssignment
    Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.numericAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.decodedAssignment
  rw [NumericBridge.finiteColumnIndex_sourceColumn_of_lt
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumns_positive
    columnLtSource]
  unfold algebraAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.numericAssignment
  rw [dif_pos (by
    simpa [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumns]
      using columnLtLocal)]
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumnValue
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localColumnValue
  rw [dif_neg (by simpa using columnPositive),
    dif_pos (by simpa using columnLtChallenge),
    dif_neg (by simpa using columnPositive)]
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.localSlot
  rw [dif_pos (by simpa using columnLtChallenge)]
  rfl

/-- Canonical numeric view of the retained algebra assignment. -/
theorem algebraAssignment_canonical
    (assignment : Fin finalColumns → F) :
    ∀ column, algebraAssignment assignment column < goldilocksP :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.numericAssignment_canonical
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
      assignment)

/-- The normalized algebra assignment preserves the constant-one column. -/
theorem algebraAssignment_one
    (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1) :
    algebraAssignment assignment 0 = 1 := by
  have decodedOne :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
          assignment
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.one = 1 := by
    simpa [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.one]
      using
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment_zero
          assignment).trans constantOne
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.numericAssignment_one
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.decodedLocalAssignment
        assignment)
      decodedOne

/-- Input rings read by the retained algebra block. -/
def algebraInputs (assignment : Fin finalColumns → F) : Source → RingF :=
  decodedInputs
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
    (algebraAssignment assignment) (algebraAssignment_canonical assignment)

/-- Output ring read by the retained algebra block. -/
def algebraOutput (assignment : Fin finalColumns → F) : RingF :=
  ProductPiRlcRingCombinationSound.outputRing
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
    (algebraAssignment assignment) (algebraAssignment_canonical assignment)

/-- A five-symbol range proved through the carry decoder is the range needed
by the algebra decoder on the same final assignment. -/
theorem carryRange_implies_algebraRange
    (assignment : Fin finalColumns → F)
    (carryRange : ∀ source lane,
      carryAssignment assignment
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
            source lane) < 5) :
    ∀ source lane,
      algebraAssignment assignment
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
            source lane) < 5 := by
  intro source lane
  rw [← challengeAssignment_eq assignment source lane]
  exact carryRange source lane

/-- The two decoders construct the same typed challenge rings because they
read the same radix-seven slots. -/
theorem decodedChallenges_eq
    (assignment : Fin finalColumns → F)
    (carryRange : ∀ source lane,
      carryAssignment assignment
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
            source lane) < 5)
    (algebraRange : ∀ source lane,
      algebraAssignment assignment
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
            source lane) < 5) :
    decodedChallenges
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
        (algebraAssignment assignment) algebraRange =
      decodedChallenges
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
        (carryAssignment assignment) carryRange := by
  funext source lane
  change
    Phi81StrongSet.embedCoefficient
        ⟨algebraAssignment assignment
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
            source lane), algebraRange source lane⟩ =
      Phi81StrongSet.embedCoefficient
        ⟨carryAssignment assignment
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
            source lane), carryRange source lane⟩
  apply congrArg Phi81StrongSet.embedCoefficient
  apply Fin.ext
  exact (challengeAssignment_eq assignment source lane).symm

/-- Exact carry facts expressed in the algebra block's decoded assignment. -/
structure ExactCarry
    (assignment : Fin finalColumns → F)
    (before after : FamilyState) : Prop where
  range : ∀ source lane,
    algebraAssignment assignment
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
          source lane) < 5
  decoded :
    decodedChallenges
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
        (algebraAssignment assignment) range = before.challenges
  challenges : after.challenges = before.challenges
  cursor : after.familyCursor = before.familyCursor + 1

/-- Active carry rows give all carry facts in the algebra block's decoded
assignment. -/
theorem carryAccepted_implies_exact
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (before after : FamilyState)
    (beforeBound : before.familyCursor < 110)
    (statePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        carryLayout (carryAssignment assignment) before after)
    (strongSet :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.ChallengesInStrongSet
        before.challenges)
    (carryAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.ProductionAccepted
        arm assignment) :
    ExactCarry assignment before after := by
  have carryRange : ∀ source lane,
      carryAssignment assignment
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
            source lane) < 5 := by
    exact
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.productionAccepted_implies_range
        arm assignment constantOne before after statePlaced strongSet
          carryAccepted
  have carryExact :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.Exact
        carryLayout (carryAssignment assignment) carryRange before after := by
    exact
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.productionAccepted_implies_exact
        arm assignment constantOne carryRange before after beforeBound
          statePlaced carryAccepted
  have algebraRange := carryRange_implies_algebraRange assignment carryRange
  have carryDecoded :
      decodedChallenges
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
          (carryAssignment assignment) carryRange = before.challenges := by
    exact carryExact.decoded
  have algebraDecoded :
      decodedChallenges
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
          (algebraAssignment assignment) algebraRange = before.challenges :=
    (decodedChallenges_eq assignment carryRange algebraRange).trans carryDecoded
  exact {
    range := algebraRange
    decoded := algebraDecoded
    challenges := carryExact.challenges
    cursor := carryExact.cursor
  }

/-- Active algebra rows give the exact ring combination for any challenge
vector proved equal to their decoded symbols. -/
theorem algebraAccepted_implies_output
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (algebraAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.ProductionAccepted
        arm assignment)
    (range : ∀ source lane,
      algebraAssignment assignment
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.challengeSymbol
            source lane) < 5)
    (challenges : Source → RingF)
    (decoded :
      decodedChallenges
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
          (algebraAssignment assignment) range = challenges) :
    algebraOutput assignment =
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.combineOne
        challenges (algebraInputs assignment) := by
  have algebraSatisfied :
      R1CS.Satisfies
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.sourceRows
        (algebraAssignment assignment) := by
    exact
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.productionAccepted_implies_source_rows
        arm assignment algebraAccepted
  rw [← decoded]
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.local_rows_imply_combineOne
    (algebraAssignment_canonical assignment)
    (algebraAssignment_one assignment constantOne)
    range algebraSatisfied

/-- The two replay facts that the normalized Poseidon2 blocks must provide on
the same decoded inputs and output. -/
structure ReplayTransition
    (before after : FamilyState)
    (inputs : Source → RingF) (output : RingF) : Prop where
  inputReplay :
    after.inputReplay =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (phaseFields inputs) before.inputReplay
  outputReplay :
    after.outputReplay =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (ringFields output) before.outputReplay

/-- Joint retained algebra, residual, and carry acceptance imply one concrete
family phase. The residual update, challenge range, centered decoding,
challenge carry, and cursor increment are all derived from accepted rows on
the same final assignment. -/
theorem jointAccepted_implies_concrete_phase
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (algebraAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.ProductionAccepted
        arm assignment)
    (carryAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.ProductionAccepted
        arm assignment)
    (residualAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.ProductionAccepted
        arm assignment)
    (inputSetup : InputBindingSetup)
    (before after : FamilyState) (family : Family)
    (statePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        carryLayout (carryAssignment assignment) before after)
    (strongSet :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.ChallengesInStrongSet
        before.challenges)
    (residualStatePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.StateColumnsPlaced
        assignment before after)
    (phaseBindingPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.PhaseBindingPlaced
        inputSetup family (algebraInputs assignment) assignment)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (replay : ReplayTransition before after
      (algebraInputs assignment) (algebraOutput assignment)) :
    FamilyPhaseRelation inputSetup before after family
      (algebraInputs assignment) (algebraOutput assignment) := by
  have beforeBound : before.familyCursor < 110 := by
    rw [cursorExact]
    exact ProductPiRlcAlgebraRows.familyOrdinal_lt family
  have carryExact : ExactCarry assignment before after :=
    carryAccepted_implies_exact arm assignment constantOne before after
      beforeBound statePlaced strongSet carryAccepted
  have transition : FamilyTransition inputSetup before after family
      (algebraInputs assignment) (algebraOutput assignment) := {
    inputReplay := replay.inputReplay
    inputResidual :=
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.productionAccepted_implies_transition
        arm assignment constantOne inputSetup family (algebraInputs assignment)
        before after residualStatePlaced phaseBindingPlaced residualAccepted
    outputReplay := replay.outputReplay
    challenges := carryExact.challenges
    cursor := carryExact.cursor
  }
  have outputExact :
      algebraOutput assignment =
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.combineOne
          before.challenges (algebraInputs assignment) := by
    exact algebraAccepted_implies_output arm assignment constantOne
      algebraAccepted carryExact.range before.challenges carryExact.decoded
  exact And.intro cursorExact (And.intro outputExact transition)

end Normalized

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows
