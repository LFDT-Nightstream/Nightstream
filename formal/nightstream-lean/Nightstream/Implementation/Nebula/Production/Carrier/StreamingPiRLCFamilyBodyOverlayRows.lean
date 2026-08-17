import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyCompleteRows

/-!
Contract: exact body-and-overlay split for one production PiRLC family phase.

Assurance tier: generated-source and Rust-replay correspondence.

Owns two family-independent parity bodies and 110 family-position overlays.
Each body contains the arithmetic, canonical openings, shape pins, residual,
carry, cursor, and Poseidon2 replay rows. Each overlay contains only the 108
family-dependent seeded Phi81 rows. Both row sets use one linked assignment.

Does not own normalized low-norm link rows, the 400-arm selector, recursive
lifecycle integration, or the terminal zero-residual check.

Emits constraints: 310,646 body rows for even families, 311,846 body rows for
odd families, and 108 overlay rows for each exact family.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows

open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact
open Nightstream.SuperNeo.Concrete

private abbrev layout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout

private abbrev parityFor :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.parityFor

private abbrev verifierRows :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.verifierRows

/-- Family-independent source rows stored in both parity bodies. The omitted
rows are exactly the 108 family-position seeded-map rows. -/
def sourceBodyRows : List Row :=
  ProductPiRlcRingCombinationRows.rows layout.algebra ++
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.sourceRows
      layout.input.phase ++
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.shapeRows
        layout.input.phase ++
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.residualLayout
            layout.input) ++
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows
            (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
              layout))))

theorem sourceBodyRows_length : sourceBodyRows.length = 165446 := by
  simp only [sourceBodyRows, List.length_append,
    ProductPiRlcRingCombinationRows.rows_length,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.sourceRows_length,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.shapeRows_length,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_length,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows_length]

def replayRowsFor (parity : CursorParity) : List Row :=
  (arm parity).poseidon2Calls.flatMap Poseidon2Call.Call.rows

theorem replayRowsFor_length :
    ∀ parity : CursorParity,
      (replayRowsFor parity).length =
        match parity with
        | .even => 145200
        | .odd => 146400 := by
  intro parity
  change
    ((arm parity).poseidon2Calls.flatMap Poseidon2Call.Call.rows).length =
      match parity with
      | .even => 145200
      | .odd => 146400
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.replayRows_length
      parity

/-- One of the two stored body matrices. -/
def bodyRowsForParity (parity : CursorParity) : List Row :=
  sourceBodyRows ++ replayRowsFor parity

theorem bodyRowsForParity_length :
    ∀ parity : CursorParity,
      (bodyRowsForParity parity).length =
        match parity with
        | .even => 310646
        | .odd => 311846 := by
  intro parity
  rw [bodyRowsForParity, List.length_append, sourceBodyRows_length,
    replayRowsFor_length]
  cases parity <;> norm_num

/-- The exact family-selected seeded map. This is the only row family that
depends on the ordinal in `family`. -/
def overlayRows (setup : InputBindingSetup) (family : Family) : List Row :=
  (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.coordinateBlock
    setup layout.input.phase family).rows

theorem overlayRows_length
    (setup : InputBindingSetup) (family : Family) :
    (overlayRows setup family).length = 108 := by
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.coordinateRows_length
      setup layout.input.phase family

private theorem algebra_satisfies
    {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (bodyRowsForParity (parityFor family)) assignment) :
    Satisfies (ProductPiRlcRingCombinationRows.rows layout.algebra) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_left _ (List.mem_append_left _ member))

private theorem inputSource_satisfies
    {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (bodyRowsForParity (parityFor family)) assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.sourceRows
        layout.input.phase) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_left _
      (List.mem_append_right _ (List.mem_append_left _ member)))

/-- The family-independent body fixes the exact canonical input words used
by every family-selected seeded overlay. -/
theorem sourceColumnsExact_of_bodyRows
    {family : Family} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bodySatisfied : Satisfies
      (bodyRowsForParity (parityFor family)) assignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.SourceColumnsExact
      layout.input.phase assignment
      (decodedInputs layout.algebra assignment canonical) := by
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.sourceColumnsExact_of_rows
      canonical one
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.inputsPlaced
        assignment canonical)
      (inputSource_satisfies bodySatisfied)

private theorem shape_satisfies
    {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (bodyRowsForParity (parityFor family)) assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.shapeRows
        layout.input.phase) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_left _
      (List.mem_append_right _
        (List.mem_append_right _ (List.mem_append_left _ member))))

private theorem residual_satisfies
    {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (bodyRowsForParity (parityFor family)) assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.residualLayout
          layout.input))
      assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_left _
      (List.mem_append_right _
        (List.mem_append_right _
          (List.mem_append_right _ (List.mem_append_left _ member)))))

private theorem carry_satisfies
    {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (bodyRowsForParity (parityFor family)) assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
          layout)) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_left _
      (List.mem_append_right _
        (List.mem_append_right _
          (List.mem_append_right _ (List.mem_append_right _ member)))))

private theorem replay_satisfies
    {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (bodyRowsForParity (parityFor family)) assignment) :
    Satisfies (replayRowsFor (parityFor family)) assignment := by
  intro row member
  exact satisfies row (List.mem_append_right _ member)

private theorem arm_satisfied
    {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies
      (bodyRowsForParity (parityFor family)) assignment) :
    (arm (parityFor family)).Satisfied assignment := by
  intro call callMember row rowMember
  apply replay_satisfies satisfies row
  exact List.mem_flatMap.mpr ⟨call, callMember, rowMember⟩

/-- The parity body implies the exact family relation when the 108 local
commitment outputs are fixed by an authoritative seeded computation. -/
theorem bodyRows_sound_of_output_exact
    {setup : InputBindingSetup} {family : Family} {assignment : Nat → Nat}
    {before after : FamilyState}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : ∀ source lane,
      assignment (layout.algebra.challengeSymbol source lane) < 5)
    (residualsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
        layout.input assignment before.inputResidual after.inputResidual)
    (carryStatePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
          layout) assignment before after)
    (replayStatesPlaced :
      FamilyStatesPlaced (parityFor family)
        assignment before after)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (bodySatisfied : Satisfies
      (bodyRowsForParity (parityFor family)) assignment)
    (outputExact : ∀ output : Fin verifierRows,
      ∀ coordinate : Fin ringDegree,
        assignment
            (layout.input.phase.outputColumn
              (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.residualOutputIndex
                output coordinate)) =
          (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
            setup family
            (decodedInputs layout.algebra assignment canonical)
            (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.residualOutputIndex
              output coordinate)).val) :
    FamilyPhaseRelation setup before after family
      (decodedInputs layout.algebra assignment canonical)
      (outputRing layout.algebra assignment canonical) := by
  have sourceExact := sourceColumnsExact_of_bodyRows canonical one bodySatisfied
  have shapeExact :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.shape_exact
      canonical one (shape_satisfies bodySatisfied)
  have inputExact :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.Exact
        setup layout.input.phase family
      assignment (decodedInputs layout.algebra assignment canonical) := by
    refine ⟨shapeExact.1, shapeExact.2, sourceExact, ?_⟩
    exact outputExact
  have residualColumns :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.ColumnsPlaced
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.residualLayout
        layout.input) assignment
      before.inputResidual
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
        setup family (decodedInputs layout.algebra assignment canonical))
      after.inputResidual := by
    refine ⟨residualsPlaced.1, ?_, residualsPlaced.2⟩
    intro output
    exact inputExact.output_at output
  have residualExact :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_imply_concreteResidualTransition
      one residualColumns (residual_satisfies bodySatisfied)
      setup family (decodedInputs layout.algebra assignment canonical) rfl
  have beforeBound : before.familyCursor < 110 := by
    rw [cursorExact]
    exact ProductPiRlcAlgebraRows.familyOrdinal_lt family
  have carryExact :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows_sound
      canonical one range beforeBound
      carryStatePlaced (carry_satisfies bodySatisfied)
  have replayExact := family_replays_exact
    (parityFor family) assignment canonical one before after
    (decodedInputs layout.algebra assignment canonical)
    (outputRing layout.algebra assignment canonical)
    replayStatesPlaced
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.replayValuesPlaced
        _ assignment canonical)
    (arm_satisfied bodySatisfied)
  have transition : FamilyTransition setup before after family
      (decodedInputs layout.algebra assignment canonical)
      (outputRing layout.algebra assignment canonical) := {
    inputReplay := replayExact.1
    inputResidual := residualExact
    outputReplay := replayExact.2
    challenges := carryExact.challenges
    cursor := carryExact.cursor
  }
  exact local_rows_imply_concrete_phase canonical one range
    (algebra_satisfies bodySatisfied) setup before after family
    carryExact.decoded cursorExact transition

/-- Satisfaction of the parity body and the exact family overlay implies the
same semantic `FamilyPhaseRelation` as the former monolithic row list. -/
theorem rows_sound
    {setup : InputBindingSetup} {family : Family} {assignment : Nat → Nat}
    {before after : FamilyState}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : ∀ source lane,
      assignment (layout.algebra.challengeSymbol source lane) < 5)
    (residualsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
        layout.input assignment before.inputResidual after.inputResidual)
    (carryStatePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
          layout) assignment before after)
    (replayStatesPlaced :
      FamilyStatesPlaced (parityFor family)
        assignment before after)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (bodySatisfied : Satisfies
      (bodyRowsForParity (parityFor family)) assignment)
    (overlaySatisfied : Satisfies (overlayRows setup family) assignment) :
    FamilyPhaseRelation setup before after family
      (decodedInputs layout.algebra assignment canonical)
      (outputRing layout.algebra assignment canonical) := by
  have sourceExact := sourceColumnsExact_of_bodyRows
    canonical one bodySatisfied
  apply bodyRows_sound_of_output_exact canonical one range residualsPlaced
    carryStatePlaced replayStatesPlaced cursorExact bodySatisfied
  intro output coordinate
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.compact_output_exact_of_rows
      canonical one sourceExact overlaySatisfied output coordinate

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows
