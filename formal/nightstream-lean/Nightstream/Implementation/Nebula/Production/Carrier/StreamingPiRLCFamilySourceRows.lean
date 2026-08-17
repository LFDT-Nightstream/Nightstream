import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyCarryRows

/-!
Contract: complete source rows for one production PiRLC family phase.

Assurance tier: generated source-row soundness.

Owns one assignment across PiRLC arithmetic, the exact family input
commitment and residual update, centered challenge decoding and carry, and the
family cursor increment. The carry decoder reuses the arithmetic layout, so
the same 918 symbols drive both the ring combination and the carried fields.

Does not own either Poseidon2 replay, normalized selective-CCS slots, generated
artifact rows, or Rust assignment conformance.

Emits constraints: 165,554 R1CS rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Concrete

private abbrev InputLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.Layout

private abbrev CarryLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.Layout

/-- Columns for all source rows of one family phase. Challenge decoding shares
the exact arithmetic layout. -/
structure Layout where
  algebra : ProductPiRlcRingCombinationRows.Layout
  input : InputLayout
  beforeChallenge : Source → Fin ringDegree → Nat
  afterChallenge : Source → Fin ringDegree → Nat
  beforeCursor : Nat
  afterCursor : Nat

/-- Challenge-and-cursor view of the complete source layout. -/
def carryLayout (layout : Layout) : CarryLayout where
  algebra := layout.algebra
  beforeChallenge := layout.beforeChallenge
  afterChallenge := layout.afterChallenge
  beforeCursor := layout.beforeCursor
  afterCursor := layout.afterCursor

/-- Exact row order: arithmetic, input commitment and residual, then challenge
and cursor glue. -/
def rows
    (setup : InputBindingSetup) (layout : Layout) (family : Family) : List Row :=
  ProductPiRlcRingCombinationRows.rows layout.algebra ++
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.rows
      setup layout.input family ++
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows
      (carryLayout layout))

theorem rows_length
    (setup : InputBindingSetup) (layout : Layout) (family : Family) :
    (rows setup layout family).length = 165554 := by
  simp only [rows, List.length_append,
    ProductPiRlcRingCombinationRows.rows_length,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.rows_length,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows_length]

private theorem algebra_satisfies
    {setup : InputBindingSetup} {layout : Layout} {family : Family}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Satisfies (ProductPiRlcRingCombinationRows.rows layout.algebra) assignment := by
  intro row member
  exact satisfies row (List.mem_append_left _ member)

private theorem input_satisfies
    {setup : InputBindingSetup} {layout : Layout} {family : Family}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.rows
        setup layout.input family) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_right _ (List.mem_append_left _ member))

private theorem carry_satisfies
    {setup : InputBindingSetup} {layout : Layout} {family : Family}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows
        (carryLayout layout)) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_right _ (List.mem_append_right _ member))

/-- Accepted source rows imply the exact concrete family relation. Only the
two Poseidon2 replay equalities remain outside this source-row block. -/
theorem rows_sound
    {setup : InputBindingSetup} {layout : Layout} {family : Family}
    {assignment : Nat → Nat} {before after : FamilyState}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : ∀ source lane,
      assignment (layout.algebra.challengeSymbol source lane) < 5)
    (inputsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.InputsPlaced
        layout.input.phase assignment
        (decodedInputs layout.algebra assignment canonical))
    (residualsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
        layout.input assignment before.inputResidual after.inputResidual)
    (statePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        (carryLayout layout) assignment before after)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (inputReplayExact :
      after.inputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (phaseFields (decodedInputs layout.algebra assignment canonical))
          before.inputReplay)
    (outputReplayExact :
      after.outputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (ringFields (outputRing layout.algebra assignment canonical))
          before.outputReplay)
    (satisfies : Satisfies (rows setup layout family) assignment) :
    FamilyPhaseRelation setup before after family
      (decodedInputs layout.algebra assignment canonical)
      (outputRing layout.algebra assignment canonical) := by
  have beforeBound : before.familyCursor < 110 := by
    rw [cursorExact]
    exact ProductPiRlcAlgebraRows.familyOrdinal_lt family
  have carryExact :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.rows_sound
      canonical one range beforeBound statePlaced (carry_satisfies satisfies)
  exact local_rows_imply_concrete_phase_from_input_rows
    canonical one range (algebra_satisfies satisfies) setup before after family
    inputsPlaced residualsPlaced (input_satisfies satisfies)
    carryExact.decoded cursorExact inputReplayExact outputReplayExact
    carryExact.challenges carryExact.cursor

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows
