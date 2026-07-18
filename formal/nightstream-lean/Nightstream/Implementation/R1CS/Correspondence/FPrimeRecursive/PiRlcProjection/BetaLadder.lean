import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.BetaLadder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.IndexedRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.YZcolProjection

/-!
Conditional semantic refinement for the active shared PiRLC beta ladder.

Owns: interpretation of the exact 272 normalized source rows as one
`LadderTrace`, and discharge of the returned-parent `y_zcol` evaluators'
shared-power premise at the physical beta wire.

Does not own: transcript derivation of beta, rho evaluations, semantic parent
authority, whole-program row embedding, projection soundness, or row removal.

Emits constraints: no.

Assurance tier: conditional model-level semantics over artifact-checked rows.
The source artifact is exact, but its embedding in the whole production R1CS
and the transcript meaning of `betaColumns` remain explicit boundaries.

| Stage path | Exact premise | Result |
|---|---|---|
| `projection_shared.beta_ladder` | 272 source rows, canonical assignment, `z[0] = 1` | `LadderTrace.sound` for the physical beta wire |
| returned-parent output evaluators | exact 54-power prefix linkage | `SharedPowersValid` at that same wire |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.Refinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionBetaLadderData
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder

def ownedDefinitions : List Program.Definition :=
  ownedRowDefinitions.map Prod.snd

def ownedBuilderRows : List Row :=
  ownedRowDefinitions.map fun entry => entry.2.builderRow

theorem ownedDefinitions_eq_ladderDefinitions :
    ownedDefinitions = owner.ladderTrace.definitions := by
  set_option maxRecDepth 100000 in
    decide

theorem ownedDefinitions_canonical :
    ∀ definition ∈ ownedDefinitions, definition.Canonical := by
  set_option maxRecDepth 100000 in
    decide

theorem ownedSourceRows_definitionsHold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    ProjectionProgram.DefinitionsHold assignment
      owner.ladderTrace.definitions := by
  have builderRows : Satisfies ownedBuilderRows assignment := by
    exact ActiveIndexedRows.builderRows_satisfied_of_indexedRowsMatch
      FPrimeRecursivePiRlcProjectionBetaLadderData.sourceRows
      ownedRowDefinitions source_rows_match sourceSatisfies
  have normalizedBuilderRows :
      Satisfies (ownedDefinitions.map Program.Definition.builderRow)
        assignment := by
    simpa [ownedBuilderRows, ownedDefinitions, List.map_map]
      using builderRows
  have definitionsHold := Program.builderDefinitions_sound
    assignmentCanonical constantOne ownedDefinitions_canonical
    normalizedBuilderRows
  rw [ownedDefinitions_eq_ladderDefinitions] at definitionsHold
  exact definitionsHold

/-- Exact ladder rows determine all 55 advertised powers at the physical beta
wire. This theorem makes no claim that the wire was transcript-derived. -/
theorem ownedSourceRows_ladder_sound
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    owner.ladderTrace.powers.map
        (fun power => power.value assignment) =
      ProjectionProgram.K.powersFrom
        (owner.betaColumns.value assignment)
        ProjectionProgram.K.one owner.ladderTrace.powers.length := by
  exact owner.ladderTrace.sound assignment constantOne owner_valid.layout
    (ownedSourceRows_definitionsHold assignmentCanonical constantOne
      sourceSatisfies)

/-- The exact physical ladder discharges the arithmetic shared-power premise
used by both parent-`y_zcol` output evaluators. Transcript derivation remains
separate. -/
theorem ownedSourceRows_y_zcol_sharedPowers
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    FPrimeRecursiveYZcolProjection.Refinement.SharedPowersValid
      assignment (owner.betaColumns.value assignment) := by
  have ladderSound := ownedSourceRows_ladder_sound assignmentCanonical
    constantOne sourceSatisfies
  have ladderSound' :
      powerColumns.map (fun power => power.value assignment) =
        ProjectionProgram.K.powersFrom
          (owner.betaColumns.value assignment)
          ProjectionProgram.K.one powerColumns.length := by
    simpa [owner, PiRlcProjectionBetaLadderOwner.ladderTrace,
      ProjectionProgram.LadderTrace.ofColumns] using ladderSound
  have prefixLength :
      (powerColumns.take
        FPrimeRecursiveYZcolProjectionData.activeLaneCount).length =
        FPrimeRecursiveYZcolProjectionData.activeLaneCount := by
    decide
  unfold FPrimeRecursiveYZcolProjection.Refinement.SharedPowersValid
  rw [← y_zcol_power_prefix]
  rw [List.map_take]
  rw [ladderSound', prefixLength]
  exact ProjectionProgram.K.take_powersFrom
    (owner.betaColumns.value assignment) ProjectionProgram.K.one (by decide)

def SourceRowsEmbedded (fullRows : List Row) : Prop :=
  ActiveIndexedRows.SourceRowsEmbedded
    FPrimeRecursivePiRlcProjectionBetaLadderData.sourceRows fullRows

theorem ownedSourceRows_satisfied_of_embedded
    {fullRows : List Row} {assignment : Nat → Nat}
    (embedded : SourceRowsEmbedded fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    Satisfies ownedSourceRows assignment :=
  ActiveIndexedRows.sourceRows_satisfied_of_embedded embedded fullSatisfies

theorem fullRows_y_zcol_sharedPowers
    {fullRows : List Row} {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (embedded : SourceRowsEmbedded fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    FPrimeRecursiveYZcolProjection.Refinement.SharedPowersValid
      assignment (owner.betaColumns.value assignment) :=
  ownedSourceRows_y_zcol_sharedPowers assignmentCanonical constantOne
    (ownedSourceRows_satisfied_of_embedded embedded fullSatisfies)

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.Refinement
