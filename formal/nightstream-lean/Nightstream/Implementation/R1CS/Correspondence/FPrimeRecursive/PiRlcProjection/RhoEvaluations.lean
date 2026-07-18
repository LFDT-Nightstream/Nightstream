import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.RhoEvaluations
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.BetaLadder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.IndexedRows

/-!
Conditional semantic refinement for the active shared PiRLC rho evaluations.

Owns: interpretation of the exact 1,620 normalized source-R1CS rows as 15
independently reconstructed `EvalTrace` programs, and proof that every
physical output evaluates its exact 54 coefficient columns at the physical
beta wire constrained by the active ladder.

Does not own: transcript derivation or semantic meaning of the rho
coefficients, beta transcript authority, whole-program row embedding,
projection-identity soundness, encoded lowering, bad-root bounds, or row
removal.

Emits constraints: no.

Assurance tier: conditional model-level semantics over artifact-checked rows.
This is not whole-verifier Rust conformance or a security reduction.

| Stage path | Exact premise | Result |
|---|---|---|
| `projection_shared.rho_evaluations` | 1,620 source rows, canonical assignment, `z[0] = 1` | all 15 exact `EvalTrace` definitions hold |
| `projection_shared.beta_ladder` | 272 source rows under the same assignment | every evaluator uses powers of the same physical beta wire |
| each rho leaf | exact coefficient/power/output columns | output equals polynomial evaluation of those columns |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations.Refinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations

def betaOwner : PiRlcProjectionBetaLadderOwner :=
  FPrimeRecursivePiRlcProjection.BetaLadder.owner

def betaSourceRows : List Row :=
  FPrimeRecursivePiRlcProjection.BetaLadder.ownedSourceRows

def definitionsFor (chunk : List PiRlcRhoEvaluationOwner) :
    List Program.Definition :=
  (rowDefinitionsFor chunk).map Prod.snd

def builderRowsFor (chunk : List PiRlcRhoEvaluationOwner) : List Row :=
  (rowDefinitionsFor chunk).map fun entry => entry.2.builderRow

def sourceRowsFor (rows : List (Nat × Row)) : List Row :=
  rows.map Prod.snd

def ownedDefinitions : List Program.Definition :=
  owners.flatMap fun owner => owner.evalTrace.definitions

def OutputsCorrect (assignment : Nat → Nat)
    (point : ProjectionProgram.K) : Prop :=
  ∀ owner ∈ owners,
    owner.evalTrace.output.value assignment =
      ProjectionProgram.Polynomial.eval
        (ProjectionProgram.basePolynomial assignment
          owner.coefficientColumns) point

theorem definitions_hold_of_matched_source
    {chunk : List PiRlcRhoEvaluationOwner}
    {rows : List (Nat × Row)}
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (definitionsCanonical :
      ∀ definition ∈ definitionsFor chunk, definition.Canonical)
    (rowsMatch : ShardRowsMatch rows (rowDefinitionsFor chunk))
    (sourceSatisfies : Satisfies (sourceRowsFor rows) assignment) :
    ProjectionProgram.DefinitionsHold assignment (definitionsFor chunk) := by
  have builderRows : Satisfies (builderRowsFor chunk) assignment := by
    exact ActiveIndexedRows.builderRows_satisfied_of_indexedRowsMatch
      rows (rowDefinitionsFor chunk) rowsMatch sourceSatisfies
  have normalizedBuilderRows :
      Satisfies
        ((definitionsFor chunk).map Program.Definition.builderRow)
        assignment := by
    simpa [builderRowsFor, definitionsFor, List.map_map] using builderRows
  exact Program.builderDefinitions_sound assignmentCanonical constantOne
    definitionsCanonical normalizedBuilderRows

theorem shard0_definitions_exact :
    definitionsFor owners0 =
      owners0.flatMap
        (fun owner => owner.evalTrace.definitions) := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 1000000 in
    decide

theorem shard1_definitions_exact :
    definitionsFor owners1 =
      owners1.flatMap
        (fun owner => owner.evalTrace.definitions) := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 1000000 in
    decide

theorem shard2_definitions_exact :
    definitionsFor owners2 =
      owners2.flatMap
        (fun owner => owner.evalTrace.definitions) := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 1000000 in
    decide

theorem shard0_definitions_canonical :
    ∀ definition ∈ definitionsFor owners0,
      definition.Canonical := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 1000000 in
    decide

theorem shard1_definitions_canonical :
    ∀ definition ∈ definitionsFor owners1,
      definition.Canonical := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 1000000 in
    decide

theorem shard2_definitions_canonical :
    ∀ definition ∈ definitionsFor owners2,
      definition.Canonical := by
  set_option maxRecDepth 100000 in
  set_option maxHeartbeats 1000000 in
    decide

private theorem source_satisfies_shard0
    {assignment : Nat → Nat}
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    Satisfies (sourceRowsFor sourceRows0) assignment := by
  intro row member
  exact sourceSatisfies row (by
    simp only [ownedSourceRows, sourceRows, List.map_append,
      List.mem_append]
    exact Or.inl (Or.inl member))

private theorem source_satisfies_shard1
    {assignment : Nat → Nat}
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    Satisfies (sourceRowsFor sourceRows1) assignment := by
  intro row member
  exact sourceSatisfies row (by
    simp only [ownedSourceRows, sourceRows, List.map_append,
      List.mem_append]
    exact Or.inl (Or.inr member))

private theorem source_satisfies_shard2
    {assignment : Nat → Nat}
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    Satisfies (sourceRowsFor sourceRows2) assignment := by
  intro row member
  exact sourceSatisfies row (by
    simp only [ownedSourceRows, sourceRows, List.map_append,
      List.mem_append]
    exact Or.inr member)

private theorem shard0_definitions_hold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    ProjectionProgram.DefinitionsHold assignment
      (owners0.flatMap
        (fun owner => owner.evalTrace.definitions)) := by
  have definitionsHold := definitions_hold_of_matched_source
    assignmentCanonical constantOne shard0_definitions_canonical
    source_rows_match.1 (source_satisfies_shard0 sourceSatisfies)
  rw [shard0_definitions_exact] at definitionsHold
  exact definitionsHold

private theorem shard1_definitions_hold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    ProjectionProgram.DefinitionsHold assignment
      (owners1.flatMap
        (fun owner => owner.evalTrace.definitions)) := by
  have definitionsHold := definitions_hold_of_matched_source
    assignmentCanonical constantOne shard1_definitions_canonical
    source_rows_match.2.1 (source_satisfies_shard1 sourceSatisfies)
  rw [shard1_definitions_exact] at definitionsHold
  exact definitionsHold

private theorem shard2_definitions_hold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    ProjectionProgram.DefinitionsHold assignment
      (owners2.flatMap
        (fun owner => owner.evalTrace.definitions)) := by
  have definitionsHold := definitions_hold_of_matched_source
    assignmentCanonical constantOne shard2_definitions_canonical
    source_rows_match.2.2 (source_satisfies_shard2 sourceSatisfies)
  rw [shard2_definitions_exact] at definitionsHold
  exact definitionsHold

private theorem owner_definitions_hold
    {assignment : Nat → Nat} {owner : PiRlcRhoEvaluationOwner}
    (member : owner ∈ owners)
    (shard0 : ProjectionProgram.DefinitionsHold assignment
      (owners0.flatMap
        (fun current => current.evalTrace.definitions)))
    (shard1 : ProjectionProgram.DefinitionsHold assignment
      (owners1.flatMap
        (fun current => current.evalTrace.definitions)))
    (shard2 : ProjectionProgram.DefinitionsHold assignment
      (owners2.flatMap
        (fun current => current.evalTrace.definitions))) :
    ProjectionProgram.DefinitionsHold assignment owner.evalTrace.definitions := by
  have ownerShard :
      owner ∈ owners0 ∨ owner ∈ owners1 ∨ owner ∈ owners2 := by
    simpa [owners] using member
  rcases ownerShard with inShard0 | inShard1 | inShard2
  · intro definition definitionMember
    exact shard0 definition (List.mem_flatMap.mpr
      ⟨owner, inShard0, definitionMember⟩)
  · intro definition definitionMember
    exact shard1 definition (List.mem_flatMap.mpr
      ⟨owner, inShard1, definitionMember⟩)
  · intro definition definitionMember
    exact shard2 definition (List.mem_flatMap.mpr
      ⟨owner, inShard2, definitionMember⟩)

/-- The exact 1,620 rho source rows force the complete shared definition
schedule. This exposes arithmetic definitions only; transcript and semantic
rho authority remain outside the theorem. -/
theorem ownedSourceRows_definitionsHold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (rhoSourceSatisfies : Satisfies ownedSourceRows assignment) :
    ProjectionProgram.DefinitionsHold assignment ownedDefinitions := by
  have definitions0 := shard0_definitions_hold assignmentCanonical
    constantOne rhoSourceSatisfies
  have definitions1 := shard1_definitions_hold assignmentCanonical
    constantOne rhoSourceSatisfies
  have definitions2 := shard2_definitions_hold assignmentCanonical
    constantOne rhoSourceSatisfies
  intro definition member
  rcases List.mem_flatMap.mp member with ⟨owner, ownerMember, definitionMember⟩
  exact owner_definitions_hold ownerMember definitions0 definitions1
    definitions2 definition definitionMember

/-- Exact rho and ladder rows force all 15 physical outputs to equal
evaluation of their exact coefficient columns at the same physical beta wire.
No semantic interpretation or transcript authority is assigned to those
coefficient columns here. -/
theorem ownedSourceRows_outputs_correct
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (rhoSourceSatisfies : Satisfies ownedSourceRows assignment)
    (ladderSourceSatisfies :
      Satisfies betaSourceRows assignment) :
    OutputsCorrect assignment
      (betaOwner.betaColumns.value assignment) := by
  have ladderSound := FPrimeRecursivePiRlcProjection.BetaLadder.Refinement.ownedSourceRows_ladder_sound
    assignmentCanonical constantOne ladderSourceSatisfies
  have ladderValues :
      ladderPowerColumns.map (fun power => power.value assignment) =
        ProjectionProgram.K.powersFrom
          (betaOwner.betaColumns.value assignment)
          ProjectionProgram.K.one ladderPowerColumns.length := by
    simpa [ladderPowerColumns, betaOwner,
      FPrimeRecursivePiRlcProjection.BetaLadder.owner,
      PiRlcProjectionBetaLadderOwner.ladderTrace,
      ProjectionProgram.LadderTrace.ofColumns] using ladderSound
  have definitions0 := shard0_definitions_hold assignmentCanonical
    constantOne rhoSourceSatisfies
  have definitions1 := shard1_definitions_hold assignmentCanonical
    constantOne rhoSourceSatisfies
  have definitions2 := shard2_definitions_hold assignmentCanonical
    constantOne rhoSourceSatisfies
  intro owner member
  have valid := owner_valid member
  have prefixShape :
      owner.evalTrace.powers =
        ladderPowerColumns.take owner.evalTrace.coefficients.length := by
    have coefficientLength := valid.coefficient_length
    have powerPrefix := owner_power_prefix member
    simp only [PiRlcRhoEvaluationOwner.evalTrace,
      ProjectionProgram.EvalTrace.ofColumns]
    rw [coefficientLength, powerPrefix]
    simp [List.take_take, coefficientCount]
  have within := owner.evalTrace.coefficientLength_le_ladder
    ladderPowerColumns valid.layout prefixShape
  have powersValid := owner.evalTrace.powersValid_of_ladderPrefix
    assignment (betaOwner.betaColumns.value assignment)
    ladderPowerColumns prefixShape within ladderValues
  exact owner.evalTrace.sound assignment
    (betaOwner.betaColumns.value assignment) valid.layout powersValid
    (owner_definitions_hold member definitions0 definitions1 definitions2)

def SourceRowsEmbedded (fullRows : List Row) : Prop :=
  ActiveIndexedRows.SourceRowsEmbedded sourceRows fullRows

def LadderRowsEmbedded (fullRows : List Row) : Prop :=
  FPrimeRecursivePiRlcProjection.BetaLadder.Refinement.SourceRowsEmbedded fullRows

theorem fullRows_outputs_correct
    {fullRows : List Row} {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (rhoEmbedded : SourceRowsEmbedded fullRows)
    (ladderEmbedded : LadderRowsEmbedded fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    OutputsCorrect assignment
      (betaOwner.betaColumns.value assignment) := by
  apply ownedSourceRows_outputs_correct assignmentCanonical constantOne
  · exact ActiveIndexedRows.sourceRows_satisfied_of_embedded
      rhoEmbedded fullSatisfies
  · exact FPrimeRecursivePiRlcProjection.BetaLadder.Refinement.ownedSourceRows_satisfied_of_embedded
      ladderEmbedded fullSatisfies

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations.Refinement
