import NightstreamFPrime.Export.Pilot
import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.PiRLCCombinationCounts
import NightstreamFPrime.Export.Stage1.PiRLCCombinationConformance
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Conformance
import NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions
import NightstreamFPrime.Layout.PiDEC.v1_1.Preservation
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns structural proofs for the one Stage 1 pilot + PiCCS + PiRLC + PiDEC
package.

No theorem evaluates the package artifact. Counts follow from the pilot
proofs, the compact invocation compiler, the ordinary-row classifier, and the
proved PiCCS leaf footprints.
-/

namespace NightstreamFPrime.Export.Stage1.Package

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

theorem liftPilotRows_length (rows : List SparseRow) :
    (Data.liftPilotRows rows).length = rows.length := by
  simp [Data.liftPilotRows]

private theorem instantiateRow_stage1_eq_pilot
    (chain : HashChain) (invocation : Nat) (row : TemplateRow) :
    instantiateRow (Data.circuitPackage ()) chain invocation row =
      instantiateRow (PilotData.circuitPackage ()) chain invocation row := by
  rfl

private theorem hashChainHolds_pilotTemplate
    (chain : HashChain) (env : Env)
    (holds : HashChainHolds (Data.circuitPackage ()) chain env) :
    HashChainHolds (PilotData.circuitPackage ()) chain env := by
  intro invocation bound row member
  have stageMember : row ∈ (Data.circuitPackage ()).permutation.rows := by
    rw [Data.circuitPackage_permutation]
    exact member
  have stage := holds invocation bound row stageMember
  rw [instantiateRow_stage1_eq_pilot] at stage
  exact stage

private def pilotEnv (env : Env) : Env :=
  fun column => env
    (NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn column)

private theorem liftPilotCombination_eval
    (combination : SparseCombination) (env : Env) :
    (Data.liftPilotCombination combination).eval env =
      combination.eval (pilotEnv env) := by
  unfold Data.liftPilotCombination Data.liftPilotTerm
    SparseCombination.eval pilotEnv
  rw [List.map_map]
  rfl

private theorem liftPilotRow_holds
    (row : SparseRow) (env : Env) :
    (Data.liftPilotRow row).Holds env ↔ row.Holds (pilotEnv env) := by
  unfold Data.liftPilotRow SparseRow.Holds
  rw [liftPilotCombination_eval, liftPilotCombination_eval,
    liftPilotCombination_eval]

private theorem combinedAssertions_imply_pilotAssertions
    (env : Env)
    (holds : AssertionsHold (Data.circuitPackage ()) env) :
    AssertionsHold (PilotData.circuitPackage ()) (pilotEnv env) := by
  intro row member
  have liftedMember : Data.liftPilotRow row ∈
      (Data.circuitPackage ()).assertionRows := by
    rw [Data.circuitPackage_assertionRows]
    unfold Data.Components.assertionRows
    apply List.mem_append_left
    rw [Data.liftPilotRows, List.mem_map]
    exact ⟨row, member, rfl⟩
  exact (liftPilotRow_holds row env).mp (holds _ liftedMember)

private theorem chainInputValues_lift
    (chain : HashChain) (env : Env)
    (bound : chain.inputStart + chain.inputLength ≤
      NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount) :
    NightstreamFPrime.Export.Pilot.chainInputValues chain (pilotEnv env) =
      NightstreamFPrime.Export.Pilot.chainInputValues
        (Data.liftPilotChain chain) env := by
  unfold NightstreamFPrime.Export.Pilot.chainInputValues
  apply congrArg List.ofFn
  funext index
  change env (NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn
      (chain.inputStart + index.val)) =
    env (NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn
      chain.inputStart + index.val)
  apply congrArg env
  apply NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn_add_of_input
  have indexBound := index.isLt
  omega

private theorem chainOutputState_lift
    (chain : HashChain) (env : Env)
    (lower : NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount
      ≤ chain.witnessStart)
    (upper : chain.witnessStart + chain.absorbCount * 592 + 584 + 7 <
      NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount) :
    NightstreamFPrime.Export.Pilot.chainOutputState chain chain.absorbCount
        (pilotEnv env) =
      NightstreamFPrime.Export.Pilot.chainOutputState
        (Data.liftPilotChain chain) chain.absorbCount env := by
  funext lane
  unfold NightstreamFPrime.Export.Pilot.chainOutputState
    invocationLocalStart pilotEnv Data.liftPilotChain
  change env (NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn
      (chain.witnessStart + chain.absorbCount * 592 + 584 + lane.val)) =
    env (NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn
      chain.witnessStart + chain.absorbCount * 592 + 584 + lane.val)
  have offsetBound : chain.witnessStart +
      (chain.absorbCount * 592 + 584 + lane.val) <
        NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount := by
    have laneBound := lane.isLt
    omega
  have columnEq :=
    NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn_add_of_private
      chain.witnessStart (chain.absorbCount * 592 + 584 + lane.val)
      lower offsetBound
  have normalized :
      NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn
          (chain.witnessStart + chain.absorbCount * 592 + 584 + lane.val) =
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn
            chain.witnessStart + chain.absorbCount * 592 + 584 + lane.val := by
    simpa [Nat.add_assoc] using columnEq
  exact congrArg env normalized

theorem circuitPackage_template_rows :
    (Data.circuitPackage ()).permutation.rows.length = 592 := by
  rw [Data.circuitPackage_permutation]
  exact NightstreamFPrime.Export.Pilot.templateRows_length

theorem circuitPackage_hash_chains :
    (Data.circuitPackage ()).hashChains.length = 2 := by
  rw [Data.circuitPackage_hashChains]
  rfl

theorem circuitPackage_permutation_invocations :
    (Data.circuitPackage ()).permutationInvocations.length = 7613 := by
  rw [Data.circuitPackage_permutationInvocations,
    Data.components_permutationInvocations,
    Data.permutationInvocations_eq, List.length_append,
    PiCCSInvocations.invocations_length Data.logicalWidth Data.publicFits,
    PiRLCSamplerInvocations.invocations_length]

theorem proofInputStart_eq : Data.proofInputStart = 84950 := by
  rfl

theorem witnessStart_eq : Data.witnessStart = 113962 := by
  rfl

theorem witnessLength_eq : Data.witnessLength = 25555039 := by
  rfl

theorem circuitPackage_layout_values :
    let layout := (Data.circuitPackage ()).layout
    layout.rowCount = 25564086 ∧
      layout.privateColumnCount = 25714955 ∧
      layout.constantColumn = 25714955 ∧
      layout.publicColumnCount = 62 ∧
      layout.totalColumnCount = 25715018 := by
  rw [Data.circuitPackage_layout]
  dsimp [Data.physicalLayout]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem circuitPackage_jointDomain_le_twoPow25 :
    max (Data.circuitPackage ()).layout.rowCount
      ((Data.circuitPackage ()).layout.totalColumnCount - 1) ≤ 2 ^ 25 := by
  rw [Data.circuitPackage_layout]
  norm_num [Data.physicalLayout,
    NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount]

/-- The generic ordinary-row encoding classifies every arithmetic row once. -/
theorem arithmetic_partition
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits) :
    (Rows.witnessInstructions (Data.arithmeticRows ())).length +
      (Rows.assertionRows (Data.arithmeticRows ())).length = 993379 := by
  calc
    _ = (Data.arithmeticRows ()).length :=
      Rows.witnessInstructions_length_add_assertionRows_length _
    _ = 993379 := by
      rw [Data.arithmeticRows_eq, List.length_append, List.length_append,
        PiCCSArithmetic.arithmeticRows_length Data.logicalWidth
          Data.publicFits relation,
        PiRLCSamplerOrdinaryRows.rows_length,
        PiDECArithmetic.Plan.rows_length,
        PiDECArithmetic.canonicalPlan_rowCount relation]

theorem circuitPackage_ordinary_rows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits) :
    (Data.components ()).toCircuitPackage.witnessInstructions.length +
      (Data.components ()).toCircuitPackage.assertionRows.length = 993437 := by
  calc
    _ = (Data.liftPilotRows (PilotData.assertionRows ())).length +
        (Data.components ()).arithmeticRows.length :=
      Data.Components.ordinaryRows_length (Data.components ())
    _ = 58 + (Data.arithmeticRows ()).length := by
      rw [liftPilotRows_length,
        NightstreamFPrime.Export.Pilot.assertionRows_length,
        Data.components_arithmeticRows]
    _ = 58 + 993379 := by
      rw [Data.arithmeticRows_eq, List.length_append, List.length_append,
        PiCCSArithmetic.arithmeticRows_length Data.logicalWidth
          Data.publicFits relation,
        PiRLCSamplerOrdinaryRows.rows_length,
        PiDECArithmetic.Plan.rows_length,
        PiDECArithmetic.canonicalPlan_rowCount relation]
    _ = 993437 := by norm_num

/-- Construct all 7,460 PiCCS Poseidon2 invocations in their proved private
intervals. Sampler invocations have a separate package completion owner. -/
theorem complete_piCcsInvocations
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env) :
    ∃ completed,
      Invocations.AgreesOutsideInvocations env completed
          (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits) ∧
        ∀ invocation ∈
            PiCCSInvocations.invocations Data.logicalWidth Data.publicFits,
          PermutationInvocationHolds (Data.circuitPackage ()) invocation
            completed := by
  have schedule := PiCCSInvocations.invocations_scheduleWithin
    Data.logicalWidth Data.publicFits relation
  rcases Invocations.completeInvocations env
      (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        PiCCSInvocations.statementWitnessStart)
      PiCCSInvocations.invocationCeiling
      (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits)
      schedule.1 with
    ⟨completed, _coarseAgrees, exactAgrees, invocationHolds⟩
  refine ⟨completed, ?_, ?_⟩
  · exact exactAgrees
  · intro invocation member
    have pilotHolds := invocationHolds invocation member
    unfold PermutationInvocationHolds at pilotHolds ⊢
    rw [Data.circuitPackage_permutation]
    simpa [PilotData.circuitPackage] using pilotHolds

/-- Package satisfaction covers every losslessly classified PiCCS arithmetic
row. This proof is structural in the classifier and does not evaluate the
materialized row packet. -/
theorem circuitPackage_implies_arithmeticRows
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    R1CS.RowsHold env
      ((Data.arithmeticRows ()).map Rows.CompiledRow.toR1CS) := by
  apply (Rows.compiledRows_hold_iff (Data.arithmeticRows ()) env).mpr
  constructor
  · intro instruction member
    apply holds.2.2.2.1 instruction
    rw [Data.circuitPackage_witnessInstructions]
    unfold Data.Components.witnessInstructions
    rw [Rows.witnessInstructionsTR_eq, Data.components_arithmeticRows]
    exact member
  · intro assertion member
    apply holds.2.2.2.2 assertion
    rw [Data.circuitPackage_assertionRows]
    unfold Data.Components.assertionRows
      Data.Components.arithmeticAssertionRows
    apply List.mem_append_right
    rw [Rows.assertionRowsTR_eq, Data.components_arithmeticRows]
    exact member

/-- Exact compiled arithmetic-row satisfaction supplies the two ordinary-row
predicates carried by the canonical package. Pilot assertions remain under
their separate pilot completeness owner. -/
theorem arithmeticRows_imply_packageOrdinary
    (env : Env)
    (holds : R1CS.RowsHold env
      ((Data.arithmeticRows ()).map Rows.CompiledRow.toR1CS)) :
    (∀ instruction ∈ (Data.circuitPackage ()).witnessInstructions,
        instruction.Holds env) ∧
      ∀ assertion ∈ (Data.components ()).arithmeticAssertionRows,
        assertion.Holds env := by
  have classified :=
    (Rows.compiledRows_hold_iff (Data.arithmeticRows ()) env).mp holds
  constructor
  · intro instruction member
    apply classified.1 instruction
    rw [Data.circuitPackage_witnessInstructions] at member
    unfold Data.Components.witnessInstructions at member
    rw [Rows.witnessInstructionsTR_eq, Data.components_arithmeticRows]
      at member
    exact member
  · intro assertion member
    apply classified.2 assertion
    unfold Data.Components.arithmeticAssertionRows at member
    rw [Rows.assertionRowsTR_eq, Data.components_arithmeticRows] at member
    exact member

private theorem compiledRowsHold_three_iff
    (env : Env) (first second third : List Rows.CompiledRow) :
    R1CS.RowsHold env
        (((first ++ second) ++ third).map Rows.CompiledRow.toR1CS) ↔
      R1CS.RowsHold env (first.map Rows.CompiledRow.toR1CS) ∧
        R1CS.RowsHold env (second.map Rows.CompiledRow.toR1CS) ∧
        R1CS.RowsHold env (third.map Rows.CompiledRow.toR1CS) := by
  rw [List.map_append, List.map_append, R1CS.rowsHold_append,
    R1CS.rowsHold_append, and_assoc]

/-- The three phase-local ordinary packets compose into the exact classified
ordinary surface of the current canonical package. -/
theorem phaseArithmeticRows_imply_packageOrdinary
    (env : Env)
    (piCcs : R1CS.RowsHold env
      ((PiCCSArithmetic.arithmeticRows Data.logicalWidth
        Data.publicFits).map Rows.CompiledRow.toR1CS))
    (piRlc : R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)).map Rows.CompiledRow.toR1CS))
    (piDec : R1CS.RowsHold env
      ((PiDECArithmetic.canonicalPlan
        Data.logicalWidth Data.publicFits).rows.map
          Rows.CompiledRow.toR1CS)) :
    (∀ instruction ∈ (Data.circuitPackage ()).witnessInstructions,
        instruction.Holds env) ∧
      ∀ assertion ∈ (Data.components ()).arithmeticAssertionRows,
        assertion.Holds env := by
  apply arithmeticRows_imply_packageOrdinary env
  rw [Data.arithmeticRows_eq]
  exact (compiledRowsHold_three_iff env _ _ _).mpr
    ⟨piCcs, piRlc, piDec⟩

/-- The ordinary package packet preserves the exact PiCCS arithmetic prefix. -/
theorem circuitPackage_implies_piCcsArithmeticRows
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    R1CS.RowsHold env
      ((PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits).map
        Rows.CompiledRow.toR1CS) := by
  have combined := circuitPackage_implies_arithmeticRows env holds
  rw [Data.arithmeticRows_eq] at combined
  exact (compiledRowsHold_three_iff env _ _ _).mp combined |>.1

/-- The ordinary package packet preserves the exact PiRLC sampler-row
suffix selected by `Data.arithmeticRows`. -/
theorem circuitPackage_implies_piRlcSamplerOrdinaryRows
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)).map Rows.CompiledRow.toR1CS) := by
  have combined := circuitPackage_implies_arithmeticRows env holds
  rw [Data.arithmeticRows_eq] at combined
  exact (compiledRowsHold_three_iff env _ _ _).mp combined |>.2.1

/-- The ordinary package packet preserves the exact PiDEC row suffix selected
by the canonical generative plan. -/
theorem circuitPackage_implies_piDecArithmeticRows
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    R1CS.RowsHold env
      ((PiDECArithmetic.canonicalPlan
        Data.logicalWidth Data.publicFits).rows.map
          Rows.CompiledRow.toR1CS) := by
  have combined := circuitPackage_implies_arithmeticRows env holds
  rw [Data.arithmeticRows_eq] at combined
  exact (compiledRowsHold_three_iff env _ _ _).mp combined |>.2.2

/-- Canonical package rows imply the exact PiDEC verifier semantics through
the Lean-owned lowering plan and Spartan column permutation. -/
theorem circuitPackage_implies_piDecPhaseHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        (PiDECArithmetic.phaseInterface Data.logicalWidth Data.publicFits)
        NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai
      (PiDECArithmetic.phaseInterface Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have packageRows := circuitPackage_implies_piDecArithmeticRows env holds
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits)
    (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  rw [exactRows] at packageRows
  have physical :=
    (NightstreamFPrime.Layout.Stage1.Spartan.remapRows_hold env _).mp
      packageRows
  exact NightstreamFPrime.Layout.PiDEC.v1_1.physical_implies_phaseHolds
    relation ajtai
    (PiDECArithmetic.phaseInterface Data.logicalWidth Data.publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
    assumptions physical

/-- Canonical package satisfaction implies the exact First54 constraint
specification for one bounded PiRLC scalar source. -/
theorem circuitPackage_implies_piRlcFirst54Spec
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (source : Nat)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount) :
    NightstreamFPrime.Gadgets.Sampling.First54.SpecHolds
      (PiRLCFirst54Conformance.selectorInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply PiRLCFirst54Conformance.packageInvocations_imply_spec
    (Data.circuitPackage ()) ?_ source sourceLt env
  · intro invocation member
    apply holds.2.2.1 invocation
    rw [Data.circuitPackage_compactRowInvocations,
      Data.compactRowInvocations_eq]
    exact List.mem_append_left _ member
  · exact circuitPackage_implies_piRlcSamplerOrdinaryRows env holds
  · rw [Data.circuitPackage_compactRowTemplates,
      Data.compactRowTemplates_eq]
    rfl

/-- Package satisfaction implies one exact PiRLC digest-lane specification
for every bounded source, digest round, and lane. -/
theorem circuitPackage_implies_piRlcDigestLane
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (source round : Nat) (lane : Fin 4)
    (sourceLt : source < PiRLCSamplerOrdinaryRows.sourceCount)
    (roundLt : round < PiRLCSamplerOrdinaryRows.digestRoundCount)
    (assumptions : DigestLane.Assumptions
      (PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source round lane)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
        source round lane.val)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    DigestLane.SpecHolds
      (PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source round lane)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
        source round lane.val)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply PiRLCSamplerOrdinaryRows.rows_imply_laneSpec source round lane
    sourceLt roundLt env assumptions
  exact circuitPackage_implies_piRlcSamplerOrdinaryRows env holds

/-- Package satisfaction supplies every exact PiRLC sampler permutation
invocation through the one canonical Poseidon2 template. -/
theorem circuitPackage_implies_piRlcSamplerInvocations
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    ∀ invocation ∈
      PiRLCSamplerInvocations.invocations
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits),
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation env := by
  intro invocation member
  have packageInvocation : PermutationInvocationHolds
      (Data.circuitPackage ()) invocation env := by
    apply holds.2.1 invocation
    rw [Data.circuitPackage_permutationInvocations,
      Data.components_permutationInvocations,
      Data.permutationInvocations_eq]
    exact List.mem_append_right _ member
  unfold PermutationInvocationHolds at packageInvocation ⊢
  rw [Data.circuitPackage_permutation] at packageInvocation
  simpa [PilotData.circuitPackage] using packageInvocation

theorem circuitPackage_implies_piRlcEntry
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (source : Nat) (sourceLt : source < PiRLCSamplerInvocations.sourceCount) :
    TranscriptAbsorption.SpecHolds
      (Sampler.entryInterface
        (PiRLCSamplerInvocations.sourceInterface
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
          source))
      source (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply PiRLCSamplerInvocations.entryTrace_implies_spec source env
  intro invocation member
  apply circuitPackage_implies_piRlcSamplerInvocations env holds invocation
  unfold PiRLCSamplerInvocations.invocations
  apply List.mem_flatMap.mpr
  refine ⟨source, List.mem_range.mpr sourceLt, ?_⟩
  unfold PiRLCSamplerInvocations.sourceInvocations
    PiRLCSamplerInvocations.entryInvocations
  exact List.mem_append_left _ member

theorem circuitPackage_implies_piRlcWindowPermutation
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (source round : Nat)
    (sourceLt : source < PiRLCSamplerInvocations.sourceCount)
    (roundLt : round < PiRLCSamplerInvocations.digestRoundCount) :
    Permutation.Owned.SpecHolds
      (DigestWindow.permutationInterface
        (Sampler.windowInterface
          (PiRLCSamplerInvocations.sourceInterface
            (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
            source)
          source (PiRLCSamplerInvocations.sourceLogicalStart source) round)
        (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
          source round))
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
        source round)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply PiRLCSamplerInvocations.windowInvocation_implies_spec source round env
  apply circuitPackage_implies_piRlcSamplerInvocations env holds
  unfold PiRLCSamplerInvocations.invocations
  apply List.mem_flatMap.mpr
  refine ⟨source, List.mem_range.mpr sourceLt, ?_⟩
  unfold PiRLCSamplerInvocations.sourceInvocations
  apply List.mem_append_right
  unfold PiRLCSamplerInvocations.windowInvocations
  apply List.mem_map.mpr
  exact ⟨round, List.mem_range.mpr roundLt, rfl⟩

/-- Package satisfaction composes the four ordinary lane packets and the one
compact permutation invocation into the exact digest-window parent. -/
theorem circuitPackage_implies_piRlcDigestWindow
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (source round : Nat)
    (sourceLt : source < PiRLCSamplerInvocations.sourceCount)
    (roundLt : round < PiRLCSamplerInvocations.digestRoundCount)
    (assumptions : DigestWindow.Assumptions
      (PiRLCSamplerOrdinaryRows.windowInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source round)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
        source round)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    DigestWindow.SpecHolds
      (PiRLCSamplerOrdinaryRows.windowInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source round)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
        source round)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  constructor
  · intro lane
    apply circuitPackage_implies_piRlcDigestLane env holds source round lane
      sourceLt roundLt
    exact DigestWindow.laneAssumptions
      (PiRLCSamplerOrdinaryRows.windowInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source round)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
        source round)
      lane assumptions
  · simpa [PiRLCSamplerOrdinaryRows.windowInterface,
      PiRLCSamplerOrdinaryRows.sourceInterface,
      PiRLCSamplerInvocations.sourceInterface,
      PiRLCSamplerOrdinaryRows.chainInterface,
      PiRLCSamplerInvocations.chainInterface,
      PiRLCSamplerInvocations.sourceLogicalStart] using
      circuitPackage_implies_piRlcWindowPermutation env holds source round
        sourceLt roundLt

/-- Package satisfaction assembles the exact scalar-sampler prefix through
all eight digest windows. The separate compact First54 selector remains the
only sampler child not included by this theorem. -/
theorem circuitPackage_implies_piRlcSamplerPrefix
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (source : Nat) (sourceLt : source < PiRLCSamplerInvocations.sourceCount)
    (assumptions : Sampler.Assumptions
      (PiRLCSamplerInvocations.sourceInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source)
      (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    Sampler.PrefixHolds
      (PiRLCSamplerInvocations.sourceInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source)
      source (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  constructor
  · exact circuitPackage_implies_piRlcEntry env holds source sourceLt
  · intro round
    have windowAssumptions := Sampler.windowAssumptions
      (PiRLCSamplerInvocations.sourceInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source)
      source (PiRLCSamplerInvocations.sourceLogicalStart source)
      assumptions round
    simpa [PiRLCSamplerOrdinaryRows.windowInterface,
      PiRLCSamplerOrdinaryRows.sourceInterface,
      PiRLCSamplerInvocations.sourceInterface,
      PiRLCSamplerOrdinaryRows.chainInterface,
      PiRLCSamplerInvocations.chainInterface,
      PiRLCSamplerInvocations.sourceLogicalStart] using
      circuitPackage_implies_piRlcDigestWindow env holds source round.val
        sourceLt round.isLt windowAssumptions

/-- Package satisfaction assembles the exact sampler prefix and compact
First54 selector into the complete scalar-sampler specification. -/
theorem circuitPackage_implies_piRlcSamplerSpec
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (source : Nat) (sourceLt : source < PiRLCSamplerInvocations.sourceCount)
    (assumptions : Sampler.Assumptions
      (PiRLCSamplerInvocations.sourceInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source)
      (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    Sampler.SpecHolds
      (PiRLCSamplerInvocations.sourceInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source)
      source (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  let samplerInterface := PiRLCSamplerInvocations.sourceInterface
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source
  let samplerOffset := PiRLCSamplerInvocations.sourceLogicalStart source
  let completed := NightstreamFPrime.Layout.Stage1.Spartan.pullback env
  have samplerPrefix : Sampler.PrefixHolds samplerInterface source samplerOffset
      completed :=
    circuitPackage_implies_piRlcSamplerPrefix env holds source sourceLt
      assumptions
  refine { toPrefixHolds := samplerPrefix, selector := ?_ }
  apply NightstreamFPrime.Gadgets.Sampling.First54.parentCoverage
  · exact Sampler.selectorAssumptions samplerInterface source samplerOffset
      completed samplerPrefix.window
  · have selectorSpec := circuitPackage_implies_piRlcFirst54Spec env holds
      source (by
        simpa [PiRLCSamplerInvocations.sourceCount,
          PiRLCFirst54Invocations.sourceCount] using sourceLt)
    simpa [samplerInterface, samplerOffset, completed,
      PiRLCFirst54Conformance.selectorInterface,
      PiRLCFirst54Conformance.sourceInterface,
      PiRLCSamplerOrdinaryRows.sourceInterface,
      PiRLCSamplerOrdinaryRows.chainInterface,
      PiRLCSamplerInvocations.sourceInterface,
      PiRLCSamplerInvocations.chainInterface,
      PiRLCSamplerInvocations.sourceLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorLogicalStart,
      Sampler.selectorOffset, Sampler.windowBase, Sampler.entryPrivateCount,
      Sampler.digestRoundCount, DigestWindow.logicalPrivateCount] using
        selectorSpec

/-- Package satisfaction composes all 17 exact scalar samplers into the
authoritative production sampler-chain relation. -/
theorem circuitPackage_implies_piRlcSamplerChain
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (assumptions : SamplerChain.Assumptions
      (PiRLCSamplerRows.samplerInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    SamplerChain.RelationHolds
      (PiRLCSamplerRows.samplerInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply SamplerChain.parentCoverage
  intro source
  have childAssumptions := SamplerChain.childAssumptions
    (PiRLCSamplerRows.samplerInterface
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart source.val
    source.isLt (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
    assumptions
  have childSpec := circuitPackage_implies_piRlcSamplerSpec env holds
    source.val (by
      simpa [SamplerChain.sourceCount_eq,
        PiRLCSamplerInvocations.sourceCount] using source.isLt)
    (by
      simpa [PiRLCSamplerInvocations.sourceInterface,
        PiRLCSamplerInvocations.chainInterface,
        PiRLCSamplerInvocations.sourceLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart,
        SamplerChain.sourceOffset, Sampler.logicalPrivateCount] using
          childAssumptions)
  apply Sampler.parentCoverage
  simpa [PiRLCSamplerInvocations.sourceInterface,
    PiRLCSamplerInvocations.chainInterface,
    PiRLCSamplerInvocations.sourceLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart,
    SamplerChain.sourceOffset, Sampler.logicalPrivateCount] using childSpec

/-- Exact template-selection equation required to interpret every compact
PiRLC combination invocation. -/
def PiRLCCombinationTemplateSelection (package : CircuitPackage) : Prop :=
  ∀ source : Fin PiRLCCombinationInvocations.sourceCount,
    ∀ lane : Fin ringDegree,
      package.compactRowTemplates[
          PiRLCCombinationTemplates.templateIndex source.val lane.val]? =
        some (PiRLCCombinationTemplates.template
          (PiRLCCombinationInvocations.firstSource source.val) lane)

private theorem piRlcPackageTemplates_selectCombination
    (source : Nat) (lane : Fin ringDegree) :
    PiRLCFirst54Invocations.packageTemplates[
        PiRLCCombinationTemplates.templateIndex source lane.val]? =
      some (PiRLCCombinationTemplates.template
        (PiRLCCombinationInvocations.firstSource source) lane) := by
  unfold PiRLCFirst54Invocations.packageTemplates
  rw [List.getElem?_append_left
    (PiRLCCombinationTemplates.templateIndex_lt source lane)]
  simpa [PiRLCCombinationInvocations.firstSource] using
    PiRLCCombinationTemplates.template_getElem? source lane

theorem circuitPackage_piRlcCombinationTemplateSelection :
    PiRLCCombinationTemplateSelection (Data.circuitPackage ()) := by
  intro source lane
  rw [Data.circuitPackage_compactRowTemplates,
    Data.compactRowTemplates_eq]
  change PiRLCFirst54Invocations.packageTemplates[
      PiRLCCombinationTemplates.templateIndex source.val lane.val]? = _
  exact piRlcPackageTemplates_selectCombination source.val lane

theorem piRlcCombination_compactRowCount :
    compactRowCountFor PiRLCFirst54Invocations.packageTemplates
      PiRLCCombinationInvocations.invocations = 6792282 := by
  exact PiRLCCombinationInvocations.invocationsCompactRowCountFor
    PiRLCFirst54Invocations.packageTemplates
    piRlcPackageTemplates_selectCombination

private theorem familyInvocationRows_of_package
    (package : CircuitPackage) (env : Env)
    (selection : PiRLCCombinationTemplateSelection package)
    (compactHolds : ∀ invocation ∈ package.compactRowInvocations,
      CompactRowInvocationHolds package invocation env)
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    [NeZero cellCount] (valueSourceStart : Nat → Nat → Nat → Nat)
    (member : ∀ source : Fin PiRLCCombinationInvocations.sourceCount,
      ∀ index : Fin (CombinationStep.privateCount blockCount cellCount),
        let coordinates := CombinationStep.coordinates index
        PiRLCCombinationInvocations.invocation logicalStart rowStart freshStart
            blockCount cellCount valueStride source.val coordinates.1.val
              coordinates.2.1.val coordinates.2.2.val valueSourceStart ∈
          package.compactRowInvocations) :
    PiRLCCombinationConformance.FamilyInvocationRowsHold logicalStart rowStart
      freshStart blockCount cellCount valueStride valueSourceStart env := by
  intro source index
  let coordinates := CombinationStep.coordinates index
  let selected := PiRLCCombinationInvocations.invocation logicalStart rowStart
    freshStart blockCount cellCount valueStride source.val coordinates.1.val
      coordinates.2.1.val coordinates.2.2.val valueSourceStart
  have packageRows := compactHolds selected (member source index)
  unfold CompactRowInvocationHolds at packageRows
  have selectedIndex : selected.templateIndex =
      PiRLCCombinationTemplates.templateIndex source.val
        coordinates.2.1.val := by
    rfl
  rw [selectedIndex, selection source coordinates.2.1] at packageRows
  dsimp only
  rw [CompactRows.instantiateRows_eq_package]
  exact packageRows

/-- Exact instantiated compact rows for the four PiRLC combination
families, before their package-template lookup is discharged. -/
structure PiRLCCombinationRowsHold (env : Env) : Prop where
  commitment : PiRLCCombinationConformance.FamilyInvocationRowsHold
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.commitmentLogicalStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.commitmentRowStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.commitmentFreshStart 18 1 1
    PiRLCCombinationInvocations.commitmentValueSourceStart env
  publicInput : PiRLCCombinationConformance.FamilyInvocationRowsHold
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.publicInputLogicalStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.publicInputRowStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.publicInputFreshStart 1 1 1
    PiRLCCombinationInvocations.publicInputValueSourceStart env
  eval_K : PiRLCCombinationConformance.FamilyInvocationRowsHold
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalKLogicalStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalKRowStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalKFreshStart 1 2 2
    PiRLCCombinationInvocations.evalKValueSourceStart env
  eval_A : PiRLCCombinationConformance.FamilyInvocationRowsHold
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalALogicalStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalARowStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalAFreshStart 14 2 2
    PiRLCCombinationInvocations.evalAValueSourceStart env

/-- Canonical package satisfaction supplies all four exact combination-row
packets from its Lean-selected templates. -/
theorem circuitPackage_implies_piRlcCombinationRows
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    PiRLCCombinationRowsHold env := by
  let selection := circuitPackage_piRlcCombinationTemplateSelection
  constructor
  · apply familyInvocationRows_of_package (Data.circuitPackage ()) env
      selection holds.2.2.1
    intro source index
    rw [Data.circuitPackage_compactRowInvocations,
      Data.compactRowInvocations_eq]
    apply List.mem_append_right
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_append_left
    unfold PiRLCCombinationInvocations.commitmentInvocations
      PiRLCCombinationInvocations.familyInvocations
    apply List.mem_flatMap.mpr
    refine ⟨source.val, List.mem_range.mpr source.isLt, ?_⟩
    simp
  · apply familyInvocationRows_of_package (Data.circuitPackage ()) env
      selection holds.2.2.1
    intro source index
    rw [Data.circuitPackage_compactRowInvocations,
      Data.compactRowInvocations_eq]
    apply List.mem_append_right
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_append_right
    unfold PiRLCCombinationInvocations.publicInputInvocations
      PiRLCCombinationInvocations.familyInvocations
    apply List.mem_flatMap.mpr
    refine ⟨source.val, List.mem_range.mpr source.isLt, ?_⟩
    simp
  · apply familyInvocationRows_of_package (Data.circuitPackage ()) env
      selection holds.2.2.1
    intro source index
    rw [Data.circuitPackage_compactRowInvocations,
      Data.compactRowInvocations_eq]
    apply List.mem_append_right
    apply List.mem_append_left
    apply List.mem_append_right
    unfold PiRLCCombinationInvocations.evalKInvocations
      PiRLCCombinationInvocations.familyInvocations
    apply List.mem_flatMap.mpr
    refine ⟨source.val, List.mem_range.mpr source.isLt, ?_⟩
    simp
  · apply familyInvocationRows_of_package (Data.circuitPackage ()) env
      selection holds.2.2.1
    intro source index
    rw [Data.circuitPackage_compactRowInvocations,
      Data.compactRowInvocations_eq]
    apply List.mem_append_right
    apply List.mem_append_right
    unfold PiRLCCombinationInvocations.evalAInvocations
      PiRLCCombinationInvocations.familyInvocations
    apply List.mem_flatMap.mpr
    refine ⟨source.val, List.mem_range.mpr source.isLt, ?_⟩
    simp

/-- Package-backed sampler rows plus the four exact combination packets
assemble the authoritative seven-child PiRLC parent specification. -/
theorem circuitPackage_implies_piRlcSpecHolds_of_combinationRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (assumptions : Formal.Assumptions relation
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
    (combinationRows : PiRLCCombinationRowsHold env) :
    Formal.SpecHolds relation
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  refine {
    inputBinding := ?_
    sampler := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_
    outputBinding := ?_
  }
  · apply InputBinding.soundness
    intro operation member
    cases member
  · simpa [PiRLCSamplerRows.samplerInterface,
      PiRLCSamplerRows.sharedInterface] using
      circuitPackage_implies_piRlcSamplerChain env holds assumptions.sampler
  · simpa [PiRLCCombinationInvocations.productionCommitmentFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.commitmentLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.commitmentFamilyRows_imply_canonical
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        env combinationRows.commitment
  · simpa [PiRLCCombinationInvocations.productionPublicInputFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.publicInputLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.publicInputFamilyRows_imply_canonical
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        env combinationRows.publicInput
  · simpa [PiRLCCombinationInvocations.productionEvalKFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalKLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.evalKFamilyRows_imply_canonical
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        env combinationRows.eval_K
  · simpa [PiRLCCombinationInvocations.productionEvalAFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalALogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.evalAFamilyRows_imply_canonical
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        env combinationRows.eval_A
  · apply OutputBinding.soundness
    intro operation member
    cases member

/-- The same package-backed parent entails the complete deterministic PiRLC
phase semantics for the caller-selected relation and Ajtai key. -/
theorem circuitPackage_implies_piRlcPhaseHolds_of_combinationRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (assumptions : Formal.Assumptions relation
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
    (combinationRows : PiRLCCombinationRowsHold env) :
    Semantics.PhaseHolds relation ajtai
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply Semantics.spec_implies_phaseHolds
  exact circuitPackage_implies_piRlcSpecHolds_of_combinationRows relation env
    holds assumptions
    combinationRows

/-- Canonical package satisfaction implies the exact seven-child PiRLC
parent specification. -/
theorem circuitPackage_implies_piRlcSpecHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (assumptions : Formal.Assumptions relation
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    Formal.SpecHolds relation
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  exact circuitPackage_implies_piRlcSpecHolds_of_combinationRows relation env
    holds assumptions (circuitPackage_implies_piRlcCombinationRows env holds)

/-- Canonical package satisfaction entails the complete deterministic PiRLC
phase semantics. -/
theorem circuitPackage_implies_piRlcPhaseHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (assumptions : Formal.Assumptions relation
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    Semantics.PhaseHolds relation ajtai
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply Semantics.spec_implies_phaseHolds
  exact circuitPackage_implies_piRlcSpecHolds relation env holds assumptions

/-- The combined package enforces the exact two pilot hashes and the complete
prior public-input marker/tail layout after the pilot-column lift. -/
theorem circuitPackage_implies_pilotHashFacts
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    let lifted := pilotEnv env
    lifted PilotSpartan.firstPublicStart = 1 ∧
      (∀ lane : Fin 49,
        lifted (PilotSpartan.firstPublicStart + 5 + lane.val) = 0) ∧
      List.ofFn (fun lane : Fin 4 =>
        lifted (PilotData.priorChain.digestStart + lane.val)) =
          Spec.Poseidon2.hash
            (NightstreamFPrime.Export.Pilot.chainInputValues
              PilotData.priorChain lifted) ∧
      List.ofFn (fun lane : Fin 4 =>
        lifted (PilotData.outputChain.digestStart + lane.val)) =
          Spec.Poseidon2.hash
            (NightstreamFPrime.Export.Pilot.chainInputValues
              PilotData.outputChain lifted) := by
  have priorStage : HashChainHolds (Data.circuitPackage ())
      Data.priorChain env := by
    apply holds.1 Data.priorChain
    rw [Data.circuitPackage_hashChains]
    simp
  have outputStage : HashChainHolds (Data.circuitPackage ())
      Data.outputChain env := by
    apply holds.1 Data.outputChain
    rw [Data.circuitPackage_hashChains]
    simp
  have priorHolds := hashChainHolds_pilotTemplate Data.priorChain env priorStage
  have outputHolds := hashChainHolds_pilotTemplate Data.outputChain env outputStage
  have priorHash := NightstreamFPrime.Export.Pilot.canonicalChainDigest_eq_hash
    Data.priorChain env (by rfl) priorHolds
  have outputHash := NightstreamFPrime.Export.Pilot.canonicalChainDigest_eq_hash
    Data.outputChain env (by rfl) outputHolds
  have pilotAssertions :=
    combinedAssertions_imply_pilotAssertions env holds.2.2.2.2
  have assertionFacts := NightstreamFPrime.Export.Pilot.canonicalAssertions_sound
    (pilotEnv env) pilotAssertions
  have priorInputs := chainInputValues_lift PilotData.priorChain env (by
    norm_num [PilotData.priorChain,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount])
  have outputInputs := chainInputValues_lift PilotData.outputChain env (by
    norm_num [PilotData.outputChain,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount])
  have priorState := chainOutputState_lift PilotData.priorChain env (by
    norm_num [PilotData.priorChain, PilotData.priorWitnessStart,
      NightstreamFPrime.Layout.PilotValues.priorWitnessStart,
      NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
      NightstreamFPrime.Layout.PilotValues.stateHashWords,
      NightstreamFPrime.Layout.PilotValues.stateHashBaseWords,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount]) (by
    norm_num [PilotData.priorChain, PilotData.priorWitnessStart,
      NightstreamFPrime.Layout.PilotValues.priorWitnessStart,
      NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
      NightstreamFPrime.Layout.PilotValues.absorbCount,
      NightstreamFPrime.Layout.PilotValues.stateHashWords,
      NightstreamFPrime.Layout.PilotValues.stateHashBaseWords,
      Spec.Poseidon2.rate,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount])
  have outputState := chainOutputState_lift PilotData.outputChain env (by
    norm_num [PilotData.outputChain, PilotData.outputWitnessStart,
      NightstreamFPrime.Layout.PilotValues.outputWitnessStart,
      NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
      NightstreamFPrime.Layout.PilotValues.hashWitnessCount,
      NightstreamFPrime.Layout.PilotValues.absorbCount,
      NightstreamFPrime.Layout.PilotValues.permutationRecipeCount,
      NightstreamFPrime.Layout.PilotValues.stateHashWords,
      NightstreamFPrime.Layout.PilotValues.stateHashBaseWords,
      Spec.Poseidon2.rate,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount]) (by
    norm_num [PilotData.outputChain, PilotData.outputWitnessStart,
      PilotData.priorWitnessStart,
      NightstreamFPrime.Layout.PilotValues.outputWitnessStart,
      NightstreamFPrime.Layout.PilotValues.priorWitnessStart,
      NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
      NightstreamFPrime.Layout.PilotValues.hashWitnessCount,
      NightstreamFPrime.Layout.PilotValues.absorbCount,
      NightstreamFPrime.Layout.PilotValues.permutationRecipeCount,
      NightstreamFPrime.Layout.PilotValues.stateHashWords,
      NightstreamFPrime.Layout.PilotValues.stateHashBaseWords,
      Spec.Poseidon2.rate,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount])
  dsimp only
  refine ⟨assertionFacts.1, assertionFacts.2.2.1, ?_, ?_⟩
  · calc
      List.ofFn (fun lane : Fin 4 =>
          pilotEnv env (PilotData.priorChain.digestStart + lane.val)) =
          List.ofFn (fun lane : Fin 4 =>
            NightstreamFPrime.Export.Pilot.chainOutputState
              PilotData.priorChain PilotData.priorChain.absorbCount
              (pilotEnv env)
              ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
        apply congrArg List.ofFn
        funext lane
        exact assertionFacts.2.1 lane
      _ = List.ofFn (fun lane : Fin 4 =>
            NightstreamFPrime.Export.Pilot.chainOutputState
              (Data.liftPilotChain PilotData.priorChain)
              PilotData.priorChain.absorbCount env
              ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
        rw [priorState]
      _ = Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            (Data.liftPilotChain PilotData.priorChain) env) := by
        simpa [Data.priorChain] using priorHash
      _ = Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.priorChain (pilotEnv env)) := by
        rw [priorInputs]
  · calc
      List.ofFn (fun lane : Fin 4 =>
          pilotEnv env (PilotData.outputChain.digestStart + lane.val)) =
          List.ofFn (fun lane : Fin 4 =>
            NightstreamFPrime.Export.Pilot.chainOutputState
              PilotData.outputChain PilotData.outputChain.absorbCount
              (pilotEnv env)
              ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
        apply congrArg List.ofFn
        funext lane
        exact assertionFacts.2.2.2 lane
      _ = List.ofFn (fun lane : Fin 4 =>
            NightstreamFPrime.Export.Pilot.chainOutputState
              (Data.liftPilotChain PilotData.outputChain)
              PilotData.outputChain.absorbCount env
              ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
        rw [outputState]
      _ = Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            (Data.liftPilotChain PilotData.outputChain) env) := by
        simpa [Data.outputChain] using outputHash
      _ = Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.outputChain (pilotEnv env)) := by
        rw [outputInputs]

/-- Satisfaction of the combined pilot+PiCCS package implies both logical
pilot builder specifications. -/
theorem circuitPackage_implies_pilotSpec
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    Lifecycle.Pilot.SpecHolds PilotProduction.interface
      PilotProduction.witnessOffset
      (PilotSpartan.pullback (pilotEnv env)) := by
  exact NightstreamFPrime.Export.Pilot.hashFacts_imply_spec
    (pilotEnv env) (circuitPackage_implies_pilotHashFacts env holds)

/-- The combined package rows, together with the fixed protocol ABI
representation below the pilot witness boundary, imply both concrete recursive
hash slots. -/
theorem circuitPackage_implies_recursive_hash_slots
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * PaperAlgebra.publicRingColumns <=
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (priorFixed : PilotProduction.FixedPreimage
      (priorHashPreimage (setup relation ajtai vk) input))
    (outputFixed : PilotProduction.FixedPreimage
      (nextHashPreimage (setup relation ajtai vk) input output))
    (digestFixed : output.x.length = PilotProduction.digestWords)
    (env : Env)
    (agrees : PilotProduction.AgreesBelow
      (PilotSpartan.pullback (pilotEnv env))
      (PilotProduction.protocolEnv
        (priorHashPreimage (setup relation ajtai vk) input)
        ((machine publicFits F).freshPublic input.fresh)
        (nextHashPreimage (setup relation ajtai vk) input output)
        output.x priorFixed outputFixed digestFixed)
      PilotProduction.witnessOffset)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    (machine publicFits F).freshPublic input.fresh =
        (machine publicFits F).encodeInstance
          ((machine publicFits F).hash
            (priorHashPreimage (setup relation ajtai vk) input)) ∧
      OutputHolds (setup relation ajtai vk) (machine publicFits F)
        input output := by
  exact NightstreamFPrime.Export.Pilot.hashFacts_imply_recursive_hash_slots
    relation ajtai vk F input output priorFixed outputFixed digestFixed
      (pilotEnv env) agrees (circuitPackage_implies_pilotHashFacts env holds)

/-- Package satisfaction supplies all four exact PiCCS transcript predicates.
The rewrite below changes only the package wrapper; both packages use the one
canonical Poseidon2 permutation template. -/
theorem circuitPackage_implies_piCcsTranscriptSpecs
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    PiCCSInvocations.TranscriptSpecs Data.logicalWidth Data.publicFits env := by
  apply PiCCSInvocations.invocations_imply_transcriptSpecs Data.logicalWidth
    Data.publicFits relation env
  intro invocation member
  have packageInvocation : PermutationInvocationHolds
      (Data.circuitPackage ()) invocation env := by
    apply holds.2.1 invocation
    rw [Data.circuitPackage_permutationInvocations,
      Data.components_permutationInvocations,
      Data.permutationInvocations_eq]
    exact List.mem_append_left _ member
  unfold PermutationInvocationHolds at packageInvocation ⊢
  rw [Data.circuitPackage_permutation] at packageInvocation
  simpa [PilotData.circuitPackage] using packageInvocation

/-- Satisfaction of the one emitted package covers all twelve children of the
canonical PiCCS `FormalCircuit`. The parent proof owns only packet projection
and child wiring. -/
theorem circuitPackage_implies_piCcsSpecHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds relation
      (PiCCSInvocations.parentInterface Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  let parent := PiCCSInvocations.parentInterface Data.logicalWidth Data.publicFits
  have assumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation parent
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
        Data.logicalWidth Data.publicFits)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  have transcripts := circuitPackage_implies_piCcsTranscriptSpecs relation env
    holds
  have arithmeticRows := circuitPackage_implies_piCcsArithmeticRows env holds
  have packets := PiCCSArithmetic.arithmeticRows_imply_packetHolds
    Data.logicalWidth Data.publicFits env arithmeticRows
  have arithmetic := PiCCSArithmetic.packetHolds_imply_arithmeticSpecs
    Data.logicalWidth Data.publicFits relation env assumptions packets
  refine {
    statementBinding := ?_
    statementAbsorption := ?_
    challenge := ?_
    roundTranscript := ?_
    initialClaim := ?_
    sumcheck := ?_
    eval_K := ?_
    eval_A := ?_
    ccs := ?_
    norm := ?_
    finalIdentity := ?_
    outputBinding := ?_ }
  · exact arithmetic.statementBinding_parent
  · exact transcripts.statementAbsorption_parent
  · exact transcripts.challengeDerivation_parent
  · exact transcripts.roundTranscript_parent
  · exact arithmetic.initialClaim_parent
  · exact arithmetic.sumcheck_parent
  · exact arithmetic.evalK_parent
  · exact arithmetic.evalA_parent
  · exact arithmetic.ccs_parent
  · exact arithmetic.norm_parent
  · exact arithmetic.finalIdentity_parent
  · exact transcripts.outputBinding_parent relation

/-- The production verifier selects the four public context words by
recomputing them from its static authority. -/
def SelectedVerifierContext
    (authority : VerifierContext.Authority) (env : Env) : Prop :=
  ∀ lane : Fin 4,
    env (NightstreamFPrime.Layout.Stage1.Spartan.expectedContextPublicStart +
      lane.val) = (VerifierContext.digest authority).getD lane.val 0

/-- Package satisfaction binds both canonical state preimages to the exact
verifier-selected context digest. The digest is a public verifier input; the
prover cannot choose a different state context that still satisfies these
rows. -/
theorem circuitPackage_implies_selectedVerifierContext
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (authority : VerifierContext.Authority)
    (env : Env)
    (selected : SelectedVerifierContext authority env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    (∀ lane : Fin 4,
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.priorStateWord
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart +
          lane.val)).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
        (VerifierContext.digest authority).getD lane.val 0) ∧
    (∀ lane : Fin 4,
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.outputStateWord
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart +
          lane.val)).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
        (VerifierContext.digest authority).getD lane.val 0) := by
  have specification := circuitPackage_implies_piCcsSpecHolds relation env holds
  constructor
  · intro lane
    have context := specification.statementBinding.state.priorContext lane
    change
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.priorStateWord
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart +
          lane.val)).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
        (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContext lane).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) at context
    rw [context]
    change env (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart +
        lane.val)) = _
    rw [NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_expectedContext]
    exact selected lane
  · intro lane
    have context := specification.statementBinding.state.outputContext lane
    change
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.outputStateWord
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart +
          lane.val)).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
        (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContext lane).eval
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) at context
    rw [context]
    change env (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart +
        lane.val)) = _
    rw [NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_expectedContext]
    exact selected lane

/-- Authoritative emitted-package soundness edge for the exact SuperNeo v1_1
PiCCS phase. -/
theorem circuitPackage_implies_piCcsPhaseHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (PiCCSInvocations.parentInterface Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) template := by
  apply NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.spec_implies_phaseHolds
    relation ajtai
  exact circuitPackage_implies_piCcsSpecHolds relation env holds

private theorem hashChain_rows :
    Data.priorChain.witnessLength + Data.outputChain.witnessLength =
      12574080 := by
  have pilot := NightstreamFPrime.Export.Pilot.circuitPackage_row_coverage
  change PilotData.priorChain.witnessLength +
      PilotData.outputChain.witnessLength + 58 = 12574138 at pilot
  change PilotData.priorChain.witnessLength +
      PilotData.outputChain.witnessLength = 12574080
  omega

theorem circuitPackage_compactRowCount :
    (Data.components ()).toCircuitPackage.compactRowCount = 7489673 := by
  unfold CircuitPackage.compactRowCount
  rw [Data.Components.toCircuitPackage_compactRowTemplates,
    Data.Components.toCircuitPackage_compactRowInvocations,
    Data.compactRowTemplates_eq, Data.compactRowInvocations_eq]
  change compactRowCountFor PiRLCFirst54Invocations.packageTemplates
    (PiRLCFirst54Invocations.invocations ++
      PiRLCCombinationInvocations.invocations) = 7489673
  rw [compactRowCountFor_append,
    PiRLCFirst54Invocations.compactRowCount,
    piRlcCombination_compactRowCount]

/-- All template and ordinary row families account for the exact physical
row count. Exact row-index ordering is proved by the phase compilers and is
also checked by the strict Rust loader. -/
theorem circuitPackage_row_coverage
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits) :
    ((Data.components ()).toCircuitPackage.hashChains.map
        (fun chain => chain.witnessLength)).sum +
      (Data.components ()).toCircuitPackage.permutationInvocations.length *
        (Data.components ()).toCircuitPackage.permutation.rows.length +
      (Data.components ()).toCircuitPackage.compactRowCount +
      (Data.components ()).toCircuitPackage.witnessInstructions.length +
      (Data.components ()).toCircuitPackage.assertionRows.length =
        (Data.components ()).toCircuitPackage.layout.rowCount := by
  apply Data.Components.rowCoverage (Data.components ())
  · rw [Data.components_arithmeticRows, Data.arithmeticRows_eq,
      List.length_append, List.length_append,
      PiCCSArithmetic.arithmeticRows_length Data.logicalWidth
        Data.publicFits relation,
      PiRLCSamplerOrdinaryRows.rows_length,
      PiDECArithmetic.Plan.rows_length,
      PiDECArithmetic.canonicalPlan_rowCount relation]
  · rw [Data.components_permutationInvocations,
      Data.permutationInvocations_eq, List.length_append,
      PiCCSInvocations.invocations_length Data.logicalWidth Data.publicFits,
      PiRLCSamplerInvocations.invocations_length]
  · exact NightstreamFPrime.Export.Pilot.templateRows_length
  · rw [liftPilotRows_length,
      NightstreamFPrime.Export.Pilot.assertionRows_length]
  · exact hashChain_rows
  · exact circuitPackage_compactRowCount

end NightstreamFPrime.Export.Stage1.Package
