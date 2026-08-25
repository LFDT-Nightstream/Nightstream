import NightstreamFPrime.Export.Pilot
import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns structural proofs for the one Stage 1 pilot + PiCCS package.

No theorem evaluates the package artifact. Counts follow from the pilot
proofs, the compact invocation compiler, the ordinary-row classifier, and the
proved PiCCS leaf footprints.
-/

namespace NightstreamFPrime.Export.Stage1.Package

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
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
    (Data.circuitPackage ()).permutationInvocations.length = 7460 := by
  rw [Data.circuitPackage_permutationInvocations,
    Data.components_permutationInvocations,
    Data.permutationInvocations_eq]
  exact PiCCSInvocations.invocations_length Data.logicalWidth Data.publicFits

theorem proofInputStart_eq : Data.proofInputStart = 84950 := by
  rfl

theorem witnessStart_eq : Data.witnessStart = 113962 := by
  rfl

theorem witnessLength_eq : Data.witnessLength = 17755558 := by
  rfl

theorem circuitPackage_layout_values :
    let layout := (Data.circuitPackage ()).layout
    layout.rowCount = 17755828 ∧
      layout.privateColumnCount = 17869520 ∧
      layout.constantColumn = 17869520 ∧
      layout.publicColumnCount = 62 ∧
      layout.totalColumnCount = 17869583 := by
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
      (Rows.assertionRows (Data.arithmeticRows ())).length = 765370 := by
  calc
    _ = (Data.arithmeticRows ()).length :=
      Rows.witnessInstructions_length_add_assertionRows_length _
    _ = 765370 := by
      rw [Data.arithmeticRows_eq]
      exact PiCCSArithmetic.arithmeticRows_length Data.logicalWidth
        Data.publicFits relation

theorem circuitPackage_ordinary_rows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits) :
    (Data.components ()).toCircuitPackage.witnessInstructions.length +
      (Data.components ()).toCircuitPackage.assertionRows.length = 765428 := by
  calc
    _ = (Data.liftPilotRows (PilotData.assertionRows ())).length +
        (Data.components ()).arithmeticRows.length :=
      Data.Components.ordinaryRows_length (Data.components ())
    _ = 58 + (Data.arithmeticRows ()).length := by
      rw [liftPilotRows_length,
        NightstreamFPrime.Export.Pilot.assertionRows_length,
        Data.components_arithmeticRows]
    _ = 58 + 765370 := by
      rw [Data.arithmeticRows_eq,
        PiCCSArithmetic.arithmeticRows_length Data.logicalWidth
          Data.publicFits relation]
    _ = 765428 := by norm_num

/-- Construct all 7,460 compact PiCCS Poseidon2 invocations in their proved
private intervals. Every non-invocation column, including the relocated public
suffix and the arithmetic gaps, keeps its input value. -/
theorem complete_piCcsInvocations
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env) :
    ∃ completed,
      Invocations.AgreesOutsideInvocations env completed
          (Data.circuitPackage ()).permutationInvocations ∧
        ∀ invocation ∈ (Data.circuitPackage ()).permutationInvocations,
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
  · rw [Data.circuitPackage_permutationInvocations,
      Data.components_permutationInvocations, Data.permutationInvocations_eq]
    exact exactAgrees
  · intro invocation member
    have sourceMember : invocation ∈
        PiCCSInvocations.invocations Data.logicalWidth Data.publicFits := by
      rw [Data.circuitPackage_permutationInvocations,
        Data.components_permutationInvocations,
        Data.permutationInvocations_eq] at member
      exact member
    have pilotHolds := invocationHolds invocation sourceMember
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
    apply holds.2.2.1 instruction
    rw [Data.circuitPackage_witnessInstructions]
    unfold Data.Components.witnessInstructions
    rw [Rows.witnessInstructionsTR_eq, Data.components_arithmeticRows]
    exact member
  · intro assertion member
    apply holds.2.2.2 assertion
    rw [Data.circuitPackage_assertionRows]
    unfold Data.Components.assertionRows
      Data.Components.arithmeticAssertionRows
    apply List.mem_append_right
    rw [Rows.assertionRowsTR_eq, Data.components_arithmeticRows]
    exact member

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
  have pilotAssertions := combinedAssertions_imply_pilotAssertions env holds.2.2.2
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
    exact member
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
  have arithmeticRows := circuitPackage_implies_arithmeticRows env holds
  rw [Data.arithmeticRows_eq] at arithmeticRows
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
      (Data.components ()).toCircuitPackage.witnessInstructions.length +
      (Data.components ()).toCircuitPackage.assertionRows.length =
        (Data.components ()).toCircuitPackage.layout.rowCount := by
  apply Data.Components.rowCoverage (Data.components ())
  · rw [Data.components_arithmeticRows, Data.arithmeticRows_eq]
    exact PiCCSArithmetic.arithmeticRows_length Data.logicalWidth
      Data.publicFits relation
  · rw [Data.components_permutationInvocations,
      Data.permutationInvocations_eq]
    exact PiCCSInvocations.invocations_length Data.logicalWidth Data.publicFits
  · exact NightstreamFPrime.Export.Pilot.templateRows_length
  · rw [liftPilotRows_length,
      NightstreamFPrime.Export.Pilot.assertionRows_length]
  · exact hashChain_rows

end NightstreamFPrime.Export.Stage1.Package
