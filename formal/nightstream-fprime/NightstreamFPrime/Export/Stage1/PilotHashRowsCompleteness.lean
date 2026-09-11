import NightstreamFPrime.Export.Stage1.PilotPoseidonCompleteness
import NightstreamFPrime.Export.Stage1.PermutationCompilerTransport
import NightstreamFPrime.Layout.Poseidon2.HashInvocationRows
import NightstreamFPrime.Layout.Stage1.SpartanRows

/-!
Owns the pilot compiler-row connection from the actual completed Spartan
prefix to the existing package hash-chain predicates and compact pilot rows.
Every input and local value comes from the same physical target.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PilotHashRowsCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem canonical_input (value : ColumnRef → F) (lane : Fin 8) :
    Pilot.canonicalTemplateEnv value lane.val = value (.input lane.val) := by
  unfold Pilot.canonicalTemplateEnv
  rw [show PilotData.columnRef lane.val = .input lane.val by
    simp [PilotData.columnRef, lane.isLt]]
  rfl

private theorem canonical_local (value : ColumnRef → F) (index : Nat) :
    Pilot.canonicalTemplateEnv value (8 + index) = value (.local index) := by
  unfold Pilot.canonicalTemplateEnv
  rw [show PilotData.columnRef (8 + index) = .local index by simp [PilotData.columnRef]]
  rfl

private theorem chunk_word (input : List Expr) (invocation lane : Nat)
    (bound : invocation < (Hash.inputChunks input).length) :
    ((Hash.inputChunks input).getD invocation []).getD lane 0 =
      if lane < Poseidon2.rate ∧ invocation * Poseidon2.rate + lane < input.length then
        input.getD (invocation * Poseidon2.rate + lane) 0 else 0 := by
  have chunk : (Hash.inputChunks input).getD invocation [] =
      (input.drop (invocation * Poseidon2.rate)).take Poseidon2.rate := by
    rw [List.getD_eq_getElem (l := Hash.inputChunks input) (d := []) bound]
    simp only [Hash.inputChunks, List.getElem_map, List.getElem_range]
  rw [chunk]
  by_cases present : lane < Poseidon2.rate ∧ invocation * Poseidon2.rate + lane < input.length
  · have localBound : lane < ((input.drop (invocation * Poseidon2.rate)).take Poseidon2.rate).length := by
      simp only [List.length_take, List.length_drop]
      omega
    rw [if_pos present, List.getD_eq_getElem (l :=
      (input.drop (invocation * Poseidon2.rate)).take Poseidon2.rate) (d := 0) localBound,
      List.getElem_take, List.getElem_drop,
      List.getD_eq_getElem (l := input) (d := 0) present.2]
  · have outside : ((input.drop (invocation * Poseidon2.rate)).take Poseidon2.rate).length ≤ lane := by
      simp only [List.length_take, List.length_drop]
      omega
    rw [if_neg present, List.getD_eq_default (l :=
      (input.drop (invocation * Poseidon2.rate)).take Poseidon2.rate) (d := 0) outside]

private theorem chain_input
    (chain : HashChain) (target source : Env) (start : Nat) (input : List Expr)
    (inputLength : input.length = chain.inputLength)
    (count : (Hash.inputChunks input).length = chain.absorbCount)
    (inputValues : ∀ index, index < input.length →
      (input.getD index 0).eval source = target (chain.inputStart + index))
    (localValues : ∀ index, index < (chain.absorbCount + 1) * 592 →
      target (chain.witnessStart + index) = source (start + index))
    (invocation : Nat) (bound : invocation ≤ chain.absorbCount) (lane : Fin 8) :
    (invocationInput (PilotData.circuitPackage ()) chain invocation lane.val).eval target =
      ((if invocation < (Hash.inputChunks input).length then
          Hash.absorbE
            (if invocation = 0 then Hash.zeroE else
              Permutation.freshState (start + (invocation - 1) * 592 + 584))
            ((Hash.inputChunks input).getD invocation [])
        else Hash.padE
          (if invocation = 0 then Hash.zeroE else
            Permutation.freshState (start + (invocation - 1) * 592 + 584))) lane).eval source := by
  have localCount : (PilotData.circuitPackage ()).permutation.localColumnCount = 592 := rfl
  have outputStart : (PilotData.circuitPackage ()).permutation.outputLocalStart = 584 := rfl
  have rate : (PilotData.circuitPackage ()).poseidon.rate = Poseidon2.rate := rfl
  have previous :
      (if invocation = 0 then R1CS.LinearCombination.zero else
        R1CS.LinearCombination.ofVar
          (chain.witnessStart + (invocation - 1) * 592 + 584 + lane.val)).eval target =
      ((if invocation = 0 then Hash.zeroE else
        Permutation.freshState (start + (invocation - 1) * 592 + 584)) lane).eval source := by
    by_cases first : invocation = 0
    · simp [first, Hash.zeroE, R1CS.LinearCombination.eval_zero]
    · have localBound : (invocation - 1) * 592 + 584 + lane.val <
          (chain.absorbCount + 1) * 592 := by
        have laneBound := lane.isLt
        omega
      simpa only [if_neg first, R1CS.LinearCombination.eval_ofVar,
        Permutation.freshState, Expr.eval_var, Nat.add_assoc] using
        localValues ((invocation - 1) * 592 + 584 + lane.val) localBound
  by_cases absorbing : invocation < chain.absorbCount
  · have sourceBound : invocation < (Hash.inputChunks input).length := by simpa only [count] using absorbing
    have word := chunk_word input invocation lane.val sourceBound
    rw [count]
    simp only [invocationInput, localCount, outputStart, rate, if_pos absorbing]
    change _ = (Hash.absorbE _ _ lane).eval source
    rw [Hash.absorbE, Expr.eval_hadd, word]
    by_cases present : lane.val < Poseidon2.rate ∧ invocation * Poseidon2.rate + lane.val < input.length
    · have chainPresent : lane.val < Poseidon2.rate ∧
          invocation * Poseidon2.rate + lane.val < chain.inputLength := by
        simpa only [inputLength] using present
      rw [if_pos present, if_pos chainPresent, R1CS.LinearCombination.eval_add,
        R1CS.LinearCombination.eval_ofVar, previous, inputValues _ present.2]
    · have chainAbsent : ¬(lane.val < Poseidon2.rate ∧
          invocation * Poseidon2.rate + lane.val < chain.inputLength) := by
        simpa only [inputLength] using present
      rw [if_neg present, if_neg chainAbsent, previous]
      simp
  · rw [count]
    simp only [invocationInput, localCount, outputStart, rate, if_neg absorbing]
    change _ = (Hash.padE _ lane).eval source
    unfold Hash.padE
    by_cases zero : lane.val = 0
    · rw [if_pos zero, if_pos zero, R1CS.LinearCombination.eval_add,
        R1CS.LinearCombination.eval_one, Expr.eval_hadd, previous]
      rfl
    · rw [if_neg zero, if_neg zero, previous]

private theorem hashChain_of_sourceRows
    (chain : HashChain) (target source : Env) (start : Nat) (input : List Expr)
    (inputLength : input.length = chain.inputLength)
    (count : (Hash.inputChunks input).length = chain.absorbCount)
    (inputValues : ∀ index, index < input.length →
      (input.getD index 0).eval source = target (chain.inputStart + index))
    (localValues : ∀ index, index < (chain.absorbCount + 1) * 592 →
      target (chain.witnessStart + index) = source (start + index))
    (rows : ConstraintsHold source (recipeConstraints start (Hash.compile start input).recipes)) :
    HashChainHolds (PilotData.circuitPackage ()) chain target := by
  intro invocation bound
  have sourceRows := NightstreamFPrime.Layout.Poseidon2.HashInvocationRows.hash_rows source start input rows invocation (by
    simpa only [count] using bound)
  apply (Pilot.canonicalTemplateInvocation_iff chain invocation target).mpr
  unfold PilotData.canonicalRows
  apply R1CS.lowerConstraints_complete_of_noFresh
  · apply R1CS.recipeConstraints_noFresh
    exact NightstreamFPrime.Layout.Poseidon2.compile_schedule_direct 8 PilotData.canonicalState Pilot.canonicalState_affine
  · change ConstraintsHold _ (recipeConstraints 8
      (Permutation.compile 8 PilotData.canonicalState Permutation.schedule).recipes)
    refine PermutationCompilerTransport.compileConstraintsHold_of_transport _ source
      8 (start + invocation * 592) PilotData.canonicalState _ Permutation.schedule ?_ ?_ sourceRows
    · funext lane
      change Pilot.canonicalTemplateEnv _ lane.val = _
      rw [canonical_input]
      exact chain_input chain target source start input inputLength count
        inputValues localValues invocation bound lane
    · intro index indexBound
      change Pilot.canonicalTemplateEnv _ (8 + index) = _
      rw [canonical_local]
      change target (chain.witnessStart + invocation * 592 + index) =
        source (start + invocation * 592 + index)
      have scheduleWidth : Permutation.scheduleSize Permutation.schedule = 592 :=
        (Permutation.compile_recipes_length 8 PilotData.canonicalState
          Permutation.schedule).symm.trans
            (Permutation.compile_schedule_recipe_count 8 PilotData.canonicalState)
      have width : index < 592 := by
        simpa only [scheduleWidth] using indexBound
      simpa only [Nat.add_assoc] using localValues (invocation * 592 + index) (by omega)

private theorem prior_input_map (index : Fin PilotProduction.stateHashWords) :
    Spartan.sourceToSpartan (PilotProduction.priorPreimageStart + index.val) =
      Data.priorChain.inputStart + index.val := by
  have bound : index.val < 49393 := by simpa only [PilotProduction.stateHashWords_eq] using index.isLt
  change Spartan.sourceToSpartan (0 + index.val) = Spartan.liftPilotColumn 0 + index.val
  unfold Spartan.sourceToSpartan
  rw [if_pos (by change 0 + index.val < 14722512; omega)]
  unfold PilotSpartan.sourceToSpartan
  rw [if_pos (by change 0 + index.val < 49393; omega)]
  exact Spartan.liftPilotColumn_add_of_input 0 index.val (by
    change 0 + index.val < 98786
    omega)

private theorem output_input_map (index : Fin PilotProduction.stateHashWords) :
    Spartan.sourceToSpartan (PilotProduction.outputPreimageStart + index.val) =
      Data.outputChain.inputStart + index.val := by
  have bound : index.val < 49393 := by simpa only [PilotProduction.stateHashWords_eq] using index.isLt
  change Spartan.sourceToSpartan (49663 + index.val) = Spartan.liftPilotColumn 49393 + index.val
  unfold Spartan.sourceToSpartan
  rw [if_pos (by change 49663 + index.val < 14722512; omega)]
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by change ¬49663 + index.val < 49393; omega)]
  rw [if_neg (by change ¬49663 + index.val < 49663; omega)]
  rw [if_pos (by change 49663 + index.val < 99056; omega)]
  change Spartan.liftPilotColumn (49393 + ((49663 + index.val) - 49663)) = _
  rw [Nat.add_sub_cancel_left]
  exact Spartan.liftPilotColumn_add_of_input 49393 index.val (by
    change 49393 + index.val < 98786
    omega)

private theorem pilot_witness_map (start index : Nat)
    (afterInputs : PilotProduction.witnessOffset ≤ start)
    (beforeEnd : start + index < Spartan.pilotSourceColumnCount) :
    Spartan.sourceToSpartan (start + index) =
      Spartan.liftPilotColumn (PilotSpartan.witnessPrivateStart +
        (start - PilotProduction.witnessOffset + index)) := by
  unfold Spartan.sourceToSpartan
  rw [if_pos beforeEnd]
  have position : start + index = PilotProduction.witnessOffset +
      (start - PilotProduction.witnessOffset + index) := by omega
  rw [position, PilotSpartan.sourceToSpartan_pilotWitness]

private theorem local_values (chain : HashChain) (start : Nat)
    (startEq : chain.witnessStart = PilotSpartan.witnessPrivateStart +
      (start - PilotProduction.witnessOffset))
    (afterInputs : PilotProduction.witnessOffset ≤ start)
    (sourceEnd : start + (chain.absorbCount + 1) * 592 ≤ Spartan.pilotSourceColumnCount)
    (privateStart : Spartan.pilotInputPrivateColumnCount ≤ chain.witnessStart)
    (privateEnd : chain.witnessStart + (chain.absorbCount + 1) * 592 ≤ Spartan.pilotPrivateColumnCount)
    (target : Env) (index : Nat) (bound : index < (chain.absorbCount + 1) * 592) :
    target ((Data.liftPilotChain chain).witnessStart + index) =
      Spartan.pullback target (start + index) := by
  change target (Spartan.liftPilotColumn chain.witnessStart + index) =
    target (Spartan.sourceToSpartan (start + index))
  apply congrArg target
  rw [pilot_witness_map start index afterInputs (by omega)]
  have lifted := Spartan.liftPilotColumn_add_of_private chain.witnessStart index
    privateStart (by omega)
  simpa only [startEq, Nat.add_assoc] using lifted.symm

private theorem prior_local_values (target : Env) (index : Nat)
    (bound : index < (Data.priorChain.absorbCount + 1) * 592) :
    target (Data.priorChain.witnessStart + index) =
      Spartan.pullback target (PilotProduction.witnessOffset + index) := by
  apply local_values PilotData.priorChain PilotProduction.witnessOffset
    (target := target) (index := index) (bound := bound)
  · rw [Nat.sub_self, Nat.add_zero]
    rfl
  · exact Nat.le_refl _
  · rw [PilotProduction.witnessOffset_eq]
    change 99060 + (12349 + 1) * 592 ≤ 14722512
    norm_num
  · change 98786 ≤ 98786
    exact Nat.le_refl _
  · change 98786 + (12349 + 1) * 592 ≤ 14722238
    norm_num

private theorem output_local_values (target : Env) (index : Nat)
    (bound : index < (Data.outputChain.absorbCount + 1) * 592) :
    target (Data.outputChain.witnessStart + index) =
      Spartan.pullback target (PilotProduction.lifecycleOutputOffset + index) := by
  apply local_values PilotData.outputChain PilotProduction.lifecycleOutputOffset
    (target := target) (index := index) (bound := bound)
  · rw [PilotProduction.lifecycleOutputOffset_eq, PilotProduction.witnessOffset_eq]
    change 7410250 = 98786 + (7410524 - 99060)
    norm_num
  · rw [PilotProduction.lifecycleOutputOffset_eq, PilotProduction.witnessOffset_eq]
    norm_num
  · rw [PilotProduction.lifecycleOutputOffset_eq]
    change 7410524 + (12349 + 1) * 592 ≤ 14722512
    norm_num
  · change 98786 ≤ 7410250
    norm_num
  · change 7410250 + (12349 + 1) * 592 ≤ 14722238
    norm_num

private theorem variable_word (env : Env) (start count index : Nat)
    (bound : index < count) :
    ((PilotProduction.variableExprs start count).getD index 0).eval env = env (start + index) := by
  have selected := PriorStateHash.ofFn_getD
    (fun position : Fin count => Expr.var (start + position.val)) ⟨index, bound⟩ (0 : Expr)
  change ((List.ofFn fun position : Fin count => Expr.var (start + position.val)).getD index 0).eval env = _
  rw [selected]
  rfl

private theorem stage_template (chain : HashChain) (target : Env)
    (held : HashChainHolds (PilotData.circuitPackage ()) chain target) :
    HashChainHolds (Data.circuitPackage ()) chain target := by
  intro invocation bound row member
  have pilotMember := member
  rw [Data.circuitPackage_permutation] at pilotMember
  have rowHolds := held invocation bound row pilotMember
  change (instantiateRow (Data.circuitPackage ()) chain invocation row).Holds target
  exact rowHolds

private theorem pilot_hashRows (target : Env)
    (rows : NightstreamFPrime.Layout.Pilot.PhysicalHolds PilotProduction.interface
      PilotProduction.witnessOffset (Spartan.pullback target)) :
    ConstraintsHold (Spartan.pullback target)
      (recipeConstraints PilotProduction.witnessOffset
        (Hash.compile PilotProduction.witnessOffset
          (PilotProduction.priorPreimage PilotProduction.witnessOffset)).recipes) ∧
    ConstraintsHold (Spartan.pullback target)
      (recipeConstraints PilotProduction.lifecycleOutputOffset
        (Hash.compile PilotProduction.lifecycleOutputOffset
          (PilotProduction.outputPreimage PilotProduction.lifecycleOutputOffset)).recipes) := by
  have logical := R1CS.lowerConstraints_sound (Spartan.pullback target)
    (NightstreamFPrime.Layout.Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset)
    (NightstreamFPrime.Layout.Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset) rows
  have phases := (constraintsHold_append (Spartan.pullback target) _ _).mp logical
  constructor
  · intro expression member
    have hashMember : expression ∈ flatConstraints
        [PriorStateHash.hashOp PilotProduction.priorInterface PilotProduction.witnessOffset] := by
      rw [PriorStateHash.hashOp_flatConstraints_eq, PilotProduction.priorInterface_preimage_apply]
      exact member
    apply phases.1 expression
    rw [NightstreamFPrime.Layout.Pilot.priorConstraints_eq]
    exact List.mem_append_left _ (List.mem_append_left _ hashMember)
  · have output := phases.2
    rw [NightstreamFPrime.Layout.Pilot.outputConstraints_eq] at output
    have recipes := NightstreamFPrime.Layout.Poseidon2.HashInvocationRows.recipeRows_of_hashConstraints
      (OutputHash.hashInterface PilotProduction.outputInterface)
      (NightstreamFPrime.Layout.Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset)
      (Spartan.pullback target) output
    rw [PilotProduction.outputHashInterface_input,
      ← PilotProduction.lifecycleOutputOffset_matches_layout] at recipes
    exact recipes

/-- Actual physical pilot rows enforce both package hash-chain predicates.
The input and local equalities are proved from the existing Spartan maps. -/
theorem hashChains_of_pilotRows (target : Env)
    (rows : NightstreamFPrime.Layout.Pilot.PhysicalHolds PilotProduction.interface
      PilotProduction.witnessOffset (Spartan.pullback target)) :
    HashChainHolds (Data.circuitPackage ()) Data.priorChain target ∧
      HashChainHolds (Data.circuitPackage ()) Data.outputChain target := by
  have packets := pilot_hashRows target rows
  constructor
  · apply stage_template
    apply hashChain_of_sourceRows Data.priorChain target (Spartan.pullback target)
      PilotProduction.witnessOffset (PilotProduction.priorPreimage PilotProduction.witnessOffset)
    · simp only [PilotProduction.priorPreimage, PilotProduction.variableExprs_length]
      rfl
    · rw [PilotProduction.priorPreimage_chunkCount]
      rfl
    · intro index bound
      have width : index < PilotProduction.stateHashWords := by
        simpa only [PilotProduction.priorPreimage, PilotProduction.variableExprs_length] using bound
      change ((PilotProduction.variableExprs PilotProduction.priorPreimageStart
        PilotProduction.stateHashWords).getD index 0).eval (Spartan.pullback target) = _
      rw [variable_word (Spartan.pullback target) _ _ index width]
      exact congrArg target (prior_input_map ⟨index, width⟩)
    · exact prior_local_values target
    · exact packets.1
  · apply stage_template
    apply hashChain_of_sourceRows Data.outputChain target (Spartan.pullback target)
      PilotProduction.lifecycleOutputOffset (PilotProduction.outputPreimage PilotProduction.lifecycleOutputOffset)
    · simp only [PilotProduction.outputPreimage, PilotProduction.variableExprs_length]
      rfl
    · rw [PilotProduction.outputPreimage_chunkCount]
      rfl
    · intro index bound
      have width : index < PilotProduction.stateHashWords := by
        simpa only [PilotProduction.outputPreimage, PilotProduction.variableExprs_length] using bound
      change ((PilotProduction.variableExprs PilotProduction.outputPreimageStart
        PilotProduction.stateHashWords).getD index 0).eval (Spartan.pullback target) = _
      rw [variable_word (Spartan.pullback target) _ _ index width]
      exact congrArg target (output_input_map ⟨index, width⟩)
    · exact output_local_values target
    · exact packets.2

/-- The complete physical prefix supplies the pilot assumptions of the
checked canonical assignment consumer. No separate hash-row premise remains. -/
theorem rowsZero_of_spartanRows
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (application : Lifecycle.Stage1.Application.Program) (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (rows : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    (PilotPoseidonPlan.plan (DirectPrefixPlan.pilotGeometry
      (PerApplicationCanonicalEncodes.poseidonGeometry application))).RowsZero
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application
        (PerApplicationSourceAssignment.ofCompleted application target applicationPrivate)).assignment := by
  have complete := (Spartan.remappedRows_hold relation target).mp rows
  have beforeRunning := (PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation _).mp complete
  have beforeD := (PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation _).mp beforeRunning.1
  have beforeR := (PilotPiCCSPiRLC.physicalHolds_iff relation _).mp beforeD.1
  have beforeC := (PilotPiCCS.physicalHolds_iff relation _).mp beforeR.1
  have hashes := hashChains_of_pilotRows target beforeC.1
  exact PilotPoseidonCompleteness.rowsZero_of_completed_hashRows application target applicationPrivate
    hashes.1 hashes.2

end NightstreamFPrime.Export.Stage1.PilotHashRowsCompleteness
