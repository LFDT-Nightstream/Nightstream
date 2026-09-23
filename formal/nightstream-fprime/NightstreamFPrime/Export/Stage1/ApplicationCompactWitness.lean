import NightstreamFPrime.Export.Stage1.ApplicationPoseidonSoundness
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment

/-! Construct only the three variable application permutations in their retained source positions. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationCompactWitness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open Poseidon2HashChainV1Package (application)

/-- Evaluate each variable permutation once and share its completed state. -/
def sources (prior message : Fin 4 → F) : Fin 3 → Env :=
  let first := PoseidonCompactWitness.source (Poseidon2HashChainCompactWitness.firstInput prior)
  let firstOutput := Gadgets.Poseidon2.Layer.evalState first
    (Gadgets.Poseidon2.Permutation.scheduleOutput PoseidonScheduleTrace.inputCount)
  let second := PoseidonCompactWitness.source (fun lane =>
    firstOutput lane + Poseidon2HashChainCompactWitness.blockValue message lane)
  let secondOutput := Gadgets.Poseidon2.Layer.evalState second
    (Gadgets.Poseidon2.Permutation.scheduleOutput PoseidonScheduleTrace.inputCount)
  let third := PoseidonCompactWitness.source (fun lane =>
    if lane.val = 0 then secondOutput lane + 1 else secondOutput lane)
  fun invocation => if invocation.val = 0 then first else if invocation.val = 1 then second else third

theorem sources_eq (prior message : Fin 4 → F) (invocation : Fin 3) :
    sources prior message invocation = PoseidonCompactWitness.source
      (Poseidon2HashChainCompactWitness.permutationInput prior message invocation) := by
  fin_cases invocation <;> simp [sources, Poseidon2HashChainCompactWitness.permutationInput]
  all_goals apply congrArg PoseidonCompactWitness.source
  all_goals funext lane
  all_goals dsimp only [Poseidon2HashChainCompactWitness.secondInput,
    Poseidon2HashChainCompactWitness.finalInput, PoseidonCompactWitness.output]
  all_goals rw [show Poseidon2HashChainCompactWitness.secondInput prior message =
    (fun lane => Gadgets.Poseidon2.Layer.evalState
      (PoseidonCompactWitness.source (Poseidon2HashChainCompactWitness.firstInput prior))
      (Gadgets.Poseidon2.Permutation.scheduleOutput PoseidonScheduleTrace.inputCount) lane +
        Poseidon2HashChainCompactWitness.blockValue message lane) from rfl]
  all_goals simp

/-- The four message words and last three physical permutation intervals.
The unused ten-permutation interval is zero and is never computed. -/
def suffix (prior message : Fin 4 → F) : Fin 7700 → F :=
  let values := sources prior message
  fun index =>
    if word : index.val < 4 then message ⟨index.val, word⟩
    else if variableRegion : 5924 ≤ index.val then
      let relative := index.val - 5924
      let invocation : Fin 3 := ⟨relative / 592, by
        have bound := index.isLt
        omega⟩
      values invocation (8 + relative % 592)
    else 0

theorem suffix_message (prior message : Fin 4 → F) (lane : Fin 4) :
    suffix prior message ⟨lane.val, by omega⟩ = message lane := by
  simp [suffix, lane.isLt]

theorem suffix_retained (prior message : Fin 4 → F)
    (invocation : Fin 3) (row : Fin PoseidonRetainedSlots.rows.length) :
    suffix prior message ⟨5924 + invocation.val * 592 + (PoseidonRetainedSlots.localOutput row).val, by
      have localBound := (PoseidonRetainedSlots.localOutput row).isLt
      change _ < 592 at localBound
      omega⟩ = Poseidon2HashChainCompactWitness.witness prior message invocation row := by
  have localBound := (PoseidonRetainedSlots.localOutput row).isLt
  change _ < 592 at localBound
  dsimp only [suffix]
  rw [dif_neg (by omega), dif_pos (by omega)]
  have indexEq : (5924 + invocation.val * 592 + (PoseidonRetainedSlots.localOutput row).val - 5924) / 592 =
      invocation.val := by omega
  have remainderEq : (5924 + invocation.val * 592 + (PoseidonRetainedSlots.localOutput row).val - 5924) % 592 =
      (PoseidonRetainedSlots.localOutput row).val := by omega
  simp only [indexEq, remainderEq, sources_eq, Poseidon2HashChainCompactWitness.witness,
    PoseidonCompactWitness.retained, PoseidonRetainedSlots.output_eq_input_add_local]
  rfl

/-- Place the computed values in the existing physical-source ABI. -/
def privateSuffix (prior message : Fin 4 → F) :
    Fin (PerApplicationPackage.addedPrivateColumnCount application) → F :=
  fun index => suffix prior message (Fin.cast Poseidon2HashChainV1Package.addedPrivateColumnCount index)

def priorValues (target : Env) (lane : Fin 4) : F := target (ApplicationInputs.inputColumn lane)

def raw (target : Env) (message : Fin 4 → F) : PerApplicationCanonicalAssignment.RawValues application :=
  PerApplicationAssignmentTransportExecution.canonicalRawValues application
    (PerApplicationSourceAssignment.ofCompleted application target (privateSuffix (priorValues target) message))

private theorem source_before (target : Env) (message : Fin 4 → F)
    (column : Fin (ApplicationRetainedBlocks.sourceWidth application))
    (before : column.val < Spartan.privateColumnCount) :
    (raw target message).applicationSource column = target column.val := by
  change PerApplicationSourceAssignment.ofCompleted application target
    (privateSuffix (priorValues target) message) ⟨column.val, _⟩ = _
  unfold PerApplicationSourceAssignment.ofCompleted
  rw [dif_pos (by
    have constant : PerApplicationPackage.basePackage.layout.constantColumn = Spartan.privateColumnCount := by
      rw [Spartan.privateColumnCount_eq]
      exact Package.circuitPackage_layout_values.2.2.1
    rw [constant]
    exact before)]

private theorem source_private (target : Env) (message : Fin 4 → F)
    (column : Fin (ApplicationRetainedBlocks.sourceWidth application)) (index : Fin 7700)
    (position : column.val = Spartan.privateColumnCount + index.val) :
    (raw target message).applicationSource column = suffix (priorValues target) message index := by
  let slot : Fin (PerApplicationPackage.addedPrivateColumnCount application) :=
    Fin.cast Poseidon2HashChainV1Package.addedPrivateColumnCount.symm index
  have stored := PerApplicationSourceAssignment.application_ofCompleted application target
    (privateSuffix (priorValues target) message) slot
  change PerApplicationSourceAssignment.ofCompleted application target
    (privateSuffix (priorValues target) message) ⟨column.val, _⟩ = _
  simpa only [position, slot, privateSuffix, Fin.val_cast, Fin.cast_cast, Fin.cast_eq_self] using stored

theorem witnessValue (target : Env) (message : Fin 4 → F) :
    Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
      (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv (raw target message).base) =
        List.ofFn message := by
  apply congrArg List.ofFn
  funext lane
  have stored := source_private target message ((ApplicationRetainedBlocks.witnessBlock application).source lane)
    ⟨lane.val, by have := lane.isLt; change lane.val < 4 at this; omega⟩ rfl
  rw [suffix_message] at stored
  exact (DirectApplicationPrefixPlan.applicationSource_eq_sourceEnv application
    (raw target message).base ((ApplicationRetainedBlocks.witnessBlock application).source lane)).symm.trans stored

private def certificate : ApplicationPoseidonRetainedBlock.Certificate application := ⟨rfl, fun _ => rfl⟩

private theorem selected : application.compactHashChain = some certificate := rfl

private theorem localEncoding {program : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry program columns)
    (certificate : ApplicationPoseidonRetainedBlock.Certificate program)
    (selected : program.compactHashChain = some certificate)
    (assignment : Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.Assignment F columns)
    (source : Fin (ApplicationRetainedBlocks.sourceWidth program) → F)
    (encodes : ApplicationRetainedGeometry.Encodes geometry assignment source) :
    (ApplicationPoseidonRetainedBlock.block program certificate).EncodesAt
      (ApplicationPoseidonRetainedGeometry.localStart program)
      (ApplicationPoseidonRetainedGeometry.localFits
        (ApplicationRetainedGeometry.poseidonGeometry geometry certificate selected)) assignment source := by
  have encoded (fits : ApplicationRetainedGeometry.localStart program +
      (ApplicationSelectedBlocks.localBlock program).coordinateCount ≤ columns) :
      (ApplicationSelectedBlocks.localBlock program).EncodesAt
        (ApplicationRetainedGeometry.localStart program) fits assignment source := encodes.localValues
  rw [ApplicationSelectedBlocks.localBlock_some program certificate selected] at encoded
  exact encoded _

/-- The direct three-permutation witness completes the selected application relation. -/
theorem complete (target : Env) (message : Fin 4 → F)
    (step : (List.ofFn fun lane : Fin 4 => target (ApplicationInputs.outputColumn lane)) =
      application.step (List.ofFn (priorValues target)) (List.ofFn message)) :
    (ApplicationDirectPlan.plan Poseidon2HashChainV1Package.fits.package
      (PerApplicationFixedPoint.geometry application)).RowsZero (raw target message).assignment := by
  let geometry := ApplicationRetainedGeometry.poseidonGeometry
    (PerApplicationFixedPoint.geometry application) certificate selected
  have encoding := (PerApplicationCanonicalEncodes.encodes (raw target message)).applicationEncoding
  rw [ApplicationDirectPlan.plan_some _ _ certificate selected]
  apply ApplicationPoseidonSoundness.complete_of_encoding geometry (raw target message).assignment
    (priorValues target) message (PerApplicationCanonicalAssignment.assignment_one _) ?_ ?_ ?_ ?_
  · intro lane
    change ((ApplicationRetainedBlocks.inputBlock application).form _ _ lane).eval _ = _
    rw [LowNormBlock.Block.form_eval _ _ _ _ _ encoding.input]
    apply source_before
    change ApplicationInputs.inputColumn lane < Spartan.privateColumnCount
    rw [ApplicationInputs.inputColumn_value, Spartan.privateColumnCount_eq]
    simp only [ApplicationInputs.currentWordStart]
    have := lane.isLt
    omega
  · intro lane
    change ((ApplicationRetainedBlocks.witnessBlock application).form _ _ lane).eval _ = _
    rw [LowNormBlock.Block.form_eval _ _ _ _ _ encoding.witness]
    exact (source_private target message _ ⟨lane.val, by have := lane.isLt; omega⟩ rfl).trans
      (suffix_message _ _ lane)
  · have outputs (lane : Fin 4) :
        ((ApplicationPoseidonRetainedGeometry.interface geometry).digest lane).eval (raw target message).assignment =
          target (ApplicationInputs.outputColumn lane) := by
      change ((ApplicationRetainedBlocks.outputBlock application).form _ _ lane).eval _ = _
      rw [LowNormBlock.Block.form_eval _ _ _ _ _ encoding.output]
      apply source_before
      change ApplicationInputs.outputColumn lane < Spartan.privateColumnCount
      rw [ApplicationInputs.outputColumn_value, Spartan.privateColumnCount_eq]
      have := lane.isLt
      omega
    simpa only [outputs] using step
  · intro invocation row
    have localValues := localEncoding _ certificate selected _ _ encoding
    dsimp only [ApplicationPoseidonRetainedGeometry.interface]
    rw [LowNormBlock.Block.form_eval _ _ _ _ _ localValues]
    have position :
        ((ApplicationPoseidonRetainedBlock.block application certificate).source
          (Fin.encodeProd (invocation, row))).val = Spartan.privateColumnCount +
            (5924 + invocation.val * 592 + (PoseidonRetainedSlots.localOutput row).val) := by
      have parsed := PoseidonRetainedBlock.block_source
        (ApplicationDirectSource.sourceWidth application) 3
        (ApplicationPoseidonRetainedBlock.witnessStart application)
        (ApplicationPoseidonRetainedBlock.witnessStart_bound application certificate)
        (Fin.encodeProd (invocation, row))
      rw [Fin.decodeProd_encodeProd] at parsed
      calc
        _ = ApplicationPoseidonRetainedBlock.witnessStart application invocation +
            (PoseidonRetainedSlots.localOutput row).val := parsed
        _ = _ := by
          unfold ApplicationPoseidonRetainedBlock.witnessStart ApplicationInputs.localStart
            ApplicationInputs.witnessStart
          change Spartan.privateColumnCount + 4 + 5920 + invocation.val * 592 + _ = _
          omega
    exact (source_private target message _ _ position).trans (suffix_retained _ _ invocation row)

end NightstreamFPrime.Export.Stage1.ApplicationCompactWitness
