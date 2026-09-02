import NightstreamFPrime.Export.Stage1.PackageSourceRows
import NightstreamFPrime.Export.Stage1.ApplicationDirectSource
import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram
import NightstreamFPrime.Export.Stage1.PiDECOrdinaryDirectSource
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSchedule
import NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram
/-!
Owns source-row custody for one Lean-authored per-application package. Typed
row transports follow the exact pilot lift and application-private suffix
shift used by package emission.

This module does not select an application or claim final conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceRows

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program

/-- Shape-only witness for row-count theorems on the fixed prefix. Its zero
matrices are never used as semantic authority. -/
def baseShapeRelation :
    Lifecycle.ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits where
  matrices := fun _ _ _ => 0
  cubeFits := by
    norm_num [Data.logicalWidth, VerifierContext.candidateLogicalWidth,
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth,
      Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount,
      Lifecycle.cubeVariables, Spec.ringDegree]

def shiftCompiledRow (application : ApplicationProgram) :
    Rows.CompiledRow → Rows.CompiledRow
  | .witness instruction => .witness
      (PerApplicationPackage.shiftWitnessInstruction application instruction)
  | .assertion row => .assertion
      (PerApplicationPackage.shiftSparseRow application row)

@[simp] theorem shiftCompiledRow_rowIndex (application : ApplicationProgram)
    (row : Rows.CompiledRow) :
    (shiftCompiledRow application row).rowIndex = row.rowIndex := by
  cases row <;> rfl

theorem shiftCompiledRow_toR1CS (application : ApplicationProgram)
    (row : Rows.CompiledRow) :
    (shiftCompiledRow application row).toR1CS =
      PerApplicationSourceProjection.basePackageRow application row.toR1CS := by
  cases row with
  | witness instruction =>
      simp only [shiftCompiledRow, Rows.CompiledRow.toR1CS,
        WitnessInstruction.toR1CS,
        PerApplicationSourceProjection.basePackageRow,
        PerApplicationPackage.shiftWitnessInstruction]
      rw [PerApplicationPackage.shiftSparseCombination_toR1CS,
        PerApplicationPackage.shiftSparseCombination_toR1CS]
      rfl
  | assertion row =>
      simp only [shiftCompiledRow, Rows.CompiledRow.toR1CS,
        SparseRow.toR1CS, PerApplicationSourceProjection.basePackageRow,
        PerApplicationPackage.shiftSparseRow]
      rw [PerApplicationPackage.shiftSparseCombination_toR1CS,
        PerApplicationPackage.shiftSparseCombination_toR1CS,
        PerApplicationPackage.shiftSparseCombination_toR1CS]
      rfl

def liftPilotCompiledRow : Rows.CompiledRow → Rows.CompiledRow
  | .witness instruction => .witness (Data.liftPilotInstruction instruction)
  | .assertion row => .assertion (Data.liftPilotRow row)

@[simp] theorem liftPilotCompiledRow_rowIndex (row : Rows.CompiledRow) :
    (liftPilotCompiledRow row).rowIndex = row.rowIndex := by
  cases row <;> rfl

private theorem liftPilotCombination_toR1CS
    (combination : SparseCombination) :
    (Data.liftPilotCombination combination).toR1CS =
      mapCombinationColumns Spartan.liftPilotColumn combination.toR1CS := by
  cases combination
  simp [Data.liftPilotCombination, Data.liftPilotTerm,
    SparseCombination.toR1CS, mapCombinationColumns, List.map_map,
    Function.comp_def]

theorem liftPilotCompiledRow_toR1CS (row : Rows.CompiledRow) :
    (liftPilotCompiledRow row).toR1CS =
      mapRowColumns Spartan.liftPilotColumn row.toR1CS := by
  cases row with
  | witness instruction =>
      simp only [liftPilotCompiledRow, Rows.CompiledRow.toR1CS,
        WitnessInstruction.toR1CS, Data.liftPilotInstruction]
      rw [liftPilotCombination_toR1CS, liftPilotCombination_toR1CS]
      rfl
  | assertion row =>
      simp only [liftPilotCompiledRow, Rows.CompiledRow.toR1CS,
        SparseRow.toR1CS, Data.liftPilotRow]
      rw [liftPilotCombination_toR1CS, liftPilotCombination_toR1CS,
        liftPilotCombination_toR1CS]
      rfl

theorem shiftLiftPilotCompiledRow_toR1CS
    (application : ApplicationProgram) (row : Rows.CompiledRow) :
    (shiftCompiledRow application (liftPilotCompiledRow row)).toR1CS =
      PerApplicationSourceProjection.pilotPackageRow application
        row.toR1CS := by
  rw [shiftCompiledRow_toR1CS, liftPilotCompiledRow_toR1CS]
  rfl

theorem decodedRows_shift (application : ApplicationProgram)
    (instructions : List WitnessInstruction) (assertions : List SparseRow) :
    PackageSourceRows.decodedRows
        (instructions.map
          (PerApplicationPackage.shiftWitnessInstruction application))
        (assertions.map (PerApplicationPackage.shiftSparseRow application)) =
      (PackageSourceRows.decodedRows instructions assertions).map
        (shiftCompiledRow application) := by
  simp [PackageSourceRows.decodedRows, shiftCompiledRow, List.map_map,
    Function.comp_def, List.map_append]

theorem decodedRows_liftPilot (instructions : List WitnessInstruction)
    (assertions : List SparseRow) :
    PackageSourceRows.decodedRows
        (Data.liftPilotInstructions instructions)
        (Data.liftPilotRows assertions) =
      (PackageSourceRows.decodedRows instructions assertions).map
        liftPilotCompiledRow := by
  simp [PackageSourceRows.decodedRows, Data.liftPilotInstructions,
    Data.liftPilotRows, liftPilotCompiledRow, List.map_map,
    Function.comp_def, List.map_append]

def pilotRows : List Rows.CompiledRow :=
  PackageSourceRows.decodedRows (PilotData.witnessInstructions ())
    (PilotData.assertionRows ())

def pilotDigestRows : List Rows.CompiledRow :=
  (PilotData.digestRows PilotData.outputChain).map
    Rows.CompiledRow.assertion

theorem priorExtraRows_rowIndices :
    (PilotData.priorExtraRows ()).map Rows.CompiledRow.rowIndex =
      List.range' PilotData.priorBindingRowStart 1326 := by
  have loweredLength := Pilot.priorExtraRows_length
  unfold PilotData.priorExtraRows at loweredLength
  simp only [List.length_map, Rows.compileRowsTR_length] at loweredLength
  let compiled := Rows.compileRowsTR PilotValues.logicalColumnCount
    PilotData.priorBindingRowStart
    (Rows.lowerConstraintsTR (PilotData.priorExtraConstraints ())
      PilotValues.logicalColumnCount).rows
  change (compiled.map PilotData.remapCompiledRow).map
      Rows.CompiledRow.rowIndex = _
  rw [List.map_map]
  calc
    compiled.map (Rows.CompiledRow.rowIndex ∘
        PilotData.remapCompiledRow) =
        compiled.map Rows.CompiledRow.rowIndex := by
      apply List.map_congr_left
      intro row member
      cases row <;> rfl
    _ = List.range' PilotData.priorBindingRowStart 1326 := by
      simpa [compiled, loweredLength] using
        (Rows.compileRowsTR_rowIndices PilotValues.logicalColumnCount
          PilotData.priorBindingRowStart
          (Rows.lowerConstraintsTR (PilotData.priorExtraConstraints ())
            PilotValues.logicalColumnCount).rows)

theorem pilotDigestRows_rowIndices :
    pilotDigestRows.map Rows.CompiledRow.rowIndex =
      List.range'
        (PilotData.outputChain.rowStart +
          PilotData.outputChain.witnessLength) 4 := by
  simp [pilotDigestRows, PilotData.digestRows, PilotData.digestRow,
    Rows.CompiledRow.rowIndex, List.range'_eq_map_range]
  rfl

theorem pilotRows_eq :
    pilotRows =
      PackageSourceRows.classifiedRows (PilotData.priorExtraRows ()) ++
        pilotDigestRows := by
  simp [pilotRows, PackageSourceRows.decodedRows,
    PackageSourceRows.classifiedRows, PilotData.witnessInstructions,
    PilotData.assertionRows, pilotDigestRows, Rows.witnessInstructionsTR_eq,
    Rows.assertionRowsTR_eq, List.map_append, List.append_assoc]

theorem pilotRows_perm :
    List.Perm pilotRows
      (PilotData.priorExtraRows () ++ pilotDigestRows) := by
  rw [pilotRows_eq]
  exact (PackageSourceRows.classifiedRows_perm
    (PilotData.priorExtraRows ())).append (List.Perm.refl _)

theorem pilotRows_rowIndices_perm :
    List.Perm (pilotRows.map Rows.CompiledRow.rowIndex)
      (List.range' PilotData.priorBindingRowStart 1326 ++
        List.range'
          (PilotData.outputChain.rowStart +
            PilotData.outputChain.witnessLength) 4) := by
  have mapped := pilotRows_perm.map Rows.CompiledRow.rowIndex
  simpa [List.map_append, priorExtraRows_rowIndices,
    pilotDigestRows_rowIndices] using mapped

private theorem pilotIndexRanges_nodup :
    (List.range' PilotData.priorBindingRowStart 1326 ++
      List.range'
        (PilotData.outputChain.rowStart +
          PilotData.outputChain.witnessLength) 4).Nodup := by
  rw [List.nodup_append]
  refine ⟨List.nodup_range', List.nodup_range', ?_⟩
  intro first firstMember second secondMember equal
  rw [List.mem_range'_1] at firstMember secondMember
  norm_num [PilotData.priorBindingRowStart, PilotData.outputChain,
    PilotData.outputHashRowStart, PilotValues.priorBindingRowStart,
    PilotValues.outputHashRowStart, PilotValues.priorBindingRowCount,
    PilotValues.priorExtraRowCount, PilotValues.priorCanonicalRowCount,
    PilotValues.priorFixedRowCount] at firstMember secondMember
  omega

theorem pilotRows_rowIndices_nodup :
    (pilotRows.map Rows.CompiledRow.rowIndex).Nodup := by
  exact pilotRows_rowIndices_perm.nodup_iff.mpr pilotIndexRanges_nodup

private theorem piCcsRowStart_eq :
    PiCCSArithmetic.statementBindingRowStart = 14623730 := by
  unfold PiCCSArithmetic.statementBindingRowStart
    PiCCSStarts.statementBindingRowStart PiCCSStarts.rowBase
  exact PilotProduction.physicalRowCountValue_eq

theorem pilotRows_rowIndex_lt (row : Rows.CompiledRow)
    (member : row ∈ pilotRows) :
    row.rowIndex < PiCCSArithmetic.statementBindingRowStart := by
  rw [piCcsRowStart_eq]
  have indexMember : row.rowIndex ∈
      pilotRows.map Rows.CompiledRow.rowIndex :=
    List.mem_map_of_mem member
  have rangesMember := pilotRows_rowIndices_perm.mem_iff.mp indexMember
  rw [List.mem_append] at rangesMember
  rcases rangesMember with prior | digest
  · rw [List.mem_range'_1] at prior
    norm_num [PilotData.priorBindingRowStart,
      PilotValues.priorBindingRowStart, PilotValues.priorHashRowStart,
      PilotValues.hashWitnessCount, PilotValues.absorbCount,
      PilotValues.stateHashWords, PilotValues.stateHashBaseWords,
      Spec.Poseidon2.rate, PilotValues.permutationRecipeCount] at prior ⊢
    omega
  · rw [List.mem_range'_1] at digest
    norm_num [PilotData.outputChain, PilotData.outputHashRowStart,
      PilotValues.outputHashRowStart, PilotValues.priorBindingRowStart,
      PilotValues.priorBindingRowCount, PilotValues.priorExtraRowCount,
      PilotValues.priorCanonicalRowCount, PilotValues.priorFixedRowCount,
      PilotValues.hashWitnessCount, PilotValues.absorbCount,
      PilotValues.stateHashWords, PilotValues.stateHashBaseWords,
      Spec.Poseidon2.rate, PilotValues.permutationRecipeCount] at digest ⊢
    omega

theorem piDecRows_rowIndices :
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits).rows.map
        Rows.CompiledRow.rowIndex =
      List.range' PiDECStarts.phaseRowStart 25488 := by
  calc
    _ = List.range'
        (PiDECArithmetic.canonicalPlan Data.logicalWidth
          Data.publicFits).rowStart
        (PiDECArithmetic.canonicalPlan Data.logicalWidth
          Data.publicFits).rows.length :=
      PiCCSArithmetic.compilePacket_rowIndices _ _ _
    _ = List.range' PiDECStarts.phaseRowStart 25488 := by
      rw [PiDECArithmetic.Plan.rows_length,
        PiDECArithmetic.canonicalPlan_rowCount baseShapeRelation]
      rfl

theorem runningRows_rowIndices :
    (RunningTransitionArithmetic.canonicalPlan Data.logicalWidth
      Data.publicFits).rows.map Rows.CompiledRow.rowIndex =
      List.range' RunningTransitionArithmetic.rowStart 345495 := by
  calc
    _ = List.range'
        (RunningTransitionArithmetic.canonicalPlan Data.logicalWidth
          Data.publicFits).rowStart
        (RunningTransitionArithmetic.canonicalPlan Data.logicalWidth
          Data.publicFits).rows.length :=
      PiCCSArithmetic.compilePacket_rowIndices _ _ _
    _ = List.range' RunningTransitionArithmetic.rowStart 345495 := by
      rw [RunningTransitionArithmetic.Plan.rows_length,
        RunningTransitionArithmetic.canonicalPlan_rowCount baseShapeRelation]
      rfl

theorem arithmeticRows_rowIndices :
    (Data.arithmeticRows ()).map Rows.CompiledRow.rowIndex =
      PiCCSOrdinaryMatrixProgram.rowIndexReference ++
        PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference ++
        List.range' PiDECStarts.phaseRowStart 25488 ++
        List.range' RunningTransitionArithmetic.rowStart 345495 := by
  unfold Data.arithmeticRows
  simp only [List.map_append]
  rw [PiCCSOrdinaryMatrixProgram.arithmeticRows_rowIndices
      baseShapeRelation,
    PiRLCSamplerOrdinaryMatrixSchedule.arithmeticRows_rowIndices,
    piDecRows_rowIndices, runningRows_rowIndices]

private theorem piDecIndex_bounds (index : Nat)
    (member : index ∈ List.range' PiDECStarts.phaseRowStart 25488) :
    PiDECStarts.phaseRowStart ≤ index ∧
      index < RunningTransitionArithmetic.rowStart := by
  rw [List.mem_range'_1] at member
  unfold RunningTransitionArithmetic.rowStart PiDECStarts.outputRowStart
    PiDECStarts.evalARowStart PiDECStarts.evalKRowStart
    PiDECStarts.commitmentRowStart PiDECStarts.publicInputRowStart
    PiDECStarts.inputRowStart
  omega

private theorem runningIndex_lower (index : Nat)
    (member : index ∈
      List.range' RunningTransitionArithmetic.rowStart 345495) :
    RunningTransitionArithmetic.rowStart ≤ index := by
  exact (List.mem_range'_1.mp member).1

private theorem piDecRunningIndices_nodup :
    (List.range' PiDECStarts.phaseRowStart 25488 ++
      List.range' RunningTransitionArithmetic.rowStart 345495).Nodup := by
  rw [List.nodup_append]
  refine ⟨List.nodup_range', List.nodup_range', ?_⟩
  intro piDec piDecMember running runningMember equal
  have piDecBounds := piDecIndex_bounds piDec piDecMember
  have runningLower := runningIndex_lower running runningMember
  omega

private theorem samplerLaterIndices_nodup :
    (PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference ++
      (List.range' PiDECStarts.phaseRowStart 25488 ++
        List.range' RunningTransitionArithmetic.rowStart 345495)).Nodup := by
  rw [List.nodup_append]
  refine ⟨PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference_nodup,
    piDecRunningIndices_nodup, ?_⟩
  intro sampler samplerMember later laterMember equal
  have samplerBounds :=
    PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference_bounds
      sampler samplerMember
  rw [List.mem_append] at laterMember
  rcases laterMember with piDecMember | runningMember
  · have piDecBounds := piDecIndex_bounds later piDecMember
    omega
  · have runningLower := runningIndex_lower later runningMember
    unfold RunningTransitionArithmetic.rowStart PiDECStarts.outputRowStart
      PiDECStarts.evalARowStart PiDECStarts.evalKRowStart
      PiDECStarts.commitmentRowStart PiDECStarts.publicInputRowStart
      PiDECStarts.inputRowStart at runningLower
    omega

private theorem arithmeticIndexRanges_nodup :
    (PiCCSOrdinaryMatrixProgram.rowIndexReference ++
      (PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference ++
        (List.range' PiDECStarts.phaseRowStart 25488 ++
          List.range' RunningTransitionArithmetic.rowStart 345495))).Nodup := by
  rw [List.nodup_append]
  refine ⟨PiCCSOrdinaryMatrixProgram.rowIndexReference_nodup,
    samplerLaterIndices_nodup, ?_⟩
  intro piCcs piCcsMember later laterMember equal
  have piCcsBounds := PiCCSOrdinaryMatrixProgram.rowIndexReference_bounds
    piCcs piCcsMember
  have piRlcStart : PiRLCStarts.phaseRowStart = 19936967 := rfl
  rw [piRlcStart] at piCcsBounds
  rw [List.mem_append] at laterMember
  rcases laterMember with samplerMember | laterMember
  · have samplerBounds :=
      PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference_bounds
        later samplerMember
    rw [piRlcStart] at samplerBounds
    omega
  · rw [List.mem_append] at laterMember
    rcases laterMember with piDecMember | runningMember
    · have piDecBounds := piDecIndex_bounds later piDecMember
      have piDecStart : PiDECStarts.phaseRowStart = 28847041 := rfl
      rw [piDecStart] at piDecBounds
      omega
    · have runningLower := runningIndex_lower later runningMember
      have runningStart :
          RunningTransitionArithmetic.rowStart = 28872529 := rfl
      rw [runningStart] at runningLower
      omega

theorem arithmeticRows_rowIndices_nodup :
    ((Data.arithmeticRows ()).map Rows.CompiledRow.rowIndex).Nodup := by
  rw [arithmeticRows_rowIndices]
  simpa only [List.append_assoc] using arithmeticIndexRanges_nodup

theorem arithmeticRows_rowIndex_ge (index : Nat)
    (member : index ∈
      (Data.arithmeticRows ()).map Rows.CompiledRow.rowIndex) :
    PiCCSArithmetic.statementBindingRowStart ≤ index := by
  rw [piCcsRowStart_eq]
  rw [arithmeticRows_rowIndices] at member
  have normalized : index ∈
      PiCCSOrdinaryMatrixProgram.rowIndexReference ++
        (PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference ++
          (List.range' PiDECStarts.phaseRowStart 25488 ++
            List.range' RunningTransitionArithmetic.rowStart 345495)) := by
    simpa only [List.append_assoc] using member
  rw [List.mem_append] at normalized
  rcases normalized with piCcsMember | member
  · have lower := (PiCCSOrdinaryMatrixProgram.rowIndexReference_bounds
      index piCcsMember).1
    rw [piCcsRowStart_eq] at lower
    exact lower
  · rw [List.mem_append] at member
    rcases member with samplerMember | member
    · have lower :=
        (PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference_bounds
          index samplerMember).1
      have phaseStart : PiRLCStarts.phaseRowStart = 19936967 := rfl
      rw [phaseStart] at lower
      omega
    · rw [List.mem_append] at member
      rcases member with piDecMember | runningMember
      · have lower := (piDecIndex_bounds index piDecMember).1
        have phaseStart : PiDECStarts.phaseRowStart = 28847041 := rfl
        rw [phaseStart] at lower
        omega
      · have lower := runningIndex_lower index runningMember
        have phaseStart : RunningTransitionArithmetic.rowStart = 28872529 := rfl
        rw [phaseStart] at lower
        omega

private theorem baseRowCount_eq :
    PerApplicationPackage.basePackage.layout.rowCount = 29218024 := by
  unfold PerApplicationPackage.basePackage
  exact Package.circuitPackage_layout_values.1

theorem arithmeticRows_rowIndex_lt_base (index : Nat)
    (member : index ∈
      (Data.arithmeticRows ()).map Rows.CompiledRow.rowIndex) :
    index < PerApplicationPackage.basePackage.layout.rowCount := by
  rw [baseRowCount_eq]
  rw [arithmeticRows_rowIndices] at member
  have normalized : index ∈
      PiCCSOrdinaryMatrixProgram.rowIndexReference ++
        (PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference ++
          (List.range' PiDECStarts.phaseRowStart 25488 ++
            List.range' RunningTransitionArithmetic.rowStart 345495)) := by
    simpa only [List.append_assoc] using member
  rw [List.mem_append] at normalized
  rcases normalized with piCcsMember | member
  · have upper := (PiCCSOrdinaryMatrixProgram.rowIndexReference_bounds
      index piCcsMember).2
    have phaseStart : PiRLCStarts.phaseRowStart = 19936967 := rfl
    rw [phaseStart] at upper
    omega
  · rw [List.mem_append] at member
    rcases member with samplerMember | member
    · have upper :=
        (PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference_bounds
          index samplerMember).2
      have phaseStart : PiDECStarts.phaseRowStart = 28847041 := rfl
      rw [phaseStart] at upper
      omega
    · rw [List.mem_append] at member
      rcases member with piDecMember | runningMember
      · have upper := (piDecIndex_bounds index piDecMember).2
        have phaseStart : RunningTransitionArithmetic.rowStart = 28872529 := rfl
        rw [phaseStart] at upper
        omega
      · have bounds := List.mem_range'_1.mp runningMember
        have phaseStart : RunningTransitionArithmetic.rowStart = 28872529 := rfl
        rw [phaseStart] at bounds
        omega

def baseRows : List Rows.CompiledRow :=
  pilotRows.map liftPilotCompiledRow ++ Data.arithmeticRows ()

theorem baseRows_rowIndices_nodup :
    (baseRows.map Rows.CompiledRow.rowIndex).Nodup := by
  unfold baseRows
  rw [List.map_append]
  have liftedIndices :
      (pilotRows.map liftPilotCompiledRow).map Rows.CompiledRow.rowIndex =
        pilotRows.map Rows.CompiledRow.rowIndex := by
    rw [List.map_map]
    apply List.map_congr_left
    intro row member
    exact liftPilotCompiledRow_rowIndex row
  rw [liftedIndices, List.nodup_append]
  refine ⟨pilotRows_rowIndices_nodup,
    arithmeticRows_rowIndices_nodup, ?_⟩
  intro pilotIndex pilotMember arithmeticIndex arithmeticMember equal
  rcases List.mem_map.mp pilotMember with ⟨row, rowMember, rowIndexEq⟩
  have pilotBound := pilotRows_rowIndex_lt row rowMember
  have arithmeticBound := arithmeticRows_rowIndex_ge
    arithmeticIndex arithmeticMember
  rw [rowIndexEq] at pilotBound
  omega

theorem baseRows_rowIndex_lt (row : Rows.CompiledRow)
    (member : row ∈ baseRows) :
    row.rowIndex < PerApplicationPackage.basePackage.layout.rowCount := by
  rw [baseRows, List.mem_append] at member
  rcases member with pilotMember | arithmeticMember
  · rcases List.mem_map.mp pilotMember with
      ⟨source, sourceMember, sourceEq⟩
    subst row
    rw [liftPilotCompiledRow_rowIndex]
    have upper := pilotRows_rowIndex_lt source sourceMember
    rw [piCcsRowStart_eq] at upper
    rw [baseRowCount_eq]
    omega
  · have indexMember : row.rowIndex ∈
        (Data.arithmeticRows ()).map Rows.CompiledRow.rowIndex :=
      List.mem_map_of_mem arithmeticMember
    exact arithmeticRows_rowIndex_lt_base row.rowIndex indexMember

def applicationRows (application : ApplicationProgram) :
    List Rows.CompiledRow :=
  ApplicationPackage.compiledRows application
    (ApplicationPackage.productionColumns application)
    (Layout.Stage1.ApplicationInputs.localStart application)
    PerApplicationPackage.basePackage.layout.rowCount

theorem applicationRows_rowIndices (application : ApplicationProgram) :
    (applicationRows application).map Rows.CompiledRow.rowIndex =
      List.range' PerApplicationPackage.basePackage.layout.rowCount
        (applicationRows application).length := by
  unfold applicationRows ApplicationPackage.compiledRows
  simpa only [Rows.compileRowsTR_length] using
    (Rows.compileRowsTR_rowIndices _ _ _)

theorem applicationRows_rowIndices_nodup (application : ApplicationProgram) :
    ((applicationRows application).map Rows.CompiledRow.rowIndex).Nodup := by
  rw [applicationRows_rowIndices]
  exact List.nodup_range'

theorem applicationRows_rowIndex_ge (application : ApplicationProgram)
    (index : Nat)
    (member : index ∈
      (applicationRows application).map Rows.CompiledRow.rowIndex) :
    PerApplicationPackage.basePackage.layout.rowCount ≤ index := by
  rw [applicationRows_rowIndices, List.mem_range'_1] at member
  exact member.1

def nextPreimageRows (application : ApplicationProgram) :
    List Rows.CompiledRow :=
  NextPreimagePackage.compiledRows
    (PerApplicationPackage.nextPreimageRowStart application)

theorem nextPreimageRows_rowIndices (application : ApplicationProgram) :
    (nextPreimageRows application).map Rows.CompiledRow.rowIndex =
      List.range' (PerApplicationPackage.nextPreimageRowStart application) 5 :=
  NextPreimagePackage.compiledRows_rowIndices _

def canonicalRows (application : ApplicationProgram) :
    List Rows.CompiledRow :=
  (baseRows.map (shiftCompiledRow application) ++ applicationRows application) ++
    nextPreimageRows application

theorem basePackage_decodedRows_perm :
    List.Perm
      (PackageSourceRows.decodedRows
        PerApplicationPackage.basePackage.witnessInstructions
        PerApplicationPackage.basePackage.assertionRows)
      baseRows := by
  rw [PerApplicationPackage.basePackage,
    Data.circuitPackage_witnessInstructions,
    Data.circuitPackage_assertionRows]
  unfold Data.Components.witnessInstructions Data.Components.assertionRows
    Data.Components.arithmeticAssertionRows
  rw [Data.components_arithmeticRows, Rows.witnessInstructionsTR_eq,
    Rows.assertionRowsTR_eq]
  refine (PackageSourceRows.decodedRows_append_perm _ _ _ _).trans ?_
  rw [decodedRows_liftPilot]
  change List.Perm
    (pilotRows.map liftPilotCompiledRow ++
      PackageSourceRows.classifiedRows (Data.arithmeticRows ()))
    (pilotRows.map liftPilotCompiledRow ++ Data.arithmeticRows ())
  exact (PackageSourceRows.classifiedRows_perm
    (Data.arithmeticRows ())).append_left _

theorem applicationPlan_decodedRows_perm (application : ApplicationProgram) :
    List.Perm
      (PackageSourceRows.decodedRows
        (PerApplicationPackage.applicationPlan application).witnessInstructions
        (PerApplicationPackage.applicationPlan application).assertionRows)
      (applicationRows application) := by
  change List.Perm
    (PackageSourceRows.decodedRows
      (Rows.witnessInstructionsTR (applicationRows application))
      (Rows.assertionRowsTR (applicationRows application)))
    (applicationRows application)
  rw [Rows.witnessInstructionsTR_eq, Rows.assertionRowsTR_eq]
  exact PackageSourceRows.classifiedRows_perm (applicationRows application)

theorem nextPreimage_decodedRows_perm (application : ApplicationProgram) :
    List.Perm
      (PackageSourceRows.decodedRows []
        (NextPreimagePackage.assertionRows
          (PerApplicationPackage.nextPreimageRowStart application)))
      (nextPreimageRows application) := by
  let rowStart := PerApplicationPackage.nextPreimageRowStart application
  let rows := NextPreimagePackage.compiledRows rowStart
  have empty : Rows.witnessInstructionsTR rows = [] := by
    simpa [rows, rowStart, NextPreimagePackage.witnessInstructions] using
      NextPreimagePackage.witnessInstructions_eq_nil rowStart
  change List.Perm
    (PackageSourceRows.decodedRows [] (Rows.assertionRowsTR rows)) rows
  rw [← empty, Rows.witnessInstructionsTR_eq, Rows.assertionRowsTR_eq]
  exact PackageSourceRows.classifiedRows_perm _

theorem package_decodedRows_perm_canonical (application : ApplicationProgram) :
    List.Perm
      (PackageSourceRows.decodedRows
        (PerApplicationPackage.package application).witnessInstructions
        (PerApplicationPackage.package application).assertionRows)
      (canonicalRows application) := by
  rw [PerApplicationPackage.package_witnessInstructions,
    PerApplicationPackage.package_assertionRows]
  have basePerm : List.Perm
      (PackageSourceRows.decodedRows
        (PerApplicationPackage.basePackage.witnessInstructions.map
          (PerApplicationPackage.shiftWitnessInstruction application))
        (PerApplicationPackage.basePackage.assertionRows.map
          (PerApplicationPackage.shiftSparseRow application)))
      (baseRows.map (shiftCompiledRow application)) := by
    rw [decodedRows_shift]
    exact basePackage_decodedRows_perm.map (shiftCompiledRow application)
  have prefixSplit := PackageSourceRows.decodedRows_append_perm
    (PerApplicationPackage.basePackage.witnessInstructions.map
      (PerApplicationPackage.shiftWitnessInstruction application))
    (PerApplicationPackage.applicationPlan application).witnessInstructions
    (PerApplicationPackage.basePackage.assertionRows.map
      (PerApplicationPackage.shiftSparseRow application))
    (PerApplicationPackage.applicationPlan application).assertionRows
  have prefixRows := prefixSplit.trans
    (basePerm.append (applicationPlan_decodedRows_perm application))
  have outer := PackageSourceRows.decodedRows_append_perm
    ((PerApplicationPackage.basePackage.witnessInstructions.map
        (PerApplicationPackage.shiftWitnessInstruction application)) ++
      (PerApplicationPackage.applicationPlan application).witnessInstructions)
    []
    ((PerApplicationPackage.basePackage.assertionRows.map
        (PerApplicationPackage.shiftSparseRow application)) ++
      (PerApplicationPackage.applicationPlan application).assertionRows)
    (NextPreimagePackage.assertionRows
      (PerApplicationPackage.nextPreimageRowStart application))
  simp only [List.append_nil] at outer
  exact outer.trans (prefixRows.append (nextPreimage_decodedRows_perm application))

theorem canonicalRows_rowIndices_nodup (application : ApplicationProgram) :
    ((canonicalRows application).map Rows.CompiledRow.rowIndex).Nodup := by
  unfold canonicalRows
  rw [List.map_append, List.nodup_append]
  refine ⟨?_, ?_, ?_⟩
  · rw [List.map_append]
    have shiftedIndices :
        (baseRows.map (shiftCompiledRow application)).map
            Rows.CompiledRow.rowIndex =
          baseRows.map Rows.CompiledRow.rowIndex := by
      rw [List.map_map]
      apply List.map_congr_left
      intro row member
      exact shiftCompiledRow_rowIndex application row
    rw [shiftedIndices, List.nodup_append]
    refine ⟨baseRows_rowIndices_nodup,
      applicationRows_rowIndices_nodup application, ?_⟩
    intro baseIndex baseMember applicationIndex applicationMember equal
    rcases List.mem_map.mp baseMember with
      ⟨row, rowMember, rowIndexEq⟩
    have upper := baseRows_rowIndex_lt row rowMember
    rw [rowIndexEq] at upper
    have lower := applicationRows_rowIndex_ge application applicationIndex
      applicationMember
    omega
  · rw [nextPreimageRows_rowIndices]
    exact List.nodup_range'
  · intro prefixIndex prefixMember nextIndex nextMember equal
    rw [nextPreimageRows_rowIndices, List.mem_range'_1] at nextMember
    have shiftedIndices :
        (baseRows.map (shiftCompiledRow application)).map
            Rows.CompiledRow.rowIndex =
          baseRows.map Rows.CompiledRow.rowIndex := by
      rw [List.map_map]
      apply List.map_congr_left
      intro row member
      exact shiftCompiledRow_rowIndex application row
    rw [List.map_append, shiftedIndices, List.mem_append] at prefixMember
    rcases prefixMember with baseMember | applicationMember
    · rcases List.mem_map.mp baseMember with
        ⟨row, rowMember, rowIndexEq⟩
      have upper : row.rowIndex <
          PerApplicationPackage.basePackage.layout.rowCount :=
        baseRows_rowIndex_lt row rowMember
      rw [rowIndexEq] at upper
      unfold PerApplicationPackage.nextPreimageRowStart at nextMember
      omega
    · rw [applicationRows_rowIndices, List.mem_range'_1] at applicationMember
      have lengthEq : (applicationRows application).length =
          (PerApplicationPackage.applicationPlan application).rowCount := by
        have count := ApplicationDirectSource.sourceRows_length_eq_plan application
        unfold ApplicationDirectSource.sourceRows at count
        rw [List.length_map] at count
        simpa [applicationRows] using count
      rw [lengthEq] at applicationMember
      unfold PerApplicationPackage.nextPreimageRowStart at nextMember
      omega

theorem package_decodedRows_rowIndices_nodup
    (application : ApplicationProgram) :
    ((PackageSourceRows.decodedRows
      (PerApplicationPackage.package application).witnessInstructions
      (PerApplicationPackage.package application).assertionRows).map
        Rows.CompiledRow.rowIndex).Nodup := by
  have permutation :=
    (package_decodedRows_perm_canonical application).map
      Rows.CompiledRow.rowIndex
  exact permutation.nodup_iff.mpr
    (canonicalRows_rowIndices_nodup application)

theorem packageSourceRow?_eq_some (application : ApplicationProgram)
    (target : Rows.CompiledRow) (member : target ∈ canonicalRows application) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application) target.rowIndex =
      some target.toR1CS := by
  apply PackageSourceRows.sourceRow?_eq_some
  · exact package_decodedRows_rowIndices_nodup application
  · exact (package_decodedRows_perm_canonical application).mem_iff.mpr member

theorem basePackageSourceRow?_eq_some (application : ApplicationProgram)
    (target : Rows.CompiledRow) (member : target ∈ baseRows) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application) target.rowIndex =
      some (PerApplicationSourceProjection.basePackageRow application
        target.toR1CS) := by
  have canonicalMember : shiftCompiledRow application target ∈
      canonicalRows application := by
    rw [canonicalRows, List.mem_append]
    exact Or.inl (List.mem_append_left _ (List.mem_map_of_mem member))
  have recovered := packageSourceRow?_eq_some application
    (shiftCompiledRow application target) canonicalMember
  simpa only [shiftCompiledRow_rowIndex,
    shiftCompiledRow_toR1CS] using recovered

theorem pilotPackageSourceRow?_eq_some (application : ApplicationProgram)
    (target : Rows.CompiledRow) (member : target ∈ pilotRows) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application) target.rowIndex =
      some (PerApplicationSourceProjection.pilotPackageRow application
        target.toR1CS) := by
  have baseMember : liftPilotCompiledRow target ∈ baseRows := by
    rw [baseRows, List.mem_append]
    exact Or.inl (List.mem_map_of_mem member)
  have canonicalMember :
      shiftCompiledRow application (liftPilotCompiledRow target) ∈
        canonicalRows application := by
    rw [canonicalRows, List.mem_append]
    exact Or.inl (List.mem_append_left _ (List.mem_map_of_mem baseMember))
  have recovered := packageSourceRow?_eq_some application
    (shiftCompiledRow application (liftPilotCompiledRow target))
    canonicalMember
  simpa only [shiftCompiledRow_rowIndex, liftPilotCompiledRow_rowIndex,
    shiftLiftPilotCompiledRow_toR1CS] using recovered

theorem applicationPackageSourceRow?_eq_some
    (application : ApplicationProgram) (target : Rows.CompiledRow)
    (member : target ∈ applicationRows application) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application) target.rowIndex =
      some target.toR1CS := by
  apply packageSourceRow?_eq_some application target
  rw [canonicalRows, List.mem_append]
  exact Or.inl (List.mem_append_right _ member)

theorem nextPreimagePackageSourceRow?_eq_some
    (application : ApplicationProgram) (target : Rows.CompiledRow)
    (member : target ∈ nextPreimageRows application) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application) target.rowIndex =
      some target.toR1CS := by
  apply packageSourceRow?_eq_some application target
  rw [canonicalRows, List.mem_append]
  exact Or.inr member

private theorem indexedPackageStoredRow?_eq_some
    (application : ApplicationProgram) {count : Nat}
    (rows : List Rows.CompiledRow) (rowsLength : rows.length = count)
    (programRow : Fin count → R1CS.Row)
    (exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn programRow)
    (projection : R1CS.Row → R1CS.Row)
    (recovered : ∀ target ∈ rows,
      PackageSourceRows.packageSourceRow?
          (PerApplicationPackage.package application) target.rowIndex =
        some (projection target.toR1CS))
    (index : Fin count) :
    let rowFin : Fin rows.length := Fin.cast rowsLength.symm index
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (rows.get rowFin).rowIndex =
      some (projection (programRow index)) := by
  let rowFin : Fin rows.length := Fin.cast rowsLength.symm index
  let target := rows.get rowFin
  have indexBound : index.val < rows.length := by
    rw [rowsLength]
    exact index.isLt
  have targetRow : target.toR1CS = programRow index := by
    have selected := congrArg
      (fun values : List R1CS.Row =>
        values.getD index.val target.toR1CS) exactRows
    dsimp only at selected
    rw [List.getD_map,
      List.getD_eq_getElem rows target indexBound] at selected
    have referenceBound : index.val < (List.ofFn programRow).length := by
      simpa using index.isLt
    rw [List.getD_eq_getElem _ _ referenceBound] at selected
    simpa [target, rowFin] using selected
  have selected := recovered target (List.get_mem rows rowFin)
  simpa only [target, targetRow] using selected

theorem indexedPackageSourceRow?_eq_some
    (application : ApplicationProgram) {count rowStart : Nat}
    (rows : List Rows.CompiledRow) (rowsLength : rows.length = count)
    (rowIndices : rows.map Rows.CompiledRow.rowIndex =
      List.range' rowStart count)
    (programRow : Fin count → R1CS.Row)
    (exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn programRow)
    (projection : R1CS.Row → R1CS.Row)
    (recovered : ∀ target ∈ rows,
      PackageSourceRows.packageSourceRow?
          (PerApplicationPackage.package application) target.rowIndex =
        some (projection target.toR1CS))
    (index : Fin count) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (rowStart + index.val) =
      some (projection (programRow index)) := by
  let rowFin : Fin rows.length := Fin.cast rowsLength.symm index
  let target := rows.get rowFin
  have indexBound : index.val < rows.length := by
    rw [rowsLength]
    exact index.isLt
  have targetIndex : target.rowIndex = rowStart + index.val := by
    have selected := congrArg
      (fun values : List Nat =>
        values.getD index.val target.rowIndex) rowIndices
    dsimp only at selected
    rw [List.getD_map,
      List.getD_eq_getElem rows target indexBound] at selected
    have rangeBound : index.val < (List.range' rowStart count).length := by
      simpa using index.isLt
    rw [List.getD_eq_getElem _ _ rangeBound,
      List.getElem_range'_1] at selected
    simpa [target, rowFin] using selected
  have selected := indexedPackageStoredRow?_eq_some application rows rowsLength
    programRow exactRows projection recovered index
  change PackageSourceRows.packageSourceRow?
      (PerApplicationPackage.package application) target.rowIndex =
    some (projection (programRow index)) at selected
  rw [targetIndex] at selected
  exact selected

theorem indexedBasePackageStoredRow?_eq_some
    (application : ApplicationProgram) {count : Nat}
    (rows : List Rows.CompiledRow) (rowsLength : rows.length = count)
    (included : ∀ row ∈ rows, row ∈ baseRows)
    (programRow : Fin count → R1CS.Row)
    (exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn programRow)
    (index : Fin count) :
    let rowFin : Fin rows.length := Fin.cast rowsLength.symm index
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (rows.get rowFin).rowIndex =
      some (PerApplicationSourceProjection.basePackageRow application
        (programRow index)) := by
  exact indexedPackageStoredRow?_eq_some application rows rowsLength
    programRow exactRows
    (PerApplicationSourceProjection.basePackageRow application)
    (fun target member =>
      basePackageSourceRow?_eq_some application target
        (included target member)) index

theorem indexedPilotPackageStoredRow?_eq_some
    (application : ApplicationProgram) {count : Nat}
    (rows : List Rows.CompiledRow) (rowsLength : rows.length = count)
    (included : ∀ row ∈ rows, row ∈ pilotRows)
    (programRow : Fin count → R1CS.Row)
    (exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn programRow)
    (index : Fin count) :
    let rowFin : Fin rows.length := Fin.cast rowsLength.symm index
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (rows.get rowFin).rowIndex =
      some (PerApplicationSourceProjection.pilotPackageRow application
        (programRow index)) := by
  exact indexedPackageStoredRow?_eq_some application rows rowsLength
    programRow exactRows
    (PerApplicationSourceProjection.pilotPackageRow application)
    (fun target member =>
      pilotPackageSourceRow?_eq_some application target
        (included target member)) index

theorem indexedBasePackageSourceRow?_eq_some
    (application : ApplicationProgram) {count rowStart : Nat}
    (rows : List Rows.CompiledRow) (rowsLength : rows.length = count)
    (rowIndices : rows.map Rows.CompiledRow.rowIndex =
      List.range' rowStart count)
    (included : ∀ row ∈ rows, row ∈ baseRows)
    (programRow : Fin count → R1CS.Row)
    (exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn programRow)
    (index : Fin count) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (rowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (programRow index)) := by
  exact indexedPackageSourceRow?_eq_some application rows rowsLength
    rowIndices programRow exactRows
    (PerApplicationSourceProjection.basePackageRow application)
    (fun target member =>
      basePackageSourceRow?_eq_some application target
        (included target member)) index

theorem indexedPilotPackageSourceRow?_eq_some
    (application : ApplicationProgram) {count rowStart : Nat}
    (rows : List Rows.CompiledRow) (rowsLength : rows.length = count)
    (rowIndices : rows.map Rows.CompiledRow.rowIndex =
      List.range' rowStart count)
    (included : ∀ row ∈ rows, row ∈ pilotRows)
    (programRow : Fin count → R1CS.Row)
    (exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn programRow)
    (index : Fin count) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (rowStart + index.val) =
      some (PerApplicationSourceProjection.pilotPackageRow application
        (programRow index)) := by
  exact indexedPackageSourceRow?_eq_some application rows rowsLength
    rowIndices programRow exactRows
    (PerApplicationSourceProjection.pilotPackageRow application)
    (fun target member =>
      pilotPackageSourceRow?_eq_some application target
        (included target member)) index

theorem indexedApplicationPackageSourceRow?_eq_some
    (application : ApplicationProgram) {count rowStart : Nat}
    (rows : List Rows.CompiledRow) (rowsLength : rows.length = count)
    (rowIndices : rows.map Rows.CompiledRow.rowIndex =
      List.range' rowStart count)
    (included : ∀ row ∈ rows, row ∈ applicationRows application)
    (programRow : Fin count → R1CS.Row)
    (exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn programRow)
    (index : Fin count) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (rowStart + index.val) = some (programRow index) := by
  exact indexedPackageSourceRow?_eq_some application rows rowsLength
    rowIndices programRow exactRows id
    (fun target member => by
      simpa using applicationPackageSourceRow?_eq_some application target
        (included target member)) index

theorem piCcsPackageSourceRow?_eq_some
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (application : ApplicationProgram)
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits)
    (index : Fin 811669) (sourceIndex : Nat)
    (selected : PiCCSOrdinaryMatrixProgram.rowSchedule.index? index.val =
      some sourceIndex) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application) sourceIndex =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiCCSOrdinaryDirectSource.programRow relation index)) := by
  let rows := PiCCSArithmetic.arithmeticRows logicalWidth publicFits
  have rowsLength : rows.length = 811669 :=
    PiCCSArithmetic.arithmeticRows_length logicalWidth publicFits relation
  have included : ∀ row ∈ rows, row ∈ baseRows := by
    intro row member
    rw [baseRows, List.mem_append]
    apply Or.inr
    have rowsEq : rows = PiCCSArithmetic.arithmeticRows
        Data.logicalWidth Data.publicFits := by
      rfl
    rw [rowsEq] at member
    unfold Data.arithmeticRows
    simp only [List.mem_append]
    exact Or.inl (Or.inl (Or.inl member))
  have exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn (PiCCSOrdinaryDirectSource.programRow relation) := by
    change PiCCSOrdinaryDirectSource.sourceRows logicalWidth publicFits = _
    exact (PiCCSOrdinaryDirectSource.programRows_eq relation).symm
  have recovered := indexedBasePackageStoredRow?_eq_some application rows
    rowsLength included (PiCCSOrdinaryDirectSource.programRow relation)
    exactRows index
  let rowFin : Fin rows.length := Fin.cast rowsLength.symm index
  let target := rows.get rowFin
  have indexBound : index.val < rows.length := by
    rw [rowsLength]
    exact index.isLt
  have scheduleEq :=
    PiCCSOrdinaryMatrixProgram.rowSchedule_index?_eq_arithmeticRowIndex?
      relation index.val
  rw [selected, List.getElem?_eq_getElem indexBound] at scheduleEq
  simp only [Option.map_some] at scheduleEq
  change some sourceIndex = some target.rowIndex at scheduleEq
  have targetIndex : target.rowIndex = sourceIndex := by
    exact Option.some.inj scheduleEq.symm
  change PackageSourceRows.packageSourceRow?
      (PerApplicationPackage.package application) target.rowIndex =
    some (PerApplicationSourceProjection.basePackageRow application
      (PiCCSOrdinaryDirectSource.programRow relation index)) at recovered
  rw [targetIndex] at recovered
  exact recovered

theorem samplerPackageSourceRow?_eq_some
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (application : ApplicationProgram) (index : Fin 220881)
    (sourceIndex : Nat)
    (selected :
      PiRLCSamplerOrdinaryMatrixSchedule.rowSchedule.index? index.val =
        some sourceIndex) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application) sourceIndex =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiRLCSamplerOrdinaryDirectSource.programRow
          (logicalWidth := logicalWidth) (publicFits := publicFits) index)) := by
  let rows := PiRLCSamplerOrdinaryRows.rows
    (logicalWidth := logicalWidth) (publicFits := publicFits)
  have rowsLength : rows.length = 220881 :=
    PiRLCSamplerOrdinaryRows.rows_length
  have included : ∀ row ∈ rows, row ∈ baseRows := by
    intro row member
    rw [baseRows, List.mem_append]
    apply Or.inr
    have rowsEq : rows = PiRLCSamplerOrdinaryRows.rows
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) := by
      rfl
    rw [rowsEq] at member
    unfold Data.arithmeticRows
    simp only [List.mem_append]
    exact Or.inl (Or.inl (Or.inr member))
  have exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn (PiRLCSamplerOrdinaryDirectSource.programRow
        (logicalWidth := logicalWidth) (publicFits := publicFits)) := by
    change PiRLCSamplerOrdinaryDirectSource.sourceRows
      (logicalWidth := logicalWidth) (publicFits := publicFits) = _
    exact (PiRLCSamplerOrdinaryDirectSource.programRows_eq
      (logicalWidth := logicalWidth) (publicFits := publicFits)).symm
  have recovered := indexedBasePackageStoredRow?_eq_some application rows
    rowsLength included
    (PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    exactRows index
  let rowFin : Fin rows.length := Fin.cast rowsLength.symm index
  let target := rows.get rowFin
  have indexBound : index.val < rows.length := by
    rw [rowsLength]
    exact index.isLt
  have scheduleEq :=
    PiRLCSamplerOrdinaryMatrixSchedule.rowSchedule_index?_eq_arithmeticRowIndex?
      (logicalWidth := logicalWidth) (publicFits := publicFits) index.val
  rw [selected, List.getElem?_eq_getElem indexBound] at scheduleEq
  simp only [Option.map_some] at scheduleEq
  change some sourceIndex = some target.rowIndex at scheduleEq
  have targetIndex : target.rowIndex = sourceIndex :=
    Option.some.inj scheduleEq.symm
  change PackageSourceRows.packageSourceRow?
      (PerApplicationPackage.package application) target.rowIndex =
    some (PerApplicationSourceProjection.basePackageRow application
      (PiRLCSamplerOrdinaryDirectSource.programRow
        (logicalWidth := logicalWidth) (publicFits := publicFits) index))
      at recovered
  rw [targetIndex] at recovered
  exact recovered

theorem pilotPackageSourceRowAt?_eq_some
    (application : ApplicationProgram) (index : Fin 1330) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (PilotOrdinaryMatrixProgram.rowIndexAt index) =
      some (PerApplicationSourceProjection.pilotPackageRow application
        (PilotOrdinaryDirectSource.programRow index)) := by
  let rows := pilotRows
  have exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn PilotOrdinaryDirectSource.programRow := by
    calc
      rows.map Rows.CompiledRow.toR1CS =
          PilotOrdinaryDirectSource.sourceRows := by
        unfold rows pilotRows PackageSourceRows.decodedRows
          PilotOrdinaryDirectSource.sourceRows
          PilotOrdinaryDirectSource.instructionRows
          PilotOrdinaryDirectSource.assertionRows
        simp only [List.map_append, List.map_map, Function.comp_def]
      _ = List.ofFn PilotOrdinaryDirectSource.programRow :=
        (PilotOrdinaryDirectSource.programRows_eq).symm
  have rowsLength : rows.length = 1330 := by
    have lengths := congrArg List.length exactRows
    simpa only [List.length_map, List.length_ofFn] using lengths
  have recovered := indexedPilotPackageStoredRow?_eq_some application rows
    rowsLength (fun _ member => member) PilotOrdinaryDirectSource.programRow
    exactRows index
  let rowFin : Fin rows.length := Fin.cast rowsLength.symm index
  let target := rows.get rowFin
  have rowIndices : rows.map Rows.CompiledRow.rowIndex =
      PilotOrdinaryMatrixProgram.rowIndexReference := by
    unfold rows pilotRows PackageSourceRows.decodedRows
      PilotOrdinaryMatrixProgram.rowIndexReference
      PilotOrdinaryMatrixProgram.instructionIndices
      PilotOrdinaryMatrixProgram.assertionIndices
    simp only [List.map_append, List.map_map, Function.comp_def,
      Rows.CompiledRow.rowIndex]
  have indexBound : index.val < rows.length := by
    rw [rowsLength]
    exact index.isLt
  have selected := congrArg
    (fun values : List Nat => values.getD index.val target.rowIndex)
    rowIndices
  dsimp only at selected
  rw [List.getD_map,
    List.getD_eq_getElem rows target indexBound] at selected
  have referenceBound : index.val <
      PilotOrdinaryMatrixProgram.rowIndexReference.length := by
    rw [PilotOrdinaryMatrixProgram.rowIndexReference_length]
    exact index.isLt
  rw [List.getD_eq_getElem _ _ referenceBound] at selected
  change target.rowIndex = PilotOrdinaryMatrixProgram.rowIndexAt index
    at selected
  change PackageSourceRows.packageSourceRow?
      (PerApplicationPackage.package application) target.rowIndex =
    some (PerApplicationSourceProjection.pilotPackageRow application
      (PilotOrdinaryDirectSource.programRow index)) at recovered
  rw [selected] at recovered
  exact recovered

private theorem ofFn_cast_get {Alpha : Type} (rows : List Alpha) {count : Nat}
    (lengthEq : rows.length = count) :
    List.ofFn (fun index : Fin count =>
      rows.get (Fin.cast lengthEq.symm index)) = rows := by
  subst count
  simpa using List.ofFn_get rows

def piDecProgramRow
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 25488) : R1CS.Row :=
  (PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits).getD
    index.val Spartan.zeroRow

theorem piDecProgramRows_eq
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) :
    List.ofFn (piDecProgramRow relation) =
      PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits := by
  let rows := PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits
  have rowsLength : rows.length = 25488 :=
    PiDECOrdinaryDirectSource.sourceRows_length relation
  change List.ofFn (fun index : Fin 25488 =>
    rows.getD index.val Spartan.zeroRow) = rows
  apply List.ext_get
  · rw [List.length_ofFn, rowsLength]
  · intro position leftBound rightBound
    rw [List.get_ofFn]
    exact List.getD_eq_get rows Spartan.zeroRow ⟨position, rightBound⟩

theorem piDecPackageSourceRow?_eq_some
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (application : ApplicationProgram)
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 25488) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (PiDECStarts.phaseRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (piDecProgramRow relation index)) := by
  let rows := (PiDECArithmetic.canonicalPlan logicalWidth publicFits).rows
  have rowsLength : rows.length = 25488 := by
    rw [PiDECArithmetic.Plan.rows_length,
      PiDECArithmetic.canonicalPlan_rowCount relation]
  have rowIndices : rows.map Rows.CompiledRow.rowIndex =
      List.range' PiDECStarts.phaseRowStart 25488 := by
    calc
      _ = List.range'
          (PiDECArithmetic.canonicalPlan logicalWidth publicFits).rowStart
          rows.length := PiCCSArithmetic.compilePacket_rowIndices _ _ _
      _ = List.range' PiDECStarts.phaseRowStart 25488 := by
        rw [rowsLength]
        rfl
  have included : ∀ row ∈ rows, row ∈ baseRows := by
    intro row member
    rw [baseRows, List.mem_append]
    apply Or.inr
    have rowsEq : rows =
        (PiDECArithmetic.canonicalPlan Data.logicalWidth
          Data.publicFits).rows := by
      rfl
    rw [rowsEq] at member
    unfold Data.arithmeticRows
    simp only [List.mem_append]
    exact Or.inl (Or.inr member)
  have exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn (piDecProgramRow relation) := by
    calc
      _ = PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits :=
        (PiDECOrdinaryDirectSource.sourceRows_eq_canonical
          (logicalWidth := logicalWidth) (publicFits := publicFits)).symm
      _ = List.ofFn (piDecProgramRow relation) :=
        (piDecProgramRows_eq relation).symm
  exact indexedBasePackageSourceRow?_eq_some application rows rowsLength
    rowIndices included (piDecProgramRow relation) exactRows index

def piDecPublicIndex (index : Fin 22680) : Fin 25488 :=
  ⟨index.val, by omega⟩

theorem piDecProgramRow_public
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 22680) :
    piDecProgramRow relation (piDecPublicIndex index) =
      PiDECOrdinaryDirectSource.publicProgramRow relation index := by
  change (PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits).getD
      index.val Spartan.zeroRow = _
  unfold PiDECOrdinaryDirectSource.sourceRows
  rw [List.getD_append _ _ _ _ (by
    rw [List.length_append, List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation,
      PiDECOrdinaryDirectSource.evalKRows_length relation]
    omega)]
  rw [List.getD_append _ _ _ _ (by
    rw [List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation]
    omega)]
  rw [List.getD_append _ _ _ _ (by
    rw [PiDECOrdinaryDirectSource.publicRows_length relation]
    exact index.isLt)]
  unfold PiDECOrdinaryDirectSource.publicProgramRow
    PiDECOrdinaryDirectSource.publicListIndex
  have bound : index.val <
      (PiDECOrdinaryDirectSource.publicRows logicalWidth publicFits).length := by
    rw [PiDECOrdinaryDirectSource.publicRows_length relation]
    exact index.isLt
  rw [List.getD_eq_getElem _ Spartan.zeroRow bound,
    List.get_eq_getElem]
  simp only [Fin.val_cast]

def piDecCommitmentIndex (index : Fin 1188) : Fin 25488 :=
  ⟨22680 + index.val, by omega⟩

theorem piDecProgramRow_commitment
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 1188) :
    piDecProgramRow relation (piDecCommitmentIndex index) =
      PiDECOrdinaryDirectSource.commitmentProgramRow relation index := by
  change (PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits).getD
      (22680 + index.val) Spartan.zeroRow = _
  unfold PiDECOrdinaryDirectSource.sourceRows
  rw [List.getD_append _ _ _ _ (by
    rw [List.length_append, List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation,
      PiDECOrdinaryDirectSource.evalKRows_length relation]
    omega)]
  rw [List.getD_append _ _ _ _ (by
    rw [List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation]
    omega)]
  rw [List.getD_append_right _ _ _ _ (by
    rw [PiDECOrdinaryDirectSource.publicRows_length relation]
    omega)]
  have offsetEq : 22680 + index.val -
      (PiDECOrdinaryDirectSource.publicRows logicalWidth publicFits).length =
        index.val := by
    rw [PiDECOrdinaryDirectSource.publicRows_length relation]
    omega
  rw [offsetEq]
  unfold PiDECOrdinaryDirectSource.commitmentProgramRow
    PiDECOrdinaryDirectSource.commitmentListIndex
  have bound : index.val <
      (PiDECOrdinaryDirectSource.commitmentRows logicalWidth
        publicFits).length := by
    rw [PiDECOrdinaryDirectSource.commitmentRows_length relation]
    exact index.isLt
  rw [List.getD_eq_getElem _ Spartan.zeroRow bound,
    List.get_eq_getElem]
  simp only [Fin.val_cast]

def piDecEvalKIndex (index : Fin 108) : Fin 25488 :=
  ⟨23868 + index.val, by omega⟩

theorem piDecProgramRow_evalK
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 108) :
    piDecProgramRow relation (piDecEvalKIndex index) =
      PiDECOrdinaryDirectSource.evalKProgramRow relation index := by
  change (PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits).getD
      (23868 + index.val) Spartan.zeroRow = _
  unfold PiDECOrdinaryDirectSource.sourceRows
  rw [List.getD_append _ _ _ _ (by
    rw [List.length_append, List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation,
      PiDECOrdinaryDirectSource.evalKRows_length relation]
    omega)]
  rw [List.getD_append_right _ _ _ _ (by
    rw [List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation]
    omega)]
  have offsetEq : 23868 + index.val -
      (PiDECOrdinaryDirectSource.publicRows logicalWidth publicFits ++
        PiDECOrdinaryDirectSource.commitmentRows logicalWidth
          publicFits).length = index.val := by
    rw [List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation]
    omega
  rw [offsetEq]
  unfold PiDECOrdinaryDirectSource.evalKProgramRow
    PiDECOrdinaryDirectSource.evalKListIndex
  have bound : index.val <
      (PiDECOrdinaryDirectSource.evalKRows logicalWidth publicFits).length := by
    rw [PiDECOrdinaryDirectSource.evalKRows_length relation]
    exact index.isLt
  rw [List.getD_eq_getElem _ Spartan.zeroRow bound,
    List.get_eq_getElem]
  simp only [Fin.val_cast]

def piDecEvalAIndex (index : Fin 1512) : Fin 25488 :=
  ⟨23976 + index.val, by omega⟩

theorem piDecProgramRow_evalA
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 1512) :
    piDecProgramRow relation (piDecEvalAIndex index) =
      PiDECOrdinaryDirectSource.evalAProgramRow relation index := by
  change (PiDECOrdinaryDirectSource.sourceRows logicalWidth publicFits).getD
      (23976 + index.val) Spartan.zeroRow = _
  unfold PiDECOrdinaryDirectSource.sourceRows
  rw [List.getD_append_right _ _ _ _ (by
    rw [List.length_append, List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation,
      PiDECOrdinaryDirectSource.evalKRows_length relation]
    omega)]
  have offsetEq : 23976 + index.val -
      ((PiDECOrdinaryDirectSource.publicRows logicalWidth publicFits ++
        PiDECOrdinaryDirectSource.commitmentRows logicalWidth publicFits) ++
          PiDECOrdinaryDirectSource.evalKRows logicalWidth publicFits).length =
        index.val := by
    rw [List.length_append, List.length_append,
      PiDECOrdinaryDirectSource.publicRows_length relation,
      PiDECOrdinaryDirectSource.commitmentRows_length relation,
      PiDECOrdinaryDirectSource.evalKRows_length relation]
    omega
  rw [offsetEq]
  unfold PiDECOrdinaryDirectSource.evalAProgramRow
    PiDECOrdinaryDirectSource.evalAListIndex
  have bound : index.val <
      (PiDECOrdinaryDirectSource.evalARows logicalWidth publicFits).length := by
    rw [PiDECOrdinaryDirectSource.evalARows_length relation]
    exact index.isLt
  rw [List.getD_eq_getElem _ Spartan.zeroRow bound,
    List.get_eq_getElem]
  simp only [Fin.val_cast]

theorem piDecPublicPackageSourceRow?_eq_some
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (application : ApplicationProgram)
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 22680) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (PiDECStarts.publicInputRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.publicProgramRow relation index)) := by
  have recovered := piDecPackageSourceRow?_eq_some application relation
    (piDecPublicIndex index)
  rw [piDecProgramRow_public relation index] at recovered
  simpa [piDecPublicIndex, PiDECStarts.publicInputRowStart,
    PiDECStarts.inputRowStart] using recovered

theorem piDecCommitmentPackageSourceRow?_eq_some
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (application : ApplicationProgram)
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 1188) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (PiDECStarts.commitmentRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.commitmentProgramRow relation index)) := by
  have recovered := piDecPackageSourceRow?_eq_some application relation
    (piDecCommitmentIndex index)
  rw [piDecProgramRow_commitment relation index] at recovered
  have rowEq : PiDECStarts.phaseRowStart +
      (piDecCommitmentIndex index).val =
        PiDECStarts.commitmentRowStart + index.val := by
    change PiDECStarts.phaseRowStart + (22680 + index.val) =
      PiDECStarts.commitmentRowStart + index.val
    unfold PiDECStarts.commitmentRowStart
      PiDECStarts.publicInputRowStart PiDECStarts.inputRowStart
    omega
  rw [rowEq] at recovered
  exact recovered

theorem piDecEvalKPackageSourceRow?_eq_some
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (application : ApplicationProgram)
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 108) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (PiDECStarts.evalKRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.evalKProgramRow relation index)) := by
  have recovered := piDecPackageSourceRow?_eq_some application relation
    (piDecEvalKIndex index)
  rw [piDecProgramRow_evalK relation index] at recovered
  have rowEq : PiDECStarts.phaseRowStart + (piDecEvalKIndex index).val =
      PiDECStarts.evalKRowStart + index.val := by
    change PiDECStarts.phaseRowStart + (23868 + index.val) =
      PiDECStarts.evalKRowStart + index.val
    unfold PiDECStarts.evalKRowStart
      PiDECStarts.commitmentRowStart PiDECStarts.publicInputRowStart
      PiDECStarts.inputRowStart
    omega
  rw [rowEq] at recovered
  exact recovered

theorem piDecEvalAPackageSourceRow?_eq_some
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (application : ApplicationProgram)
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) (index : Fin 1512) :
    PackageSourceRows.packageSourceRow?
        (PerApplicationPackage.package application)
        (PiDECStarts.evalARowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.evalAProgramRow relation index)) := by
  have recovered := piDecPackageSourceRow?_eq_some application relation
    (piDecEvalAIndex index)
  rw [piDecProgramRow_evalA relation index] at recovered
  have rowEq : PiDECStarts.phaseRowStart + (piDecEvalAIndex index).val =
      PiDECStarts.evalARowStart + index.val := by
    change PiDECStarts.phaseRowStart + (23976 + index.val) =
      PiDECStarts.evalARowStart + index.val
    unfold PiDECStarts.evalARowStart
      PiDECStarts.evalKRowStart PiDECStarts.commitmentRowStart
      PiDECStarts.publicInputRowStart PiDECStarts.inputRowStart
    omega
  rw [rowEq] at recovered
  exact recovered

end NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceRows
