import NightstreamFPrime.Export.Stage1.Wide.ArchiveRows
import NightstreamFPrime.Export.Stage1.Wide.ApplicationPackageCounts
import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixRecovery
import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSemantics

/-! Construct source custody for the successful wide physical package with
the selected Poseidon2 hash-chain application. The package result is the only
archive precondition; row ownership and inverse relocation are proved here. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalSourceCustody

open NightstreamFPrime.Layout NightstreamFPrime.Export.Package
open ArchiveRows PhysicalRelabel

abbrev application := Poseidon2HashChainV1Package.application

private def extra (program : ApplicationPackage.Program) : Nat :=
  PerApplicationPackage.directAddedPrivateColumnCount program

private def nextRows (program : ApplicationPackage.Program) : List Rows.CompiledRow :=
  PackageSourceRows.decodedRows []
    (NextPreimagePackage.assertionRows (PerApplicationPackage.nextPreimageRowStart program))

private structure BuilderRows (program : ApplicationPackage.Program) (commonPackage package : CircuitPackage) where
  base : CircuitPackage
  common : List Rows.CompiledRow
  inserted : List Rows.CompiledRow
  applicationRows : List Rows.CompiledRow
  next : List Rows.CompiledRow
  commonPairs : List.Forall₂ (fun a b => ordinaryMap.compiledRow a = .ok b)
    (stored commonPackage) common
  baseRows : (stored base).Perm (common ++ PhysicalSampler.rows ())
  insertPairs : List.Forall₂ (fun a b => (ApplicationPackage.insertApplication (extra program)).compiledRow a = .ok b)
    (stored base) inserted
  nextPairs : List.Forall₂ (fun a b => ordinaryMap.compiledRow a = .ok b) (nextRows program) next
  applicationBounds : ∀ row ∈ applicationRows,
    27716409 ≤ row.rowIndex ∧
      row.rowIndex < 27716409 + (PerApplicationPackage.applicationPlan program).rowCount
  finalRows : (stored package).Perm (inserted ++ applicationRows ++ next)

private theorem insertion_count (program : ApplicationPackage.Program) (plan : Stage1.ApplicationPackage.Plan)
    (same : plan.privateCount = (PerApplicationPackage.applicationPlan program).privateCount) :
    program.witnessWordCount + plan.privateCount = PerApplicationPackage.directAddedPrivateColumnCount program := by
  rw [same, PerApplicationPackage.directAddedPrivateColumnCount_eq_addedPrivateColumnCount]
  rfl

private theorem common_layout : (PhysicalPackage.common ()).layout = Data.physicalLayout := rfl

private theorem builder_rows_of_base (program : ApplicationPackage.Program)
    (commonPackage base package : CircuitPackage) (plan : Stage1.ApplicationPackage.Plan)
    (commonCount : commonPackage.layout.rowCount = 28666318)
    (counts : ∀ start, (ApplicationPackage.plan program start).privateCount =
      (PerApplicationPackage.applicationPlan program).privateCount ∧
      (ApplicationPackage.plan program start).rowCount = (PerApplicationPackage.applicationPlan program).rowCount)
    (baseResult : PhysicalPackage.ofCommon commonPackage = .ok base)
    (extended : ApplicationPackage.ofBase program base = .ok (package, plan)) :
    Nonempty (BuilderRows program commonPackage package) := by
  obtain ⟨commonInstructions, commonAssertions, commonInstructionMap, commonAssertionMap,
    countMap, baseRows⟩ := physical_fields commonPackage base baseResult
  obtain ⟨planEq, instructions, assertions, next, instructionMap, assertionMap, nextMap, finalRows⟩ :=
    application_fields program base package plan extended
  have baseCount : base.layout.rowCount = 27716409 := by
    rw [commonCount] at countMap
    change Except.ok 27716409 = Except.ok base.layout.rowCount at countMap
    exact (Except.ok.inj countMap).symm
  have countEq : program.witnessWordCount + plan.privateCount = extra program := by
    apply insertion_count program plan
    rw [planEq]
    exact (counts _).1
  rw [countEq] at instructionMap assertionMap
  refine ⟨{
    base := base
    common := PackageSourceRows.decodedRows commonInstructions commonAssertions
    inserted := PackageSourceRows.decodedRows instructions assertions
    applicationRows := PackageSourceRows.decodedRows plan.witnessInstructions plan.assertionRows
    next := PackageSourceRows.decodedRows [] next
    commonPairs := decoded_pairs ordinaryMap _ _ _ _ commonInstructionMap commonAssertionMap
    baseRows := baseRows
    insertPairs := decoded_pairs (ApplicationPackage.insertApplication (extra program)) _ _ _ _ instructionMap assertionMap
    nextPairs := decoded_pairs ordinaryMap [] [] _ _ rfl nextMap
    applicationBounds := ?_
    finalRows := finalRows }⟩
  intro row member
  rw [planEq] at member
  have bounds := application_bounds program base.layout.rowCount row member
  rw [(counts _).2, baseCount] at bounds
  exact bounds

private theorem bind_result {α β : Type} (input : Except String α)
    (next : α → Except String β) (result : β) (built : (input >>= next) = .ok result) :
    ∃ value, input = .ok value ∧ next value = .ok result := by
  cases input with
  | error message => simp [Bind.bind, Except.bind] at built
  | ok value => exact ⟨value, rfl, built⟩

private theorem builder_rows (program : ApplicationPackage.Program)
    (package : CircuitPackage) (plan : Stage1.ApplicationPackage.Plan)
    (counts : ∀ start, (ApplicationPackage.plan program start).privateCount =
      (PerApplicationPackage.applicationPlan program).privateCount ∧
      (ApplicationPackage.plan program start).rowCount = (PerApplicationPackage.applicationPlan program).rowCount)
    (built : ApplicationPackage.package program = .ok (package, plan)) :
    Nonempty (BuilderRows program (PhysicalPackage.common ()) package) := by
  obtain ⟨base, baseResult, extended⟩ := bind_result (PhysicalPackage.circuitPackage ())
    (ApplicationPackage.ofBase program) (package, plan) built
  apply builder_rows_of_base program (PhysicalPackage.common ()) base package plan _ counts baseResult extended
  rw [common_layout]
  rfl

private theorem insertion_injective (program : ApplicationPackage.Program) (left right value : Nat)
    (first : (ApplicationPackage.insertApplication (extra program)).row left = .ok value)
    (second : (ApplicationPackage.insertApplication (extra program)).row right = .ok value) : left = right := by
  exact (Except.ok.inj first).trans (Except.ok.inj second).symm

private theorem before_location (index : Nat)
    (beforeApplication : index < PerApplicationPackage.basePackage.layout.rowCount)
    (outsideSampler : index < Layout.Stage1.PiRLCStarts.phaseRowStart ∨
      Layout.Stage1.PiDECStarts.phaseRowStart ≤ index) :
    ∃ moved, PhysicalRelabel.row index = .ok moved ∧ moved < 27716409 ∧
      (moved < 19385261 ∨ 19444200 ≤ moved) := by
  have baseCount : PerApplicationPackage.basePackage.layout.rowCount = 28666318 :=
    PerApplicationPackage.basePackage_rowCount_eq
  rw [baseCount] at beforeApplication
  change index < 19385261 ∨ 28295335 ≤ index at outsideSampler
  change ∃ moved,
    (if index < 19385261 then Except.ok index else if 20394109 ≤ index then
      Except.ok (index - 949909) else Except.error _) = .ok moved ∧ _
  rcases outsideSampler with early | late
  · exact ⟨index, by simp [early], by omega, Or.inl early⟩
  · refine ⟨index - 949909, ?_, by omega, Or.inr (by omega)⟩
    rw [if_neg (by omega), if_pos (by omega)]

private theorem after_location (program : ApplicationPackage.Program) (index : Nat)
    (afterApplication : PerApplicationPackage.nextPreimageRowStart program ≤ index) :
    ∃ moved, PhysicalRelabel.row index = .ok moved ∧
      27716409 + (PerApplicationPackage.applicationPlan program).rowCount ≤ moved := by
  rw [PerApplicationPackage.nextPreimageRowStart, PerApplicationPackage.basePackage_rowCount_eq] at afterApplication
  refine ⟨index - 949909, ?_, by omega⟩
  change (if index < 19385261 then Except.ok index else if 20394109 ≤ index then
    Except.ok (index - 949909) else Except.error _) = .ok _
  rw [if_neg (by omega), if_pos (by omega)]

private theorem sampler_empty (index : Nat) (outside : index < 19385261 ∨ 19444200 ≤ index) :
    selected (PhysicalSampler.rows ()) index = [] := by
  apply selected_nil
  intro row member equal
  have bounds := sampler_bounds row member
  omega

private theorem base_source (program : ApplicationPackage.Program)
    (package : CircuitPackage) (archive : BuilderRows program (PhysicalPackage.common ()) package)
    (index : Nat) (value : R1CS.Row)
    (read : PackageSourceRows.packageSourceRow? (PerApplicationPackage.package program) index = some value)
    (beforeApplication : index < PerApplicationPackage.basePackage.layout.rowCount)
    (outsideSampler : index < Layout.Stage1.PiRLCStarts.phaseRowStart ∨
      Layout.Stage1.PiDECStarts.phaseRowStart ≤ index) :
    PhysicalMatrixSemantics.referenceSource (extra program) (PackageSourceRows.packageSourceRow? package) index = some value := by
  obtain ⟨original, originalSingle, valueEq⟩ :=
    common_of_reference program index value read beforeApplication outsideSampler
  obtain ⟨movedIndex, rowMap, beforeNewApplication, outsideNewSampler⟩ :=
    before_location index beforeApplication outsideSampler
  obtain ⟨moved, movedSingle, mapped⟩ := mapped_single ordinaryMap _ _ archive.commonPairs
    index movedIndex rowMap row_injective original originalSingle
  have baseSingle : selected (stored archive.base) movedIndex = [moved] := by
    apply selected_eq_of_perm archive.baseRows.symm movedIndex moved
    rw [selected_append, movedSingle, sampler_empty movedIndex outsideNewSampler, List.append_nil]
  obtain ⟨result, insertedSingle, inserted⟩ := mapped_single
    (ApplicationPackage.insertApplication (extra program)) _ _ archive.insertPairs
    movedIndex movedIndex rfl (insertion_injective program) moved baseSingle
  have applicationEmpty : selected archive.applicationRows movedIndex = [] := by
    apply selected_nil
    intro row member equal
    have bound := (archive.applicationBounds row member).1
    omega
  have nextEmpty : selected archive.next movedIndex = [] := by
    apply mapped_empty ordinaryMap _ _ archive.nextPairs index movedIndex rowMap row_injective
    apply selected_nil
    intro row member equal
    have bound := next_bounds program row member
    unfold PerApplicationPackage.nextPreimageRowStart at bound
    omega
  have finalSingle : selected (stored package) movedIndex = [result] := by
    apply selected_eq_of_perm archive.finalRows.symm movedIndex result
    simp only [selected_append, insertedSingle, applicationEmpty, nextEmpty, List.append_nil]
  have sourceRead := source_eq_of_selected package movedIndex result finalSingle
  have recovered := PhysicalMatrixRecovery.ordinary_row_recovered_inserted program original moved result mapped inserted
  rw [valueEq] at recovered
  simpa only [PhysicalMatrixSemantics.referenceSource, rowMap, Except.toOption,
    Bind.bind, Option.bind_some, sourceRead] using recovered

private theorem next_source (program : ApplicationPackage.Program)
    (package : CircuitPackage) (archive : BuilderRows program (PhysicalPackage.common ()) package)
    (index : Nat) (value : R1CS.Row)
    (read : PackageSourceRows.packageSourceRow? (PerApplicationPackage.package program) index = some value)
    (afterApplication : PerApplicationPackage.nextPreimageRowStart program ≤ index) :
    PhysicalMatrixSemantics.referenceSource (extra program) (PackageSourceRows.packageSourceRow? package) index = some value := by
  obtain ⟨original, originalSingle, valueEq⟩ := next_of_reference program index value read afterApplication
  obtain ⟨movedIndex, rowMap, afterNewApplication⟩ := after_location program index afterApplication
  obtain ⟨result, nextSingle, mapped⟩ := mapped_single ordinaryMap _ _ archive.nextPairs
    index movedIndex rowMap row_injective original originalSingle
  have commonEmpty : selected archive.common movedIndex = [] := by
    apply mapped_empty ordinaryMap _ _ archive.commonPairs index movedIndex rowMap row_injective
    apply selected_nil
    intro row member equal
    have commonMember := common_stored.mem_iff.mp member
    have bound := PerApplicationPackageSourceRows.baseRows_rowIndex_lt row (commonRows_sublist.subset commonMember)
    unfold PerApplicationPackage.nextPreimageRowStart at afterApplication
    omega
  have baseEmpty : selected (stored archive.base) movedIndex = [] := by
    have permutation := selected_perm archive.baseRows movedIndex
    rw [selected_append, commonEmpty, sampler_empty movedIndex (Or.inr (by omega)), List.nil_append] at permutation
    exact List.perm_nil.mp permutation
  have insertedEmpty := mapped_empty (ApplicationPackage.insertApplication (extra program)) _ _ archive.insertPairs
    movedIndex movedIndex rfl (insertion_injective program) baseEmpty
  have applicationEmpty : selected archive.applicationRows movedIndex = [] := by
    apply selected_nil
    intro row member equal
    have bound := (archive.applicationBounds row member).2
    omega
  have finalSingle : selected (stored package) movedIndex = [result] := by
    apply selected_eq_of_perm archive.finalRows.symm movedIndex result
    simp only [selected_append, insertedEmpty, applicationEmpty, nextSingle, List.nil_append]
  have sourceRead := source_eq_of_selected package movedIndex result finalSingle
  have recovered := PhysicalMatrixRecovery.ordinary_row_recovered (extra program) original result mapped
  rw [valueEq] at recovered
  simpa only [PhysicalMatrixSemantics.referenceSource, rowMap, Except.toOption,
    Bind.bind, Option.bind_some, sourceRead] using recovered

/-- Every reused ordinary source row is read from the successfully emitted
wide archive. No caller supplies source custody or source-row equalities. -/
theorem custody (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (package : CircuitPackage) (plan : Stage1.ApplicationPackage.Plan)
    (built : ApplicationPackage.package application = .ok (package, plan)) :
    ReusedMatrixPrograms.SourceCustody application (FixedPoint.relation application compiled fits)
      fits.package (PhysicalMatrixSemantics.referenceSource
        (PerApplicationPackage.directAddedPrivateColumnCount application)
        (PackageSourceRows.packageSourceRow? package)) := by
  obtain ⟨archive⟩ := builder_rows application package plan ApplicationPackageCounts.plan_counts built
  have old := ReusedMatrixPrograms.source_custody application (FixedPoint.relation application compiled fits)
    fits
  have baseCount : PerApplicationPackage.basePackage.layout.rowCount = 28666318 :=
    PerApplicationPackage.basePackage_rowCount_eq
  refine {
    piCcsOrdinary := ?_
    pilotOrdinary := ?_
    piDecPublic := ?_
    piDecCommitment := ?_
    piDecEvalK := ?_
    piDecEvalA := ?_
    applicationRows := ?_
    nextPreimage := ?_ }
  · intro index sourceIndex selectedRow
    have member : sourceIndex ∈ PiCCSOrdinaryMatrixProgram.rowIndexReference := by
      rw [PiCCSOrdinaryMatrixProgram.rowSchedule_index?] at selectedRow
      exact List.mem_of_getElem? selectedRow
    have bound := (PiCCSOrdinaryMatrixProgram.rowIndexReference_bounds sourceIndex member).2
    apply base_source application package archive sourceIndex _ (old.piCcsOrdinary index sourceIndex selectedRow)
    · rw [baseCount]
      change sourceIndex < 19385261 at bound
      omega
    · exact Or.inl bound
  · intro index
    have member : PilotOrdinaryMatrixProgram.rowIndexAt index ∈
        PerApplicationPackageSourceRows.pilotRows.map Rows.CompiledRow.rowIndex := by
      have member := List.get_mem PilotOrdinaryMatrixProgram.rowIndexReference
        (Fin.cast PilotOrdinaryMatrixProgram.rowIndexReference_length.symm index)
      simpa only [PilotOrdinaryMatrixProgram.rowIndexAt, PilotOrdinaryMatrixProgram.rowIndexReference,
        PilotOrdinaryMatrixProgram.instructionIndices, PilotOrdinaryMatrixProgram.assertionIndices,
        PerApplicationPackageSourceRows.pilotRows, PackageSourceRows.decodedRows,
        List.map_append, List.map_map, Function.comp_def, Rows.CompiledRow.rowIndex] using member
    obtain ⟨row, rowMember, indexEq⟩ := List.mem_map.mp member
    have bound := PerApplicationPackageSourceRows.pilotRows_rowIndex_lt row rowMember
    rw [indexEq] at bound
    have start : PiCCSArithmetic.statementBindingRowStart = 14623730 := by
      unfold PiCCSArithmetic.statementBindingRowStart Layout.Stage1.PiCCSStarts.statementBindingRowStart
        Layout.Stage1.PiCCSStarts.rowBase
      exact Layout.PilotProduction.physicalRowCountValue_eq
    rw [start] at bound
    apply base_source application package archive _ _ (old.pilotOrdinary index)
    · rw [baseCount]; omega
    · left; change _ < 19385261; omega
  · intro index
    apply base_source application package archive _ _ (old.piDecPublic index)
    · rw [baseCount]
      change 28295335 + index.val < 28666318
      have bound := index.isLt
      omega
    · right; change 28295335 ≤ 28295335 + index.val; omega
  · intro index
    apply base_source application package archive _ _ (old.piDecCommitment index)
    · rw [baseCount]
      change 28318015 + index.val < 28666318
      have bound := index.isLt
      omega
    · right; change 28295335 ≤ 28318015 + index.val; omega
  · intro index
    apply base_source application package archive _ _ (old.piDecEvalK index)
    · rw [baseCount]
      change 28319203 + index.val < 28666318
      have bound := index.isLt
      omega
    · right; change 28295335 ≤ 28319203 + index.val; omega
  · intro index
    apply base_source application package archive _ _ (old.piDecEvalA index)
    · rw [baseCount]
      change 28319311 + index.val < 28666318
      have bound := index.isLt
      omega
    · right; change 28295335 ≤ 28319311 + index.val; omega
  · intro absent
    cases absent
  · intro index
    apply next_source application package archive _ _ (old.nextPreimage index)
    omega

/-- The complete relocated matrix program is exact for the selected wide
structural plan and the actual emitted physical archive. -/
theorem exact (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (package : CircuitPackage) (plan : Stage1.ApplicationPackage.Plan)
    (built : ApplicationPackage.package application = .ok (package, plan))
    (program : Layout.MatrixProgram.Program)
    (emitted : PhysicalMatrixSource.program application compiled = .ok program) :
    Layout.MatrixProgram.Exact program (FixedPoint.structuralPlan application compiled fits)
      (PackageSourceRows.packageSourceRow? package) :=
  PhysicalMatrixSemantics.exact application compiled fits program emitted
    (PackageSourceRows.packageSourceRow? package) (custody compiled fits package plan built)

end NightstreamFPrime.Export.Stage1.Wide.PhysicalSourceCustody
