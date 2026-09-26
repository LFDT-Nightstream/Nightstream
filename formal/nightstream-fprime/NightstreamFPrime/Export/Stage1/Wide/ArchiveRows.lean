import NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage
import NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceRows
import NightstreamFPrime.Layout.PiRlcWideSampler.RangePhysical
import NightstreamFPrime.Layout.PiRlcWideSampler.DigitProjection

/-! The physical archive's ordinary row lists and their checked relocation.
The proofs use list structure and child footprint theorems, never emitted data. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ArchiveRows

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open PhysicalRelabel

def stored (package : CircuitPackage) : List Rows.CompiledRow :=
  PackageSourceRows.decodedRows package.witnessInstructions package.assertionRows

def selected (rows : List Rows.CompiledRow) (index : Nat) : List Rows.CompiledRow :=
  rows.filter (fun row => row.rowIndex == index)

theorem selected_perm {before after : List Rows.CompiledRow}
    (permutation : before.Perm after) (index : Nat) :
    (selected before index).Perm (selected after index) :=
  permutation.filter _

theorem selected_eq_of_perm {before after : List Rows.CompiledRow}
    (permutation : before.Perm after) (index : Nat) (row : Rows.CompiledRow)
    (single : selected before index = [row]) : selected after index = [row] := by
  have same := selected_perm permutation index
  rw [single] at same
  simpa using same.symm

theorem source_eq_of_selected (package : CircuitPackage) (index : Nat)
    (row : Rows.CompiledRow) (single : selected (stored package) index = [row]) :
    PackageSourceRows.packageSourceRow? package index = some row.toR1CS := by
  change (match selected (stored package) index with | [row] => some row.toR1CS | _ => none) = _
  rw [single]

private theorem selected_of_source (package : CircuitPackage) (index : Nat) (value : R1CS.Row)
    (read : PackageSourceRows.packageSourceRow? package index = some value) :
    ∃ row, selected (stored package) index = [row] ∧ row.toR1CS = value := by
  change (match selected (stored package) index with | [row] => some row.toR1CS | _ => none) = _ at read
  cases found : selected (stored package) index with
  | nil => simp [found] at read
  | cons row rest =>
    cases rest with
    | nil =>
      rw [found] at read
      exact ⟨row, rfl, Option.some.inj read⟩
    | cons next tail => simp [found] at read

theorem selected_nil (rows : List Rows.CompiledRow) (index : Nat)
    (absent : ∀ row ∈ rows, row.rowIndex ≠ index) : selected rows index = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro row member
  simpa using absent row member

theorem selected_append (left right : List Rows.CompiledRow) (index : Nat) :
    selected (left ++ right) index = selected left index ++ selected right index :=
  List.filter_append left right

theorem row_injective (left right target : Nat)
    (first : PhysicalRelabel.row left = .ok target)
    (second : PhysicalRelabel.row right = .ok target) : left = right := by
  have oldStart : Layout.Stage1.PiRLCStarts.phaseRowStart = 19385261 := rfl
  have oldEnd : Layout.Stage1.PiRLCStarts.commitmentRowStart = 20394109 := rfl
  have newEnd : Layout.Stage1.Wide.PiRLCStarts.commitmentRowStart = 19444200 := rfl
  simp only [PhysicalRelabel.row, oldStart, oldEnd, newEnd] at first second
  split_ifs at first second <;> simp_all <;> omega

private theorem mapped_selected (mapping : Map) (before after : List Rows.CompiledRow)
    (pairs : List.Forall₂ (fun a b => mapping.compiledRow a = .ok b) before after)
    (index target : Nat) (moved : mapping.row index = .ok target)
    (injective : ∀ left right value, mapping.row left = .ok value →
      mapping.row right = .ok value → left = right) :
    List.Forall₂ (fun a b => mapping.compiledRow a = .ok b)
      (selected before index) (selected after target) := by
  apply List.rel_filter (R := fun a b => mapping.compiledRow a = .ok b) _ pairs
  intro a b pair
  have indices := (mapping.compiledRow_correct a b pair).1
  simp only [beq_iff_eq]
  constructor
  · intro equal
    rw [equal, moved] at indices
    exact (Except.ok.inj indices).symm
  · intro equal
    rw [equal] at indices
    exact injective a.rowIndex index target indices moved

theorem mapped_single (mapping : Map) (before after : List Rows.CompiledRow)
    (pairs : List.Forall₂ (fun a b => mapping.compiledRow a = .ok b) before after)
    (index target : Nat) (moved : mapping.row index = .ok target)
    (injective : ∀ left right value, mapping.row left = .ok value →
      mapping.row right = .ok value → left = right)
    (row : Rows.CompiledRow) (single : selected before index = [row]) :
    ∃ relocated, selected after target = [relocated] ∧ mapping.compiledRow row = .ok relocated := by
  have filtered := mapped_selected mapping before after pairs index target moved injective
  rw [single] at filtered
  obtain ⟨relocated, tail, relation, empty, equation⟩ := List.forall₂_cons_left_iff.mp filtered
  have emptyEq := List.forall₂_nil_left_iff.mp empty
  exact ⟨relocated, by simpa only [emptyEq] using equation, relation⟩

theorem mapped_empty (mapping : Map) (before after : List Rows.CompiledRow)
    (pairs : List.Forall₂ (fun a b => mapping.compiledRow a = .ok b) before after)
    (index target : Nat) (moved : mapping.row index = .ok target)
    (injective : ∀ left right value, mapping.row left = .ok value →
      mapping.row right = .ok value → left = right)
    (empty : selected before index = []) : selected after target = [] := by
  have filtered := mapped_selected mapping before after pairs index target moved injective
  rw [empty] at filtered
  exact List.forall₂_nil_left_iff.mp filtered

theorem decoded_pairs (mapping : Map)
    (instructions movedInstructions : List WitnessInstruction)
    (assertions movedAssertions : List SparseRow)
    (instructionMap : instructions.mapM mapping.instruction = .ok movedInstructions)
    (assertionMap : assertions.mapM mapping.assertion = .ok movedAssertions) :
    List.Forall₂ (fun a b => mapping.compiledRow a = .ok b)
      (PackageSourceRows.decodedRows instructions assertions)
      (PackageSourceRows.decodedRows movedInstructions movedAssertions) := by
  have witnessPairs := mapM_pairs mapping.instruction instructions movedInstructions instructionMap
  have assertionPairs := mapM_pairs mapping.assertion assertions movedAssertions assertionMap
  apply List.rel_append
  · apply List.rel_map _ witnessPairs
    intro before after emitted
    simp [Map.compiledRow, emitted]
  · apply List.rel_map _ assertionPairs
    intro before after emitted
    simp [Map.compiledRow, emitted]

private theorem bind_ok {α β : Type} (input : Except String α) (next : α → Except String β) (output : β) :
    (input >>= next) = .ok output ↔ ∃ value, input = .ok value ∧ next value = .ok output := by
  cases input <;> simp [Bind.bind, Except.bind]

private theorem inserted_perm {α : Type} (predicate : α → Bool) (before added : List α) :
    ((before.span predicate).1 ++ added ++ (before.span predicate).2).Perm (before ++ added) := by
  rw [List.span_eq_takeWhile_dropWhile]
  have swapped : (before.takeWhile predicate ++ added ++ before.dropWhile predicate).Perm
      ((before.takeWhile predicate ++ before.dropWhile predicate) ++ added) := by
    simpa only [List.append_assoc] using
      (List.perm_append_comm (l₁ := added) (l₂ := before.dropWhile predicate)).append_left
        (before.takeWhile predicate)
  simpa only [List.takeWhile_append_dropWhile] using swapped

/-- The successful physical builder carries the checked common rows and the
new sampler rows. The ordering split does not change the ordinary-row multiset. -/
theorem physical_fields (before after : CircuitPackage)
    (emitted : PhysicalPackage.ofCommon before = .ok after) :
    ∃ instructions assertions,
      before.witnessInstructions.mapM ordinaryMap.instruction = .ok instructions ∧
      before.assertionRows.mapM ordinaryMap.assertion = .ok assertions ∧
      PhysicalRelabel.row before.layout.rowCount = .ok after.layout.rowCount ∧
      (stored after).Perm
        (PackageSourceRows.decodedRows instructions assertions ++ PhysicalSampler.rows ()) := by
  unfold PhysicalPackage.ofCommon at emitted
  simp only [bind_ok] at emitted
  obtain ⟨count, countMap, privateSegments, privateMap, publicSegments, publicMap,
    batches, batchMap, instructions, instructionMap, assertions, assertionMap,
    chains, chainMap, permutations, permutationMap, sampler, samplerMap,
    compact, compactMap, result⟩ := emitted
  simp only [Pure.pure, Except.pure, Except.ok.injEq] at result
  subst after
  refine ⟨instructions, assertions, instructionMap, assertionMap, countMap, ?_⟩
  change List.Perm (PackageSourceRows.decodedRows _ _) _
  have instructionPerm := inserted_perm (fun item : WitnessInstruction =>
    item.rowIndex < Layout.Stage1.Wide.PiRLCStarts.samplerRowStart)
    instructions (Rows.witnessInstructionsTR (PhysicalSampler.rows ()))
  have assertionPerm := inserted_perm (fun item : SparseRow =>
    item.rowIndex < Layout.Stage1.Wide.PiRLCStarts.samplerRowStart)
    assertions (Rows.assertionRowsTR (PhysicalSampler.rows ()))
  have rearranged := (instructionPerm.map Rows.CompiledRow.witness).append
    (assertionPerm.map Rows.CompiledRow.assertion)
  refine rearranged.trans ((PackageSourceRows.decodedRows_append_perm _ _ _ _).trans ?_)
  rw [Rows.witnessInstructionsTR_eq, Rows.assertionRowsTR_eq]
  exact (PackageSourceRows.classifiedRows_perm _).append_left _

/-- The application builder adds exactly its classified rows and the checked
next-preimage assertions to the checked base archive. -/
theorem application_fields (application : ApplicationPackage.Program) (base final : CircuitPackage)
    (plan : Stage1.ApplicationPackage.Plan)
    (emitted : ApplicationPackage.ofBase application base = .ok (final, plan)) :
    plan = ApplicationPackage.plan application base.layout.rowCount ∧
    ∃ instructions assertions nextRows,
      base.witnessInstructions.mapM
        (ApplicationPackage.insertApplication (application.witnessWordCount + plan.privateCount)).instruction =
        .ok instructions ∧
      base.assertionRows.mapM
        (ApplicationPackage.insertApplication (application.witnessWordCount + plan.privateCount)).assertion =
        .ok assertions ∧
      (NextPreimagePackage.assertionRows (PerApplicationPackage.nextPreimageRowStart application)).mapM
        ordinaryMap.assertion = .ok nextRows ∧
      (stored final).Perm (PackageSourceRows.decodedRows instructions assertions ++
        PackageSourceRows.decodedRows plan.witnessInstructions plan.assertionRows ++
        PackageSourceRows.decodedRows [] nextRows) := by
  unfold ApplicationPackage.ofBase at emitted
  simp only [bind_ok] at emitted
  obtain ⟨nextRows, nextMap, segments, segmentMap, chains, chainMap, permutations, permutationMap,
    compact, compactMap, batches, batchMap, instructions, instructionMap,
    assertions, assertionMap, result⟩ := emitted
  simp only [Pure.pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at result
  obtain ⟨result, planEq⟩ := result
  subst final
  subst plan
  refine ⟨rfl, instructions, assertions, nextRows, instructionMap, assertionMap, nextMap, ?_⟩
  unfold stored
  rw [TerminalPackage.install_witnessInstructions, TerminalPackage.install_assertionRows]
  have outer := PackageSourceRows.decodedRows_append_perm
    (instructions ++ (ApplicationPackage.plan application base.layout.rowCount).witnessInstructions) []
    (assertions ++ (ApplicationPackage.plan application base.layout.rowCount).assertionRows) nextRows
  simp only [List.append_nil] at outer
  exact outer.trans ((PackageSourceRows.decodedRows_append_perm _ _ _ _).append_right _)

def commonRows : List Rows.CompiledRow :=
  PerApplicationPackageSourceRows.pilotRows.map PerApplicationPackageSourceRows.liftPilotCompiledRow ++
    (PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits ++
      (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits).rows ++
      (RunningTransitionArithmetic.canonicalPlan Data.logicalWidth Data.publicFits).rows)

theorem commonRows_sublist : List.Sublist commonRows PerApplicationPackageSourceRows.baseRows := by
  unfold commonRows PerApplicationPackageSourceRows.baseRows Data.arithmeticRows
  simp only [List.append_assoc]
  exact (List.Sublist.refl _).append ((List.Sublist.refl _).append (List.sublist_append_right _ _))

private theorem commonRows_unique : (commonRows.map Rows.CompiledRow.rowIndex).Nodup :=
  List.Nodup.sublist (commonRows_sublist.map _) PerApplicationPackageSourceRows.baseRows_rowIndices_nodup

theorem common_stored : (stored (PhysicalPackage.common ())).Perm commonRows := by
  change List.Perm
    (PackageSourceRows.decodedRows
      (Data.liftPilotInstructions (PilotData.witnessInstructions ()) ++ Rows.witnessInstructionsTR _)
      (Data.liftPilotRows (PilotData.assertionRows ()) ++ Rows.assertionRowsTR _)) _
  refine (PackageSourceRows.decodedRows_append_perm _ _ _ _).trans ?_
  rw [PerApplicationPackageSourceRows.decodedRows_liftPilot,
    Rows.witnessInstructionsTR_eq, Rows.assertionRowsTR_eq]
  exact (PackageSourceRows.classifiedRows_perm _).append_left _

private theorem selected_unique (rows : List Rows.CompiledRow)
    (unique : (rows.map Rows.CompiledRow.rowIndex).Nodup)
    (target : Rows.CompiledRow) (member : target ∈ rows) :
    selected rows target.rowIndex = [target] := by
  induction rows with
  | nil => simp at member
  | cons head tail ih =>
    obtain ⟨absent, tailUnique⟩ := List.nodup_cons.mp unique
    rcases List.mem_cons.mp member with equal | member
    · subst target
      have empty := selected_nil tail head.rowIndex (by
        intro row member equal
        apply absent
        rw [← equal]
        exact List.mem_map_of_mem member)
      simpa [selected] using congrArg (List.cons head) empty
    · have different : head.rowIndex ≠ target.rowIndex := by
        intro equal
        apply absent
        rw [equal]
        exact List.mem_map_of_mem member
      simpa [selected, different] using ih tailUnique member

/-- Recover the actual compiled common row selected by an existing source
custody theorem. The old sampler and application suffix cannot own this index. -/
theorem common_of_reference (application : ApplicationPackage.Program) (index : Nat) (value : R1CS.Row)
    (read : PackageSourceRows.packageSourceRow? (PerApplicationPackage.package application) index = some value)
    (beforeApplication : index < PerApplicationPackage.basePackage.layout.rowCount)
    (outsideSampler : index < Layout.Stage1.PiRLCStarts.phaseRowStart ∨
      Layout.Stage1.PiDECStarts.phaseRowStart ≤ index) :
    ∃ row, selected (stored (PhysicalPackage.common ())) index = [row] ∧
      PerApplicationSourceProjection.basePackageRow application row.toR1CS = value := by
  obtain ⟨target, single, valueEq⟩ := selected_of_source _ index value read
  have canonicalSingle := selected_eq_of_perm
    (PerApplicationPackageSourceRows.package_decodedRows_perm_canonical application) index target single
  have member : target ∈ PerApplicationPackageSourceRows.canonicalRows application ∧ target.rowIndex = index := by
    have found : target ∈ selected (PerApplicationPackageSourceRows.canonicalRows application) index := by
      rw [canonicalSingle]
      exact List.mem_singleton_self _
    simpa only [selected, List.mem_filter, beq_iff_eq] using found
  rcases member with ⟨member, indexEq⟩
  simp only [PerApplicationPackageSourceRows.canonicalRows, List.mem_append] at member
  rcases member with (baseMember | applicationMember) | nextMember
  · obtain ⟨original, originalMember, rfl⟩ := List.mem_map.mp baseMember
    rw [PerApplicationPackageSourceRows.shiftCompiledRow_rowIndex] at indexEq
    have commonMember : original ∈ commonRows := by
      simp only [PerApplicationPackageSourceRows.baseRows, Data.arithmeticRows,
        List.append_assoc, List.mem_append] at originalMember
      simp only [commonRows, List.append_assoc, List.mem_append]
      rcases originalMember with pilot | ccs | sampler | dec | running
      · exact Or.inl pilot
      · exact Or.inr (Or.inl ccs)
      · have bound := PiRLCSamplerOrdinaryMatrixSchedule.rowIndexReference_bounds original.rowIndex (by
          rw [← PiRLCSamplerOrdinaryMatrixSchedule.arithmeticRows_rowIndices
            (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)]
          exact List.mem_map_of_mem sampler)
        rw [indexEq] at bound
        have endpoint : Layout.Stage1.PiRLCStarts.outputRowStart = Layout.Stage1.PiDECStarts.phaseRowStart := rfl
        rw [endpoint] at bound
        omega
      · exact Or.inr (Or.inr (Or.inl dec))
      · exact Or.inr (Or.inr (Or.inr running))
    refine ⟨original, ?_, ?_⟩
    · apply selected_eq_of_perm common_stored.symm index original
      rw [← indexEq]
      exact selected_unique commonRows commonRows_unique original commonMember
    · simpa only [PerApplicationPackageSourceRows.shiftCompiledRow_toR1CS] using valueEq
  · have lower := PerApplicationPackageSourceRows.applicationRows_rowIndex_ge application target.rowIndex
      (List.mem_map_of_mem applicationMember)
    omega
  · have lower : PerApplicationPackage.nextPreimageRowStart application ≤ target.rowIndex := by
      have found := List.mem_map_of_mem (f := Rows.CompiledRow.rowIndex) nextMember
      rw [PerApplicationPackageSourceRows.nextPreimageRows_rowIndices, List.mem_range'_1] at found
      exact found.1
    unfold PerApplicationPackage.nextPreimageRowStart at lower
    omega

private theorem packet_bounds (rowStart freshStart : Nat) (constraints : List Expr)
    (row : Rows.CompiledRow) (member : row ∈ PhysicalSampler.compilePacket rowStart freshStart constraints) :
    rowStart ≤ row.rowIndex ∧ row.rowIndex < rowStart + R1CS.totalRowCount constraints := by
  have found := List.mem_map_of_mem (f := Rows.CompiledRow.rowIndex) member
  rw [PhysicalSampler.compilePacket, Rows.compileRowsTR_rowIndices, List.length_map,
    Rows.lowerConstraintsTR_eq, R1CS.lowerConstraints_rows_length, List.mem_range'_1] at found
  exact found

theorem sampler_bounds (row : Rows.CompiledRow) (member : row ∈ PhysicalSampler.rows ()) :
    19385261 ≤ row.rowIndex ∧ row.rowIndex < 19444200 := by
  rw [PhysicalSampler.rows, List.mem_append] at member
  rcases member with rangeMember | digitMember
  · obtain ⟨source, sourceMember, member⟩ := List.mem_flatMap.mp rangeMember
    have bound := packet_bounds _ _ _ row member
    have count : R1CS.totalRowCount (flatConstraints (PhysicalSampler.rangeOperations source)) = 2229 := by
      rw [PhysicalSampler.rangeOperations, Gadgets.Sampling.WideReduction.Program.constraints_eq]
      exact (PiRlcWideSampler.RangePhysical.counts _ _ _ (fun _ => R1CS.isAffine_var _)).2
    rw [count] at bound
    have sourceBound := List.mem_range.mp sourceMember
    simp only [Layout.Stage1.Wide.PiRLCStarts.rangeRowStart,
      Layout.Stage1.Wide.PiRLCStarts.samplerSourceRowStart,
      Layout.Stage1.Wide.PiRLCStarts.samplerRowStart,
      Layout.Stage1.Wide.PiRLCStarts.phaseRowStart] at bound
    omega
  · have bound := packet_bounds _ _ _ row digitMember
    rw [PhysicalSampler.digitOperations,
      (PiRlcWideSampler.DigitProjection.r1cs_counts _ _).2] at bound
    change 19385261 + 17 * 3413 ≤ row.rowIndex ∧ row.rowIndex < 19385261 + 17 * 3413 + 918 at bound
    omega

theorem application_bounds (application : ApplicationPackage.Program) (rowStart : Nat)
    (row : Rows.CompiledRow)
    (member : row ∈ PackageSourceRows.decodedRows
      (ApplicationPackage.plan application rowStart).witnessInstructions
      (ApplicationPackage.plan application rowStart).assertionRows) :
    rowStart ≤ row.rowIndex ∧ row.rowIndex < rowStart + (ApplicationPackage.plan application rowStart).rowCount := by
  let rows := Stage1.ApplicationPackage.compiledRows application (ApplicationPackage.columns application)
    (Layout.Stage1.Wide.SourceOrder.privateColumns + application.witnessWordCount) rowStart
  have actual : row ∈ rows := by
    apply (PackageSourceRows.classifiedRows_perm rows).mem_iff.mp
    change row ∈ PackageSourceRows.decodedRows (Rows.witnessInstructionsTR rows) (Rows.assertionRowsTR rows) at member
    simpa only [PackageSourceRows.classifiedRows, Rows.witnessInstructionsTR_eq, Rows.assertionRowsTR_eq]
      using member
  have found := List.mem_map_of_mem (f := Rows.CompiledRow.rowIndex) actual
  have indices : rows.map Rows.CompiledRow.rowIndex = List.range' rowStart rows.length := by
    unfold rows Stage1.ApplicationPackage.compiledRows
    rw [Rows.compileRowsTR_rowIndices, Rows.compileRowsTR_length]
  rw [indices, List.mem_range'_1] at found
  exact found

theorem next_bounds (application : ApplicationPackage.Program) (row : Rows.CompiledRow)
    (member : row ∈ PackageSourceRows.decodedRows []
      (NextPreimagePackage.assertionRows (PerApplicationPackage.nextPreimageRowStart application))) :
    PerApplicationPackage.nextPreimageRowStart application ≤ row.rowIndex := by
  have actual := (PerApplicationPackageSourceRows.nextPreimage_decodedRows_perm application).mem_iff.mp member
  have found := List.mem_map_of_mem (f := Rows.CompiledRow.rowIndex) actual
  rw [PerApplicationPackageSourceRows.nextPreimageRows_rowIndices, List.mem_range'_1] at found
  exact found.1

theorem next_of_reference (application : ApplicationPackage.Program) (index : Nat) (value : R1CS.Row)
    (read : PackageSourceRows.packageSourceRow? (PerApplicationPackage.package application) index = some value)
    (afterApplication : PerApplicationPackage.nextPreimageRowStart application ≤ index) :
    ∃ row, selected (PackageSourceRows.decodedRows []
      (NextPreimagePackage.assertionRows (PerApplicationPackage.nextPreimageRowStart application))) index = [row] ∧
      row.toR1CS = value := by
  obtain ⟨target, single, valueEq⟩ := selected_of_source _ index value read
  have canonicalSingle := selected_eq_of_perm
    (PerApplicationPackageSourceRows.package_decodedRows_perm_canonical application) index target single
  have member : target ∈ PerApplicationPackageSourceRows.canonicalRows application ∧ target.rowIndex = index := by
    have found : target ∈ selected (PerApplicationPackageSourceRows.canonicalRows application) index := by
      rw [canonicalSingle]
      exact List.mem_singleton_self _
    simpa only [selected, List.mem_filter, beq_iff_eq] using found
  rcases member with ⟨member, indexEq⟩
  simp only [PerApplicationPackageSourceRows.canonicalRows, List.mem_append] at member
  rcases member with (baseMember | applicationMember) | nextMember
  · obtain ⟨original, originalMember, rfl⟩ := List.mem_map.mp baseMember
    rw [PerApplicationPackageSourceRows.shiftCompiledRow_rowIndex] at indexEq
    have bound := PerApplicationPackageSourceRows.baseRows_rowIndex_lt original originalMember
    unfold PerApplicationPackage.nextPreimageRowStart at afterApplication
    omega
  · have found := List.mem_map_of_mem (f := Rows.CompiledRow.rowIndex) applicationMember
    rw [PerApplicationPackageSourceRows.applicationRows_rowIndices, List.mem_range'_1] at found
    have lengthEq : (PerApplicationPackageSourceRows.applicationRows application).length =
        (PerApplicationPackage.applicationPlan application).rowCount := by
      have count := ApplicationDirectSource.sourceRows_length_eq_plan application
      unfold ApplicationDirectSource.sourceRows at count
      rw [List.length_map] at count
      exact count
    rw [lengthEq] at found
    unfold PerApplicationPackage.nextPreimageRowStart at afterApplication
    omega
  · refine ⟨target, ?_, valueEq⟩
    apply selected_eq_of_perm (PerApplicationPackageSourceRows.nextPreimage_decodedRows_perm application).symm
    rw [← indexEq]
    apply selected_unique _ _ target nextMember
    rw [PerApplicationPackageSourceRows.nextPreimageRows_rowIndices]
    exact List.nodup_range'

end NightstreamFPrime.Export.Stage1.Wide.ArchiveRows
