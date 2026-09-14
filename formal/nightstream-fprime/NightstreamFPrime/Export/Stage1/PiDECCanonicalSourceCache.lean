import Std.Data.HashMap.Lemmas
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.PerApplicationCachedShift

/-!
Index the existing canonical source rows once. Keys remain physical source
indices and values remain R1CS.Row. No source row is read from an artifact.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache

open NightstreamFPrime.Layout

/-- Apply the existing shared column-shift context to a compiled row. -/
def shiftCompiledRow (context : PerApplicationCachedShift.Context) :
    Rows.CompiledRow → Rows.CompiledRow
  | .witness instruction => .witness
      (PerApplicationCachedShift.shiftWitnessInstruction context instruction)
  | .assertion row => .assertion
      (PerApplicationCachedShift.shiftSparseRow context row)

theorem shiftCompiledRow_eq (context : PerApplicationCachedShift.Context)
    (row : Rows.CompiledRow) :
    shiftCompiledRow context row =
      PerApplicationPackageSourceRows.shiftCompiledRow context.program row := by
  cases row <;>
    simp only [shiftCompiledRow,
      PerApplicationPackageSourceRows.shiftCompiledRow,
      PerApplicationCachedShift.shiftWitnessInstruction_eq,
      PerApplicationCachedShift.shiftSparseRow_eq]

private def applicationRows (application : Lifecycle.Stage1.Application.Program) :
    List Rows.CompiledRow :=
  ApplicationPackage.compiledRows application
    (ApplicationPackage.productionColumns application)
    (Layout.Stage1.ApplicationInputs.localStart application)
    Data.physicalLayout.rowCount

private theorem applicationRows_eq (application : Lifecycle.Stage1.Application.Program) :
    applicationRows application =
      PerApplicationPackageSourceRows.applicationRows application := by
  unfold applicationRows PerApplicationPackageSourceRows.applicationRows
    PerApplicationPackage.basePackage
  rw [Data.circuitPackage_layout]

/-- Construct the canonical collection with one shared application delta.
Application rows use the proved layout count without constructing a package. -/
def rows (application : Lifecycle.Stage1.Application.Program) :
    List Rows.CompiledRow :=
  let context := PerApplicationCachedShift.Context.ofProgram application
  (PerApplicationPackageSourceRows.baseRows.map (shiftCompiledRow context) ++
    applicationRows application) ++
    PerApplicationPackageSourceRows.nextPreimageRows application

theorem rows_eq_canonicalRows
    (application : Lifecycle.Stage1.Application.Program) :
    rows application =
      PerApplicationPackageSourceRows.canonicalRows application := by
  have functionEq :
      shiftCompiledRow (PerApplicationCachedShift.Context.ofProgram application) =
        PerApplicationPackageSourceRows.shiftCompiledRow application := by
    funext row
    exact shiftCompiledRow_eq
      (PerApplicationCachedShift.Context.ofProgram application) row
  unfold rows PerApplicationPackageSourceRows.canonicalRows
  dsimp only
  rw [functionEq, applicationRows_eq]

/-- A transient index of the already projected canonical package rows.
Bind this value once outside the matrix-row loop. -/
def stored (application : Lifecycle.Stage1.Application.Program) :
    Std.HashMap Nat R1CS.Row :=
  Std.HashMap.ofList
    ((rows application).map fun row => (row.rowIndex, row.toR1CS))

private theorem indexed_value (rows : List Rows.CompiledRow)
    (instructions : List Package.WitnessInstruction)
    (assertions : List Package.SparseRow)
    (unique : (rows.map Rows.CompiledRow.rowIndex).Nodup)
    (permuted : (PackageSourceRows.decodedRows instructions assertions).Perm rows)
    (source : Nat) :
    (Std.HashMap.ofList (rows.map fun row => (row.rowIndex, row.toR1CS)))[source]? =
      PackageSourceRows.sourceRow? instructions assertions source := by
  have distinct :
      (rows.map fun row => (row.rowIndex, row.toR1CS)).Pairwise
        (fun left right => (left.1 == right.1) = false) := by
    have indexDistinct : rows.Pairwise
        (fun left right => left.rowIndex ≠ right.rowIndex) :=
      List.pairwise_map.mp (List.nodup_iff_pairwise_ne.mp unique)
    apply List.pairwise_map.mpr
    exact indexDistinct.imp fun different =>
      Bool.eq_false_iff.mpr fun found => different (beq_iff_eq.mp found)
  by_cases present : source ∈ rows.map Rows.CompiledRow.rowIndex
  · obtain ⟨target, member, rfl⟩ := List.mem_map.mp present
    calc
      _ = some target.toR1CS :=
        Std.HashMap.getElem?_ofList_of_mem
          (l := rows.map fun row => (row.rowIndex, row.toR1CS))
          (k := target.rowIndex) (k' := target.rowIndex)
          (v := target.toR1CS) (BEq.refl target.rowIndex) distinct
          (List.mem_map_of_mem member)
      _ = PackageSourceRows.sourceRow? instructions assertions target.rowIndex := by
        apply (PackageSourceRows.sourceRow?_eq_some instructions assertions _ target
          (permuted.mem_iff.mpr member)).symm
        exact (permuted.map Rows.CompiledRow.rowIndex).nodup_iff.mpr unique
  · have missing :
        ((rows.map fun row => (row.rowIndex, row.toR1CS)).map
          Prod.fst).contains source = false := by
      apply Bool.eq_false_iff.mpr
      intro found
      apply present
      have member := List.contains_iff_mem.mp found
      simpa only [List.map_map, Function.comp_def] using member
    rw [Std.HashMap.getElem?_ofList_of_contains_eq_false missing]
    let decoded := PackageSourceRows.decodedRows instructions assertions
    have decodedAbsent : source ∉ decoded.map Rows.CompiledRow.rowIndex := by
      intro member
      apply present
      exact (permuted.map Rows.CompiledRow.rowIndex).mem_iff.mp member
    have filtered :
        decoded.filter (fun row => row.rowIndex == source) = [] := by
      apply List.filter_eq_nil_iff.mpr
      intro row member matched
      apply decodedAbsent
      have equal : row.rowIndex = source := beq_iff_eq.mp matched
      have mapped : row.rowIndex ∈ decoded.map Rows.CompiledRow.rowIndex :=
        List.mem_map_of_mem member
      exact equal ▸ mapped
    change none = (match decoded.filter
      (fun row => row.rowIndex == source) with
      | [row] => some row.toR1CS
      | _ => none)
    rw [filtered]

/-- The index preserves every canonical lookup, including missing indices.
Canonical ownership proves distinct keys; map overwrite behavior is unused. -/
theorem stored_value (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application) (source : Nat) :
    (stored application)[source]? =
      PerApplicationCanonicalPackage.sourceRow application fits source := by
  rw [stored, rows_eq_canonicalRows,
    PerApplicationCanonicalPackage.sourceRow_eq_packageSource]
  exact indexed_value
    (PerApplicationPackageSourceRows.canonicalRows application)
    (PerApplicationPackage.package application).witnessInstructions
    (PerApplicationPackage.package application).assertionRows
    (PerApplicationPackageSourceRows.canonicalRows_rowIndices_nodup application)
    (PerApplicationPackageSourceRows.package_decodedRows_perm_canonical application) source

end NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
