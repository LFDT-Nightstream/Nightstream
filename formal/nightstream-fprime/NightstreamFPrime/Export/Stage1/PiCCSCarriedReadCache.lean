import NightstreamFPrime.Export.Stage1.PiCCSCarriedRead
import NightstreamFPrime.Export.Stage1.PiCCSLinearRows
import Std.Data.HashMap.Lemmas

/-! Cache complete carried blocks and their K-valued column reads for one
invocation. Repeated keys reuse computed values; misses use the original read. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-- Compute each distinct requested key once. Every stored value comes from
`blocks`; the key list supplies no values or correctness claims. -/
def prepareCache {Value : Type} (keys : List Nat) (blocks : Nat → Value) :
    Std.HashMap Nat Value :=
  keys.foldl (fun cache key =>
    if cache.contains key then cache else cache.insert key (blocks key)) ∅

/-- The original function is evaluated only when the key is absent. -/
def cachedRead {Value : Type} (cache : Std.HashMap Nat Value)
    (blocks : Nat → Value) (key : Nat) : Value :=
  match cache[key]? with
  | some value => value
  | none => blocks key

private theorem insert_read {Value : Type} (cache : Std.HashMap Nat Value)
    (blocks : Nat → Value)
    (valid : ∀ key, cachedRead cache blocks key = blocks key)
    (inserted : Nat) :
    ∀ key, cachedRead (cache.insert inserted (blocks inserted)) blocks key =
      blocks key := by
  intro key
  by_cases equal : inserted = key
  · subst key
    simp [cachedRead]
  · simpa [cachedRead, Std.HashMap.getElem?_insert, equal] using valid key

private theorem fold_read {Value : Type} (keys : List Nat) (blocks : Nat → Value)
    (cache : Std.HashMap Nat Value)
    (valid : ∀ key, cachedRead cache blocks key = blocks key) :
    ∀ key, cachedRead
      (keys.foldl (fun current inserted =>
        if current.contains inserted then current
        else current.insert inserted (blocks inserted)) cache) blocks key =
      blocks key := by
  induction keys generalizing cache with
  | nil => exact valid
  | cons inserted keys inductionHypothesis =>
      simp only [List.foldl_cons]
      apply inductionHypothesis
      by_cases present : cache.contains inserted
      · simpa only [if_pos present] using valid
      · simpa only [if_neg present] using insert_read cache blocks valid inserted

/-- Cache preparation preserves every original read, including repeated
keys and keys absent from the preparation list. There is no support premise. -/
theorem cachedRead_prepareCache {Value : Type} (keys : List Nat)
    (blocks : Nat → Value) (key : Nat) :
    cachedRead (prepareCache keys blocks) blocks key = blocks key := by
  unfold prepareCache
  apply fold_read keys blocks ∅
  intro requested
  simp [cachedRead]

/-- Request blocks from the existing invocation inputs and retained cells.
Missing keys remain correct through the original-read fallback. -/
def interfaceKeys {columns : Nat} (interface : PoseidonSboxPlan.Interface columns) : List Nat :=
  interface.oneColumn.val / ringDegree ::
    ((List.ofFn interface.input ++ List.ofFn interface.sboxOutput ++ List.ofFn interface.output).flatMap
      fun form => form.entries.map (fun entry => entry.column.val / ringDegree))

/-- The same invocation fields supply the exact physical column keys.
Only these requested scalar reads are prepared; missing keys retain fallback. -/
def interfaceColumnKeys {columns : Nat}
    (interface : PoseidonSboxPlan.Interface columns) : List Nat :=
  interface.oneColumn.val ::
    ((List.ofFn interface.input ++ List.ofFn interface.sboxOutput ++ List.ofFn interface.output).flatMap
      fun form => form.entries.map (fun entry => entry.column.val))

private def columnValue {columns : Nat} (read : Fin columns → K) (index : Nat) : K :=
  if bounded : index < columns then read ⟨index, bounded⟩ else K.zero

private theorem cached_columns_eq {columns : Nat}
    (keys : List Nat) (read : Fin columns → K) :
    (fun column : Fin columns =>
      cachedRead (prepareCache keys (columnValue read)) (columnValue read) column.val) = read := by
  funext column
  rw [cachedRead_prepareCache]
  simp only [columnValue, dif_pos column.isLt]

/-- Share each requested block and each complete K column read across both
scalar projections and all 94 numeric rows of the same invocation. -/
def invocation {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (interface : PoseidonSboxPlan.Interface columns) :
    Vector (Vector K Spec.ProductionRelation.matrixCount) 94 :=
  let blockCache := prepareCache (interfaceKeys interface) blocks
  let read : Fin columns → K :=
    PiCCSCarriedRead.read basis (cachedRead blockCache blocks)
  let columnCache := prepareCache (interfaceColumnKeys interface) (columnValue read)
  PiCCSLinearRows.invocation
    (fun column => cachedRead columnCache (columnValue read) column.val) interface

/-- Both caches preserve every stored row and matrix port, for arbitrary
original blocks and interfaces. No requested-support premise is needed. -/
theorem invocation_eq {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (interface : PoseidonSboxPlan.Interface columns) :
    invocation basis blocks interface =
      PiCCSLinearRows.invocation (PiCCSCarriedRead.read basis blocks) interface := by
  have reads : cachedRead (prepareCache (interfaceKeys interface) blocks) blocks = blocks :=
    funext (cachedRead_prepareCache (interfaceKeys interface) blocks)
  dsimp only [invocation]
  rw [cached_columns_eq, reads]

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram

/-- Physical columns read by the already compiled meaningful port forms.
Repeated entries remain in the forms; cache preparation shares only reads. -/
def rowColumnKeys {columns : Nat} (forms : RowForms columns) : List Nat :=
  (List.ofFn forms).flatMap fun form =>
    form.entries.map fun entry => entry.column.val

/-- Cache complete blocks and K column reads for one loaded sparse row.
All fourteen port values use the existing sparse evaluator. -/
def sparseRow {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (forms : RowForms columns) :
    Vector K Spec.ProductionRelation.matrixCount :=
  let keys := rowColumnKeys forms
  let blockCache := prepareCache (keys.map (fun key => key / ringDegree)) blocks
  let read : Fin columns → K :=
    PiCCSCarriedRead.read basis (cachedRead blockCache blocks)
  let columnCache := prepareCache keys (columnValue read)
  Vector.ofFn fun port =>
    PiCCSSparseEvaluation.evaluateK
      (match meaningfulPort? port with
        | some meaningful => forms meaningful
        | none => SparseForm.empty)
      (fun column => cachedRead columnCache (columnValue read) column.val)

/-- The caches preserve each original sparse port, including the empty
fourteenth port. No assignment or requested-support premise is required. -/
theorem sparseRow_value {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (forms : RowForms columns)
    (port : Fin Spec.ProductionRelation.matrixCount) :
    (sparseRow basis blocks forms).get port =
      PiCCSSparseEvaluation.evaluateK
        (match meaningfulPort? port with
          | some meaningful => forms meaningful
          | none => SparseForm.empty)
        (PiCCSCarriedRead.read basis blocks) := by
  have reads :
      cachedRead
        (prepareCache ((rowColumnKeys forms).map (fun key => key / ringDegree)) blocks)
        blocks = blocks :=
    funext (cachedRead_prepareCache
      ((rowColumnKeys forms).map (fun key => key / ringDegree)) blocks)
  dsimp only [sparseRow]
  rw [cached_columns_eq, reads]
  change (Vector.ofFn _)[port.val] = _
  rw [Vector.getElem_ofFn]

/-- Load the original program row once, then share its carried reads.
Every original lookup failure is retained. -/
def row? (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row)
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (ordinal : Nat) :
    Option (Vector K Spec.ProductionRelation.matrixCount) :=
  (program.row? columns sourceRow ordinal).map (sparseRow basis blocks)

/-- Exact optional-output equality with both scalar passes of the existing
linear row interpreter. The source accessor, row order, guards, and all
fourteen K values are unchanged, for arbitrary programs and carried blocks. -/
theorem row?_eq (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row)
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (ordinal : Nat) :
    row? program (columns := columns) sourceRow basis blocks ordinal =
      PiCCSLinearRows.row? program (columns := columns) sourceRow
        (PiCCSCarriedRead.read basis blocks) ordinal := by
  have equal : ∀ port : Fin Spec.ProductionRelation.matrixCount,
      (row? program (columns := columns) sourceRow basis blocks ordinal).map (fun values => values.get port) =
        (PiCCSLinearRows.row? program (columns := columns) sourceRow
          (PiCCSCarriedRead.read basis blocks) ordinal).map
            (fun values => values.get port) := by
    intro port
    rw [PiCCSLinearRows.row?_value]
    simp only [row?, Option.map_map, Function.comp_def, sparseRow_value]
    cases meaningfulPort? port <;> simp only
  cases cached : row? program (columns := columns) sourceRow basis blocks ordinal with
  | none =>
      cases original : PiCCSLinearRows.row? program (columns := columns) sourceRow
          (PiCCSCarriedRead.read basis blocks) ordinal with
      | none => rfl
      | some oldValues =>
          have impossible := equal ⟨0, by decide⟩
          rw [cached, original] at impossible
          cases impossible
  | some values =>
      cases original : PiCCSLinearRows.row? program (columns := columns) sourceRow
          (PiCCSCarriedRead.read basis blocks) ordinal with
      | none =>
          have impossible := equal ⟨0, by decide⟩
          rw [cached, original] at impossible
          cases impossible
      | some oldValues =>
          apply congrArg some
          apply Vector.ext
          intro lane bounded
          have component := equal ⟨lane, bounded⟩
          exact Option.some.inj (by
            simpa only [cached, original, Option.map_some] using! component)

/-- Retain the original optional forms for all rows of one product
invocation. The interface is loaded by the caller only once. -/
private def productInvocationRows {columns : Nat}
    (interface : ProductSumPlan.Interface columns) :
    Vector (Option (RowForms columns)) 34 :=
  Vector.ofFn fun row =>
    (PiDECProductRow.row? interface row.val).map ProductSumPlan.Row.meaningfulForm

/-- Share complete carried blocks and K column reads across the existing
34-row product invocation. Each row keeps its original optional result. -/
def productInvocation {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree)
    (interface : ProductSumPlan.Interface columns) :
    Vector (Option (Vector K Spec.ProductionRelation.matrixCount)) 34 :=
  let rows := productInvocationRows interface
  let keys := rows.toList.flatMap fun selected =>
    match selected with
    | none => []
    | some forms => rowColumnKeys forms
  let blockCache := prepareCache (keys.map (fun key => key / ringDegree)) blocks
  let read : Fin columns → K :=
    PiCCSCarriedRead.read basis (cachedRead blockCache blocks)
  let columnCache := prepareCache keys (columnValue read)
  Vector.ofFn fun row =>
    (rows.get row).map fun forms =>
      Vector.ofFn fun port =>
        PiCCSSparseEvaluation.evaluateK
          (match meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => SparseForm.empty)
          (fun column => cachedRead columnCache (columnValue read) column.val)

/-- Each complete optional row equals the existing direct product row
under the original carried read. All fourteen ports and all lookup failures
are preserved, without a cache-support or interface-validity premise. -/
theorem productInvocation_value {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree)
    (interface : ProductSumPlan.Interface columns) (row : Fin 34) :
    (productInvocation basis blocks interface).get row =
      (PiDECProductRow.row? interface row.val).map (fun selected =>
        Vector.ofFn fun port : Fin Spec.ProductionRelation.matrixCount =>
          PiCCSSparseEvaluation.evaluateK (selected.portForm port)
            (PiCCSCarriedRead.read basis blocks)) := by
  let keys := (productInvocationRows interface).toList.flatMap fun selected =>
    match selected with
    | none => []
    | some forms => rowColumnKeys forms
  have reads :
      cachedRead (prepareCache (keys.map (fun key => key / ringDegree)) blocks)
        blocks = blocks :=
    funext (cachedRead_prepareCache (keys.map (fun key => key / ringDegree)) blocks)
  dsimp only [productInvocation]
  rw [cached_columns_eq, reads]
  change (Vector.ofFn _)[row.val] = _
  rw [Vector.getElem_ofFn]
  change ((productInvocationRows interface)[row.val]).map _ = _
  simp only [productInvocationRows, Vector.getElem_ofFn, Option.map_map,
    Function.comp_def, ProductSumPlan.Row.portForm]
  apply congrArg (fun evaluate : ProductSumPlan.Row columns →
      Vector K Spec.ProductionRelation.matrixCount =>
    (PiDECProductRow.row? interface row.val).map evaluate)
  funext selected
  apply Vector.ext
  intro index bounded
  simp only [Vector.getElem_ofFn]
  cases meaningfulPort? ⟨index, bounded⟩ <;> rfl

end NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache
