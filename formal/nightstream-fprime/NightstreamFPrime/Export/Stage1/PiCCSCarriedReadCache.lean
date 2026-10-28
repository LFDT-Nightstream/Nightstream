import NightstreamFPrime.Export.Stage1.PiCCSCarriedRead
import NightstreamFPrime.Export.Stage1.PiCCSLinearRows
import Std.Data.HashMap.Lemmas

/-! Cache complete carried blocks and their K-valued column reads for one
invocation. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

def prepareCache {Value : Type} (keys : List Nat) (blocks : Nat → Value) :
    Std.HashMap Nat Value :=
  keys.foldl (fun cache key =>
    if cache.contains key then cache else cache.insert key (blocks key)) ∅

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

theorem cachedRead_prepareCache {Value : Type} (keys : List Nat)
    (blocks : Nat → Value) (key : Nat) :
    cachedRead (prepareCache keys blocks) blocks key = blocks key := by
  unfold prepareCache
  apply fold_read keys blocks ∅
  intro requested
  simp [cachedRead]

def interfaceKeys {columns : Nat} (interface : PoseidonSboxPlan.Interface columns) : List Nat :=
  interface.oneColumn.val / ringDegree ::
    ((List.ofFn interface.input ++ List.ofFn interface.sboxOutput ++ List.ofFn interface.output).flatMap
      fun form => form.entries.map (fun entry => entry.column.val / ringDegree))

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

def invocation {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (interface : PoseidonSboxPlan.Interface columns) :
    Vector (Vector K Spec.ProductionRelation.matrixCount) 150 :=
  let blockCache := prepareCache (interfaceKeys interface) blocks
  let read : Fin columns → K :=
    PiCCSCarriedRead.read basis (cachedRead blockCache blocks)
  let columnCache := prepareCache (interfaceColumnKeys interface) (columnValue read)
  PiCCSLinearRows.invocation
    (fun column => cachedRead columnCache (columnValue read) column.val) interface

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

def rowColumnKeys {columns : Nat} (forms : RowForms columns) : List Nat :=
  (List.ofFn forms).flatMap fun form =>
    form.entries.map fun entry => entry.column.val

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

def row? (program : MatrixProgram.Program) {columns : Nat}
    (sourceRow : Nat → Option R1CS.Row)
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (ordinal : Nat) :
    Option (Vector K Spec.ProductionRelation.matrixCount) :=
  (program.row? columns sourceRow ordinal).map (sparseRow basis blocks)

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

end NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache
