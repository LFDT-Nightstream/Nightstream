import Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding
import Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRowsFor
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCount
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor

/-!
Contract: exponent-indexed Poseidon2 rows for one production F-prime
successor state.

One `rowVariables` value owns the running-state width, physical layout,
successor frame, sponge census, and emitted rows. The rows start from the
verifier-key-bound statement state. They absorb the complete successor value
and apply the terminal gate. The just-consumed fresh claim is absent because
HyperNova Construction 2 hashes the updated running state.

`Placed` states only physical serialization. It does not assume NIFS
acceptance, state continuity, a digest equality, or the desired soundness
result.

Does not own the link from the typed PiCCS output to the running window,
application transitions, generated-artifact containment, terminal
verification, Poseidon2 security, or Rust refinement.

Assurance tier: exponent-indexed row implementation.

Emits constraints: yes; the exact count is
`successorPermutationCount rowVariables * 352`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev LinComb := LinCombNormal.LinComb

/-! ## Exponent-indexed physical successor carrier -/

def invocationOffset : Nat := 0
def realRowsOffset : Nat := 1
def initialApplicationOffset : Nat := 2
def applicationOffset : Nat := 87
def runningOffset : Nat := 172

def initialCarryOffset (rowVariables : Nat) : Nat :=
  runningOffset + ProductNifsCodec.runningFieldCountFor rowVariables

def carryOffset (rowVariables : Nat) : Nat :=
  initialCarryOffset rowVariables + 59

def endOffset (rowVariables : Nat) : Nat :=
  carryOffset rowVariables + 59

theorem section_offsets_exact (rowVariables : Nat) :
    realRowsOffset = invocationOffset + 1 /\
      initialApplicationOffset = realRowsOffset + 1 /\
      applicationOffset = initialApplicationOffset + 85 /\
      runningOffset = applicationOffset + 85 /\
      initialCarryOffset rowVariables =
        runningOffset + ProductNifsCodec.runningFieldCountFor rowVariables /\
      carryOffset rowVariables = initialCarryOffset rowVariables + 59 /\
      endOffset rowVariables = carryOffset rowVariables + 59 := by
  simp [invocationOffset, realRowsOffset, initialApplicationOffset,
    applicationOffset, runningOffset, initialCarryOffset, carryOffset,
    endOffset]

structure Layout (rowVariables : Nat) where
  start : Nat
  startPositive : 0 < start
deriving Repr

def Layout.invocationColumn {rowVariables} (layout : Layout rowVariables) : Nat :=
  layout.start + invocationOffset

def Layout.realRowsColumn {rowVariables} (layout : Layout rowVariables) : Nat :=
  layout.start + realRowsOffset

def Layout.initialApplicationColumn {rowVariables}
    (layout : Layout rowVariables) (index : Fin 85) : Nat :=
  layout.start + initialApplicationOffset + index.val

def Layout.applicationColumn {rowVariables}
    (layout : Layout rowVariables) (index : Fin 85) : Nat :=
  layout.start + applicationOffset + index.val

def Layout.runningColumn {rowVariables}
    (layout : Layout rowVariables)
    (index : Fin (ProductNifsCodec.runningFieldCountFor rowVariables)) : Nat :=
  layout.start + runningOffset + index.val

def Layout.initialCarryColumn {rowVariables}
    (layout : Layout rowVariables) (index : Fin 59) : Nat :=
  layout.start + initialCarryOffset rowVariables + index.val

def Layout.carryColumn {rowVariables}
    (layout : Layout rowVariables) (index : Fin 59) : Nat :=
  layout.start + carryOffset rowVariables + index.val

theorem runningNativeFields_length
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (running : ProductionFieldNativeFullClaim.Running fullShape) :
    (ProductionSuccessorStateBinding.runningNativeFields running).length =
      ProductNifsCodec.runningFieldCountFor rowVariables := by
  have exactLength :=
    ProductionSuccessorStateBinding.runningNativeFields_lengthFor
      contract.toShape running
  simpa only [contract.rowVariablesExact] using exactLength

/-- A field-native running-carrier coordinate and the natural-number
successor-frame coordinate are the same canonical Goldilocks representative.
This is the exact bridge used by the recursive F-prime successor. -/
theorem runningNativeFields_get_eq_codec_getD
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (running : ProductionFieldNativeFullClaim.Running fullShape)
    (index : Fin (ProductNifsCodec.runningFieldCountFor rowVariables)) :
    (ProductionSuccessorStateBinding.runningNativeFields running).get
        (Fin.cast (runningNativeFields_length contract running).symm index) =
      (((ProductNifsCodec.runningCodecFor fullShape.rowVariables fullShape).encode
        running).getD index.val 0).val := by
  let fields :=
    (ProductNifsCodec.runningCodecFor fullShape.rowVariables fullShape).encode
      running
  have fieldsLength : fields.length =
      ProductNifsCodec.runningFieldCountFor fullShape.rowVariables := by
    dsimp only [fields]
    rw [(ProductNifsCodec.runningCodecFor fullShape.rowVariables
      fullShape).encode_length]
    exact ProductNifsCodec.runningCodecFor_width contract.toShape
  let castIndex :=
    Fin.cast (runningNativeFields_length contract running).symm index
  have castIndexValue : castIndex.val = index.val := rfl
  have mappedBound : castIndex.val < (fields.map Fin.val).length :=
    castIndex.isLt
  change (fields.map Fin.val).get castIndex =
    (fields.getD index.val 0).val
  have leftGetD :
      (fields.map Fin.val).get castIndex =
        (fields.map Fin.val).getD castIndex.val 0 := by
    symm
    rw [List.getD_eq_getElem (fields.map Fin.val) 0 mappedBound]
    rfl
  rw [leftGetD, castIndexValue]
  simpa using
    (List.getD_map (l := fields) (d := (0 : F))
      (n := index.val) Fin.val)

structure Placed
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (layout : Layout rowVariables) (assignment : Nat -> Nat)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape) : Prop where
  invocation : assignment layout.invocationColumn =
    value.augmentedInvocationIndex
  realRows : assignment layout.realRowsColumn = value.realApplicationRowCount
  initialApplication : forall index : Fin 85,
    assignment (layout.initialApplicationColumn index) =
      (ProductionWasmStateFields.encode value.initialApplicationState).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length
            value.initialApplicationState).symm index)
  application : forall index : Fin 85,
    assignment (layout.applicationColumn index) =
      (ProductionWasmStateFields.encode value.applicationState).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length value.applicationState).symm
          index)
  running : forall index :
      Fin (ProductNifsCodec.runningFieldCountFor rowVariables),
    assignment (layout.runningColumn index) =
      (ProductionSuccessorStateBinding.runningNativeFields value.running).get
        (Fin.cast (runningNativeFields_length contract value.running).symm index)
  initialCarry : forall index : Fin 59,
    assignment (layout.initialCarryColumn index) =
      (ProductionMemoryCarryFields.encode value.initialMemoryCarry).get
        (Fin.cast
          (ProductionMemoryCarryFields.encode_length
            value.initialMemoryCarry).symm index)
  carry : forall index : Fin 59,
    assignment (layout.carryColumn index) =
      (ProductionMemoryCarryFields.encode value.memoryCarry).get
        (Fin.cast
          (ProductionMemoryCarryFields.encode_length value.memoryCarry).symm
          index)

/-! ## Direct symbolic frame -/

def columnField (column : Nat) : LinComb := [(column, 1)]

def fixedValues (candidate : Id) : List Nat :=
  [ ProductionSuccessorStateBinding.successorTag
  , ProductionSuccessorStateBinding.successorVersion
  ] ++ ProductionMemoryTranscriptHashFrame.profileFields candidate

def applicationColumns {rowVariables} (layout : Layout rowVariables) : List Nat :=
  List.ofFn layout.applicationColumn

def initialApplicationColumns {rowVariables}
    (layout : Layout rowVariables) : List Nat :=
  List.ofFn layout.initialApplicationColumn

def runningColumns {rowVariables} (layout : Layout rowVariables) : List Nat :=
  List.ofFn layout.runningColumn

def initialCarryColumns {rowVariables}
    (layout : Layout rowVariables) : List Nat :=
  List.ofFn layout.initialCarryColumn

def carryColumns {rowVariables} (layout : Layout rowVariables) : List Nat :=
  List.ofFn layout.carryColumn

def preCarryFields
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables) :
    List LinComb :=
  (fixedValues candidate).map ProductPiCcsTranscriptRowsFor.word ++
    [columnField layout.invocationColumn,
      columnField layout.realRowsColumn] ++
    (initialApplicationColumns layout).map columnField ++
    (applicationColumns layout).map columnField ++
    (runningColumns layout).map columnField

def carryFields
    {rowVariables : Nat} (layout : Layout rowVariables) : List LinComb :=
    (initialCarryColumns layout).map columnField ++
    (carryColumns layout).map columnField

def successorFields
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables) :
    List LinComb :=
  preCarryFields candidate layout ++ carryFields layout

theorem fixedValues_length (candidate : Id) :
    (fixedValues candidate).length = 6 := by
  simp [fixedValues,
    ProductionMemoryTranscriptHashFrame.profileFields_length]

theorem preCarryFields_length
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables) :
    (preCarryFields candidate layout).length =
      ProductNifsCodec.runningFieldCountFor rowVariables + 178 := by
  simp [preCarryFields, fixedValues_length, applicationColumns,
    initialApplicationColumns, runningColumns]
  omega

theorem carryFields_length
    {rowVariables : Nat} (layout : Layout rowVariables) :
    (carryFields layout).length = 118 := by
  simp [carryFields, initialCarryColumns, carryColumns]

theorem successorFields_length
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables) :
    (successorFields candidate layout).length =
      ProductNifsCodec.runningFieldCountFor rowVariables + 296 := by
  rw [successorFields, List.length_append, preCarryFields_length,
    carryFields_length]

def successorStart
    (statementId : ProductPoseidon2.StatementId) : SymbolicDuplex.Builder :=
  SymbolicDuplex.start
    (ProductPiCcsTranscriptRowsFor.initialLanes statementId)
    (ProductPoseidon2.initialStateForStatement statementId).absorbed

theorem decoded_successorStart
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId) :
    decodedBuilder assignment (successorStart statementId) =
      ProductPoseidon2.initialStateForStatement statementId := by
  rw [Poseidon2Duplex.State.mk.injEq]
  refine ⟨?_, rfl⟩
  funext lane
  change lcEval assignment
      (ProductPiCcsTranscriptRowsFor.word
        ((ProductPoseidon2.initialStateForStatement statementId).lanes lane)) =
    (ProductPoseidon2.initialStateForStatement statementId).lanes lane
  rw [ProductPiCcsTranscriptRowsFor.word,
    ProductPiCcsTranscriptSemantics.lcEval_word assignment one]
  exact Nat.mod_eq_of_lt
    (ProductPiCcsTranscriptSemantics.initialStateForStatement_canonical
      statementId lane)

def preCarryBuilder
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany base (preCarryFields candidate layout)
    (successorStart statementId)

def absorbedBuilder
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) : SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany base (carryFields layout)
    (preCarryBuilder candidate base layout statementId)

def builder
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) : SymbolicDuplex.Builder :=
  SymbolicDuplex.gate base
    (absorbedBuilder candidate base layout statementId)

def rows
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) : List Row :=
  SymbolicDuplex.rows base ProductPoseidon2.constants
    (builder candidate base layout statementId)

/-! ## Direct frame semantics -/

def fieldValues (assignment : Nat -> Nat) (fields : List LinComb) : List Nat :=
  fields.map (lcEval assignment)

private theorem fieldValues_append
    (assignment : Nat -> Nat) (left right : List LinComb) :
    fieldValues assignment (left ++ right) =
      fieldValues assignment left ++ fieldValues assignment right := by
  exact List.map_append

private theorem fieldValues_columnFields
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (columns : List Nat) :
    fieldValues assignment (columns.map columnField) =
      columns.map assignment := by
  induction columns with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change lcEval assignment (columnField head) ::
          fieldValues assignment (tail.map columnField) =
        assignment head :: tail.map assignment
      rw [inductionHypothesis]
      simp [columnField, lcEval, Nat.mod_eq_of_lt (canonical head)]

private theorem mapOfFn_assignment_eq_values
    {n : Nat} {assignment : Nat -> Nat}
    (columns : Fin n -> Nat) (values : List Nat)
    (lengthExact : values.length = n)
    (placed : forall coordinate,
      assignment (columns coordinate) =
        values.get (Fin.cast lengthExact.symm coordinate)) :
    (List.ofFn columns).map assignment = values := by
  rw [List.map_ofFn]
  have functionsEqual :
      assignment ∘ columns =
        fun coordinate : Fin n =>
          values.get (Fin.cast lengthExact.symm coordinate) := by
    funext coordinate
    exact placed coordinate
  rw [functionsEqual]
  have reindexed := List.ofFn_congr lengthExact
    (fun coordinate : Fin values.length => values.get coordinate)
  rw [List.ofFn_get] at reindexed
  exact reindexed.symm

private theorem fixedValues_canonical
    (candidate : Id) {value : Nat} (member : value ∈ fixedValues candidate) :
    value < goldilocksP := by
  simp only [fixedValues, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with (rfl | rfl) | member
  · norm_num [ProductionSuccessorStateBinding.successorTag, goldilocksP]
  · norm_num [ProductionSuccessorStateBinding.successorVersion, goldilocksP]
  · simp only [ProductionMemoryTranscriptHashFrame.profileFields,
      List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl <;>
      cases candidate <;>
      norm_num [version, checkedStepsPerFreshClaim, goldilocksP]

private theorem fixedFieldValues
    (candidate : Id) (assignment : Nat -> Nat)
    (one : assignment 0 = 1) :
    fieldValues assignment
        ((fixedValues candidate).map ProductPiCcsTranscriptRowsFor.word) =
      fixedValues candidate := by
  change ProductPiCcsTranscriptSemantics.fieldValues assignment
      ((fixedValues candidate).map ProductPiCcsTranscriptRows.word) = _
  rw [ProductPiCcsTranscriptSemantics.fieldValues_words assignment one]
  calc
    List.map (fun value => value % goldilocksP) (fixedValues candidate) =
        List.map id (fixedValues candidate) := by
      apply List.map_congr_left
      intro value member
      exact Nat.mod_eq_of_lt (fixedValues_canonical candidate member)
    _ = fixedValues candidate := List.map_id _

/-- The challenge-independent symbolic prefix is the exact typed pre-carry
frame.  This theorem uses only source-column placement; it does not assume a
digest or a challenge value. -/
theorem fieldValues_eq_preCarryFrame
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape)
    (placed : Placed contract layout assignment value) :
    fieldValues assignment (preCarryFields candidate layout) =
      ProductionSuccessorStateBinding.preCarryFrame value.preCarry := by
  have counters :
      fieldValues assignment
          [columnField layout.invocationColumn,
            columnField layout.realRowsColumn] =
        [value.augmentedInvocationIndex, value.realApplicationRowCount] := by
    have invocationLt : value.augmentedInvocationIndex < goldilocksP := by
      rw [← placed.invocation]
      exact canonical layout.invocationColumn
    have realRowsLt : value.realApplicationRowCount < goldilocksP := by
      rw [← placed.realRows]
      exact canonical layout.realRowsColumn
    simp [fieldValues, columnField, lcEval,
      placed.invocation, placed.realRows,
      Nat.mod_eq_of_lt invocationLt, Nat.mod_eq_of_lt realRowsLt]
  have initialApplication :
      fieldValues assignment
          ((initialApplicationColumns layout).map columnField) =
        ProductionWasmStateFields.encode value.initialApplicationState := by
    rw [fieldValues_columnFields canonical]
    exact mapOfFn_assignment_eq_values layout.initialApplicationColumn
      (ProductionWasmStateFields.encode value.initialApplicationState)
      (ProductionWasmStateFields.encode_length value.initialApplicationState)
      placed.initialApplication
  have application :
      fieldValues assignment ((applicationColumns layout).map columnField) =
        ProductionWasmStateFields.encode value.applicationState := by
    rw [fieldValues_columnFields canonical]
    exact mapOfFn_assignment_eq_values layout.applicationColumn
      (ProductionWasmStateFields.encode value.applicationState)
      (ProductionWasmStateFields.encode_length value.applicationState)
      placed.application
  have running :
      fieldValues assignment ((runningColumns layout).map columnField) =
        ProductionSuccessorStateBinding.runningNativeFields value.running := by
    rw [fieldValues_columnFields canonical]
    exact mapOfFn_assignment_eq_values layout.runningColumn
      (ProductionSuccessorStateBinding.runningNativeFields value.running)
      (runningNativeFields_length contract value.running) placed.running
  rw [preCarryFields, fieldValues_append, fieldValues_append,
    fieldValues_append, fieldValues_append]
  rw [fixedFieldValues candidate assignment one, counters,
    initialApplication, application, running]
  simp [fixedValues, ProductionSuccessorStateBinding.preCarryFrame,
    ProductionSuccessorStateBinding.preCarryBlocks,
    ProductionSuccessorStateBinding.Value.preCarry]

theorem fieldValues_eq_carryFrame
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape)
    (placed : Placed contract layout assignment value) :
    fieldValues assignment (carryFields layout) =
      ProductionSuccessorStateBinding.carryFrame value := by
  have initialCarry :
      fieldValues assignment ((initialCarryColumns layout).map columnField) =
        ProductionMemoryCarryFields.encode value.initialMemoryCarry := by
    rw [fieldValues_columnFields canonical]
    exact mapOfFn_assignment_eq_values layout.initialCarryColumn
      (ProductionMemoryCarryFields.encode value.initialMemoryCarry)
      (ProductionMemoryCarryFields.encode_length value.initialMemoryCarry)
      placed.initialCarry
  have carry :
      fieldValues assignment ((carryColumns layout).map columnField) =
        ProductionMemoryCarryFields.encode value.memoryCarry := by
    rw [fieldValues_columnFields canonical]
    exact mapOfFn_assignment_eq_values layout.carryColumn
      (ProductionMemoryCarryFields.encode value.memoryCarry)
      (ProductionMemoryCarryFields.encode_length value.memoryCarry)
      placed.carry
  rw [carryFields, fieldValues_append, initialCarry, carry]
  rfl

theorem fieldValues_eq_successorFrame
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape)
    (placed : Placed contract layout assignment value) :
    fieldValues assignment (successorFields candidate layout) =
      ProductionSuccessorStateBinding.successorFrame value := by
  rw [successorFields, fieldValues_append,
    fieldValues_eq_preCarryFrame contract canonical one value placed]
  rw [fieldValues_eq_carryFrame contract canonical value placed]
  exact (ProductionSuccessorStateBinding.successorFrame_eq_preCarry_append
    value).symm

theorem successorFrame_fields_canonical_of_placed
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape)
    (placed : Placed contract layout assignment value) :
    forall field,
      field ∈ ProductionSuccessorStateBinding.successorFrame value ->
        field < goldilocksP := by
  intro field member
  have frameExact := fieldValues_eq_successorFrame contract canonical one
    value placed
  rw [← frameExact] at member
  rcases List.mem_map.mp member with ⟨terms, _, rfl⟩
  exact Nat.mod_lt _ (by norm_num [goldilocksP])

theorem preCarryFrame_fields_canonical_of_placed
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape)
    (placed : Placed contract layout assignment value) :
    forall field,
      field ∈ ProductionSuccessorStateBinding.preCarryFrame value.preCarry ->
        field < goldilocksP := by
  intro field member
  have frameExact := fieldValues_eq_preCarryFrame contract canonical one
    value placed
  rw [← frameExact] at member
  rcases List.mem_map.mp member with ⟨terms, _, rfl⟩
  exact Nat.mod_lt _ (by norm_num [goldilocksP])

/-! ## Exact structural census -/

private theorem value_absorbList_absorbed
    (constants : Poseidon2Schedule.Constants)
    (values : List Nat) (state : Poseidon2Duplex.State) :
    (Poseidon2Duplex.absorbList constants values state).absorbed =
      SymbolicDuplexCursor.after state.absorbed values.length := by
  induction values generalizing state with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [Poseidon2Duplex.absorbList, inductionHypothesis]
      simp only [SymbolicDuplexCursor.after]
      unfold Poseidon2Duplex.absorbElem Poseidon2Duplex.guarded
        SymbolicDuplexCursor.step
      by_cases full : Poseidon2Sponge.rate <= state.absorbed
      · simp [full, Poseidon2Duplex.permute]
      · simp [full]

private theorem after_two_four_mul (blocks : Nat) :
    SymbolicDuplexCursor.after 2 (4 * blocks) = 2 := by
  induction blocks with
  | zero => rfl
  | succ blocks inductionHypothesis =>
      rw [Nat.mul_succ, SymbolicDuplexCursor.after_add,
        inductionHypothesis]
      decide

theorem initialStateForStatement_absorbed
    (statementId : ProductPoseidon2.StatementId) :
    (ProductPoseidon2.initialStateForStatement statementId).absorbed = 2 := by
  rw [ProductPoseidon2.initialStateForStatement,
    value_absorbList_absorbed,
    ProductPoseidon2.statementIdentifierFields,
    ProductPoseidon2.proofPrefixFields_length]
  change SymbolicDuplexCursor.after 0 366 = 2
  rw [show 366 = 2 + 4 * 91 by decide,
    SymbolicDuplexCursor.after_add]
  change SymbolicDuplexCursor.after 2 (4 * 91) = 2
  exact after_two_four_mul 91

private theorem successorStart_control
    (statementId : ProductPoseidon2.StatementId) :
    SymbolicDuplexCount.ofBuilder (successorStart statementId) =
      { entries := 0, absorbed := 2 } := by
  simp only [successorStart, SymbolicDuplexCount.ofBuilder,
    SymbolicDuplex.start, List.length_nil]
  rw [initialStateForStatement_absorbed]

def preCarryControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  SymbolicDuplexCount.absorbMany
    (ProductNifsCodec.runningFieldCountFor rowVariables + 178)
    { entries := 0, absorbed := 2 }

def absorbedControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  SymbolicDuplexCount.absorbMany 118 (preCarryControl rowVariables)

def finalControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  SymbolicDuplexCount.gate (absorbedControl rowVariables)

def successorPermutationCount (rowVariables : Nat) : Nat :=
  (finalControl rowVariables).entries

/-- The exponent-indexed census specializes to the checked fixed-25 reference
count. This is a regression fact, not a production exponent selection. -/
theorem successorPermutationCount_25 :
    successorPermutationCount 25 = 20878 := by
  have preCarryExact : preCarryControl 25 =
      { entries := 20847, absorbed := 2 } := by
    rw [preCarryControl, SymbolicDuplexCount.absorbMany_eq_fast]
    decide
  have absorbedExact : absorbedControl 25 =
      { entries := 20876, absorbed := 4 } := by
    rw [absorbedControl, preCarryExact,
      SymbolicDuplexCount.absorbMany_eq_fast]
    decide
  rw [successorPermutationCount, finalControl, absorbedExact]
  decide

/-- The first exponent not ruled out by the mandatory recursive-core census
has the same complete successor permutation count as exponent 25. -/
theorem successorPermutationCount_26 :
    successorPermutationCount 26 = 20878 := by
  have preCarryExact : preCarryControl 26 =
      { entries := 20847, absorbed := 4 } := by
    rw [preCarryControl, SymbolicDuplexCount.absorbMany_eq_fast]
    decide
  have absorbedExact : absorbedControl 26 =
      { entries := 20877, absorbed := 2 } := by
    rw [absorbedControl, preCarryExact,
      SymbolicDuplexCount.absorbMany_eq_fast]
    decide
  rw [successorPermutationCount, finalControl, absorbedExact]
  decide

theorem preCarryBuilder_control
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    SymbolicDuplexCount.ofBuilder
        (preCarryBuilder candidate base layout statementId) =
      preCarryControl rowVariables := by
  change SymbolicDuplexCount.ofBuilder
      (SymbolicDuplex.absorbMany base (preCarryFields candidate layout)
        (successorStart statementId)) = _
  rw [SymbolicDuplexCount.ofBuilder_absorbMany,
    preCarryFields_length, successorStart_control]
  rfl

theorem absorbedBuilder_control
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    SymbolicDuplexCount.ofBuilder
      (absorbedBuilder candidate base layout statementId) =
      absorbedControl rowVariables := by
  change SymbolicDuplexCount.ofBuilder
      (SymbolicDuplex.absorbMany base (carryFields layout)
        (preCarryBuilder candidate base layout statementId)) = _
  rw [SymbolicDuplexCount.ofBuilder_absorbMany,
    carryFields_length, preCarryBuilder_control]
  rfl

theorem builder_control
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    SymbolicDuplexCount.ofBuilder
        (builder candidate base layout statementId) =
      finalControl rowVariables := by
  change SymbolicDuplexCount.ofBuilder
      (SymbolicDuplex.gate base
        (absorbedBuilder candidate base layout statementId)) = _
  rw [SymbolicDuplexCount.ofBuilder_gate,
    absorbedBuilder_control]
  rfl

theorem builder_entries_length
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    (builder candidate base layout statementId).entries.length =
      successorPermutationCount rowVariables := by
  have exactControl := builder_control candidate base layout statementId
  have entriesEqual :=
    congrArg SymbolicDuplexCount.Control.entries exactControl
  simpa [SymbolicDuplexCount.ofBuilder, successorPermutationCount] using
    entriesEqual

theorem rows_length_exact
    (candidate : Id) (base : Nat) {rowVariables : Nat}
    (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    (rows candidate base layout statementId).length =
      successorPermutationCount rowVariables * 352 := by
  rw [rows, SymbolicDuplex.rows_length,
    builder_entries_length]

/-! ## Row soundness -/

/-- The complete successor rows validate the shared pre-carry builder.  This
is a row-derived prefix fact, not a receipt supplied by the caller. -/
theorem rows_imply_preCarryBuilder_valid
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId)
    (base : Nat)
    (successorRows : Satisfies
      (rows candidate base layout statementId) assignment) :
    Valid base ProductPoseidon2.constants assignment
      (preCarryBuilder candidate base layout statementId) := by
  have finalValid :
      Valid base ProductPoseidon2.constants assignment
        (builder candidate base layout statementId) :=
    valid_of_satisfied base ProductPoseidon2.constants
      (builder candidate base layout statementId) assignment canonical one
      successorRows
  have absorbedValid :
      Valid base ProductPoseidon2.constants assignment
        (absorbedBuilder candidate base layout statementId) :=
    finalValid.of_extends
      (gate_extends base
        (absorbedBuilder candidate base layout statementId))
  exact absorbedValid.of_extends
    (absorbMany_extends base (carryFields layout)
      (preCarryBuilder candidate base layout statementId))

/-- Satisfying successor rows recover the exact challenge-independent
Construction-2 prefix state before its dedicated domain gate. -/
theorem rows_imply_preCarryAbsorbedState
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate fullShape)
    (successorPlaced : Placed contract layout assignment successor)
    (base : Nat)
    (successorRows : Satisfies
      (rows candidate base layout statementId) assignment) :
    decodedBuilder assignment
        (preCarryBuilder candidate base layout statementId) =
      ProductionSuccessorStateBinding.preCarryAbsorbedState statementId
        successor.preCarry := by
  have preCarryValid := rows_imply_preCarryBuilder_valid canonical one
    statementId base successorRows
  have absorbedEq := decodedBuilder_absorbMany base
    ProductPoseidon2.constants assignment (preCarryFields candidate layout)
    (successorStart statementId) preCarryValid
  change decodedBuilder assignment
      (SymbolicDuplex.absorbMany base (preCarryFields candidate layout)
        (successorStart statementId)) =
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (fieldValues assignment (preCarryFields candidate layout))
      (decodedBuilder assignment (successorStart statementId)) at absorbedEq
  rw [decoded_successorStart assignment one,
    fieldValues_eq_preCarryFrame contract canonical one successor
      successorPlaced] at absorbedEq
  simpa [preCarryBuilder,
    ProductionSuccessorStateBinding.preCarryAbsorbedState] using absorbedEq

theorem rows_imply_outputState
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate fullShape)
    (successorPlaced : Placed contract layout assignment successor)
    (base : Nat)
    (successorRows : Satisfies
      (rows candidate base layout statementId) assignment) :
    decodedBuilder assignment (builder candidate base layout statementId) =
      ProductionSuccessorStateBinding.outputState statementId successor := by
  have finalValid :
      Valid base ProductPoseidon2.constants assignment
        (builder candidate base layout statementId) := by
    apply valid_of_satisfied base ProductPoseidon2.constants
      (builder candidate base layout statementId) assignment canonical one
    exact successorRows
  have absorbedValid :
      Valid base ProductPoseidon2.constants assignment
        (absorbedBuilder candidate base layout statementId) :=
    finalValid.of_extends
      (gate_extends base
        (absorbedBuilder candidate base layout statementId))
  have preCarryExact := rows_imply_preCarryAbsorbedState contract canonical
    one statementId successor successorPlaced base successorRows
  have absorbedEq := decodedBuilder_absorbMany base
    ProductPoseidon2.constants assignment (carryFields layout)
    (preCarryBuilder candidate base layout statementId) absorbedValid
  change decodedBuilder assignment
      (SymbolicDuplex.absorbMany base (carryFields layout)
        (preCarryBuilder candidate base layout statementId)) =
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (fieldValues assignment (carryFields layout))
      (decodedBuilder assignment
        (preCarryBuilder candidate base layout statementId)) at absorbedEq
  rw [preCarryExact,
    fieldValues_eq_carryFrame contract canonical successor successorPlaced]
      at absorbedEq
  have absorbedExact :
      decodedBuilder assignment
          (absorbedBuilder candidate base layout statementId) =
        Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (ProductionSuccessorStateBinding.carryFrame successor)
          (ProductionSuccessorStateBinding.preCarryAbsorbedState statementId
            successor.preCarry) := by
    simpa [absorbedBuilder] using absorbedEq
  have gateEq := decodedBuilder_gate base ProductPoseidon2.constants
    assignment (absorbedBuilder candidate base layout statementId) one
    finalValid
  rw [absorbedExact] at gateEq
  simpa [builder, absorbedBuilder,
    ProductionSuccessorStateBinding.outputState] using gateEq

theorem rows_imply_outputDigest_lane
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate fullShape)
    (successorPlaced : Placed contract layout assignment successor)
    (base : Nat)
    (successorRows : Satisfies
      (rows candidate base layout statementId) assignment)
    (lane : Fin 4) :
    lcEval assignment
        ((builder candidate base layout statementId).lanes
          (ProductionSuccessorStateBinding.outputLane lane)) =
      (ProductionSuccessorStateBinding.outputDigest
        statementId successor lane).val := by
  have stateEqual := rows_imply_outputState contract canonical one statementId
    successor successorPlaced base successorRows
  exact congrArg
    (fun state => state.lanes
      (ProductionSuccessorStateBinding.outputLane lane)) stateEqual

end Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor
