import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Core.Poseidon2SpongeReceipt

/-!
Contract: fused typed occurrence of the production rate-four Poseidon2
sponge on the selected 23-field plain/stateless XOut preimage.

Assurance tier: artifact-checked.

Owns:
- the source-selected six absorb rounds and mandatory padding round;
- one normalized straight-line program with 4,225 definition rows;
- elimination of internal permutation-output gates inside the fused sponge;
- four activation-gated copies to the visible digest core;
- exact row/temporary cost and a nonoptional receipt;
- active soundness, honest active completion, and inactive completion.

Does not own: XOut serialization, alignment checks, optional-digest presence,
either hash `CallRecipe`, generated call-site placement, native Poseidon2
parity, or collision resistance.

Emits constraints: exactly 4,229 rows and 4,225 auxiliary temporary columns.
The four visible outputs are allocated by the enclosing call receipt.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe

set_option maxRecDepth 131072
set_option maxHeartbeats 8000000

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace NumericSponge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Call
open Nightstream.Implementation.R1CS.Poseidon2Sponge

/-- Rename one normalized RHS without changing coefficient order. -/
def renameRhs (columnMap : Nat -> Nat) : Rhs -> Rhs
  | .linear terms =>
      .linear (terms.map fun term => (columnMap term.1, term.2))
  | .product left right =>
      .product
        (left.map fun term => (columnMap term.1, term.2))
        (right.map fun term => (columnMap term.1, term.2))

/-- Rename one deterministic SSA definition through an exact column map. -/
def renameDefinition
    (columnMap : Nat -> Nat)
    (definition : Definition) : Definition where
  output := columnMap definition.output
  rhs := renameRhs columnMap definition.rhs

/-- Deterministic definitions represented by one compact sponge round. -/
def roundDefinitions (round : Round) : List Definition :=
  let wrapper :=
    match round.kind with
    | .absorb chunkColumns =>
        (List.range chunkColumns.length).map fun lane => {
          output := round.permutationInputColumns.getD lane 0
          rhs := .linear
            [ (round.stateBeforeColumns.getD lane 0, 1)
            , (chunkColumns.getD lane 0, 1) ]
        }
    | .pad =>
        [{
          output := round.permutationInputColumns.getD 0 0
          rhs := .linear
            [(round.stateBeforeColumns.getD 0 0, 1), (0, 1)]
        }]
  wrapper ++
    Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions.map
      (renameDefinition round.call.columnMap)

/-- Complete deterministic definition program represented by a sponge
trace, including its explicit zero-state allocation. -/
def traceDefinitions (trace : Trace) : List Definition :=
  {
    output := trace.zeroColumn
    rhs := .linear []
  } :: trace.rounds.flatMap roundDefinitions

private theorem renameDefinition_builderRow
    (columnMap : Nat -> Nat)
    (mapsZero : columnMap 0 = 0)
    (definition : Definition) :
    (renameDefinition columnMap definition).builderRow =
      renameRow columnMap definition.builderRow := by
  cases definition with
  | mk output rhs =>
      cases rhs <;>
        simp [renameDefinition, renameRhs, Definition.builderRow,
          builderLinearRow, renameRow, renameTerms, negateTerms,
          mapsZero, Function.comp_def]

private theorem roundDefinitions_builderRows
    (round : Round) :
    (roundDefinitions round).map Definition.builderRow =
      round.rows := by
  have permutationRows :
      (Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions.map
          (renameDefinition round.call.columnMap)).map
            Definition.builderRow =
        round.call.rows := by
    rw [Call.rows,
      Nightstream.Implementation.R1CS.Poseidon2Permutation.rows,
      List.map_map, List.map_map]
    apply List.map_congr_left
    intro definition member
    exact renameDefinition_builderRow round.call.columnMap
      round.call.columnMap_zero definition
  cases kind : round.kind with
  | absorb chunkColumns =>
      simp only [roundDefinitions, kind, List.map_append]
      rw [permutationRows]
      simp [Round.rows, Round.expectedDefinitionRows, kind,
        Definition.builderRow]
  | pad =>
      simp only [roundDefinitions, kind, List.map_append]
      rw [permutationRows]
      simp [Round.rows, Round.expectedDefinitionRows, kind,
        Definition.builderRow]

private theorem traceDefinitions_builderRows
    (trace : Trace) :
    (traceDefinitions trace).map Definition.builderRow =
      trace.rows := by
  simp [traceDefinitions, Trace.rows, Trace.zeroDefinitionRows,
    List.map_flatMap, roundDefinitions_builderRows,
    Definition.builderRow]

/-- The selected normalized source schedule. Columns `1..23` are inputs;
columns `24..4248` are allocated exactly once in emission order. -/
def trace : Trace where
  inputColumns := List.range' 1 23
  zeroColumn := 24
  zeroRow := 0
  rounds := [
    {
      kind := .absorb [1, 2, 3, 4]
      stateBeforeColumns := List.replicate 8 24
      permutationInputColumns := [25, 26, 27, 28, 24, 24, 24, 24]
      permutationOutputColumns := List.range' 621 8
      definingRows := List.range' 1 4
      call := {
        rowStart := 5
        rowEnd := 605
        inputColumns := [25, 26, 27, 28, 24, 24, 24, 24]
        firstAllocatedColumn := 29
      }
    },
    {
      kind := .absorb [5, 6, 7, 8]
      stateBeforeColumns := List.range' 621 8
      permutationInputColumns :=
        [629, 630, 631, 632, 625, 626, 627, 628]
      permutationOutputColumns := List.range' 1225 8
      definingRows := List.range' 605 4
      call := {
        rowStart := 609
        rowEnd := 1209
        inputColumns := [629, 630, 631, 632, 625, 626, 627, 628]
        firstAllocatedColumn := 633
      }
    },
    {
      kind := .absorb [9, 10, 11, 12]
      stateBeforeColumns := List.range' 1225 8
      permutationInputColumns :=
        [1233, 1234, 1235, 1236, 1229, 1230, 1231, 1232]
      permutationOutputColumns := List.range' 1829 8
      definingRows := List.range' 1209 4
      call := {
        rowStart := 1213
        rowEnd := 1813
        inputColumns :=
          [1233, 1234, 1235, 1236, 1229, 1230, 1231, 1232]
        firstAllocatedColumn := 1237
      }
    },
    {
      kind := .absorb [13, 14, 15, 16]
      stateBeforeColumns := List.range' 1829 8
      permutationInputColumns :=
        [1837, 1838, 1839, 1840, 1833, 1834, 1835, 1836]
      permutationOutputColumns := List.range' 2433 8
      definingRows := List.range' 1813 4
      call := {
        rowStart := 1817
        rowEnd := 2417
        inputColumns :=
          [1837, 1838, 1839, 1840, 1833, 1834, 1835, 1836]
        firstAllocatedColumn := 1841
      }
    },
    {
      kind := .absorb [17, 18, 19, 20]
      stateBeforeColumns := List.range' 2433 8
      permutationInputColumns :=
        [2441, 2442, 2443, 2444, 2437, 2438, 2439, 2440]
      permutationOutputColumns := List.range' 3037 8
      definingRows := List.range' 2417 4
      call := {
        rowStart := 2421
        rowEnd := 3021
        inputColumns :=
          [2441, 2442, 2443, 2444, 2437, 2438, 2439, 2440]
        firstAllocatedColumn := 2445
      }
    },
    {
      kind := .absorb [21, 22, 23]
      stateBeforeColumns := List.range' 3037 8
      permutationInputColumns :=
        [3045, 3046, 3047, 3040, 3041, 3042, 3043, 3044]
      permutationOutputColumns := List.range' 3640 8
      definingRows := List.range' 3021 3
      call := {
        rowStart := 3024
        rowEnd := 3624
        inputColumns :=
          [3045, 3046, 3047, 3040, 3041, 3042, 3043, 3044]
        firstAllocatedColumn := 3048
      }
    },
    {
      kind := .pad
      stateBeforeColumns := List.range' 3640 8
      permutationInputColumns :=
        [3648, 3641, 3642, 3643, 3644, 3645, 3646, 3647]
      permutationOutputColumns := List.range' 4241 8
      definingRows := [3624]
      call := {
        rowStart := 3625
        rowEnd := 4225
        inputColumns :=
          [3648, 3641, 3642, 3643, 3644, 3645, 3646, 3647]
        firstAllocatedColumn := 3649
      }
    }
  ]
  outputColumns := List.range' 4241 4

def definitions : List Definition :=
  traceDefinitions trace

def known : List Nat :=
  List.range 24

/-- Every output is allocated at the next source column in program order. -/
def SequentialOutputs : Nat -> List Definition -> Prop
  | _, [] => True
  | next, definition :: tail =>
      definition.output = next ∧
        SequentialOutputs (next + 1) tail

private def sequentialOutputsDecidable :
    (next : Nat) -> (source : List Definition) ->
      Decidable (SequentialOutputs next source)
  | _, [] =>
      isTrue True.intro
  | next, definition :: tail =>
      if output : definition.output = next then
        match sequentialOutputsDecidable (next + 1) tail with
        | isTrue rest =>
            isTrue ⟨output, rest⟩
        | isFalse notRest =>
            isFalse fun accepted => notRest accepted.2
      else
        isFalse fun accepted => output accepted.1

instance (next : Nat) (source : List Definition) :
    Decidable (SequentialOutputs next source) :=
  sequentialOutputsDecidable next source

/-- Every definition reads only strictly earlier source columns. -/
def ReferencesBeforeOutput (source : List Definition) : Prop :=
  ∀ definition ∈ source,
    ∀ column ∈ definition.rhs.refs,
      column < definition.output

instance (source : List Definition) :
    Decidable (ReferencesBeforeOutput source) := by
  unfold ReferencesBeforeOutput
  infer_instance

private theorem wellFormed_of_sequential
    (next : Nat)
    (known : List Nat)
    (source : List Definition)
    (knownExact : ∀ column, column ∈ known ↔ column < next)
    (sequential : SequentialOutputs next source)
    (referencesBefore : ReferencesBeforeOutput source) :
    WellFormed known source := by
  induction source generalizing next known with
  | nil =>
      exact WellFormed.nil known
  | cons head tail inductionHypothesis =>
      have headOutput : head.output = next :=
        sequential.1
      have tailSequential :
          SequentialOutputs (next + 1) tail :=
        sequential.2
      have headReferences : ReferencesOnly known head := by
        intro column member
        apply (knownExact column).2
        rw [← headOutput]
        exact referencesBefore head (by simp) column member
      have headFresh : head.output ∉ known := by
        intro member
        have below := (knownExact head.output).1 member
        omega
      have tailReferences : ReferencesBeforeOutput tail := by
        intro definition definitionMember column columnMember
        exact referencesBefore definition
          (List.mem_cons_of_mem head definitionMember)
          column columnMember
      apply WellFormed.cons headReferences headFresh
      apply inductionHypothesis (next + 1) (head.output :: known)
      · intro column
        rw [List.mem_cons, knownExact, headOutput]
        constructor
        · intro member
          rcases member with equal | below
          · omega
          · omega
        · intro below
          by_cases equal : column = next
          · exact Or.inl equal
          · exact Or.inr (by omega)
      · exact tailSequential
      · exact tailReferences

def sourceSchedule : List ValueSchedule :=
  [.absorb 4, .absorb 4, .absorb 4, .absorb 4, .absorb 4, .absorb 3, .pad]

theorem inputColumns_exact :
    trace.inputColumns = List.range' 1 23 :=
  rfl

theorem valueSchedule_exact :
    valueSchedules trace.rounds = sourceSchedule := by
  decide

theorem rows_eq_builderRows :
    trace.rows = definitions.map Definition.builderRow :=
  (traceDefinitions_builderRows trace).symm

theorem sequential_outputs :
    SequentialOutputs 24 definitions := by
  decide

theorem references_before_output :
    ReferencesBeforeOutput definitions := by
  decide

theorem definitions_wellFormed :
    WellFormed known definitions :=
  wellFormed_of_sequential 24 known definitions
    (by
      intro column
      exact List.mem_range)
    sequential_outputs references_before_output

theorem definitions_length :
    definitions.length = 4225 := by
  decide

theorem definitions_canonical :
    ∀ definition ∈ definitions, definition.Canonical := by
  decide

theorem definition_outputs_exact :
    definitions.map Definition.output = List.range' 24 4225 := by
  decide

theorem trace_valid :
    trace.Valid trace.rows := by
  constructor <;> decide

theorem emissionReceipt :
    EmissionReceipt trace trace.rows 23 0 24 := by
  constructor <;> decide

end NumericSponge

def inputWidth : Nat := 23

def outputWidth : Nat := 4

def temporaryWidth : Nat := 4225

def coreRowCount : Nat := 4225

def gateRowCount : Nat := 4

def recurringRows : Nat := coreRowCount + gateRowCount

def inputLayout : Layout :=
  auxiliaryLayout inputWidth

def outputLayout : Layout :=
  auxiliaryLayout outputWidth

def temporaryLayout : Layout :=
  auxiliaryLayout temporaryWidth

/-- Exact physical data of one fused sponge occurrence. Only the 4,225
internal coordinates may be changed by honest completion. -/
structure Frame where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  active : ColumnId
  input : ColumnBundle inputLayout
  output : ColumnBundle outputLayout
  temporaries : ColumnBundle temporaryLayout
  allocationsNodup :
    (output.ids ++ temporaries.ids).Nodup
  temporariesDisjointVisible :
    IdsDisjoint temporaries.ids
      ([one, active] ++ input.ids ++ output.ids)
  outputsDisjointPreexisting :
    IdsDisjoint output.ids ([one, active] ++ input.ids)
  allocationsOwned :
    ∀ column,
      column ∈ output.columns ++ temporaries.columns ->
        column.id.owner = owner

namespace Frame

def visibleIds (frame : Frame) : List ColumnId :=
  [frame.one, frame.active] ++ frame.input.ids ++ frame.output.ids

def allocations (frame : Frame) : List OwnedColumn :=
  frame.output.columns ++ frame.temporaries.columns

@[simp] theorem input_ids_length (frame : Frame) :
    frame.input.ids.length = inputWidth := by
  rw [ColumnBundle.ids, List.length_map, frame.input.length_eq]
  simp [inputLayout, inputWidth, auxiliaryLayout, ownedLayout]

@[simp] theorem output_ids_length (frame : Frame) :
    frame.output.ids.length = outputWidth := by
  rw [ColumnBundle.ids, List.length_map, frame.output.length_eq]
  simp [outputLayout, outputWidth, auxiliaryLayout, ownedLayout]

@[simp] theorem temporary_ids_length (frame : Frame) :
    frame.temporaries.ids.length = temporaryWidth := by
  rw [ColumnBundle.ids, List.length_map, frame.temporaries.length_eq]
  change
    (List.replicate temporaryWidth Ownership.auxiliaryColumn).length =
      temporaryWidth
  exact List.length_replicate ..

end Frame

def inputColumn (frame : Frame) (index : Nat) : ColumnId :=
  frame.input.ids.getD index frame.one

def outputColumn (frame : Frame) (lane : Nat) : ColumnId :=
  frame.output.ids.getD lane frame.one

def temporaryColumn (frame : Frame) (index : Nat) : ColumnId :=
  frame.temporaries.ids.getD index frame.one

/-- Normalized source column zero is constant one, `1..23` are the ordered
input fields, and `24..4248` are the exact straight-line temporaries. -/
def columnMap (frame : Frame) (source : Nat) : ColumnId :=
  if source = 0 then frame.one
  else if source < 24 then inputColumn frame (source - 1)
  else temporaryColumn frame (source - 24)

/-- The four core digest lanes are the first four outputs of the terminal
padding permutation, at normalized columns `4241..4244`. -/
def internalOutputColumn (frame : Frame) (lane : Nat) : ColumnId :=
  temporaryColumn frame (4217 + lane)

@[simp] theorem columnMap_zero (frame : Frame) :
    columnMap frame 0 = frame.one := by
  simp [columnMap]

theorem columnMap_input
    (frame : Frame)
    (index : Nat)
    (indexLt : index < inputWidth) :
    columnMap frame (index + 1) = inputColumn frame index := by
  unfold inputWidth at indexLt
  unfold columnMap
  rw [if_neg (by omega), if_pos (by omega)]
  congr 1

theorem columnMap_internalOutput
    (frame : Frame)
    (lane : Nat) :
    columnMap frame (4241 + lane) = internalOutputColumn frame lane := by
  unfold columnMap internalOutputColumn
  rw [if_neg (by omega), if_neg (by omega)]
  congr 1
  omega

def coreRows (frame : Frame) : List OwnedRow :=
  ownedRowsFrom frame.owner frame.firstOrdinal (columnMap frame)
    NumericSponge.trace.rows

def gateRow (frame : Frame) (lane : Nat) : OwnedRow where
  id := {
    owner := frame.owner
    ordinal := frame.firstOrdinal + coreRowCount + lane
  }
  row := {
    a := singleton frame.active 1
    b := difference
      (internalOutputColumn frame lane)
      (outputColumn frame lane)
    c := []
  }

def gateRows (frame : Frame) : List OwnedRow :=
  (List.range gateRowCount).map (gateRow frame)

def rows (frame : Frame) : List OwnedRow :=
  coreRows frame ++ gateRows frame

def footprint : CallFootprint where
  recurringRows := recurringRows
  temporaries := [temporaryLayout]

/-- Mandatory output/temporary/row receipt of one fused occurrence. -/
def receipt (frame : Frame) : CallReceipt where
  outputBundles := [frame.output.columns]
  temporaryBundles := [frame.temporaries.columns]
  rows := rows frame

theorem receipt_exact (frame : Frame) :
    receipt frame =
      { outputBundles := [frame.output.columns]
        temporaryBundles := [frame.temporaries.columns]
        rows := rows frame } :=
  rfl

theorem coreRows_length (frame : Frame) :
    (coreRows frame).length = coreRowCount := by
  rw [coreRows, ownedRowsFrom_length]
  rw [NumericSponge.rows_eq_builderRows, List.length_map,
    NumericSponge.definitions_length]
  rfl

theorem gateRows_length (frame : Frame) :
    (gateRows frame).length = gateRowCount := by
  simp [gateRows]

theorem rows_length (frame : Frame) :
    (rows frame).length = recurringRows := by
  rw [rows, List.length_append, coreRows_length, gateRows_length]
  rfl

theorem receipt_row_count (frame : Frame) :
    (receipt frame).rows.length = footprint.recurringRows :=
  rows_length frame

theorem rows_owned
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ rows frame) :
    owned.id.owner = frame.owner := by
  rcases List.mem_append.mp member with coreMember | gateMember
  · exact ownedRowsFrom_owned frame.owner frame.firstOrdinal
      (columnMap frame) NumericSponge.trace.rows owned coreMember
  · rcases List.mem_map.mp gateMember with
      ⟨lane, laneMember, equal⟩
    subst owned
    rfl

private theorem gateIds_nodup_of
    (frame : Frame)
    (lanes : List Nat)
    (nodup : lanes.Nodup) :
    ((lanes.map (gateRow frame)).map (fun row => row.id)).Nodup := by
  rw [List.map_map]
  change
    (lanes.map (fun lane => (gateRow frame lane).id)).Nodup
  induction lanes with
  | nil =>
      exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      have split : head ∉ tail ∧ tail.Nodup := by
        simpa only [List.nodup_cons] using nodup
      rw [List.map_cons, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_map.mp member with
          ⟨lane, laneMember, equal⟩
        have ordinalEqual := congrArg RowId.ordinal equal
        simp only [gateRow] at ordinalEqual
        have laneEqual : lane = head := by omega
        exact split.1 (laneEqual ▸ laneMember)
      · exact inductionHypothesis split.2

private theorem gateRows_ids_nodup (frame : Frame) :
    ((gateRows frame).map (fun row => row.id)).Nodup :=
  gateIds_nodup_of frame (List.range gateRowCount) List.nodup_range

private theorem coreRow_ordinal_lt
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ coreRows frame) :
    owned.id.ordinal < frame.firstOrdinal + coreRowCount := by
  have mappedMember :
      owned.id ∈ (coreRows frame).map (fun row => row.id) :=
    List.mem_map.mpr ⟨owned, member, rfl⟩
  rw [coreRows,
    ownedRowsFrom_ids_exact frame.owner frame.firstOrdinal
      (columnMap frame) NumericSponge.trace.rows] at mappedMember
  rcases List.mem_map.mp mappedMember with
    ⟨ordinal, ordinalMember, equal⟩
  rcases List.mem_range'.mp ordinalMember with
    ⟨offset, offsetLt, ordinalEqual⟩
  have exactOrdinal : ordinal = owned.id.ordinal := by
    simpa using congrArg RowId.ordinal equal
  have sourceLength :
      NumericSponge.trace.rows.length = coreRowCount := by
    rw [NumericSponge.rows_eq_builderRows, List.length_map,
      NumericSponge.definitions_length]
    rfl
  have ordinalLt :
      ordinal <
        frame.firstOrdinal + NumericSponge.trace.rows.length := by
    simp only [Nat.one_mul] at ordinalEqual
    omega
  calc
    owned.id.ordinal = ordinal := exactOrdinal.symm
    _ < frame.firstOrdinal + NumericSponge.trace.rows.length := ordinalLt
    _ = frame.firstOrdinal + coreRowCount := by rw [sourceLength]

private theorem gateRow_ordinal_ge
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ gateRows frame) :
    frame.firstOrdinal + coreRowCount ≤ owned.id.ordinal := by
  rcases List.mem_map.mp member with
    ⟨lane, laneMember, equal⟩
  subst owned
  simp [gateRow]

theorem rowIds_nodup (frame : Frame) :
    ((rows frame).map (fun row => row.id)).Nodup := by
  rw [rows, List.map_append, List.nodup_append]
  refine ⟨
    ownedRowsFrom_ids_nodup frame.owner frame.firstOrdinal
      (columnMap frame) NumericSponge.trace.rows,
    gateRows_ids_nodup frame,
    ?_⟩
  intro coreId coreMember gateId gateMember equal
  rcases List.mem_map.mp coreMember with
    ⟨coreRow, coreRowMember, coreEqual⟩
  rcases List.mem_map.mp gateMember with
    ⟨gate, gateRowMember, gateEqual⟩
  have below := coreRow_ordinal_lt frame coreRow coreRowMember
  have above := gateRow_ordinal_ge frame gate gateRowMember
  have rowIdsEqual : coreRow.id = gate.id :=
    coreEqual.trans (equal.trans gateEqual.symm)
  have ordinalEqual := congrArg RowId.ordinal rowIdsEqual
  omega

/-! ## Executable sponge semantics -/

def initialNumeric
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    Nat -> Nat :=
  numericAssignment (columnMap frame) assignment

def execution
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    Nat -> Nat :=
  Nightstream.Implementation.R1CS.Program.run
    (initialNumeric frame assignment) NumericSponge.definitions

/-- Pure four-lane sponge result on the ordered 23 visible input fields. -/
def semanticLane
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (lane : Nat) : Field :=
  residue
    (Nightstream.Implementation.R1CS.Poseidon2Sponge.runValueRounds
      NumericSponge.trace.rounds
      (NumericSponge.trace.inputColumns.map
        (initialNumeric frame assignment))
      (fun _ => 0) lane)

theorem initialNumeric_canonical
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (source : Nat) :
    initialNumeric frame assignment source <
      Numeric.modulus :=
  numericAssignment_canonical (columnMap frame) assignment source

theorem initialNumeric_zero
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1) :
    initialNumeric frame assignment 0 = 1 := by
  change (assignment (columnMap frame 0)).val = 1
  rw [columnMap_zero, constantOne]
  rfl

theorem initialNumeric_input
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (index : Nat)
    (indexLt : index < inputWidth) :
    initialNumeric frame assignment (index + 1) =
      (assignment (inputColumn frame index)).val := by
  rw [initialNumeric, numericAssignment,
    columnMap_input frame index indexLt]

private theorem trace_output_getD
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    NumericSponge.trace.outputColumns.getD lane 0 =
      4241 + lane := by
  unfold outputWidth at laneLt
  have cases : lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 := by
    omega
  rcases cases with rfl | rfl | rfl | rfl <;>
    decide

/-! ## Activation gates -/

private theorem satisfies_iff_forall
    (source : List OwnedRow)
    (assignment : ColumnId -> Field) :
    Satisfies source assignment ↔
      ∀ owned, owned ∈ source -> owned.row.Holds assignment := by
  induction source with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      rw [satisfies_cons, inductionHypothesis]
      constructor
      · rintro ⟨headHolds, tailHolds⟩ owned member
        rcases List.mem_cons.mp member with equal | tailMember
        · subst owned
          exact headHolds
        · exact tailHolds owned tailMember
      · intro all
        exact ⟨
          all head (by simp),
          fun owned member => all owned (by simp [member])
        ⟩

theorem gateRow_active_iff
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (activeOne : assignment frame.active = 1)
    (lane : Nat) :
    (gateRow frame lane).row.Holds assignment ↔
      assignment (internalOutputColumn frame lane) =
        assignment (outputColumn frame lane) := by
  simp only [gateRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    activeOne, Fin.one_mul, Fin.mul_one,
    Fin.add_zero, Lean.Grind.Fin.neg_mul]
  simpa only [Fin.sub_eq_add_neg] using
    (Lean.Grind.AddCommGroup.sub_eq_zero_iff :
      assignment (internalOutputColumn frame lane) -
            assignment (outputColumn frame lane) = 0 ↔
        assignment (internalOutputColumn frame lane) =
          assignment (outputColumn frame lane))

theorem gateRow_complete_of_inactive
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (activeZero : assignment frame.active = 0)
    (lane : Nat) :
    (gateRow frame lane).row.Holds assignment := by
  simp only [gateRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    activeZero, Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul]
  exact Fin.zero_mul _

theorem gateRows_active_output_eq
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (activeOne : assignment frame.active = 1)
    (holds : Satisfies (gateRows frame) assignment)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    assignment (internalOutputColumn frame lane) =
      assignment (outputColumn frame lane) := by
  apply (gateRow_active_iff frame assignment activeOne lane).1
  apply (satisfies_iff_forall (gateRows frame) assignment).1 holds
  apply List.mem_map.mpr
  refine ⟨lane, ?_, rfl⟩
  apply List.mem_range.mpr
  simpa [outputWidth, gateRowCount] using laneLt

theorem gateRows_complete_of_inactive
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (activeZero : assignment frame.active = 0) :
    Satisfies (gateRows frame) assignment := by
  apply (satisfies_iff_forall (gateRows frame) assignment).2
  intro owned member
  rcases List.mem_map.mp member with
    ⟨lane, laneMember, equal⟩
  subst owned
  exact gateRow_complete_of_inactive frame assignment activeZero lane

/-- Active satisfaction forces each visible output to equal the pure
rate-four sponge on exactly the ordered 23 input fields. -/
theorem active_sound
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (holds : Satisfies (rows frame) assignment)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    assignment (outputColumn frame lane) =
      semanticLane frame assignment lane := by
  have split :=
    (satisfies_append_iff (coreRows frame) (gateRows frame)
      assignment).1 holds
  have numericSatisfies :
      Numeric.satisfies NumericSponge.trace.rows
        (initialNumeric frame assignment) :=
    (ownedRowsFrom_satisfies_iff frame.owner frame.firstOrdinal
      (columnMap frame) NumericSponge.trace.rows assignment).1 split.1
  have numericSound :=
    Nightstream.Implementation.R1CS.Poseidon2Sponge.trace_values_sound
      NumericSponge.trace_valid
      (fun source =>
        initialNumeric_canonical frame assignment source)
      (initialNumeric_zero frame assignment constantOne)
      numericSatisfies lane (by simpa [outputWidth] using laneLt)
  have sourceExact :=
    trace_output_getD lane laneLt
  have internalEqualsSemantic :
      assignment (internalOutputColumn frame lane) =
        semanticLane frame assignment lane := by
    calc
      assignment (internalOutputColumn frame lane) =
          residue
            (assignment (internalOutputColumn frame lane)).val :=
        (residue_field_val
          (assignment (internalOutputColumn frame lane))).symm
      _ =
          residue
            (initialNumeric frame assignment
              (NumericSponge.trace.outputColumns.getD lane 0)) := by
        rw [sourceExact, initialNumeric, numericAssignment,
          columnMap_internalOutput]
      _ =
          residue
            (Nightstream.Implementation.R1CS.Poseidon2Sponge.runValueRounds
              NumericSponge.trace.rounds
              (NumericSponge.trace.inputColumns.map
                (initialNumeric frame assignment))
              (fun _ => 0) lane) :=
        congrArg residue numericSound
      _ = semanticLane frame assignment lane :=
        rfl
  have gateEquality :=
    gateRows_active_output_eq frame assignment activeOne split.2
      lane laneLt
  exact gateEquality.symm.trans internalEqualsSemantic

/-! ## Honest temporary-only completion -/

def temporaryValues
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    List Field :=
  (List.range temporaryWidth).map fun index =>
    residue (execution frame assignment (24 + index))

def complete
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    ColumnId -> Field :=
  writeColumns assignment frame.temporaries.ids
    (temporaryValues frame assignment)

@[simp] theorem temporaryValues_length
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    (temporaryValues frame assignment).length = temporaryWidth := by
  simp [temporaryValues]

theorem Frame.temporary_ids_nodup (frame : Frame) :
    frame.temporaries.ids.Nodup := by
  have split := frame.allocationsNodup
  rw [List.nodup_append] at split
  exact split.2.1

theorem complete_changesOnly
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    ChangesOnly frame.temporaries.ids assignment
      (complete frame assignment) :=
  writeColumns_changesOnly assignment frame.temporaries.ids
    (temporaryValues frame assignment)

theorem complete_agrees_visible
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    AgreesOn frame.visibleIds assignment
      (complete frame assignment) :=
  writeColumns_agreesOn assignment frame.temporaries.ids
    frame.visibleIds (temporaryValues frame assignment)
    frame.temporariesDisjointVisible

private theorem temporaryValues_getD
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (index : Nat)
    (indexLt : index < temporaryWidth)
    (fallback : Field) :
    (temporaryValues frame assignment).getD index fallback =
      residue (execution frame assignment (24 + index)) := by
  have valuesLt :
      index < (temporaryValues frame assignment).length := by
    rw [temporaryValues_length]
    exact indexLt
  rw [← List.getElem_eq_getD
    (l := temporaryValues frame assignment)
    (i := index) (h := valuesLt) fallback]
  simp [temporaryValues]

theorem complete_temporary
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (index : Nat)
    (indexLt : index < temporaryWidth) :
    complete frame assignment (temporaryColumn frame index) =
      residue (execution frame assignment (24 + index)) := by
  have recovered :
      frame.temporaries.ids.map (complete frame assignment) =
        temporaryValues frame assignment := by
    apply writeColumns_map_eq
    · rw [Frame.temporary_ids_length, temporaryValues_length]
    · exact frame.temporary_ids_nodup
  have atIndex := congrArg
    (fun values : List Field =>
      values.getD index (complete frame assignment frame.one))
    recovered
  have idsLt : index < frame.temporaries.ids.length := by
    rw [Frame.temporary_ids_length]
    exact indexLt
  have mappedIdsLt :
      index <
        (frame.temporaries.ids.map
          (complete frame assignment)).length := by
    simpa using idsLt
  have valuesLt :
      index < (temporaryValues frame assignment).length := by
    rw [temporaryValues_length]
    exact indexLt
  change
    (frame.temporaries.ids.map (complete frame assignment)).getD
          index (complete frame assignment frame.one) =
        (temporaryValues frame assignment).getD
          index (complete frame assignment frame.one)
    at atIndex
  rw [← List.getElem_eq_getD
      (l := frame.temporaries.ids.map (complete frame assignment))
      (i := index) (h := mappedIdsLt)
      (complete frame assignment frame.one),
    ← List.getElem_eq_getD
      (l := temporaryValues frame assignment)
      (i := index) (h := valuesLt)
      (complete frame assignment frame.one)] at atIndex
  simp only [List.getElem_map] at atIndex
  rw [List.getElem_eq_getD
      (l := frame.temporaries.ids) (i := index)
      (h := idsLt) frame.one,
    List.getElem_eq_getD
      (l := temporaryValues frame assignment) (i := index)
      (h := valuesLt) (complete frame assignment frame.one)]
    at atIndex
  rw [temporaryValues_getD frame assignment index indexLt] at atIndex
  exact atIndex

theorem execution_canonical
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (source : Nat) :
    execution frame assignment source < Numeric.modulus :=
  Nightstream.Implementation.R1CS.Program.run_canonical
    (fun column =>
      initialNumeric_canonical frame assignment column)
    source

private theorem inputColumn_mem
    (frame : Frame)
    (index : Nat)
    (indexLt : index < inputWidth) :
    inputColumn frame index ∈ frame.input.ids := by
  have idsLt : index < frame.input.ids.length := by
    rw [Frame.input_ids_length]
    exact indexLt
  unfold inputColumn
  rw [← List.getElem_eq_getD
    (l := frame.input.ids) (i := index) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem outputColumn_mem
    (frame : Frame)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    outputColumn frame lane ∈ frame.output.ids := by
  have idsLt : lane < frame.output.ids.length := by
    rw [Frame.output_ids_length]
    exact laneLt
  unfold outputColumn
  rw [← List.getElem_eq_getD
    (l := frame.output.ids) (i := lane) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem columnMap_mem_visible_of_lt
    (frame : Frame)
    (source : Nat)
    (sourceLt : source < 24) :
    columnMap frame source ∈ frame.visibleIds := by
  by_cases sourceZero : source = 0
  · subst source
    simp [Frame.visibleIds, columnMap_zero]
  · have laneLt : source - 1 < inputWidth := by
      unfold inputWidth
      omega
    have sourceEq : source - 1 + 1 = source := by
      omega
    have mapped :
        columnMap frame source =
          inputColumn frame (source - 1) := by
      calc
        columnMap frame source =
            columnMap frame (source - 1 + 1) := by rw [sourceEq]
        _ = inputColumn frame (source - 1) :=
          columnMap_input frame (source - 1) laneLt
    rw [mapped]
    simp [Frame.visibleIds,
      inputColumn_mem frame (source - 1) laneLt]

private theorem columnMap_temporary
    (frame : Frame)
    (source : Nat)
    (sourceGe : 24 ≤ source) :
    columnMap frame source =
      temporaryColumn frame (source - 24) := by
  unfold columnMap
  rw [if_neg (by omega), if_neg (by omega)]

private theorem residue_val_of_lt
    (value : Nat)
    (valueLt : value < Numeric.modulus) :
    (residue value).val = value := by
  change
    value % Nightstream.SuperNeo.Concrete.goldilocksModulus = value
  apply Nat.mod_eq_of_lt
  simpa [Numeric.modulus,
    Nightstream.Implementation.R1CS.goldilocksP,
    Nightstream.SuperNeo.Concrete.goldilocksModulus] using valueLt

private theorem run_eq_of_not_output
    (state : Nat -> Nat)
    (definitions :
      List Nightstream.Implementation.R1CS.Program.Definition)
    (source : Nat)
    (notOutput :
      ∀ definition, definition ∈ definitions ->
        definition.output ≠ source) :
    Nightstream.Implementation.R1CS.Program.run
        state definitions source =
      state source := by
  induction definitions generalizing state with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have headDifferent : head.output ≠ source :=
        notOutput head (by simp)
      have tailDifferent :
          ∀ definition, definition ∈ tail ->
            definition.output ≠ source := by
        intro definition member
        exact notOutput definition (by simp [member])
      calc
        Nightstream.Implementation.R1CS.Program.run
              state (head :: tail) source =
            Nightstream.Implementation.R1CS.Program.run
              (Nightstream.Implementation.R1CS.Program.execute state head)
              tail source :=
          rfl
        _ =
            Nightstream.Implementation.R1CS.Program.execute
              state head source :=
          inductionHypothesis
            (Nightstream.Implementation.R1CS.Program.execute state head)
            tailDifferent
        _ = state source := by
          unfold Nightstream.Implementation.R1CS.Program.execute
          exact
            Nightstream.Implementation.R1CS.Program.setColumn_other
              state headDifferent.symm

private theorem definition_outputs_lt_columnCount :
    ∀ definition ∈ NumericSponge.definitions,
      definition.output < 4249 := by
  intro definition member
  have outputMember :
      definition.output ∈
        NumericSponge.definitions.map
          Nightstream.Implementation.R1CS.Program.Definition.output :=
    List.mem_map.mpr ⟨definition, member, rfl⟩
  rw [NumericSponge.definition_outputs_exact] at outputMember
  rcases List.mem_range'.mp outputMember with
    ⟨offset, offsetLt, outputExact⟩
  omega

private theorem columnMap_eq_one_of_columnCount_le
    (frame : Frame)
    (source : Nat)
    (sourceGe : 4249 ≤ source) :
    columnMap frame source = frame.one := by
  unfold columnMap temporaryColumn
  rw [if_neg (by omega), if_neg (by omega)]
  have outside :
      frame.temporaries.ids.length ≤ source - 24 := by
    rw [Frame.temporary_ids_length]
    unfold temporaryWidth
    omega
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_none outside]
  rfl

/-- Pulling the honest typed completion through the source map recovers the
exact deterministic 4,225-definition execution. -/
theorem completedNumeric_eq_execution
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    numericAssignment (columnMap frame) (complete frame assignment) =
      execution frame assignment := by
  funext source
  by_cases sourceLtKnown : source < 24
  · have preserved :=
      complete_agrees_visible frame assignment
        (columnMap frame source)
        (columnMap_mem_visible_of_lt frame source sourceLtKnown)
    have runPreserved :=
      Nightstream.Implementation.R1CS.Program.run_preserves_known
        NumericSponge.definitions_wellFormed
        (initialNumeric frame assignment)
        source (List.mem_range.mpr sourceLtKnown)
    calc
      numericAssignment (columnMap frame)
            (complete frame assignment) source =
          (complete frame assignment (columnMap frame source)).val :=
        rfl
      _ = (assignment (columnMap frame source)).val :=
        congrArg Fin.val preserved
      _ = initialNumeric frame assignment source :=
        rfl
      _ = execution frame assignment source :=
        runPreserved.symm
  · by_cases sourceLtColumnCount : source < 4249
    · have sourceGe : 24 ≤ source :=
        Nat.le_of_not_gt sourceLtKnown
      have indexLt : source - 24 < temporaryWidth := by
        unfold temporaryWidth
        omega
      calc
        numericAssignment (columnMap frame)
              (complete frame assignment) source =
            (complete frame assignment
              (temporaryColumn frame (source - 24))).val := by
          rw [numericAssignment,
            columnMap_temporary frame source sourceGe]
        _ =
            (residue
              (execution frame assignment
                (24 + (source - 24)))).val := by
          rw [complete_temporary frame assignment
            (source - 24) indexLt]
        _ = execution frame assignment source := by
          rw [show 24 + (source - 24) = source by omega]
          exact residue_val_of_lt _
            (execution_canonical frame assignment source)
    · have sourceGe : 4249 ≤ source :=
        Nat.le_of_not_gt sourceLtColumnCount
      have mapped :=
        columnMap_eq_one_of_columnCount_le frame source sourceGe
      have oneVisible : frame.one ∈ frame.visibleIds := by
        simp [Frame.visibleIds]
      have onePreserved :=
        complete_agrees_visible frame assignment frame.one oneVisible
      have runPreserved :
          execution frame assignment source =
            initialNumeric frame assignment source := by
        apply run_eq_of_not_output
        intro definition member
        have definitionLt :=
          definition_outputs_lt_columnCount definition member
        omega
      calc
        numericAssignment (columnMap frame)
              (complete frame assignment) source =
            (complete frame assignment frame.one).val := by
          rw [numericAssignment, mapped]
        _ = (assignment frame.one).val :=
          congrArg Fin.val onePreserved
        _ = initialNumeric frame assignment source := by
          rw [initialNumeric, numericAssignment, mapped]
        _ = execution frame assignment source :=
          runPreserved.symm

theorem core_complete
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1) :
    Satisfies (coreRows frame) (complete frame assignment) := by
  apply
    (ownedRowsFrom_satisfies_iff frame.owner frame.firstOrdinal
      (columnMap frame) NumericSponge.trace.rows
      (complete frame assignment)).2
  rw [completedNumeric_eq_execution,
    NumericSponge.rows_eq_builderRows]
  exact
    Nightstream.Implementation.R1CS.Program.run_satisfies_builder_rows
      NumericSponge.definitions_wellFormed
      (fun source =>
        initialNumeric_canonical frame assignment source)
      (by simp [NumericSponge.known])
      (initialNumeric_zero frame assignment constantOne)
      NumericSponge.definitions_canonical

private theorem execution_inputs_eq_initial
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    NumericSponge.trace.inputColumns.map
        (execution frame assignment) =
      NumericSponge.trace.inputColumns.map
        (initialNumeric frame assignment) := by
  rw [NumericSponge.inputColumns_exact]
  apply List.map_congr_left
  intro source member
  apply
    Nightstream.Implementation.R1CS.Program.run_preserves_known
      NumericSponge.definitions_wellFormed
      (initialNumeric frame assignment)
  rcases List.mem_range'.mp member with
    ⟨offset, offsetLt, sourceExact⟩
  apply List.mem_range.mpr
  omega

theorem execution_output_eq_semantic
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    residue
        (execution frame assignment
          (NumericSponge.trace.outputColumns.getD lane 0)) =
      semanticLane frame assignment lane := by
  have zeroPreserved :=
    Nightstream.Implementation.R1CS.Program.run_preserves_known
      NumericSponge.definitions_wellFormed
      (initialNumeric frame assignment)
      0 (by simp [NumericSponge.known])
  have programSatisfies :
      Numeric.satisfies NumericSponge.trace.rows
        (execution frame assignment) := by
    rw [NumericSponge.rows_eq_builderRows]
    exact
      Nightstream.Implementation.R1CS.Program.run_satisfies_builder_rows
        NumericSponge.definitions_wellFormed
        (fun source =>
          initialNumeric_canonical frame assignment source)
        (by simp [NumericSponge.known])
        (initialNumeric_zero frame assignment constantOne)
        NumericSponge.definitions_canonical
  have outputEquals :=
    Nightstream.Implementation.R1CS.Poseidon2Sponge.trace_values_sound
      NumericSponge.trace_valid
      (execution_canonical frame assignment)
      (zeroPreserved.trans
        (initialNumeric_zero frame assignment constantOne))
      programSatisfies lane (by simpa [outputWidth] using laneLt)
  rw [execution_inputs_eq_initial frame assignment] at outputEquals
  exact congrArg residue outputEquals

theorem gateRows_complete_of_active
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (outputsCorrect :
      ∀ lane, lane < outputWidth ->
        assignment (outputColumn frame lane) =
          semanticLane frame assignment lane) :
    Satisfies (gateRows frame) (complete frame assignment) := by
  apply
    (satisfies_iff_forall (gateRows frame)
      (complete frame assignment)).2
  intro owned member
  rcases List.mem_map.mp member with
    ⟨lane, laneMember, equal⟩
  subst owned
  have laneLtGate : lane < gateRowCount :=
    List.mem_range.mp laneMember
  have laneLt : lane < outputWidth := by
    simpa [gateRowCount, outputWidth] using laneLtGate
  have activeVisible : frame.active ∈ frame.visibleIds := by
    simp [Frame.visibleIds]
  have completedActive :
      complete frame assignment frame.active = 1 :=
    (complete_agrees_visible frame assignment
      frame.active activeVisible).trans activeOne
  apply
    (gateRow_active_iff frame (complete frame assignment)
      completedActive lane).2
  have internalIndexLt : 4217 + lane < temporaryWidth := by
    unfold temporaryWidth outputWidth at *
    omega
  have outputVisible :
      outputColumn frame lane ∈ frame.visibleIds := by
    simp [Frame.visibleIds, outputColumn_mem frame lane laneLt]
  have outputPreserved :=
    complete_agrees_visible frame assignment
      (outputColumn frame lane) outputVisible
  calc
    complete frame assignment (internalOutputColumn frame lane) =
        residue
          (execution frame assignment
            (24 + (4217 + lane))) := by
      exact complete_temporary frame assignment
        (4217 + lane) internalIndexLt
    _ =
        residue
          (execution frame assignment
            (NumericSponge.trace.outputColumns.getD lane 0)) := by
      have sourceArithmetic :
          24 + (4217 + lane) = 4241 + lane := by
        omega
      rw [sourceArithmetic, trace_output_getD lane laneLt]
    _ = semanticLane frame assignment lane :=
      execution_output_eq_semantic
        frame assignment constantOne lane laneLt
    _ = assignment (outputColumn frame lane) :=
      (outputsCorrect lane laneLt).symm
    _ = complete frame assignment (outputColumn frame lane) :=
      outputPreserved.symm

/-- Honest active inputs extend by writing only receipt-owned temporaries. -/
theorem active_complete
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (outputsCorrect :
      ∀ lane, lane < outputWidth ->
        assignment (outputColumn frame lane) =
          semanticLane frame assignment lane) :
    Satisfies (rows frame) (complete frame assignment) := by
  apply
    (satisfies_append_iff (coreRows frame) (gateRows frame)
      (complete frame assignment)).2
  exact ⟨
    core_complete frame assignment constantOne,
    gateRows_complete_of_active frame assignment constantOne
      activeOne outputsCorrect
  ⟩

/-- Inactive occurrences still execute the complete deterministic sponge;
only the four visible output copies become vacuous. -/
theorem inactive_complete
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeZero : assignment frame.active = 0) :
    Satisfies (rows frame) (complete frame assignment) := by
  have activeVisible : frame.active ∈ frame.visibleIds := by
    simp [Frame.visibleIds]
  have completedActive :
      complete frame assignment frame.active = 0 :=
    (complete_agrees_visible frame assignment
      frame.active activeVisible).trans activeZero
  apply
    (satisfies_append_iff (coreRows frame) (gateRows frame)
      (complete frame assignment)).2
  exact ⟨
    core_complete frame assignment constantOne,
    gateRows_complete_of_inactive frame
      (complete frame assignment) completedActive
  ⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe
