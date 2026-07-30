import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPlacement
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership

/-!
Contract: exact physical receipts, temporary allocation, and column
conservation for a well-placed symbolic duplex program.

Owns:

* one structured row receipt per call-local Poseidon2 receipt;
* the 352 internal columns per call (344 S-box values and eight carried
  outputs), with no duplicates;
* classification of every column mentioned by every emitted row; and
* conservation into the caller prefix or that exact allocation.

Does not own a protocol transcript schedule or its honest assignment.  A
caller constructs `WellPlaced` and `WellOwned` while building that schedule.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexHonest
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPlacement

/-! ## Call-order invariant -/

/-- Entry call IDs are exactly their list positions.  Unlike `WellPlaced`,
this invariant says nothing about the sources read by those calls. -/
def CallOrdered (builder : SymbolicDuplex.Builder) : Prop :=
  builder.entries.map SymbolicDuplex.Entry.call =
    List.range builder.entries.length

theorem callOrdered_start (lanes : State) (absorbed : Nat) :
    CallOrdered (SymbolicDuplex.start lanes absorbed) :=
  rfl

theorem callOrdered_permute
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered (SymbolicDuplex.permute base builder) := by
  unfold CallOrdered at ordered ⊢
  unfold SymbolicDuplex.permute
  simp only [List.map_append, List.map_cons, List.map_nil,
    List.length_append, List.length_cons, List.length_nil, ordered]
  rw [List.range_succ]

theorem callOrdered_guarded
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered (SymbolicDuplex.guarded base builder) := by
  unfold SymbolicDuplex.guarded
  split
  · exact callOrdered_permute base builder ordered
  · exact ordered

theorem callOrdered_absorb
    (base : Nat) (value : LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered (SymbolicDuplex.absorb base value builder) := by
  unfold SymbolicDuplex.absorb
  exact callOrdered_guarded base builder ordered

theorem callOrdered_absorbMany
    (base : Nat) :
    ∀ (values : List LinCombNormal.LinComb)
      (builder : SymbolicDuplex.Builder),
      CallOrdered builder →
        CallOrdered (SymbolicDuplex.absorbMany base values builder)
  | [], _, ordered => ordered
  | value :: rest, builder, ordered =>
      callOrdered_absorbMany base rest
        (SymbolicDuplex.absorb base value builder)
        (callOrdered_absorb base value builder ordered)

theorem callOrdered_gate
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered (SymbolicDuplex.gate base builder) :=
  callOrdered_permute base
    (SymbolicDuplex.absorb base SymbolicDuplex.one builder)
    (callOrdered_absorb base SymbolicDuplex.one builder ordered)

/-! ## Exact internal allocation -/

def perCallTemporaries : Nat := 352
def sboxTemporaries : Nat := 344

abbrev TemporaryPosition (calls : Nat) :=
  Fin (calls * perCallTemporaries)

def temporaryCall {calls : Nat} (position : TemporaryPosition calls) : Nat :=
  position.val / perCallTemporaries

def temporaryWithinCall {calls : Nat}
    (position : TemporaryPosition calls) : Nat :=
  position.val % perCallTemporaries

/-- Bound output ports occupy local offsets `0..7`; S-box columns occupy
`8..351`.  Input ports are absent from a carried-entry call. -/
def temporaryOffset {calls : Nat}
    (position : TemporaryPosition calls) : Nat :=
  if temporaryWithinCall position < sboxTemporaries
  then width + temporaryWithinCall position
  else temporaryWithinCall position - sboxTemporaries

def temporaryColumn (base calls : Nat)
    (position : TemporaryPosition calls) : Nat :=
  base + temporaryCall position * SymbolicDuplex.stride +
    temporaryOffset position

def temporaryColumns (base calls : Nat) : List Nat :=
  List.ofFn (temporaryColumn base calls)

private theorem nodup_ofFn_of_injective
    {α : Type} :
    ∀ {count : Nat} (function : Fin count → α),
      Function.Injective function → (List.ofFn function).Nodup
  | 0, _, _ => by simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal => Fin.succ_inj.mp (injective equal))

theorem temporaryCall_lt {calls : Nat}
    (position : TemporaryPosition calls) :
    temporaryCall position < calls := by
  have positionLt := position.isLt
  simp only [temporaryCall, perCallTemporaries] at *
  omega

theorem temporaryWithinCall_lt {calls : Nat}
    (position : TemporaryPosition calls) :
    temporaryWithinCall position < perCallTemporaries :=
  Nat.mod_lt _ (by decide)

theorem temporaryOffset_lt {calls : Nat}
    (position : TemporaryPosition calls) :
    temporaryOffset position < SymbolicDuplex.stride := by
  have withinLt := temporaryWithinCall_lt position
  unfold temporaryOffset
  split
  · rw [SymbolicDuplex.stride_eq]
    simp only [perCallTemporaries, sboxTemporaries, width] at *
    omega
  · rw [SymbolicDuplex.stride_eq]
    simp only [perCallTemporaries, sboxTemporaries, width] at *
    omega

private theorem position_eq_div_mul_add_mod {calls : Nat}
    (position : TemporaryPosition calls) :
    position.val =
      temporaryCall position * perCallTemporaries +
        temporaryWithinCall position := by
  unfold temporaryCall temporaryWithinCall
  calc
    position.val =
        position.val % perCallTemporaries +
          perCallTemporaries * (position.val / perCallTemporaries) :=
      (Nat.mod_add_div position.val perCallTemporaries).symm
    _ =
        position.val / perCallTemporaries * perCallTemporaries +
          position.val % perCallTemporaries := by
      rw [Nat.add_comm,
        Nat.mul_comm perCallTemporaries
          (position.val / perCallTemporaries)]

private theorem temporaryColumn_div {calls : Nat} (base : Nat)
    (position : TemporaryPosition calls) :
    (temporaryColumn base calls position - base) / SymbolicDuplex.stride =
      temporaryCall position := by
  unfold temporaryColumn
  have removeBase :
      base + temporaryCall position * SymbolicDuplex.stride +
          temporaryOffset position - base =
        temporaryCall position * SymbolicDuplex.stride +
          temporaryOffset position := by
    omega
  rw [removeBase]
  rw [Nat.mul_comm (temporaryCall position) SymbolicDuplex.stride,
    Nat.mul_add_div (by
      rw [SymbolicDuplex.stride_eq]
      decide),
    Nat.div_eq_of_lt (temporaryOffset_lt position), Nat.add_zero]

private theorem temporaryColumn_mod {calls : Nat} (base : Nat)
    (position : TemporaryPosition calls) :
    (temporaryColumn base calls position - base) % SymbolicDuplex.stride =
      temporaryOffset position := by
  unfold temporaryColumn
  have removeBase :
      base + temporaryCall position * SymbolicDuplex.stride +
          temporaryOffset position - base =
        temporaryCall position * SymbolicDuplex.stride +
          temporaryOffset position := by
    omega
  rw [removeBase]
  exact Nat.mul_add_mod_of_lt (temporaryOffset_lt position)

theorem temporaryColumn_injective (base calls : Nat) :
    Function.Injective (temporaryColumn base calls) := by
  intro first second equal
  have callsEqual : temporaryCall first = temporaryCall second := by
    have shifted :=
      congrArg
        (fun column => (column - base) / SymbolicDuplex.stride) equal
    simpa only [temporaryColumn_div] using shifted
  have offsetsEqual :
      temporaryOffset first = temporaryOffset second := by
    have shifted :=
      congrArg
        (fun column => (column - base) % SymbolicDuplex.stride) equal
    simpa only [temporaryColumn_mod] using shifted
  have firstWithin := temporaryWithinCall_lt first
  have secondWithin := temporaryWithinCall_lt second
  have withinEqual :
      temporaryWithinCall first = temporaryWithinCall second := by
    simp only [perCallTemporaries] at firstWithin secondWithin
    unfold temporaryOffset at offsetsEqual
    split at offsetsEqual <;> split at offsetsEqual
    all_goals
      simp only [sboxTemporaries, width] at *
      omega
  apply Fin.ext
  calc
    first.val =
        temporaryCall first * perCallTemporaries +
          temporaryWithinCall first :=
      position_eq_div_mul_add_mod first
    _ =
        temporaryCall second * perCallTemporaries +
          temporaryWithinCall second := by
      rw [callsEqual, withinEqual]
    _ = second.val :=
      (position_eq_div_mul_add_mod second).symm

theorem temporaryColumns_nodup (base calls : Nat) :
    (temporaryColumns base calls).Nodup :=
  nodup_ofFn_of_injective _ (temporaryColumn_injective base calls)

theorem temporaryColumns_length (base calls : Nat) :
    (temporaryColumns base calls).length =
      calls * perCallTemporaries := by
  simp only [temporaryColumns, List.length_ofFn]

theorem temporaryColumns_length_eq_cost
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    (temporaryColumns base builder.entries.length).length =
      (SymbolicDuplex.cost builder).auxiliaryColumns := by
  rw [temporaryColumns_length]
  rfl

theorem temporaryColumn_lt_end {calls : Nat}
    (base : Nat) (position : TemporaryPosition calls) :
    temporaryColumn base calls position <
      base + calls * SymbolicDuplex.stride := by
  have callLt := temporaryCall_lt position
  have offsetLt := temporaryOffset_lt position
  unfold temporaryColumn
  rw [SymbolicDuplex.stride_eq] at offsetLt ⊢
  omega

theorem temporaryColumns_lt_end
    (base calls column : Nat)
    (member : column ∈ temporaryColumns base calls) :
    column < base + calls * SymbolicDuplex.stride := by
  rw [temporaryColumns, List.mem_ofFn] at member
  rcases member with ⟨position, rfl⟩
  exact temporaryColumn_lt_end base position

theorem sboxColumn_eq_temporary
    (base calls call : Nat) (callLt : call < calls)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    sboxColumn (SymbolicDuplex.layoutAt base call) index slot =
      temporaryColumn base calls
        ⟨call * perCallTemporaries +
            columnsPerSbox * index.val + slot.val,
          by
            have indexLt := index.isLt
            have slotLt := slot.isLt
            simp only [perCallTemporaries, columnsPerSbox, sboxCount,
              externalRounds, width, partialRounds] at *
            omega⟩ := by
  have indexLt := index.isLt
  have slotLt := slot.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  simp only [columnsPerSbox] at slotLt
  simp only [temporaryColumn, temporaryCall, temporaryWithinCall,
    temporaryOffset, perCallTemporaries, sboxTemporaries,
    SymbolicDuplex.layoutAt, sboxColumn,
    columnsPerSbox]
  rw [SymbolicDuplex.stride_eq]
  have withinLt : 4 * index.val + slot.val < 352 := by omega
  have divEq :
      (call * 352 + (4 * index.val + slot.val)) / 352 = call := by
    rw [Nat.mul_comm call 352, Nat.mul_add_div (by decide : 0 < 352),
      Nat.div_eq_of_lt withinLt, Nat.add_zero]
  have modEq :
      (call * 352 + (4 * index.val + slot.val)) % 352 =
        4 * index.val + slot.val :=
    Nat.mul_add_mod_of_lt withinLt
  rw [show call * 352 + 4 * index.val + slot.val =
      call * 352 + (4 * index.val + slot.val) by omega,
    divEq, modEq, if_pos (by omega)]
  omega

theorem outputColumn_eq_temporary
    (base calls call : Nat) (callLt : call < calls)
    (lane : Fin width) :
    (SymbolicDuplex.layoutAt base call).outputPort lane =
      temporaryColumn base calls
        ⟨call * perCallTemporaries + sboxTemporaries + lane.val,
          by
            have laneLt := lane.isLt
            simp only [perCallTemporaries, sboxTemporaries, width] at *
            omega⟩ := by
  have laneLt := lane.isLt
  simp only [width] at laneLt
  simp only [temporaryColumn, temporaryCall, temporaryWithinCall,
    temporaryOffset, perCallTemporaries, sboxTemporaries,
    SymbolicDuplex.layoutAt]
  rw [SymbolicDuplex.stride_eq]
  have withinLt : 344 + lane.val < 352 := by omega
  have divEq :
      (call * 352 + (344 + lane.val)) / 352 = call := by
    rw [Nat.mul_comm call 352, Nat.mul_add_div (by decide : 0 < 352),
      Nat.div_eq_of_lt withinLt, Nat.add_zero]
  have modEq :
      (call * 352 + (344 + lane.val)) % 352 =
        344 + lane.val :=
    Nat.mul_add_mod_of_lt withinLt
  rw [show call * 352 + 344 + lane.val =
      call * 352 + (344 + lane.val) by omega,
    divEq, modEq, if_neg (by omega)]
  omega

theorem sboxColumn_mem
    (base calls call : Nat) (callLt : call < calls)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    sboxColumn (SymbolicDuplex.layoutAt base call) index slot ∈
      temporaryColumns base calls := by
  rw [temporaryColumns, List.mem_ofFn]
  exact
    ⟨⟨call * perCallTemporaries +
          columnsPerSbox * index.val + slot.val,
        by
          have indexLt := index.isLt
          have slotLt := slot.isLt
          simp only [perCallTemporaries, columnsPerSbox, sboxCount,
            externalRounds, width, partialRounds] at *
          omega⟩,
      (sboxColumn_eq_temporary base calls call callLt index slot).symm⟩

theorem outputColumn_mem
    (base calls call : Nat) (callLt : call < calls)
    (lane : Fin width) :
    (SymbolicDuplex.layoutAt base call).outputPort lane ∈
      temporaryColumns base calls := by
  rw [temporaryColumns, List.mem_ofFn]
  exact
    ⟨⟨call * perCallTemporaries + sboxTemporaries + lane.val,
        by
          have laneLt := lane.isLt
          simp only [perCallTemporaries, sboxTemporaries, width] at *
          omega⟩,
      (outputColumn_eq_temporary base calls call callLt lane).symm⟩

/-- Compact duplex temporaries occupy the complete contiguous call span.
Their list order is S-box-first/output-last, but membership is exactly the
numeric interval. -/
theorem temporaryColumns_mem_iff
    (base calls column : Nat) :
    column ∈ temporaryColumns base calls ↔
      base ≤ column ∧
        column < base + calls * SymbolicDuplex.stride := by
  constructor
  · intro member
    constructor
    · rw [temporaryColumns, List.mem_ofFn] at member
      rcases member with ⟨position, rfl⟩
      unfold temporaryColumn
      omega
    · exact temporaryColumns_lt_end base calls column member
  · intro window
    let shifted := column - base
    let call := shifted / SymbolicDuplex.stride
    let offset := shifted % SymbolicDuplex.stride
    have shiftedEq : column = base + shifted := by
      simp only [shifted]
      omega
    have offsetLt : offset < SymbolicDuplex.stride :=
      Nat.mod_lt _ (by rw [SymbolicDuplex.stride_eq]; decide)
    have shiftedSplit :
        shifted = call * SymbolicDuplex.stride + offset := by
      have split := Nat.div_add_mod shifted SymbolicDuplex.stride
      simp only [call, offset] at split ⊢
      rw [Nat.mul_comm] at split
      exact split.symm
    have callLt : call < calls := by
      rw [SymbolicDuplex.stride_eq] at window shiftedSplit
      omega
    by_cases isOutput : offset < width
    · let lane : Fin width := ⟨offset, isOutput⟩
      have columnEq :
          column =
            (SymbolicDuplex.layoutAt base call).outputPort lane := by
        simp only [SymbolicDuplex.layoutAt, lane]
        omega
      rw [columnEq]
      exact outputColumn_mem base calls call callLt lane
    · let within := offset - width
      have withinLt : within < sboxCount * columnsPerSbox := by
        rw [SymbolicDuplex.stride_eq] at offsetLt
        simp only [within, width, sboxCount, externalRounds, partialRounds,
          columnsPerSbox] at isOutput ⊢
        omega
      let index : Fin sboxCount :=
        ⟨within / columnsPerSbox, by
          have remLt :
              within % columnsPerSbox < columnsPerSbox :=
            Nat.mod_lt _ (by simp [columnsPerSbox])
          have split := Nat.div_add_mod within columnsPerSbox
          simp only [sboxCount, externalRounds, width, partialRounds,
            columnsPerSbox] at withinLt remLt split ⊢
          omega⟩
      let slot : Fin columnsPerSbox :=
        ⟨within % columnsPerSbox,
          Nat.mod_lt _ (by simp [columnsPerSbox])⟩
      have withinSplit :
          within = columnsPerSbox * index.val + slot.val := by
        have split := Nat.div_add_mod within columnsPerSbox
        simp only [index, slot] at split ⊢
        exact split.symm
      have offsetEq : offset = width + within := by
        simp only [within]
        omega
      have columnEq :
          column =
            sboxColumn (SymbolicDuplex.layoutAt base call) index slot := by
        calc
          column = base + shifted := shiftedEq
          _ = base + (call * SymbolicDuplex.stride + offset) := by
            rw [shiftedSplit]
          _ = base + call * SymbolicDuplex.stride + width +
              columnsPerSbox * index.val + slot.val := by
            rw [offsetEq, withinSplit]
            omega
          _ =
              sboxColumn (SymbolicDuplex.layoutAt base call) index slot :=
            rfl
      rw [columnEq]
      exact sboxColumn_mem base calls call callLt index slot

/-! ## Every declared temporary is emitted -/

/-- Every compact duplex temporary is either an S-box chain column or one of
the eight bound output columns of exactly one call. -/
theorem temporaryColumns_classify
    (base calls column : Nat)
    (member : column ∈ temporaryColumns base calls) :
    (∃ call, call < calls ∧
      ∃ index : Fin sboxCount, ∃ slot : Fin columnsPerSbox,
        column =
          sboxColumn (SymbolicDuplex.layoutAt base call) index slot) ∨
    (∃ call, call < calls ∧ ∃ lane : Fin width,
      column = (SymbolicDuplex.layoutAt base call).outputPort lane) := by
  rw [temporaryColumns, List.mem_ofFn] at member
  rcases member with ⟨position, rfl⟩
  have callLt := temporaryCall_lt position
  have withinLt := temporaryWithinCall_lt position
  by_cases isSbox : temporaryWithinCall position < sboxTemporaries
  · let index : Fin sboxCount :=
      ⟨temporaryWithinCall position / columnsPerSbox, by
        simp only [sboxTemporaries, sboxCount, externalRounds, width,
          partialRounds, columnsPerSbox] at isSbox ⊢
        omega⟩
    let slot : Fin columnsPerSbox :=
      ⟨temporaryWithinCall position % columnsPerSbox,
        Nat.mod_lt _ (by simp [columnsPerSbox])⟩
    refine Or.inl ⟨temporaryCall position, callLt, index, slot, ?_⟩
    rw [sboxColumn_eq_temporary base calls (temporaryCall position)
      callLt index slot]
    congr 1
    apply Fin.ext
    have splitWithin :
        temporaryWithinCall position =
          temporaryWithinCall position / columnsPerSbox *
              columnsPerSbox +
            temporaryWithinCall position % columnsPerSbox := by
      rw [Nat.mul_comm]
      exact (Nat.div_add_mod (temporaryWithinCall position)
        columnsPerSbox).symm
    rw [position_eq_div_mul_add_mod position]
    simp only [index, slot, columnsPerSbox]
    omega
  · let lane : Fin width :=
      ⟨temporaryWithinCall position - sboxTemporaries, by
        simp only [perCallTemporaries, sboxTemporaries, width] at withinLt isSbox ⊢
        omega⟩
    refine Or.inr ⟨temporaryCall position, callLt, lane, ?_⟩
    rw [outputColumn_eq_temporary base calls (temporaryCall position)
      callLt lane]
    congr 1
    apply Fin.ext
    rw [position_eq_div_mul_add_mod position]
    simp only [lane, perCallTemporaries, sboxTemporaries] at isSbox ⊢
    omega

private theorem entryRows_mem_rowsFrom
    (base : Nat) (constants : Constants) :
    ∀ (entries : List SymbolicDuplex.Entry)
      (entry : SymbolicDuplex.Entry),
      entry ∈ entries →
      ∀ row, row ∈ SymbolicDuplex.entryRows base constants entry →
        row ∈ SymbolicDuplex.rowsFrom base constants entries
  | [], _, member, _, _ => by cases member
  | head :: rest, entry, member, row, rowMember => by
      rcases List.mem_cons.1 member with rfl | inRest
      · exact List.mem_append_left _ rowMember
      · exact List.mem_append_right _
          (entryRows_mem_rowsFrom base constants rest entry inRest
            row rowMember)

/-- Call-order alone is enough to show that every declared compact duplex
temporary occurs in an emitted normalized row.  Source placement is not
needed for this converse-to-conservation direction. -/
theorem temporaryColumns_written_of_calls
    (base : Nat) (constants : Constants)
    (builder : SymbolicDuplex.Builder)
    (calls :
      builder.entries.map SymbolicDuplex.Entry.call =
        List.range builder.entries.length)
    (column : Nat)
    (member : column ∈ temporaryColumns base builder.entries.length) :
    ∃ row ∈ SymbolicDuplex.rows base constants builder,
      Mentions row.c column := by
  have findEntry :
      ∀ call, call < builder.entries.length →
        ∃ entry ∈ builder.entries, entry.call = call := by
    intro call callLt
    have callMember : call ∈ List.range builder.entries.length :=
      List.mem_range.2 callLt
    rw [← calls] at callMember
    rcases List.mem_map.1 callMember with ⟨entry, entryMember, equal⟩
    exact ⟨entry, entryMember, equal⟩
  rcases temporaryColumns_classify base builder.entries.length column member with
    sbox | output
  · rcases sbox with ⟨call, callLt, index, slot, columnEq⟩
    rcases findEntry call callLt with ⟨entry, entryMember, entryCall⟩
    have declared :
        sboxColumn (SymbolicDuplex.layoutAt base entry.call) index slot ∈
          Poseidon2Program.auxiliaryColumns
            (SymbolicDuplex.layoutAt base entry.call) :=
      List.mem_flatMap.2 ⟨index, List.mem_finRange _,
        List.mem_map.2 ⟨slot, List.mem_finRange _, rfl⟩⟩
    rcases Poseidon2Program.permutationProgram_writes_auxiliaryColumns
        (SymbolicDuplex.layoutAt base entry.call)
        (scheduleOfFrom (SymbolicDuplex.layoutAt base entry.call)
          entry.state constants)
        (finalState (SymbolicDuplex.layoutAt base entry.call))
        _ declared with
      ⟨raw, rawMember, write⟩
    let normalized := normalizeRow raw
    have normalizedMember :
        normalized ∈ SymbolicDuplex.entryRows base constants entry := by
      unfold normalized SymbolicDuplex.entryRows
        normalizedCanonicalProgramFrom normalizeProgram canonicalProgramFrom
      exact List.mem_map.2 ⟨raw, rawMember, rfl⟩
    refine ⟨normalized, ?_, ?_⟩
    · unfold SymbolicDuplex.rows
      exact entryRows_mem_rowsFrom base constants builder.entries entry
        entryMember normalized normalizedMember
    · rw [columnEq, ← entryCall]
      exact mentions_normalizeRow_singleton raw _ write
  · rcases output with ⟨call, callLt, lane, columnEq⟩
    rcases findEntry call callLt with ⟨entry, entryMember, entryCall⟩
    let raw :=
      bindRow
        (finalState (SymbolicDuplex.layoutAt base entry.call) lane)
        ((SymbolicDuplex.layoutAt base entry.call).outputPort lane)
    let normalized := normalizeRow raw
    have rawMember :
        raw ∈ canonicalProgramFrom
          (SymbolicDuplex.layoutAt base entry.call) entry.state constants := by
      unfold raw canonicalProgramFrom Poseidon2Program.permutationProgram
        Poseidon2Program.bindingProgram terminalBindingRows
      exact List.mem_append_right _
        (List.mem_map.2 ⟨lane, List.mem_finRange _, rfl⟩)
    have normalizedMember :
        normalized ∈ SymbolicDuplex.entryRows base constants entry := by
      unfold normalized SymbolicDuplex.entryRows
        normalizedCanonicalProgramFrom normalizeProgram
      exact List.mem_map.2 ⟨raw, rawMember, rfl⟩
    refine ⟨normalized, ?_, ?_⟩
    · unfold SymbolicDuplex.rows
      exact entryRows_mem_rowsFrom base constants builder.entries entry
        entryMember normalized normalizedMember
    · rw [columnEq, ← entryCall]
      apply mentions_normalizeRow_singleton raw
      rfl

/-- Every declared compact duplex temporary occurs in an emitted normalized
row.  S-box columns survive normalization because their defining writes have
coefficient one; output columns survive for the same reason in the terminal
binding rows. -/
theorem temporaryColumns_written
    (base : Nat) (constants : Constants)
    (builder : SymbolicDuplex.Builder)
    (placed : WellPlaced base builder)
    (column : Nat)
    (member : column ∈ temporaryColumns base builder.entries.length) :
    ∃ row ∈ SymbolicDuplex.rows base constants builder,
      Mentions row.c column :=
  temporaryColumns_written_of_calls base constants builder placed.calls
    column member

/-! ## Structured positional row receipts -/

inductive RowOwner : List SymbolicDuplex.Entry → Type
  | head {entry rest} (receipt : Poseidon2Ownership.RowOwner) :
      RowOwner (entry :: rest)
  | tail {entry rest} (owner : RowOwner rest) :
      RowOwner (entry :: rest)

def owners : (entries : List SymbolicDuplex.Entry) → List (RowOwner entries)
  | [] => []
  | _ :: rest =>
      Poseidon2Ownership.allOwners.map RowOwner.head ++
        (owners rest).map RowOwner.tail

def ownedRow (base : Nat) (constants : Constants) :
    {entries : List SymbolicDuplex.Entry} → RowOwner entries → Row
  | entry :: _, .head receipt =>
      normalizeRow
        (Poseidon2Ownership.ownedRowFrom
          (SymbolicDuplex.layoutAt base entry.call) entry.state constants receipt)
  | _ :: _, .tail owner => ownedRow base constants owner

theorem entryRows_eq_map_owners
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) :
    SymbolicDuplex.entryRows base constants entry =
      Poseidon2Ownership.allOwners.map
        (fun receipt =>
          normalizeRow
            (Poseidon2Ownership.ownedRowFrom
              (SymbolicDuplex.layoutAt base entry.call)
              entry.state constants receipt)) := by
  unfold SymbolicDuplex.entryRows normalizedCanonicalProgramFrom
    normalizeProgram
  rw [Poseidon2Ownership.canonicalProgramFrom_eq_map_owners, List.map_map]
  rfl

theorem rowsFrom_eq_map_owners
    (base : Nat) (constants : Constants) :
    ∀ entries : List SymbolicDuplex.Entry,
      SymbolicDuplex.rowsFrom base constants entries =
        (owners entries).map (ownedRow base constants)
  | [] => rfl
  | entry :: rest => by
      rw [SymbolicDuplex.rowsFrom, owners, List.map_append, List.map_map,
        entryRows_eq_map_owners, rowsFrom_eq_map_owners base constants rest]
      rw [List.map_map]
      rfl

theorem owners_nodup :
    ∀ entries : List SymbolicDuplex.Entry, (owners entries).Nodup
  | [] => by simp [owners]
  | entry :: rest => by
      rw [owners, List.nodup_append]
      refine
        ⟨Poseidon2Ownership.nodup_map_of_injective RowOwner.head
            (fun first second equal => by cases equal; rfl)
            Poseidon2Ownership.allOwners_nodup,
          Poseidon2Ownership.nodup_map_of_injective RowOwner.tail
            (fun first second equal => by cases equal; rfl)
            (owners_nodup rest),
          ?_⟩
      intro left leftMember right rightMember equal
      rcases List.mem_map.1 leftMember with ⟨receipt, _, rfl⟩
      rcases List.mem_map.1 rightMember with ⟨owner, _, rfl⟩
      cases equal

theorem ownership_is_positional
    (base : Nat) (constants : Constants)
    (builder : SymbolicDuplex.Builder) :
    (SymbolicDuplex.rows base constants builder).length =
        (owners builder.entries).length
      ∧ (owners builder.entries).Nodup
      ∧ SymbolicDuplex.rows base constants builder =
          (owners builder.entries).map (ownedRow base constants) := by
  have emitted :=
    rowsFrom_eq_map_owners base constants builder.entries
  refine ⟨?_, owners_nodup builder.entries, ?_⟩
  · unfold SymbolicDuplex.rows
    rw [emitted, List.length_map]
  · exact emitted

/-! ## Whole-program conservation -/

def LocalColumn (base : Nat) (entry : SymbolicDuplex.Entry)
    (column : Nat) : Prop :=
  column = 0
    ∨ (∃ lane : Fin width, Mentions (entry.state lane) column)
    ∨ (∃ lane : Fin width,
        column = (SymbolicDuplex.layoutAt base entry.call).outputPort lane)
    ∨ (∃ index : Fin sboxCount, ∃ slot : Fin columnsPerSbox,
        column =
          sboxColumn (SymbolicDuplex.layoutAt base entry.call) index slot)

private theorem singleton_local_sbox
    (base : Nat) (entry : SymbolicDuplex.Entry)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox)
    (column : Nat)
    (mentioned :
      Mentions
        [(sboxColumn
          (SymbolicDuplex.layoutAt base entry.call) index slot, 1)]
        column) :
    LocalColumn base entry column := by
  simp only [Mentions, List.map_cons, List.map_nil,
    List.mem_singleton] at mentioned
  exact Or.inr (Or.inr (Or.inr ⟨index, slot, mentioned⟩))

private theorem rawOwnedRow_conservation
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry)
    (owner : Poseidon2Ownership.RowOwner)
    (column : Nat)
    (mentioned :
      Mentions
          (Poseidon2Ownership.ownedRowFrom
            (SymbolicDuplex.layoutAt base entry.call)
            entry.state constants owner).a column
        ∨ Mentions
          (Poseidon2Ownership.ownedRowFrom
            (SymbolicDuplex.layoutAt base entry.call)
            entry.state constants owner).b column
        ∨ Mentions
          (Poseidon2Ownership.ownedRowFrom
            (SymbolicDuplex.layoutAt base entry.call)
            entry.state constants owner).c column) :
    LocalColumn base entry column := by
  cases owner with
  | sbox index step =>
      have scheduled :=
        Poseidon2Conservation.scheduleOfFrom_columns
          (SymbolicDuplex.layoutAt base entry.call)
          entry.state constants index column
      have viaSchedule :
          Mentions
              (scheduleOfFrom
                (SymbolicDuplex.layoutAt base entry.call)
                entry.state constants index)
              column →
            LocalColumn base entry column := by
        intro inInput
        rcases scheduled inInput with wire | source | output
        · exact Or.inl wire
        · exact Or.inr (Or.inl source)
        · rcases output with ⟨other, otherLt, image⟩
          exact Or.inr (Or.inr (Or.inr
            ⟨⟨other, by
                simpa [sboxCount, externalRounds, width, partialRounds]
                  using otherLt⟩,
              ⟨3, by decide⟩,
              image⟩))
      match step with
      | ⟨0, _⟩ =>
          simp only [Poseidon2Ownership.ownedRowFrom,
            Poseidon2Ownership.sboxRowAt, rowSquare, frameAt] at mentioned
          rcases mentioned with input | input | target
          · exact viaSchedule input
          · exact viaSchedule input
          · exact singleton_local_sbox base entry index ⟨0, by decide⟩
              column target
      | ⟨1, _⟩ =>
          simp only [Poseidon2Ownership.ownedRowFrom,
            Poseidon2Ownership.sboxRowAt, rowFourth, frameAt] at mentioned
          rcases mentioned with target | target | target
          · exact singleton_local_sbox base entry index ⟨0, by decide⟩
              column target
          · exact singleton_local_sbox base entry index ⟨0, by decide⟩
              column target
          · exact singleton_local_sbox base entry index ⟨1, by decide⟩
              column target
      | ⟨2, _⟩ =>
          simp only [Poseidon2Ownership.ownedRowFrom,
            Poseidon2Ownership.sboxRowAt, rowSixth, frameAt] at mentioned
          rcases mentioned with target | target | target
          · exact singleton_local_sbox base entry index ⟨0, by decide⟩
              column target
          · exact singleton_local_sbox base entry index ⟨1, by decide⟩
              column target
          · exact singleton_local_sbox base entry index ⟨2, by decide⟩
              column target
      | ⟨3, _⟩ =>
          simp only [Poseidon2Ownership.ownedRowFrom,
            Poseidon2Ownership.sboxRowAt, rowSeventh, frameAt] at mentioned
          rcases mentioned with input | target | target
          · exact viaSchedule input
          · exact singleton_local_sbox base entry index ⟨2, by decide⟩
              column target
          · exact singleton_local_sbox base entry index ⟨3, by decide⟩
              column target
  | binding lane =>
      simp only [Poseidon2Ownership.ownedRowFrom, bindRow] at mentioned
      rcases mentioned with final | wire | port
      · rcases Poseidon2Conservation.terminalState_columns
          (SymbolicDuplex.layoutAt base entry.call)
          halfFullRounds (Nat.le_refl _) lane column final with
          ⟨index, bound, image⟩
        exact Or.inr (Or.inr (Or.inr
          ⟨⟨index, by
              simpa [sboxCount, externalRounds, width, partialRounds]
                using bound⟩,
            ⟨3, by decide⟩,
            image⟩))
      · simp only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] at wire
        exact Or.inl wire
      · simp only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] at port
        exact Or.inr (Or.inr (Or.inl ⟨lane, port⟩))

private theorem ownedRow_conservation
    (base : Nat) (constants : Constants) :
    ∀ {entries : List SymbolicDuplex.Entry}
      (owner : RowOwner entries) (column : Nat),
      (Mentions (ownedRow base constants owner).a column
        ∨ Mentions (ownedRow base constants owner).b column
        ∨ Mentions (ownedRow base constants owner).c column) →
      ∃ entry ∈ entries, LocalColumn base entry column
  | entry :: rest, .head receipt, column, mentioned => by
      refine ⟨entry, List.mem_cons_self, ?_⟩
      apply rawOwnedRow_conservation base constants entry receipt column
      rcases mentioned with inA | inB | inC
      · exact Or.inl ((mentions_normalizeRow _ column).1 inA)
      · exact Or.inr (Or.inl ((mentions_normalizeRow _ column).2.1 inB))
      · exact Or.inr (Or.inr ((mentions_normalizeRow _ column).2.2 inC))
  | _ :: _, .tail owner, column, mentioned => by
      rcases ownedRow_conservation base constants owner column mentioned with
        ⟨entry, member, classification⟩
      exact ⟨entry, by simp [member], classification⟩

theorem rows_conservation
    (base : Nat) (constants : Constants)
    (builder : SymbolicDuplex.Builder)
    (positive : 0 < base)
    (placed : WellPlaced base builder)
    (owned : WellOwned base builder)
    (row : Row)
    (member : row ∈ SymbolicDuplex.rows base constants builder)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < base ∨
      column ∈ temporaryColumns base builder.entries.length := by
  rw [ownership_is_positional base constants builder |>.2.2] at member
  rcases List.mem_map.1 member with ⟨owner, _, rfl⟩
  rcases ownedRow_conservation base constants owner column mentioned with
    ⟨entry, entryMember, classification⟩
  have callMember :
      entry.call ∈ List.range builder.entries.length := by
    rw [← placed.calls]
    exact List.mem_map.2 ⟨entry, entryMember, rfl⟩
  have callLt : entry.call < builder.entries.length :=
    List.mem_range.mp callMember
  rcases classification with wire | source | output | sbox
  · exact Or.inl (by omega)
  · rcases source with ⟨lane, inState⟩
    rcases owned.entrySources entry entryMember lane column inState with
      inPrefix | prior
    · exact Or.inl inPrefix
    · rcases prior with ⟨previous, priorLane, previousLt, rfl⟩
      exact Or.inr
        (outputColumn_mem base builder.entries.length previous
          (Nat.lt_trans previousLt callLt) priorLane)
  · rcases output with ⟨lane, rfl⟩
    exact Or.inr
      (outputColumn_mem base builder.entries.length entry.call callLt lane)
  · rcases sbox with ⟨index, slot, rfl⟩
    exact Or.inr
      (sboxColumn_mem base builder.entries.length entry.call callLt index slot)

end Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical
