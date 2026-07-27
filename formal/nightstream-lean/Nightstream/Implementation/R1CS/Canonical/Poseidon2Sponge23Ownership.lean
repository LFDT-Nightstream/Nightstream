import Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest

/-!
Contract: exact row/column ownership and conservation for the fixed-23
canonical sponge core.

Owns: positional row receipts, the 2,464-column internal allocation, its
injectivity and separation from the 23 visible inputs, and the theorem that
every operand of every emitted row belongs to one of those declared sources.

Does not own: typed call output ports or activation rows.
-/

set_option autoImplicit false
set_option maxRecDepth 4096

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23

def rows : Nat := 2464
def temporaries : Nat := 2464
def perCallTemporaries : Nat := 352
def sboxTemporaries : Nat := 344

abbrev RowPosition := Fin rows
abbrev TemporaryPosition := Fin temporaries

private theorem nodup_ofFn_of_injective
    {α : Type} :
    ∀ {n : Nat} (function : Fin n → α),
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

/-! ## Positional row receipts -/

def emptyRow : Row := ⟨[], [], []⟩

def rowAt (constants : Constants) (position : RowPosition) : Row :=
  (program constants).getD position.val emptyRow

def rowOwners : List RowPosition := List.ofFn id

theorem rowOwners_length : rowOwners.length = rows := by
  simp only [rowOwners, List.length_ofFn]

theorem rowOwners_nodup : rowOwners.Nodup :=
  nodup_ofFn_of_injective id (fun _ _ equal => equal)

/-- Every emitted row is exactly the row at its unique receipt position. -/
theorem program_eq_positional_receipts (constants : Constants) :
    program constants = List.ofFn (rowAt constants) := by
  symm
  unfold rowAt
  apply List.ext_get
  · rw [List.length_ofFn, program_length]
    rfl
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem rightLt]
    rfl

/-! ## Internal column allocator -/

def temporaryCall (position : TemporaryPosition) : Nat :=
  position.val / perCallTemporaries

def temporaryWithinCall (position : TemporaryPosition) : Nat :=
  position.val % perCallTemporaries

def temporaryOffset (position : TemporaryPosition) : Nat :=
  if temporaryWithinCall position < sboxTemporaries
  then 17 + temporaryWithinCall position
  else 9 + (temporaryWithinCall position - sboxTemporaries)

/-- First 344 positions of each call own S-box columns in their emitted order;
the last eight own that call's bound output ports. -/
def temporaryColumn (position : TemporaryPosition) : Nat :=
  temporaryCall position * callStride + temporaryOffset position

def temporaryColumns : List Nat := List.ofFn temporaryColumn
def inputColumns : List Nat :=
  List.ofFn (fun index : Fin sponge23Fields => inputColumn index.val)

theorem temporaryCall_lt (position : TemporaryPosition) :
    temporaryCall position < calls := by
  have positionLt := position.isLt
  simp only [temporaryCall, temporaries, rows, perCallTemporaries, calls] at *
  omega

theorem temporaryWithinCall_lt (position : TemporaryPosition) :
    temporaryWithinCall position < perCallTemporaries := by
  exact Nat.mod_lt _ (by decide)

theorem temporaryOffset_lt (position : TemporaryPosition) :
    temporaryOffset position < callStride := by
  have withinLt := temporaryWithinCall_lt position
  unfold temporaryOffset
  split
  · rw [callStride_eq]
    simp only [perCallTemporaries, sboxTemporaries] at *
    omega
  · rw [callStride_eq]
    simp only [perCallTemporaries, sboxTemporaries] at *
    omega

theorem temporaryColumn_lt_inputBase (position : TemporaryPosition) :
    temporaryColumn position < inputBase := by
  unfold temporaryColumn
  have callLt := temporaryCall_lt position
  have offsetLt := temporaryOffset_lt position
  rw [callStride_eq] at offsetLt ⊢
  rw [inputBase_eq]
  simp only [calls] at callLt
  omega

theorem inputColumn_ge_inputBase (index : Nat) :
    inputBase ≤ inputColumn index := by
  unfold inputColumn
  omega

private theorem position_eq_div_mul_add_mod (position : TemporaryPosition) :
    position.val =
      temporaryCall position * perCallTemporaries
        + temporaryWithinCall position := by
  unfold temporaryCall temporaryWithinCall
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod position.val perCallTemporaries).symm

private theorem block_position_div
    (call within : Nat) (withinLt : within < perCallTemporaries) :
    (call * perCallTemporaries + within) / perCallTemporaries = call := by
  rw [Nat.mul_comm call perCallTemporaries,
    Nat.mul_add_div (by decide : 0 < perCallTemporaries),
    Nat.div_eq_of_lt withinLt, Nat.add_zero]

private theorem block_position_mod
    (call within : Nat) (withinLt : within < perCallTemporaries) :
    (call * perCallTemporaries + within) % perCallTemporaries = within :=
  Nat.mul_add_mod_of_lt withinLt

private theorem temporaryColumn_div (position : TemporaryPosition) :
    temporaryColumn position / callStride = temporaryCall position := by
  unfold temporaryColumn
  rw [Nat.mul_comm (temporaryCall position) callStride,
    Nat.mul_add_div (by decide : 0 < callStride),
    Nat.div_eq_of_lt (temporaryOffset_lt position),
    Nat.add_zero]

private theorem temporaryColumn_mod (position : TemporaryPosition) :
    temporaryColumn position % callStride = temporaryOffset position :=
  Nat.mul_add_mod_of_lt (temporaryOffset_lt position)

theorem temporaryColumn_injective : Function.Injective temporaryColumn := by
  intro first second equal
  have callsEqual : temporaryCall first = temporaryCall second := by
    have := congrArg (fun column => column / callStride) equal
    simpa only [temporaryColumn_div] using this
  have offsetsEqual : temporaryOffset first = temporaryOffset second := by
    have := congrArg (fun column => column % callStride) equal
    simpa only [temporaryColumn_mod] using this
  have firstWithin := temporaryWithinCall_lt first
  have secondWithin := temporaryWithinCall_lt second
  have withinEqual :
      temporaryWithinCall first = temporaryWithinCall second := by
    simp only [perCallTemporaries, sboxTemporaries] at firstWithin
    simp only [perCallTemporaries, sboxTemporaries] at secondWithin
    unfold temporaryOffset at offsetsEqual
    split at offsetsEqual <;> split at offsetsEqual
    all_goals
      simp only [sboxTemporaries] at *
      omega
  apply Fin.ext
  have firstEq := position_eq_div_mul_add_mod first
  have secondEq := position_eq_div_mul_add_mod second
  calc
    first.val =
        temporaryCall first * perCallTemporaries
          + temporaryWithinCall first := firstEq
    _ = temporaryCall second * perCallTemporaries
          + temporaryWithinCall second := by rw [callsEqual, withinEqual]
    _ = second.val := secondEq.symm

theorem temporaryColumns_nodup : temporaryColumns.Nodup :=
  nodup_ofFn_of_injective temporaryColumn temporaryColumn_injective

theorem temporaryColumns_length :
    temporaryColumns.length = temporaries := by
  simp [temporaryColumns]

theorem inputColumns_nodup : inputColumns.Nodup := by
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  change inputBase + first.val = inputBase + second.val at equal
  omega

theorem inputColumns_length : inputColumns.length = sponge23Fields := by
  simp [inputColumns]

theorem inputs_disjoint_temporaries :
    ∀ input, input ∈ inputColumns → input ∉ temporaryColumns := by
  intro input inputMember temporaryMember
  rcases List.mem_ofFn.mp inputMember with ⟨index, rfl⟩
  rcases List.mem_ofFn.mp temporaryMember with ⟨position, equal⟩
  have below := temporaryColumn_lt_inputBase position
  have above := inputColumn_ge_inputBase index.val
  omega

/-! ## Exact allocator correspondence -/

theorem sboxColumn_eq_temporary
    (call : Nat) (callLt : call < calls)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    sboxColumn (layout.call call) index slot =
      temporaryColumn
        ⟨call * perCallTemporaries
            + columnsPerSbox * index.val + slot.val,
          by
            have indexLt := index.isLt
            have slotLt := slot.isLt
            simp only [calls, perCallTemporaries, columnsPerSbox,
              temporaries, sboxCount, externalRounds, width, partialRounds]
              at callLt indexLt slotLt ⊢
            omega⟩ := by
  have indexLt := index.isLt
  have slotLt := slot.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  simp only [columnsPerSbox] at slotLt
  simp only [temporaryColumn, temporaryCall, temporaryWithinCall,
    temporaryOffset, perCallTemporaries, sboxTemporaries, columnsPerSbox]
  change (call * callStride + 17) + 4 * index.val + slot.val =
    (call * 352 + 4 * index.val + slot.val) / 352 * callStride +
      (if (call * 352 + 4 * index.val + slot.val) % 352 < 344
       then 17 + (call * 352 + 4 * index.val + slot.val) % 352
       else 9 + ((call * 352 + 4 * index.val + slot.val) % 352 - 344))
  have withinLt : 4 * index.val + slot.val < 352 := by omega
  have divEq :
      (call * 352 + (4 * index.val + slot.val)) / 352 = call := by
    rw [Nat.mul_comm call 352, Nat.mul_add_div (by decide : 0 < 352),
      Nat.div_eq_of_lt withinLt, Nat.add_zero]
  have modEq :
      (call * 352 + (4 * index.val + slot.val)) % 352
        = 4 * index.val + slot.val :=
    Nat.mul_add_mod_of_lt withinLt
  rw [show call * 352 + 4 * index.val + slot.val
      = call * 352 + (4 * index.val + slot.val) by omega,
    divEq, modEq, if_pos (by omega)]
  omega

theorem outputColumn_eq_temporary
    (call : Nat) (callLt : call < calls) (lane : Fin width) :
    (layout.call call).outputPort lane =
      temporaryColumn
        ⟨call * perCallTemporaries + sboxTemporaries + lane.val,
          by
            have laneLt := lane.isLt
            simp only [calls, perCallTemporaries, sboxTemporaries,
              temporaries, width] at callLt laneLt ⊢
            omega⟩ := by
  have laneLt := lane.isLt
  simp only [width] at laneLt
  simp only [temporaryColumn, temporaryCall, temporaryWithinCall,
    temporaryOffset, perCallTemporaries, sboxTemporaries]
  change call * callStride + 9 + lane.val =
    (call * 352 + 344 + lane.val) / 352 * callStride +
      (if (call * 352 + 344 + lane.val) % 352 < 344
       then 17 + (call * 352 + 344 + lane.val) % 352
       else 9 + ((call * 352 + 344 + lane.val) % 352 - 344))
  have withinLt : 344 + lane.val < 352 := by omega
  have divEq :
      (call * 352 + (344 + lane.val)) / 352 = call := by
    rw [Nat.mul_comm call 352, Nat.mul_add_div (by decide : 0 < 352),
      Nat.div_eq_of_lt withinLt, Nat.add_zero]
  have modEq :
      (call * 352 + (344 + lane.val)) % 352 = 344 + lane.val :=
    Nat.mul_add_mod_of_lt withinLt
  rw [show call * 352 + 344 + lane.val = call * 352 + (344 + lane.val)
      by omega,
    divEq, modEq, if_neg (by omega)]
  omega

/-! ## Whole-program conservation -/

def Allocated (column : Nat) : Prop :=
  column = 0 ∨ column ∈ inputColumns ∨ column ∈ temporaryColumns

theorem input_allocated (index : Fin sponge23Fields) :
    Allocated (inputColumn index.val) := by
  exact Or.inr (Or.inl (List.mem_ofFn.2 ⟨index, rfl⟩))

theorem temporary_allocated (position : TemporaryPosition) :
    Allocated (temporaryColumn position) := by
  exact Or.inr (Or.inr (List.mem_ofFn.2 ⟨position, rfl⟩))

theorem sbox_allocated
    (call : Nat) (callLt : call < calls)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    Allocated (sboxColumn (layout.call call) index slot) := by
  rw [sboxColumn_eq_temporary call callLt index slot]
  exact temporary_allocated _

theorem output_allocated
    (call : Nat) (callLt : call < calls) (lane : Fin width) :
    Allocated ((layout.call call).outputPort lane) := by
  rw [outputColumn_eq_temporary call callLt lane]
  exact temporary_allocated _

theorem rawSboxOutput_allocated
    (call : Nat) (callLt : call < calls)
    (index : Nat) (indexLt : index < sboxCount) :
    Allocated (sboxOutput (layout.call call) index) := by
  let bounded : Fin sboxCount := ⟨index, indexLt⟩
  rw [show sboxOutput (layout.call call) index
      = sboxColumn (layout.call call) bounded ⟨3, by decide⟩ by rfl]
  exact sbox_allocated call callLt bounded ⟨3, by decide⟩

theorem chunkColumn_allocated
    (call : Nat) (callLt : call < calls) (lane : Fin width)
    (covered : lane.val < chunkLength call) :
    Allocated (layout.chunkColumn call lane) := by
  by_cases data : call < dataCalls
  · have indexLt : call * rate + lane.val < sponge23Fields := by
      simp only [dataCalls, rate, sponge23Fields] at data ⊢
      unfold chunkLength at covered
      split at covered <;> omega
    rw [chunkColumn_data call lane data]
    exact input_allocated ⟨call * rate + lane.val, indexLt⟩
  · have callEq : call = 6 := by
      simp only [calls, dataCalls] at callLt data
      omega
    subst call
    rw [chunkColumn_padding]
    exact Or.inl rfl

theorem entry_allocated
    (call : Nat) (callLt : call < calls) (lane : Fin width) (column : Nat)
    (mentioned : Mentions (entryOf layout chunkLength call lane) column) :
    Allocated column := by
  unfold entryOf at mentioned
  cases call with
  | zero =>
      split at mentioned
      next covered =>
        simp only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] at mentioned
        rw [mentioned]
        exact chunkColumn_allocated 0 callLt lane covered
      next beyond => simp [Mentions] at mentioned
  | succ previous =>
      simp only [Mentions, List.map_cons, List.mem_cons] at mentioned
      rcases mentioned with previousPort | absorbed
      · rw [previousPort]
        exact output_allocated previous (by omega) lane
      · split at absorbed
        next covered =>
          simp only [List.map_cons, List.map_nil,
            List.mem_singleton] at absorbed
          rw [absorbed]
          exact chunkColumn_allocated (previous + 1) callLt lane covered
        next beyond => simp [Mentions] at absorbed

theorem schedule_allocated
    (constants : Constants)
    (call : Nat) (callLt : call < calls)
    (index : Fin sboxCount) (column : Nat)
    (mentioned :
      Mentions
        (scheduleOfFrom (layout.call call)
          (entryOf layout chunkLength call) constants index)
        column) :
    Allocated column := by
  rcases scheduleOfFrom_columns (layout.call call)
      (entryOf layout chunkLength call) constants index column mentioned with
    wire | fromEntry | fromSbox
  · exact Or.inl wire
  · rcases fromEntry with ⟨lane, entryMentioned⟩
    exact entry_allocated call callLt lane column entryMentioned
  · rcases fromSbox with ⟨other, otherLt, rfl⟩
    exact rawSboxOutput_allocated call callLt other otherLt

private theorem singleton_allocated
    (source column : Nat) (allocated : Allocated source)
    (mentioned : Mentions [(source, 1)] column) :
    Allocated column := by
  simp only [Mentions, List.map_cons, List.map_nil,
    List.mem_singleton] at mentioned
  rw [mentioned]
  exact allocated

theorem raw_row_allocated
    (constants : Constants)
    (call : Nat) (callLt : call < calls)
    (row : Row)
    (member :
      row ∈ canonicalProgramFrom (layout.call call)
        (entryOf layout chunkLength call) constants)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) :
    Allocated column := by
  rcases everyPermutationRow_has_owner
      (layout.call call)
      (scheduleOfFrom (layout.call call)
        (entryOf layout chunkLength call) constants)
      (finalState (layout.call call)) row member with
    fromSbox | fromBinding
  · rcases fromSbox with ⟨index, rowMember⟩
    simp only [sboxRows, List.mem_cons, List.not_mem_nil, or_false] at rowMember
    rcases rowMember with rfl | rfl | rfl | rfl
    · simp only [rowSquare, frameAt] at mentioned
      rcases mentioned with scheduled | scheduled | target
      · exact schedule_allocated constants call callLt index column scheduled
      · exact schedule_allocated constants call callLt index column scheduled
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨0, by decide⟩) target
    · simp only [rowFourth, frameAt] at mentioned
      rcases mentioned with square | square | fourth
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨0, by decide⟩) square
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨0, by decide⟩) square
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨1, by decide⟩) fourth
    · simp only [rowSixth, frameAt] at mentioned
      rcases mentioned with square | fourth | sixth
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨0, by decide⟩) square
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨1, by decide⟩) fourth
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨2, by decide⟩) sixth
    · simp only [rowSeventh, frameAt] at mentioned
      rcases mentioned with scheduled | sixth | output
      · exact schedule_allocated constants call callLt index column scheduled
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨2, by decide⟩) sixth
      · exact singleton_allocated _ _ (sbox_allocated call callLt index
          ⟨3, by decide⟩) output
  · rcases fromBinding with ⟨lane, rfl⟩
    simp only [bindRow] at mentioned
    rcases mentioned with final | wire | output
    · rcases terminalState_columns (layout.call call) halfFullRounds
          (Nat.le_refl _) lane column final with ⟨index, indexLt, rfl⟩
      exact rawSboxOutput_allocated call callLt index indexLt
    · exact singleton_allocated _ _ (Or.inl rfl) wire
    · exact singleton_allocated _ _ (output_allocated call callLt lane) output

/-- **No hidden row dependency.**  Every operand of every emitted normalized
row is owned exactly by the shared constant, one of the 23 visible inputs, or
one of the 2,464 declared temporary coordinates. -/
theorem program_conservation
    (constants : Constants) (row : Row) (member : row ∈ program constants)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) :
    Allocated column := by
  unfold program spongeProgram at member
  rcases List.mem_flatMap.1 member with ⟨call, callMember, callRowMember⟩
  have callLt : call < calls := List.mem_range.1 callMember
  rcases List.mem_map.1 callRowMember with ⟨raw, rawMember, rfl⟩
  apply raw_row_allocated constants call callLt raw rawMember column
  rcases mentioned with inA | inB | inC
  · exact Or.inl ((mentions_normalizeRow raw column).1 inA)
  · exact Or.inr (Or.inl ((mentions_normalizeRow raw column).2.1 inB))
  · exact Or.inr (Or.inr ((mentions_normalizeRow raw column).2.2 inC))

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership
