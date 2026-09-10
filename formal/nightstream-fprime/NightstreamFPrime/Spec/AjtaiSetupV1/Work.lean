import NightstreamFPrime.Spec.AjtaiSetupV1
import NightstreamFPrime.Spec.AjtaiSetupV1.WordOperations
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
Counted execution of the existing indexed ChaCha20 coefficient generator.
The clock counts arithmetic, reads, branches, constructors, and allocation.
Array writes reserve and copy the current array conservatively; uniqueness
is not assumed. Fixed-word interpretation uses the existing row32/block64
framing and canonical seed/word facts. Counts are not elapsed machine time.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.AjtaiSetupV1.Work

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

private def addWord (left right : Nat) : Result Nat :=
  ⟨(left + right) % ChaCha20.wordModulus, 3⟩

private def xorWord (left right : Nat) : Result Nat :=
  ⟨Nat.xor left right % ChaCha20.wordModulus, 3⟩

private def rotateWord (value amount : Nat) : Result Nat :=
  let shift := amount % 32
  if shift = 0 then ⟨value % ChaCha20.wordModulus, 5⟩
  else
    ⟨((Nat.shiftLeft value shift) % ChaCha20.wordModulus +
      Nat.shiftRight value (32 - shift)) % ChaCha20.wordModulus, 10⟩

private theorem addWord_value (left right : Nat) :
    (addWord left right).value = ChaCha20.add32 left right := rfl

private theorem xorWord_value (left right : Nat) :
    (xorWord left right).value = ChaCha20.xor32 left right := rfl

private theorem rotateWord_value (value amount : Nat) :
    (rotateWord value amount).value = ChaCha20.rotateLeft32 value amount := by
  by_cases zero : amount % 32 = 0 <;> simp [rotateWord, ChaCha20.rotateLeft32, zero]

private def readWord (state : Array Nat) (index : Nat) : Result Nat :=
  if live : index < state.size then ⟨state[index], 5⟩ else ⟨0, 4⟩

private theorem readWord_value (state : Array Nat) (index : Nat) :
    (readWord state index).value = ChaCha20.getWord state index := by
  by_cases live : index < state.size <;> simp [readWord, ChaCha20.getWord, Array.getD, live]

private theorem readWord_work_le (state : Array Nat) (index : Nat) : (readWord state index).work ≤ 5 := by
  unfold readWord
  split <;> simp_all

/-- Size read, bounds check/branch, allocation and a read/write copy of all
cells, the final store, and return. The current generator uses 16 cells. -/
private def writeWord (state : Array Nat) (index value : Nat) : Result (Array Nat) :=
  let size := state.size
  ⟨state.set! index value, 3 * size + 5⟩

private theorem writeWord_value (state : Array Nat) (index value : Nat) :
    (writeWord state index value).value = state.set! index value := rfl

private theorem writeWord_size (state : Array Nat) (index value : Nat) :
    (writeWord state index value).value.size = state.size := by
  simp [writeWord]

private theorem writeWord_work (state : Array Nat) (index value : Nat) :
    (writeWord state index value).work = 3 * state.size + 5 := rfl

/-- Preserve all five original reads, including the second read of B. -/
private def quarter (state : Array Nat) (ai bi ci di : Nat) : Result (Array Nat) :=
  let a0 := readWord state ai
  let b0 := readWord state bi
  let a1 := addWord a0.value b0.value
  let d0 := readWord state di
  let d1x := xorWord d0.value a1.value
  let d1 := rotateWord d1x.value 16
  let c0 := readWord state ci
  let c1 := addWord c0.value d1.value
  let b0again := readWord state bi
  let b1x := xorWord b0again.value c1.value
  let b1 := rotateWord b1x.value 12
  let a2 := addWord a1.value b1.value
  let d2x := xorWord d1.value a2.value
  let d2 := rotateWord d2x.value 8
  let c2 := addWord c1.value d2.value
  let b2x := xorWord b1.value c2.value
  let b2 := rotateWord b2x.value 7
  let sa := writeWord state ai a2.value
  let sb := writeWord sa.value bi b2.value
  let sc := writeWord sb.value ci c2.value
  let sd := writeWord sc.value di d2.value
  ⟨sd.value, a0.work + b0.work + a1.work + d0.work + d1x.work + d1.work + c0.work +
    c1.work + b0again.work + b1x.work + b1.work + a2.work + d2x.work + d2.work + c2.work +
    b2x.work + b2.work + sa.work + sb.work + sc.work + sd.work + 1⟩

private theorem quarter_value (state : Array Nat) (ai bi ci di : Nat) :
    (quarter state ai bi ci di).value = ChaCha20.quarterRound state ai bi ci di := by
  simp only [quarter, writeWord_value, rotateWord_value, xorWord_value, addWord_value, readWord_value]
  rfl

private theorem quarter_size (state : Array Nat) (ai bi ci di : Nat) :
    (quarter state ai bi ci di).value.size = state.size := by
  rw [quarter_value, ChaCha20.quarterRound_size]

private theorem quarter_work_le (state : Array Nat) (ai bi ci di : Nat) (size : state.size = 16) :
    (quarter state ai bi ci di).work ≤ 302 := by
  have a := readWord_work_le state ai
  have b := readWord_work_le state bi
  have c := readWord_work_le state ci
  have d := readWord_work_le state di
  dsimp only [quarter]
  simp only [writeWord_work, writeWord_size, size, addWord, xorWord, rotateWord,
    Nat.reduceMod, Nat.reduceEqDiff, ↓reduceIte]
  omega

private def double (state : Array Nat) : Result (Array Nat) :=
  let s0 := quarter state 0 4 8 12
  let s1 := quarter s0.value 1 5 9 13
  let s2 := quarter s1.value 2 6 10 14
  let s3 := quarter s2.value 3 7 11 15
  let s4 := quarter s3.value 0 5 10 15
  let s5 := quarter s4.value 1 6 11 12
  let s6 := quarter s5.value 2 7 8 13
  let s7 := quarter s6.value 3 4 9 14
  ⟨s7.value, s0.work + s1.work + s2.work + s3.work + s4.work + s5.work + s6.work + s7.work + 1⟩

private theorem double_value (state : Array Nat) : (double state).value = ChaCha20.doubleRound state := by
  simp only [double, quarter_value]
  rfl

private theorem double_size (state : Array Nat) : (double state).value.size = state.size := by
  simp only [double, quarter_size]

private theorem double_work_le (state : Array Nat) (size : state.size = 16) : (double state).work ≤ 2417 := by
  have q0 := quarter_work_le state 0 4 8 12 size
  have q1 := quarter_work_le (quarter state 0 4 8 12).value 1 5 9 13 (by rw [quarter_size, size])
  have q2 := quarter_work_le (quarter (quarter state 0 4 8 12).value 1 5 9 13).value
    2 6 10 14 (by simp only [quarter_size, size])
  have q3 := quarter_work_le (quarter (quarter (quarter state 0 4 8 12).value 1 5 9 13).value 2 6 10 14).value
    3 7 11 15 (by simp only [quarter_size, size])
  have q4 := quarter_work_le (quarter
    (quarter (quarter (quarter state 0 4 8 12).value 1 5 9 13).value 2 6 10 14).value 3 7 11 15).value
    0 5 10 15 (by simp only [quarter_size, size])
  have q5 := quarter_work_le (quarter (quarter
    (quarter (quarter (quarter state 0 4 8 12).value 1 5 9 13).value 2 6 10 14).value 3 7 11 15).value 0 5 10 15).value
    1 6 11 12 (by simp only [quarter_size, size])
  have q6 := quarter_work_le (quarter (quarter (quarter
    (quarter (quarter (quarter state 0 4 8 12).value 1 5 9 13).value 2 6 10 14).value 3 7 11 15).value 0 5 10 15).value
      1 6 11 12).value 2 7 8 13 (by simp only [quarter_size, size])
  have q7 := quarter_work_le (quarter (quarter (quarter (quarter
    (quarter (quarter (quarter state 0 4 8 12).value 1 5 9 13).value 2 6 10 14).value 3 7 11 15).value 0 5 10 15).value
      1 6 11 12).value 2 7 8 13).value 3 4 9 14 (by simp only [quarter_size, size])
  dsimp only [double]
  omega

private def rounds : Nat → Array Nat → Result (Array Nat)
  | 0, state => ⟨state, 2⟩
  | count + 1, state =>
      let next := double state
      let tail := rounds count next.value
      ⟨tail.value, next.work + tail.work + 3⟩

private theorem rounds_value (count : Nat) (state : Array Nat) :
    (rounds count state).value = ChaCha20.runDoubleRounds count state := by
  induction count generalizing state with
  | zero => rfl
  | succ count ih => simp only [rounds, ih, double_value, ChaCha20.runDoubleRounds]

private theorem rounds_work_le (count : Nat) (state : Array Nat) (size : state.size = 16) :
    (rounds count state).work ≤ count * 2420 + 2 := by
  induction count generalizing state with
  | zero => simp [rounds]
  | succ count ih =>
      have step := double_work_le state size
      have tail := ih (double state).value (by rw [double_size, size])
      dsimp only [rounds]
      simp only [Nat.add_mul, Nat.one_mul]
      omega

/-- A list lookup traverses its prefix; no seed-byte function gets a unit
lookup charge. Each descent includes both dispatches and index/tail work. -/
private def readByte : List Nat → Nat → Result Nat
  | [], _ => ⟨0, 4⟩
  | byte :: _, 0 => ⟨byte, 4⟩
  | _ :: bytes, offset + 1 =>
      let tail := readByte bytes offset
      ⟨tail.value, tail.work + 5⟩

private theorem readByte_value (bytes : List Nat) (offset : Nat) :
    (readByte bytes offset).value = bytes.getD offset 0 := by
  induction offset generalizing bytes with
  | zero => cases bytes <;> simp [readByte, List.getD_cons_zero]
  | succ offset ih => cases bytes <;> simp [readByte, List.getD_cons_succ, ih]

private theorem readByte_work_le (bytes : List Nat) (offset : Nat) :
    (readByte bytes offset).work ≤ offset * 5 + 4 := by
  induction offset generalizing bytes with
  | zero => cases bytes <;> simp [readByte]
  | succ offset ih =>
      cases bytes with
      | nil => simp only [readByte]; omega
      | cons byte bytes =>
          have tail := ih bytes
          simp only [readByte, Nat.add_mul, Nat.one_mul]
          omega

private def littleEndian (bytes : List Nat) (offset : Nat) : Result Nat :=
  let a := readByte bytes offset
  let b := readByte bytes (offset + 1)
  let c := readByte bytes (offset + 2)
  let d := readByte bytes (offset + 3)
  ⟨(a.value + 256 * b.value + 65536 * c.value + 16777216 * d.value) % ChaCha20.wordModulus,
    a.work + b.work + c.work + d.work + 11⟩

private theorem littleEndian_value (bytes : List Nat) (offset : Nat) :
    (littleEndian bytes offset).value = ChaCha20.littleEndian32 bytes offset := by
  simp only [littleEndian, readByte_value]
  rfl

private theorem littleEndian_work_le (bytes : List Nat) (offset : Nat) :
    (littleEndian bytes offset).work ≤ 20 * offset + 57 := by
  have a := readByte_work_le bytes offset
  have b := readByte_work_le bytes (offset + 1)
  have c := readByte_work_le bytes (offset + 2)
  have d := readByte_work_le bytes (offset + 3)
  dsimp only [littleEndian]
  omega

/-- Eight seed words, the existing counter/nonce arithmetic, and all sixteen
stored words. Construction charges sixteen slots and sixteen writes. -/
private def initial (seed : List Nat) (row block lane : Nat) : Result (Array Nat) :=
  let s0 := littleEndian seed 0
  let s1 := littleEndian seed 4
  let s2 := littleEndian seed 8
  let s3 := littleEndian seed 12
  let s4 := littleEndian seed 16
  let s5 := littleEndian seed 20
  let s6 := littleEndian seed 24
  let s7 := littleEndian seed 28
  ⟨#[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
    s0.value, s1.value, s2.value, s3.value, s4.value, s5.value, s6.value, s7.value,
    lane % ChaCha20.wordModulus, row % ChaCha20.wordModulus,
    block % ChaCha20.wordModulus, (block / ChaCha20.wordModulus) % ChaCha20.wordModulus],
    s0.work + s1.work + s2.work + s3.work + s4.work + s5.work + s6.work + s7.work + 39⟩

private theorem initial_value (seed : List Nat) (row block lane : Nat) :
    (initial seed row block lane).value = ChaCha20.initialState seed row block lane := by
  simp only [initial, littleEndian_value]
  rfl

private theorem initial_size (seed : List Nat) (row block lane : Nat) :
    (initial seed row block lane).value.size = 16 := by
  rw [initial_value, ChaCha20.initialState_size]

private theorem initial_work_le (seed : List Nat) (row block lane : Nat) :
    (initial seed row block lane).work ≤ 2735 := by
  have s0 := littleEndian_work_le seed 0
  have s1 := littleEndian_work_le seed 4
  have s2 := littleEndian_work_le seed 8
  have s3 := littleEndian_work_le seed 12
  have s4 := littleEndian_work_le seed 16
  have s5 := littleEndian_work_le seed 20
  have s6 := littleEndian_work_le seed 24
  have s7 := littleEndian_work_le seed 28
  dsimp only [initial]
  omega

private def indexAction {count : Nat} (index : Fin count) : StateM Nat Nat :=
  fun work => (index.val, work + 4)

private theorem indices_state_value (count initialWork : Nat) :
    ((List.ofFnM (indexAction (count := count))).run initialWork).1 =
      List.ofFn (fun index : Fin count => index.val) := by
  have generic : ∀ {size : Nat} (read : Fin size → Nat) (start : Nat),
      ((List.ofFnM (m := StateM Nat) (fun index => fun work => (read index, work + 4))).run start).1 =
        List.ofFn read := by
    intro size read start
    induction size generalizing start with
    | zero => rw [List.ofFnM_zero, List.ofFn_zero]; rfl
    | succ size ih =>
        rw [List.ofFnM_succ_last, List.ofFn_succ_last]
        simp only [StateT.run_bind, StateT.run_pure]
        change (((List.ofFnM (m := StateM Nat)
          (fun index => fun work => (read index.castSucc, work + 4))).run start).1) ++
          [read (Fin.last size)] = _
        rw [ih]
  exact generic Fin.val initialWork

private theorem indices_state_work (count initialWork : Nat) :
    ((List.ofFnM (indexAction (count := count))).run initialWork).2 = initialWork + count * 4 := by
  have generic : ∀ {size : Nat} (read : Fin size → Nat) (start : Nat),
      ((List.ofFnM (m := StateM Nat) (fun index => fun work => (read index, work + 4))).run start).2 =
        start + size * 4 := by
    intro size read start
    induction size generalizing start with
    | zero => rw [List.ofFnM_zero]; rfl
    | succ size ih =>
        rw [List.ofFnM_succ_last]
        simp only [StateT.run_bind, StateT.run_pure]
        change ((List.ofFnM (m := StateM Nat)
          (fun index => fun work => (read index.castSucc, work + 4))).run start).2 + 4 = _
        rw [ih]
        omega
  exact generic Fin.val initialWork

/-- Direct indices build the exact range list. Charge the final reverse
as well as every forward index read, loop operation and list constructor. -/
private def indices (count : Nat) : Result (List Nat) :=
  let result := (List.ofFnM (indexAction (count := count))).run 1
  ⟨result.1, result.2 + count * 3 + 2⟩

private theorem indices_value (count : Nat) : (indices count).value = List.range count := by
  change ((List.ofFnM (indexAction (count := count))).run 1).1 = _
  rw [indices_state_value]
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [List.ofFn_succ_last, List.range_succ]
      simpa only [Fin.val_castSucc, Fin.val_last] using congrArg (· ++ [count]) ih

private theorem indices_work (count : Nat) : (indices count).work = count * 7 + 3 := by
  simp only [indices, indices_state_work]
  omega

private def feedForward (initialState permuted : Array Nat) : List Nat → Result (List Nat)
  | [] => ⟨[], 2⟩
  | index :: remaining =>
      let left := readWord permuted index
      let right := readWord initialState index
      let sum := addWord left.value right.value
      let tail := feedForward initialState permuted remaining
      ⟨sum.value :: tail.value, left.work + right.work + sum.work + tail.work + 5⟩

private theorem feedForward_value (initialState permuted : Array Nat) (range : List Nat) :
    (feedForward initialState permuted range).value = range.map (fun index =>
      ChaCha20.add32 (ChaCha20.getWord permuted index) (ChaCha20.getWord initialState index)) := by
  induction range with
  | nil => rfl
  | cons index remaining ih =>
      simp only [feedForward, List.map_cons, readWord_value, addWord_value, ih]

private theorem feedForward_work_le (initialState permuted : Array Nat) (range : List Nat) :
    (feedForward initialState permuted range).work ≤ range.length * 18 + 2 := by
  induction range with
  | nil => simp [feedForward]
  | cons index remaining ih =>
      have left := readWord_work_le permuted index
      have right := readWord_work_le initialState index
      simp only [feedForward, addWord, List.length_cons, Nat.add_mul, Nat.one_mul]
      omega

/-- All sixteen feed-forward additions execute, before the eight-word take. -/
private def block (seed : List Nat) (row blockIndex lane : Nat) : Result (List Nat) :=
  let state := initial seed row blockIndex lane
  let permuted := rounds 10 state.value
  let range := indices 16
  let output := feedForward state.value permuted.value range.value
  ⟨output.value, state.work + permuted.work + range.work + output.work + 1⟩

private theorem block_value (seed : List Nat) (row blockIndex lane : Nat) :
    (block seed row blockIndex lane).value = ChaCha20.blockWords seed row blockIndex lane := by
  simp only [block, feedForward_value, indices_value, rounds_value, initial_value]
  rfl

private theorem block_work_le (seed : List Nat) (row blockIndex lane : Nat) :
    (block seed row blockIndex lane).work ≤ 27343 := by
  have state := initial_work_le seed row blockIndex lane
  have permuted := rounds_work_le 10 (initial seed row blockIndex lane).value
    (initial_size seed row blockIndex lane)
  have output := feedForward_work_le (initial seed row blockIndex lane).value
    (rounds 10 (initial seed row blockIndex lane).value).value (indices 16).value
  simp only [indices_value, List.length_range] at output
  dsimp only [block]
  rw [indices_work, indices_value]
  omega

private def takeWords : Nat → List Nat → Result (List Nat)
  | 0, _ => ⟨[], 3⟩
  | _ + 1, [] => ⟨[], 4⟩
  | count + 1, word :: words =>
      let tail := takeWords count words
      ⟨word :: tail.value, tail.work + 7⟩

private theorem takeWords_value (count : Nat) (words : List Nat) :
    (takeWords count words).value = words.take count := by
  induction count generalizing words with
  | zero => rfl
  | succ count ih => cases words <;> simp only [takeWords, List.take, ih]

private theorem takeWords_work_le (count : Nat) (words : List Nat) :
    (takeWords count words).work ≤ count * 7 + 4 := by
  induction count generalizing words with
  | zero => simp [takeWords]
  | succ count ih =>
      cases words with
      | nil => simp only [takeWords]; omega
      | cons word words =>
          have tail := ih words
          simp only [takeWords, Nat.add_mul, Nat.one_mul]
          omega

private def reverseLoop : List Nat → Result (List Nat) → Result (List Nat)
  | [], accumulated => ⟨accumulated.value, accumulated.work + 2⟩
  | word :: words, accumulated =>
      reverseLoop words ⟨word :: accumulated.value, accumulated.work + 5⟩

private theorem reverseLoop_value (words : List Nat) (initial : Result (List Nat)) :
    (reverseLoop words initial).value = List.reverseAux words initial.value := by
  induction words generalizing initial with
  | nil => rfl
  | cons word words ih => simp only [reverseLoop, ih, List.reverseAux]

private theorem reverseLoop_work (words : List Nat) (initial : Result (List Nat)) :
    (reverseLoop words initial).work = initial.work + words.length * 5 + 2 := by
  induction words generalizing initial with
  | nil => simp [reverseLoop]
  | cons word words ih => simp only [reverseLoop, ih, List.length_cons, Nat.add_mul, Nat.one_mul]; omega

private def reverseWords (words : List Nat) : Result (List Nat) := reverseLoop words ⟨[], 2⟩

private theorem reverseWords_value (words : List Nat) : (reverseWords words).value = words.reverse := by
  rw [reverseWords, reverseLoop_value]
  rfl

private theorem reverseWords_work (words : List Nat) : (reverseWords words).work = words.length * 5 + 4 := by
  simp only [reverseWords, reverseLoop_work]
  omega

/-- Horner packing reads each of the eight words and performs one wide
multiply/add. The accumulator never exceeds 256 bits on this caller. -/
private def packLoop : List Nat → Result Nat → Result Nat
  | [], accumulated => ⟨accumulated.value, accumulated.work + 2⟩
  | word :: words, accumulated =>
      packLoop words ⟨accumulated.value * ChaCha20.wordModulus + word, accumulated.work + 6⟩

private theorem packLoop_value (words : List Nat) (initial : Result Nat) :
    (packLoop words initial).value =
      words.foldl (fun value word => value * ChaCha20.wordModulus + word) initial.value := by
  induction words generalizing initial with
  | nil => rfl
  | cons word words ih => simp only [packLoop, ih, List.foldl_cons]

private theorem packLoop_work (words : List Nat) (initial : Result Nat) :
    (packLoop words initial).work = initial.work + words.length * 6 + 2 := by
  induction words generalizing initial with
  | nil => simp [packLoop]
  | cons word words ih => simp only [packLoop, ih, List.length_cons, Nat.add_mul, Nat.one_mul]; omega

private theorem pack_range (words : List Nat)
    (canonical : ∀ word ∈ words, word < ChaCha20.wordModulus)
    (initial digits : Nat) (bounded : initial < ChaCha20.wordModulus ^ digits) :
    words.foldl (fun value word => value * ChaCha20.wordModulus + word) initial <
      ChaCha20.wordModulus ^ (digits + words.length) := by
  induction words generalizing initial digits with
  | nil => simpa using bounded
  | cons word words ih =>
      have head := canonical word (by simp)
      have step : initial * ChaCha20.wordModulus + word < ChaCha20.wordModulus ^ (digits + 1) := by
        rw [Nat.pow_succ]
        unfold ChaCha20.wordModulus at head bounded ⊢
        omega
      have tail := ih (fun value member => canonical value (by simp [member]))
        (initial * ChaCha20.wordModulus + word) (digits + 1) step
      have order : digits + 1 + words.length = digits + (words.length + 1) := by omega
      simpa only [List.foldl_cons, List.length_cons, order] using tail

private def first256 (seed : List Nat) (row blockIndex lane : Nat) : Result Nat :=
  let words := block seed row blockIndex lane
  let first := takeWords 8 words.value
  let reversed := reverseWords first.value
  let packed := packLoop reversed.value ⟨0, 1⟩
  ⟨packed.value, words.work + first.work + reversed.work + packed.work + 1⟩

private theorem first256_value (seed : List Nat) (row blockIndex lane : Nat) :
    (first256 seed row blockIndex lane).value = ChaCha20.first256Nat seed row blockIndex lane := by
  simp only [first256, packLoop_value, reverseWords_value, takeWords_value, block_value]
  rfl

private theorem first256_lt (seed : List Nat) (row blockIndex lane : Nat) :
    (first256 seed row blockIndex lane).value < 2 ^ 256 := by
  let words := ((ChaCha20.blockWords seed row blockIndex lane).take 8).reverse
  have length : words.length = 8 := by
    simp only [words, List.length_reverse, List.length_take, ChaCha20.blockWords_length]
    rfl
  have canonical : ∀ word ∈ words, word < ChaCha20.wordModulus := by
    intro word member
    have original := List.mem_of_mem_take (List.mem_reverse.mp member)
    exact ChaCha20.blockWords_canonical seed row blockIndex lane word original
  have bounded := pack_range words canonical 0 0 (by decide)
  rw [length] at bounded
  change words.foldl (fun value word => value * ChaCha20.wordModulus + word) 0 <
    ChaCha20.wordModulus ^ 8 at bounded
  have width : ChaCha20.wordModulus ^ 8 = 2 ^ 256 := by decide
  rw [width] at bounded
  rw [first256_value]
  exact bounded

private theorem first256_work_le (seed : List Nat) (row blockIndex lane : Nat) :
    (first256 seed row blockIndex lane).work ≤ 27499 := by
  have generated := block_work_le seed row blockIndex lane
  have taken := takeWords_work_le 8 (block seed row blockIndex lane).value
  have length : (takeWords 8 (block seed row blockIndex lane).value).value.length = 8 := by
    rw [takeWords_value, block_value]
    simp only [List.length_take, ChaCha20.blockWords_length]
    rfl
  dsimp only [first256]
  simp only [reverseWords_work, packLoop_work, reverseWords_value, List.length_reverse, length]
  omega

/-- Selected 32-bit row, 64-bit block and 54-lane indices make the nonce
arithmetic fixed-size. The generator itself remains total at all indices. -/
def coefficient {verifierRows messageColumns : Nat} (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (blockIndex : Fin messageColumns) (lane : Fin ringDegree) : Result F :=
  let seed := setup.seed
  let bytes := seed.bytes
  let rowIndex := row.val
  let blockNumber := blockIndex.val
  let counter := lane.val
  let generated := first256 bytes rowIndex blockNumber counter
  let wide : Fin (2 ^ 256) := ⟨generated.value, first256_lt bytes rowIndex blockNumber counter⟩
  let reduced := wide.val % goldilocksModulus
  ⟨⟨reduced, Nat.mod_lt _ (by decide)⟩, generated.work + 10⟩

theorem coefficient_value {verifierRows messageColumns : Nat} (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (blockIndex : Fin messageColumns) (lane : Fin ringDegree) :
    (coefficient setup row blockIndex lane).value = setup.verifierKey row blockIndex lane := by
  apply Fin.ext
  simp only [coefficient, first256_value, Setup.verifierKey, Setup.coefficientNat, wideCoefficientNat]

/-- Derived from list traversal, all 80 quarter rounds, all 16 feed-forward
words, the eight-word take/reverse/packing, and the final wide reduction. -/
def coefficientWork : Nat := 27509

theorem coefficient_work_le {verifierRows messageColumns : Nat} (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (blockIndex : Fin messageColumns) (lane : Fin ringDegree)
    (_rowRange : row.val < 2 ^ 32) (_blockRange : blockIndex.val < 2 ^ 64) :
    (coefficient setup row blockIndex lane).work ≤ coefficientWork := by
  have wide := first256_work_le setup.seed.bytes row.val blockIndex.val lane.val
  dsimp only [coefficient, coefficientWork]
  omega

end NightstreamFPrime.Spec.AjtaiSetupV1.Work
