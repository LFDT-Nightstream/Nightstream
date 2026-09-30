import NightstreamFPrime.Spec.AjtaiSetupV1
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
Counted execution of the indexed SHAKE128 key-element generator. The clock
counts list steps, arithmetic, reads, constructors and returns. One Keccak
round and one rate-lane list are declared primitives whose charges count
their lane reads, lane operations and constructors. Appends copy their left
list. Counts are not elapsed machine time.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.AjtaiSetupV1.Work

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

/-- One constructor and the step dispatch per copied cell. -/
private def append {Alpha : Type} : List Alpha → List Alpha → Result (List Alpha)
  | [], right => ⟨right, 2⟩
  | head :: tail, right =>
      let rest := append tail right
      ⟨head :: rest.value, rest.work + 4⟩

private theorem append_value {Alpha : Type} (left right : List Alpha) :
    (append left right).value = left ++ right := by
  induction left with
  | nil => rfl
  | cons head tail ih => simp only [append, ih, List.cons_append]

private theorem append_work {Alpha : Type} (left right : List Alpha) :
    (append left right).work = left.length * 4 + 2 := by
  induction left with
  | nil => rfl
  | cons head tail ih => simp only [append, ih, List.length_cons]; omega

/-- One remainder, one division, one constructor and the step dispatch per
byte. In-range rows and blocks keep both operations on fixed-size words. -/
private def littleEndian : Nat → Nat → Result (List Nat)
  | 0, _ => ⟨[], 2⟩
  | count + 1, value =>
      let rest := littleEndian count (value / 256)
      ⟨value % 256 :: rest.value, rest.work + 5⟩

private theorem littleEndian_value (count value : Nat) :
    (littleEndian count value).value = littleEndianBytes count value := by
  induction count generalizing value with
  | zero => rfl
  | succ count ih => simp only [littleEndian, littleEndianBytes, ih]

private theorem littleEndian_work (count value : Nat) :
    (littleEndian count value).work = count * 5 + 2 := by
  induction count generalizing value with
  | zero => rfl
  | succ count ih => simp only [littleEndian, ih]; omega

/-- Right-nested appends copy the identifier, the seed and the row bytes once. -/
private def elementInput (seed : List Nat) (row block : Nat) : Result (List Nat) :=
  let rowBytes := littleEndian 4 row
  let blockBytes := littleEndian 8 block
  let indices := append rowBytes.value blockBytes.value
  let seeded := append seed indices.value
  let input := append setupIdBytes seeded.value
  ⟨input.value, rowBytes.work + blockBytes.work + indices.work + seeded.work + input.work + 1⟩

private theorem elementInput_value (seed : List Nat) (row block : Nat) :
    (elementInput seed row block).value = AjtaiSetupV1.elementInput seed row block := by
  simp only [elementInput, append_value, littleEndian_value, AjtaiSetupV1.elementInput,
    List.append_assoc]

private theorem elementInput_work (seed : List Nat) (row block : Nat) :
    (elementInput seed row block).work = seed.length * 4 + 235 := by
  simp only [elementInput, append_work, littleEndian_work, littleEndian_value,
    littleEndianBytes_length, setupIdBytes_length]
  omega

private def length {Alpha : Type} : List Alpha → Result Nat
  | [] => ⟨0, 2⟩
  | _ :: tail =>
      let rest := length tail
      ⟨rest.value + 1, rest.work + 4⟩

private theorem length_value {Alpha : Type} (list : List Alpha) :
    (length list).value = list.length := by
  induction list with
  | nil => rfl
  | cons head tail ih => simp only [length, ih, List.length_cons]

private theorem length_work {Alpha : Type} (list : List Alpha) :
    (length list).work = list.length * 4 + 2 := by
  induction list with
  | nil => rfl
  | cons head tail ih => simp only [length, ih, List.length_cons]; omega

private def zeros : Nat → Result (List Nat)
  | 0 => ⟨[], 2⟩
  | count + 1 =>
      let rest := zeros count
      ⟨0 :: rest.value, rest.work + 3⟩

private theorem zeros_value (count : Nat) : (zeros count).value = List.replicate count 0 := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [zeros, ih, List.replicate_succ]

private theorem zeros_work (count : Nat) : (zeros count).work = count * 3 + 2 := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [zeros, ih]; omega

/-- The comparison, the subtractions, the zero run and its final byte. -/
private def padding (used : Nat) : Result (List Nat) :=
  if used + 1 = Shake128.rateBytes then ⟨[0x9F], 5⟩
  else
    let run := zeros (Shake128.rateBytes - used - 2)
    let closed := append run.value [0x80]
    ⟨0x1F :: closed.value, run.work + closed.work + 8⟩

private theorem padding_value (used : Nat) : (padding used).value = Shake128.padding used := by
  unfold padding Shake128.padding
  split <;> simp only [append_value, zeros_value]

private theorem padding_work_le (used : Nat) : (padding used).work ≤ 1174 := by
  unfold padding
  split
  · decide
  · simp only [append_work, zeros_work, zeros_value, List.length_replicate, Shake128.rateBytes]
    omega

private def pad (message : List Nat) : Result (List Nat) :=
  let size := length message
  let tail := padding (size.value % Shake128.rateBytes)
  let padded := append message tail.value
  ⟨padded.value, size.work + tail.work + padded.work + 3⟩

private theorem pad_value (message : List Nat) : (pad message).value = Shake128.pad message := by
  simp only [pad, append_value, padding_value, length_value, Shake128.pad]

private theorem pad_work_le (message : List Nat) :
    (pad message).work ≤ message.length * 8 + 1181 := by
  have tail := padding_work_le ((length message).value % Shake128.rateBytes)
  simp only [pad, append_work, length_work]
  omega

/-- Eight cell matches, seven products, seven sums, one conversion, one
constructor and the step dispatch per lane. A short tail costs at most
eight matches and one return. -/
private def lanesOfBytes : List Nat → Result (List UInt64)
  | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest =>
      let tail := lanesOfBytes rest
      ⟨Shake128.laneOf b0 b1 b2 b3 b4 b5 b6 b7 :: tail.value, tail.work + 25⟩
  | _ => ⟨[], 9⟩

private theorem lanesOfBytes_value (bytes : List Nat) :
    (lanesOfBytes bytes).value = Shake128.lanesOfBytes bytes := by
  fun_induction Shake128.lanesOfBytes bytes with
  | case1 b0 b1 b2 b3 b4 b5 b6 b7 rest ih => simp only [lanesOfBytes, ih]
  | case2 bytes short =>
      rcases bytes with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩⟩⟩⟩⟩⟩
      all_goals first
        | exact (short _ _ _ _ _ _ _ _ _ rfl).elim
        | (unfold lanesOfBytes; rfl)
        | (unfold lanesOfBytes; simp; done)
        | (simp; done)

private theorem lanesOfBytes_work (bytes : List Nat) :
    (lanesOfBytes bytes).work = bytes.length / 8 * 25 + 9 := by
  fun_induction Shake128.lanesOfBytes bytes with
  | case1 b0 b1 b2 b3 b4 b5 b6 b7 rest ih =>
      simp only [lanesOfBytes, ih, List.length_cons]
      omega
  | case2 bytes short =>
      rcases bytes with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩⟩⟩⟩⟩⟩
      all_goals first
        | exact (short _ _ _ _ _ _ _ _ _ rfl).elim
        | (unfold lanesOfBytes; rfl)
        | (unfold lanesOfBytes; simp; done)
        | (simp; done)

/-- One Keccak round: 50 lane reads, 76 XORs, 30 rotations of four
operations each (two shifts, one subtraction and one OR), 25 complements,
25 conjunctions, one allocation, 25 lane writes and one return. -/
def roundWork : Nat := 50 + 76 + 30 * 4 + 25 + 25 + 1 + 25 + 1

private def round (s : Shake128.State) (constant : UInt64) : Result Shake128.State :=
  ⟨Shake128.round s constant, roundWork⟩

/-- One list step, the round and one accumulator update per round constant. -/
private def rounds : List UInt64 → Result Shake128.State → Result Shake128.State
  | [], state => ⟨state.value, state.work + 2⟩
  | constant :: constants, state =>
      let next := round state.value constant
      rounds constants ⟨next.value, state.work + next.work + 3⟩

private theorem rounds_value (constants : List UInt64) (state : Result Shake128.State) :
    (rounds constants state).value = constants.foldl Shake128.round state.value := by
  induction constants generalizing state with
  | nil => rfl
  | cons constant constants ih => simp only [rounds, ih, round, List.foldl_cons]

private theorem rounds_work (constants : List UInt64) (state : Result Shake128.State) :
    (rounds constants state).work = state.work + constants.length * (roundWork + 3) + 2 := by
  induction constants generalizing state with
  | nil => simp [rounds]
  | cons constant constants ih =>
      simp only [rounds, ih, round, List.length_cons]
      rw [Nat.succ_mul]
      omega

private def permute (s : Shake128.State) : Result Shake128.State :=
  rounds Shake128.roundConstants ⟨s, 1⟩

/-- Keccak-f[1600]: 24 counted rounds. -/
def permutationWork : Nat := 1 + 24 * (roundWork + 3) + 2

private theorem permute_value (s : Shake128.State) : (permute s).value = Shake128.permute s :=
  rounds_value _ _

private theorem permute_work (s : Shake128.State) : (permute s).work = permutationWork := by
  unfold permute permutationWork
  rw [rounds_work]
  rfl

/-- Per rate block: 21 cell matches, 21 lane reads and XORs, 25 lane writes
into one new state, one allocation, the step dispatch and the permutation.
A short tail costs at most 21 matches and one return. -/
private def absorbLanes (s : Shake128.State) : List UInt64 → Result Shake128.State
  | l0 :: l1 :: l2 :: l3 :: l4 :: l5 :: l6 :: l7 :: l8 :: l9 :: l10 :: l11 :: l12 :: l13 ::
      l14 :: l15 :: l16 :: l17 :: l18 :: l19 :: l20 :: rest =>
    let permuted := permute { s with
      a00 := s.a00 ^^^ l0, a10 := s.a10 ^^^ l1, a20 := s.a20 ^^^ l2,
      a30 := s.a30 ^^^ l3, a40 := s.a40 ^^^ l4,
      a01 := s.a01 ^^^ l5, a11 := s.a11 ^^^ l6, a21 := s.a21 ^^^ l7,
      a31 := s.a31 ^^^ l8, a41 := s.a41 ^^^ l9,
      a02 := s.a02 ^^^ l10, a12 := s.a12 ^^^ l11, a22 := s.a22 ^^^ l12,
      a32 := s.a32 ^^^ l13, a42 := s.a42 ^^^ l14,
      a03 := s.a03 ^^^ l15, a13 := s.a13 ^^^ l16, a23 := s.a23 ^^^ l17,
      a33 := s.a33 ^^^ l18, a43 := s.a43 ^^^ l19,
      a04 := s.a04 ^^^ l20 }
    let tail := absorbLanes permuted.value rest
    ⟨tail.value, permuted.work + tail.work + 90⟩
  | _ => ⟨s, 22⟩

private theorem absorbLanes_value (s : Shake128.State) (lanes : List UInt64) :
    (absorbLanes s lanes).value = Shake128.absorbLanes s lanes := by
  fun_induction Shake128.absorbLanes s lanes with
  | case1 s l0 l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l20 rest ih =>
      simp only [absorbLanes, permute_value, ih]
  | case2 lanes s short =>
      rcases lanes with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _
        | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _
        | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩
      all_goals first
        | exact (short _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ rfl).elim
        | (unfold absorbLanes; rfl)
        | (unfold absorbLanes; simp; done)

private theorem absorbLanes_work (s : Shake128.State) (lanes : List UInt64) :
    (absorbLanes s lanes).work = lanes.length / 21 * (permutationWork + 90) + 22 := by
  fun_induction Shake128.absorbLanes s lanes with
  | case1 s l0 l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l20 rest ih =>
      simp only [absorbLanes, permute_work, permute_value, ih, List.length_cons]
      unfold permutationWork roundWork
      omega
  | case2 lanes s short =>
      rcases lanes with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _
        | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _
        | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩
      all_goals first
        | exact (short _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ rfl).elim
        | (unfold absorbLanes; rfl)
        | (unfold absorbLanes; simp; done)

/-- The 21 rate lanes: 21 reads, 21 constructors, the empty list and the return. -/
private def rateLanes (s : Shake128.State) : Result (List UInt64) :=
  ⟨Shake128.rateLanes s, 44⟩

/-- Rate blocks are squeezed in order; the state is permuted only between
blocks, and each block's lanes are copied once into the output list. -/
private def squeezeBlocks (s : Shake128.State) : Nat → Result (List UInt64)
  | 0 => ⟨[], 2⟩
  | 1 =>
      let lanes := rateLanes s
      ⟨lanes.value, lanes.work + 2⟩
  | blocks + 2 =>
      let lanes := rateLanes s
      let permuted := permute s
      let tail := squeezeBlocks permuted.value (blocks + 1)
      let joined := append lanes.value tail.value
      ⟨joined.value, lanes.work + permuted.work + tail.work + joined.work + 2⟩

private theorem squeezeBlocks_value (s : Shake128.State) (blocks : Nat) :
    (squeezeBlocks s blocks).value = Shake128.squeezeBlocks s blocks := by
  induction blocks generalizing s with
  | zero => rfl
  | succ blocks ih =>
      cases blocks with
      | zero => rfl
      | succ blocks =>
          simp only [squeezeBlocks, Shake128.squeezeBlocks, append_value, permute_value, ih,
            rateLanes]

private theorem squeezeBlocks_work_le (s : Shake128.State) (blocks : Nat) :
    (squeezeBlocks s blocks).work ≤ blocks * (permutationWork + 132) + 2 := by
  induction blocks generalizing s with
  | zero => simp [squeezeBlocks]
  | succ blocks ih =>
      cases blocks with
      | zero =>
          simp only [squeezeBlocks, rateLanes]
          unfold permutationWork roundWork
          omega
      | succ blocks =>
          have tail := ih (permute s).value
          simp only [squeezeBlocks, rateLanes, append_work, permute_work, Shake128.rateLanes,
            List.length_cons, List.length_nil] at tail ⊢
          unfold permutationWork roundWork at tail ⊢
          rw [Nat.succ_mul]
          omega

private def take {Alpha : Type} : Nat → List Alpha → Result (List Alpha)
  | 0, _ => ⟨[], 2⟩
  | _ + 1, [] => ⟨[], 3⟩
  | count + 1, head :: tail =>
      let rest := take count tail
      ⟨head :: rest.value, rest.work + 4⟩

private theorem take_value {Alpha : Type} (count : Nat) (list : List Alpha) :
    (take count list).value = list.take count := by
  induction count generalizing list with
  | zero => rfl
  | succ count ih => cases list <;> simp only [take, ih, List.take_succ_cons, List.take_nil]

private theorem take_work_le {Alpha : Type} (count : Nat) (list : List Alpha) :
    (take count list).work ≤ count * 4 + 3 := by
  induction count generalizing list with
  | zero => simp [take]
  | succ count ih =>
      cases list with
      | nil => simp only [take]; omega
      | cons head tail =>
          have rest := ih tail
          simp only [take]
          omega

/-- The first `count` output lanes of SHAKE128(message). -/
private def lanes (message : List Nat) (count : Nat) : Result (List UInt64) :=
  let padded := pad message
  let messageLanes := lanesOfBytes padded.value
  let absorbed := absorbLanes Shake128.State.zero messageLanes.value
  let squeezed := squeezeBlocks absorbed.value ((count + 20) / 21)
  let taken := take count squeezed.value
  ⟨taken.value, padded.work + messageLanes.work + absorbed.work + squeezed.work + taken.work + 4⟩

private theorem lanes_value (message : List Nat) (count : Nat) :
    (lanes message count).value = Shake128.lanes message count := by
  simp only [lanes, take_value, squeezeBlocks_value, absorbLanes_value, lanesOfBytes_value,
    pad_value, Shake128.lanes, Shake128.absorb]

/-- One 81-byte element input fills one rate block after padding. -/
private theorem pad_length (message : List Nat) (size : message.length = 81) :
    (Shake128.pad message).length = 168 := by
  simp [Shake128.pad, Shake128.padding, Shake128.rateBytes, size]

private theorem lanesOfBytes_length (bytes : List Nat) :
    (Shake128.lanesOfBytes bytes).length = bytes.length / 8 := by
  fun_induction Shake128.lanesOfBytes bytes with
  | case1 b0 b1 b2 b3 b4 b5 b6 b7 rest ih =>
      simp only [List.length_cons, ih]
      omega
  | case2 bytes short =>
      rcases bytes with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩⟩⟩⟩⟩⟩
      all_goals first
        | exact (short _ _ _ _ _ _ _ _ _ rfl).elim
        | (unfold lanesOfBytes; rfl)
        | (unfold lanesOfBytes; simp; done)
        | (simp; done)

/-- The counted SHAKE128 call on one 81-byte element input: padding, one
absorbed block, eleven squeezed blocks and 216 retained lanes. -/
private def lanesWork : Nat :=
  (81 * 8 + 1181) + (168 / 8 * 25 + 9) + (1 * (permutationWork + 90) + 22) +
    (11 * (permutationWork + 132) + 2) + (4 * ringDegree * 4 + 3) + 4

private theorem lanes_work_le (message : List Nat) (size : message.length = 81) :
    (lanes message (4 * ringDegree)).work ≤ lanesWork := by
  have padded := pad_work_le message
  have paddedSize := pad_length message size
  have laneCount : (Shake128.lanesOfBytes (Shake128.pad message)).length = 21 := by
    rw [lanesOfBytes_length, paddedSize]
  have squeezed := squeezeBlocks_work_le
    (Shake128.absorbLanes Shake128.State.zero (Shake128.lanesOfBytes (Shake128.pad message)))
    ((4 * ringDegree + 20) / 21)
  have taken := take_work_le (4 * ringDegree)
    (Shake128.squeezeBlocks
      (Shake128.absorbLanes Shake128.State.zero (Shake128.lanesOfBytes (Shake128.pad message)))
      ((4 * ringDegree + 20) / 21))
  simp only [lanes, lanesOfBytes_work, absorbLanes_work, pad_value, lanesOfBytes_value,
    absorbLanes_value, squeezeBlocks_value]
  rw [paddedSize, laneCount]
  rw [size] at padded
  have blocks : (4 * ringDegree + 20) / 21 = 11 := rfl
  rw [blocks] at squeezed taken ⊢
  have degree : ringDegree = 54 := rfl
  unfold lanesWork
  omega

/-- The element input, one SHAKE128 call and the return. -/
def elementLanesWork : Nat := 32 * 4 + 235 + lanesWork + 2

/-- Compute all 216 SHAKE128 output lanes of one key element. -/
def elementLanes {verifierRows messageColumns : Nat} (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) : Result (List UInt64) :=
  let input := elementInput setup.seed.bytes row.val block.val
  let output := lanes input.value (4 * ringDegree)
  ⟨output.value, input.work + output.work + 2⟩

theorem elementLanes_value {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) :
    (elementLanes setup row block).value =
      AjtaiSetupV1.elementLanes setup.seed.bytes row.val block.val := by
  simp only [elementLanes, lanes_value, elementInput_value, AjtaiSetupV1.elementLanes]

/-- Selected 32-bit rows and 64-bit blocks keep the index arithmetic on
fixed-size words. -/
theorem elementLanes_work_le {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (_rowRange : row.val < 2 ^ 32) (_blockRange : block.val < 2 ^ 64) :
    (elementLanes setup row block).work ≤ elementLanesWork := by
  have inputSize : (AjtaiSetupV1.elementInput setup.seed.bytes row.val block.val).length = 81 := by
    simp [AjtaiSetupV1.elementInput, setup.seed.length_eq]
  have output := lanes_work_le _ inputSize
  simp only [elementLanes, elementInput_work, elementInput_value, setup.seed.length_eq]
  unfold elementLanesWork
  omega

/-- A list read traverses its prefix. -/
private def read : List UInt64 → Nat → Result UInt64
  | [], _ => ⟨0, 4⟩
  | lane :: _, 0 => ⟨lane, 4⟩
  | _ :: lanes, offset + 1 =>
      let tail := read lanes offset
      ⟨tail.value, tail.work + 5⟩

private theorem read_value (lanes : List UInt64) (offset : Nat) :
    (read lanes offset).value = lanes.getD offset 0 := by
  induction offset generalizing lanes with
  | zero => cases lanes <;> rfl
  | succ offset ih => cases lanes <;> simp [read, ih]

private theorem read_work_le (lanes : List UInt64) (offset : Nat) :
    (read lanes offset).work ≤ offset * 5 + 4 := by
  induction offset generalizing lanes with
  | zero => cases lanes <;> simp [read]
  | succ offset ih =>
      cases lanes with
      | nil => simp only [read]; omega
      | cons lane lanes =>
          have tail := ih lanes
          simp only [read]
          omega

/-- Four index operations, four lane reads and conversions, three wide
products and sums, one reduction, the field constructor and the return. -/
def laneCoefficient (lanes : List UInt64) (lane : Fin ringDegree) : Result F :=
  let offset := 4 * lane.val
  let w0 := read lanes offset
  let w1 := read lanes (offset + 1)
  let w2 := read lanes (offset + 2)
  let w3 := read lanes (offset + 3)
  let wide := w0.value.toNat + 2 ^ 64 * (w1.value.toNat +
    2 ^ 64 * (w2.value.toNat + 2 ^ 64 * w3.value.toNat))
  ⟨⟨wide % goldilocksModulus, Nat.mod_lt _ (by decide)⟩,
    w0.work + w1.work + w2.work + w3.work + 17⟩

/-- Every coefficient reads offsets below the 216 element lanes. -/
def laneCoefficientWork : Nat := 4 * ((4 * ringDegree - 1) * 5 + 4) + 17

theorem laneCoefficient_value {verifierRows messageColumns : Nat}
    (setup : Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) (lane : Fin ringDegree) :
    (laneCoefficient (AjtaiSetupV1.elementLanes setup.seed.bytes row.val block.val) lane).value =
      setup.verifierKey row block lane := by
  apply Fin.ext
  simp only [laneCoefficient, read_value, Setup.verifierKey, Setup.coefficientNat,
    wideCoefficientNat, laneWord]

theorem laneCoefficient_work_le (lanes : List UInt64) (lane : Fin ringDegree) :
    (laneCoefficient lanes lane).work ≤ laneCoefficientWork := by
  have w0 := read_work_le lanes (4 * lane.val)
  have w1 := read_work_le lanes (4 * lane.val + 1)
  have w2 := read_work_le lanes (4 * lane.val + 2)
  have w3 := read_work_le lanes (4 * lane.val + 3)
  have bound := lane.isLt
  have degree : ringDegree = 54 := rfl
  simp only [laneCoefficient, laneCoefficientWork]
  omega

end NightstreamFPrime.Spec.AjtaiSetupV1.Work
