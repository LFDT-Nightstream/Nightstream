import NightstreamFPrime.Spec.Algebra

/-! The width-16 Poseidon2 instance over `fieldModulus` (`decisions/poseidon2-p64-instance.md`).
`scripts/poseidon2/derive.py` takes the round constants from the Grain LFSR of the Poseidon2
reference generator, derives the internal diagonal, and writes both to
`scripts/poseidon2/params.json`. The literals below are that file's width-16 instance, as canonical
residues. The layers follow plonky3 `p3-poseidon2 0.5.3`. -/

namespace NightstreamFPrime.Spec.Poseidon2

def width : Nat := 16
def rate : Nat := 12
def digestLen : Nat := 4
def halfFullRounds : Nat := 4
def partialRounds : Nat := 43

abbrev State := List F

def ofNat (n : Nat) : F := ⟨n % fieldModulus, Nat.mod_lt _ (by decide)⟩

def initialConstants : List (List Nat) :=
  [[17625877419914948618, 7043359134273770980, 8324781800510000080, 6444119057261225485,
    17717541571092793106, 3986291954004622034, 690229812634116840, 12411915661887220387,
    6363654443139256914, 12105449363599063910, 3497590068843640987, 4222579949428206412,
    9871869144392188158, 6608483792230178088, 1361840775413654458, 8678242877107847347],
   [10884132779455466827, 2140996594960535156, 6875127883761382716, 2527369230034983186,
    13363996391325744878, 8922005884211686156, 7168778978332361576, 856389527476709175,
    737755066540146832, 13556839930373010819, 10871866087908657800, 9591461885764716448,
    12537801773373699941, 3260007846858055737, 13363583202969484245, 4218638171078850311],
   [7865032564572175278, 836093766369726329, 12527643047030536902, 1194545980657409763,
    12619589965719950681, 4280088488896281756, 15089781684263861043, 15828500408936832218,
    6419605478123467032, 2370996673254346830, 8789802724387314278, 14988192665513313034,
    4090662135633155679, 8312064199888872985, 13543758618702640312, 16933079866928560805],
   [14849793775896549215, 5585003427041122008, 4039188622500476663, 8473588348029641541,
    14983429621133257453, 10213932717772184020, 14126063598418757144, 9149271627910874493,
    10568511515541199945, 8217860339194175559, 17410842106438810972, 9293755649055318357,
    4035347364804663719, 680576525761979526, 3925491009028150975, 161661792350051539]]

def internalConstants : List Nat :=
  [1025323479709822526, 6111066563283471619, 11883645335560589882,
   5173392838145480224, 12046738740207396666, 11342877365108905038,
   9954226806870194281, 284817437666484542, 7855238300370871153,
   18340977697387037787, 12536785651043586535, 16922285383409970114,
   6304755170918535876, 11920214520673410344, 15734762412768547841,
   244322478157940423, 15954689516362665467, 17014463839077801966,
   7540924580042546453, 8685971293749213383, 11638412499393676635,
   9220075213657617084, 11350647073967129101, 10320484671217327025,
   945321669835615161, 13689403322670363205, 9683142702532056693,
   5152317403353772282, 12964243777216042008, 2572560616624767066,
   5182608262343950270, 16474692687110834850, 743293326911585472,
   1616451609802447080, 13870644991934200425, 6825873398398074331,
   8937369057000642915, 13389634706674966780, 15982711626548638113,
   12255367991153439320, 792159564585665664, 6095908108145386095,
   13648899865260407704]

def terminalConstants : List (List Nat) :=
  [[56971392948291674, 6529889566834057128, 12351691752271381888, 17534431371875637294,
    6215672481047689667, 12917561096631389626, 8117077045244781265, 11392705445391554246,
    7980869059539648603, 8342592727398070895, 12598706612339968649, 14640463469680921020,
    2631864508405068938, 6769287453291648466, 10437499808121220125, 17140876581450999214],
   [13871899794127076084, 2976067163615152067, 17978719630500811223, 1697476385485035771,
    4624025223533863106, 16863109693143641675, 11259424642272530908, 12021139148564917949,
    9937415099130005641, 13718025083052700970, 6859334593920279903, 11330696051914832680,
    7582076360953008071, 8878013572368147473, 2039859730827428652, 7936695803456373494],
   [11613291057305099669, 551761998716281552, 7542914500287710423, 8812857942695644507,
    1690102753676828056, 15311849568606065067, 8830487934816023360, 6755221570880506368,
    6171890655123760836, 12545410498335876764, 9618969152146346251, 17122735410931870227,
    16462569480944052768, 6003165932848932655, 7670196776380585973, 4049877194394998303],
   [17998679892441632083, 2244367866900113196, 9247992278828546609, 9597834897550131057,
    10546074231789401477, 4261339652036653761, 13818713594235094297, 2019448978527439247,
    14018620292311326206, 3735807046906337575, 12052117347272897982, 7812511896467295333,
    9683662180518309794, 15034282915069102610, 9372870626443950502, 15048534190854756183]]

def internalDiagonal : List Nat :=
  [1, 2, 18446744073709551555, 3,
   18446744073709551554, 4, 18446744073709551553, 18446744073709551552,
   6, 18446744073709551551, 7, 18446744073709551550,
   8, 18446744073709551549, 9, 18446744073709551548]

def constantAt (rows : List (List Nat)) (round lane : Nat) : F :=
  ofNat ((rows.getD round []).getD lane 0)

def sbox (x : F) : F :=
  x * x * x

def mat4 : List F → List F
  | [x0, x1, x2, x3] =>
    [2 * x0 + 3 * x1 + x2 + x3,
     x0 + 2 * x1 + 3 * x2 + x3,
     x0 + x1 + 2 * x2 + 3 * x3,
     3 * x0 + x1 + x2 + 2 * x3]
  | other => other

def externalLayer (s : State) : State :=
  let blocks := (List.range (width / 4)).flatMap fun block =>
    mat4 ((s.drop (4 * block)).take 4)
  (List.range width).map fun i =>
    blocks.getD i 0 +
      ((List.range (width / 4)).map fun block =>
        blocks.getD (4 * block + i % 4) 0).foldl (· + ·) 0

def internalLayer (s : State) : State :=
  let sum := s.foldl (· + ·) 0
  (List.range width).map fun i =>
    ofNat (internalDiagonal.getD i 0) * s.getD i 0 + sum

def fullRound (rows : List (List Nat)) (round : Nat) (s : State) : State :=
  externalLayer ((List.range width).map fun i =>
    sbox (s.getD i 0 + constantAt rows round i))

def partialRound (round : Nat) (s : State) : State :=
  internalLayer ((List.range width).map fun i =>
    if i = 0 then sbox (s.getD 0 0 + ofNat (internalConstants.getD round 0))
    else s.getD i 0)

def rounds (step : Nat → State → State) (count : Nat) (s : State) : State :=
  (List.range count).foldl (fun acc r => step r acc) s

def permute (s : State) : State :=
  let s := externalLayer s
  let s := rounds (fullRound initialConstants) halfFullRounds s
  let s := rounds partialRound partialRounds s
  rounds (fullRound terminalConstants) halfFullRounds s

def zeroState : State := List.replicate width 0

def absorbBlock (s : State) (block : List F) : State :=
  permute ((List.range width).map fun i => s.getD i 0 + block.getD i 0)

private theorem rounds_length (step : Nat → State → State)
    (lengths : ∀ index state, (step index state).length = width)
    (indices : List Nat) (state : State) (fixed : state.length = width) :
    (indices.foldl (fun state index => step index state) state).length = width := by
  induction indices generalizing state with
  | nil => exact fixed
  | cons index rest inductionHypothesis => exact inductionHypothesis _ (lengths index state)

theorem permute_length (state : State) : (permute state).length = width := by
  unfold permute rounds
  apply rounds_length
  · intro index state
    simp [fullRound, externalLayer]
  · apply rounds_length
    · intro index state
      simp [partialRound, internalLayer]
    · apply rounds_length
      · intro index state
        simp [fullRound, externalLayer]
      · simp [externalLayer]

theorem absorbBlock_length (state : State) (block : List F) :
    (absorbBlock state block).length = width :=
  permute_length _

theorem absorbBlock_zero (state : State) (fixed : state.length = width)
    {block : List F} (zeros : ∀ index, block.getD index 0 = 0) :
    absorbBlock state block = permute state := by
  unfold absorbBlock
  apply congrArg permute
  have values : (List.range width).map (fun index => state.getD index 0 + block.getD index 0) =
      (List.range width).map (fun index => state.getD index 0) := by
    apply List.map_congr_left
    intro index _
    rw [zeros]
    simp
  rw [values, ← fixed]
  apply List.ext_getElem
  · simp
  · intro index _bound bound
    simp only [List.getElem_map, List.getElem_range]
    exact (List.getElem_eq_getD 0).symm

def absorbBlocksFast : Nat → State → List F → State
  | 0, state, _ => state
  | count + 1, state, input =>
      absorbBlocksFast count
        (absorbBlock state (input.take rate)) (input.drop rate)

theorem absorbBlocksFast_eq_indexed (count : Nat) (state : State)
    (input : List F) :
    absorbBlocksFast count state input =
      ((List.range count).map fun block =>
        (input.drop (block * rate)).take rate).foldl absorbBlock state := by
  induction count generalizing state input with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [absorbBlocksFast, inductionHypothesis]
      rw [List.range_succ_eq_map]
      simp only [List.map_cons, List.foldl_cons, Nat.zero_mul, List.drop_zero,
        List.map_map]
      congr 1
      apply List.map_congr_left
      intro block _member
      congr 1
      rw [List.drop_drop]
      congr 1
      simp [Nat.succ_mul, Nat.add_comm]

/-- The blocks of `rate` words at the start of `input`, without a last short block. -/
def fullBlocks {Alpha : Type} (input : List Alpha) : List (List Alpha) :=
  (List.range (input.length / rate)).map fun block => (input.drop (block * rate)).take rate

/-- The words after the full blocks of `input`, fewer than `rate`. -/
def lastWords {Alpha : Type} (input : List Alpha) : List Alpha :=
  input.drop (input.length / rate * rate)

/-- `10*` padding (`decisions/poseidon2-p64-instance.md`): the input, the word 1, then zeros up
to a multiple of `rate`. -/
def pad (input : List F) : List F :=
  input ++ 1 :: List.replicate (rate - 1 - input.length % rate) 0

/-- The blocks that `hash` absorbs. -/
def blocks (input : List F) : List (List F) :=
  fullBlocks (pad input)

/-- Matches `neo_ccs::crypto::poseidon2::poseidon2_hash`. -/
def hash (input : List F) : List F :=
  ((blocks input).foldl absorbBlock zeroState).take digestLen

/-- `hash` with one absorb of the last words and the word 1: the zeros of `pad` add nothing. -/
def hashFast (input : List F) : List F :=
  (absorbBlock (absorbBlocksFast (input.length / rate) zeroState input)
    (lastWords input ++ [1])).take digestLen

private theorem rate_eq : rate = 12 := rfl

theorem pad_length (input : List F) :
    (pad input).length = (input.length / rate + 1) * rate := by
  simp only [pad, List.length_append, List.length_cons, List.length_replicate, rate_eq]
  omega

private theorem pad_blockCount (input : List F) :
    (pad input).length / rate = input.length / rate + 1 := by
  rw [pad_length, Nat.mul_div_cancel _ (by decide)]

private theorem flatten_chunks {Alpha : Type} (count : Nat) (input : List Alpha)
    (covered : input.length ≤ count * rate) :
    ((List.range count).map fun block => (input.drop (block * rate)).take rate).flatten =
      input := by
  induction count generalizing input with
  | zero =>
      rw [Nat.zero_mul] at covered
      rw [List.eq_nil_of_length_eq_zero (Nat.le_zero.mp covered)]
      rfl
  | succ count inductionHypothesis =>
      have rest := inductionHypothesis (input.drop rate) (by
        rw [List.length_drop]
        rw [Nat.succ_mul] at covered
        omega)
      rw [List.range_succ_eq_map, List.map_cons, List.map_map, List.flatten_cons, Nat.zero_mul,
        List.drop_zero]
      conv => rhs; rw [← List.take_append_drop rate input]
      rw [← rest]
      congr 2
      apply List.map_congr_left
      intro block _member
      simp only [Function.comp_apply, List.drop_drop, Nat.succ_mul]
      rw [Nat.add_comm]

/-- The padding is injective: two inputs with one padded word list are equal. -/
theorem pad_injective {left right : List F} (equal : pad left = pad right) : left = right := by
  have strip : ∀ input : List F,
      (pad input).reverse.dropWhile (fun word => word == 0) = 1 :: input.reverse := by
    intro input
    simp only [pad, List.reverse_append, List.reverse_cons, List.reverse_replicate,
      List.append_assoc, List.singleton_append]
    rw [List.dropWhile_append_of_pos (fun word member => by
      rw [List.eq_of_mem_replicate member]
      rfl)]
    simp only [List.dropWhile_cons, show ((1 : F) == 0) = false from by decide,
      Bool.false_eq_true, ↓reduceIte]
  have reversed := (strip left).symm.trans
    ((congrArg (fun padded => padded.reverse.dropWhile (fun word => word == 0)) equal).trans
      (strip right))
  exact List.reverse_inj.mp (List.cons.inj reversed).2

/-- The blocks of `hash` are the padded words. -/
theorem blocks_flatten (input : List F) : (blocks input).flatten = pad input :=
  flatten_chunks _ _ (Nat.le_of_eq (by rw [pad_blockCount, pad_length]))

/-- Two inputs with one block list are equal, so `hash` absorbs different blocks for different
inputs. -/
theorem blocks_injective {left right : List F} (equal : blocks left = blocks right) :
    left = right := by
  apply pad_injective
  rw [← blocks_flatten left, ← blocks_flatten right, equal]

/-- Each block of `hash` has `rate` words. -/
theorem blocks_length {input block : List F} (member : block ∈ blocks input) :
    block.length = rate := by
  simp only [blocks, fullBlocks, List.mem_map, List.mem_range] at member
  obtain ⟨index, inside, rfl⟩ := member
  rw [pad_blockCount] at inside
  simp only [List.length_take, List.length_drop, pad_length]
  rw [rate_eq] at inside ⊢
  have : (index + 1) * 12 ≤ (input.length / 12 + 1) * 12 := Nat.mul_le_mul_right _ inside
  omega

/-- `hash` absorbs at least one block. -/
theorem blocks_ne_nil (input : List F) : blocks input ≠ [] := by
  simp [blocks, fullBlocks, pad_blockCount]

private theorem absorbBlock_append_zeros (state : State) (block : List F) (count : Nat) :
    absorbBlock state (block ++ List.replicate count 0) = absorbBlock state block := by
  unfold absorbBlock
  congr 1
  apply List.map_congr_left
  intro lane _member
  congr 1
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD]
  by_cases inside : lane < block.length
  · rw [List.getElem?_append_left inside]
  · have outside := Nat.le_of_not_lt inside
    rw [List.getElem?_append_right outside, List.getElem?_eq_none outside,
      List.getElem?_replicate]
    split <;> rfl

/-- `hash` absorbs the full blocks of the input, then its last words and the word 1 in one
block. -/
theorem hash_eq_last (input : List F) :
    hash input = (absorbBlock ((fullBlocks input).foldl absorbBlock zeroState)
      (lastWords input ++ [1])).take digestLen := by
  have below := Nat.div_mul_le_self input.length rate
  have split : blocks input = fullBlocks input ++
      [lastWords input ++ 1 :: List.replicate (rate - 1 - input.length % rate) 0] := by
    unfold blocks fullBlocks
    rw [pad_blockCount, List.range_succ, List.map_append, List.map_singleton]
    congr 1
    · apply List.map_congr_left
      intro block member
      have inside : (block + 1) * rate ≤ input.length := by
        have := List.mem_range.mp member
        rw [rate_eq] at this below ⊢
        omega
      rw [pad, List.drop_append_of_le_length (by rw [Nat.succ_mul] at inside; omega),
        List.take_append_of_le_length (by rw [List.length_drop, Nat.succ_mul] at *; omega)]
    · rw [pad, lastWords, List.drop_append_of_le_length below, List.take_of_length_le]
      simp only [List.length_append, List.length_drop, List.length_cons, List.length_replicate,
        rate_eq] at below ⊢
      omega
  rw [hash, split, List.foldl_append, List.foldl_cons, List.foldl_nil,
    show lastWords input ++ 1 :: List.replicate (rate - 1 - input.length % rate) 0 =
      (lastWords input ++ [1]) ++ List.replicate (rate - 1 - input.length % rate) 0 by simp,
    absorbBlock_append_zeros]

theorem hash_length (input : List F) : (hash input).length = digestLen := by
  rw [hash_eq_last, List.length_take, absorbBlock_length]
  decide

@[csimp] theorem hash_eq_hashFast : @hash = @hashFast := by
  funext input
  rw [hash_eq_last, hashFast, absorbBlocksFast_eq_indexed]
  rfl

theorem constant_table_shape :
    initialConstants.length = halfFullRounds ∧
    terminalConstants.length = halfFullRounds ∧
    internalConstants.length = partialRounds ∧
    internalDiagonal.length = width ∧
    (∀ row ∈ initialConstants, row.length = width) ∧
    (∀ row ∈ terminalConstants, row.length = width) := by
  decide

/-- The internal matrix is `J + diag(D)` with the `D` of `decisions/poseidon2-p64-instance.md`. -/
theorem internalDiagonal_signed :
    internalDiagonal.map ofNat = [1, 2, -2, 3, -3, 4, -4, -5, 6, -6, 7, -7, 8, -8, 9, -9] := by
  decide

end NightstreamFPrime.Spec.Poseidon2
