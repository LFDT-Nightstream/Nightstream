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
def partialRounds : Nat := 22

abbrev State := List F

def ofNat (n : Nat) : F := ⟨n % fieldModulus, Nat.mod_lt _ (by decide)⟩

def initialConstants : List (List Nat) :=
  [[1579613653969377219, 15509510893087893340, 10090715174060125222, 5625716564419252202,
    3006005019077469174, 18314693207014427912, 10170571510627764565, 2027625550790675754,
    3983470257916202094, 3423470109396435354, 3450488264035752368, 3151070045406026687,
    13462781804006123550, 13288575772684627216, 13745549378090937523, 7780139165529418388],
   [11699769881923825909, 4349260652212695338, 8228348357294662898, 12037145371429260559,
    7743383997471469184, 7842006993218978420, 18369471830829996810, 10157815501749214163,
    11128005290408113568, 12272917402811871099, 106820976709770637, 6178857985076893256,
    14036787039817180193, 1659458828486927362, 15938183887757779615, 10060334973073197601],
   [12437259172286625042, 3143835545170851304, 14361575850007152549, 11810209965918630788,
    17667700303515849362, 13451932870729425780, 17496644301504593329, 1778669202480014256,
    11124488067381312952, 14273583775934080755, 13257379154210125715, 15182093233826554989,
    5343815959848316783, 5696193414571279556, 6089787229781325813, 3717595218375990594],
   [12355842859741995537, 14160921043125059199, 13682913322779819091, 12343504462597575239,
    14249643564311509087, 15849758561085927402, 12186564686511703262, 690520822232718983,
    684200944081655198, 6008958449718781068, 3098079696536837500, 12083506578988585239,
    6489163969976034646, 1186411428667285248, 11192474480021157091, 17095617549195818058]]

def internalConstants : List Nat :=
  [2949845317987848448, 6935557324168256069, 2055390472619727748,
   12399309956582731760, 3088058410657255303, 8790959064194804235,
   4582664136559626757, 14821962908959377092, 2002685046256118388,
   15689593639089632368, 10512046881800463697, 11973738501993105496,
   13009070935614847359, 12908033183314664096, 9530849385991780809,
   17528314872631229800, 17869527288639567155, 7829055113315023688,
   3256047469251174543, 3015723851705964382, 5177282273995529875,
   16035152896984012190]

def terminalConstants : List (List Nat) :=
  [[15766898731627159621, 6714370142978148338, 5906664861467821209, 2197120128337974909,
    9400421283898153864, 16740473708727660723, 7287335615015370599, 12399217337488717232,
    12655834778591227242, 7516917516498792203, 17291898461659961460, 3483661079725019706,
    18038566855851505564, 6166154478921479976, 8824538844032254328, 17138720803294251067],
   [4158286797471309733, 1589482450994236426, 4836112672767737407, 6108317238076822824,
    11174425919404088598, 2755155124184744585, 13202523778257173956, 6643835433125335951,
    2748965361441888477, 14297205492001640506, 12595977886586773319, 257624298448087449,
    2744326799038764281, 9097808034015176417, 5523099006222423939, 7256201493271212720],
   [9738621014989498753, 237412858668409460, 6405904454922718574, 16343835012250148340,
    9342245617122491936, 824922755340087111, 5398764608665671277, 16030622638137836774,
    4921899302518525668, 10676667276121395674, 16304445044992359788, 587812931475326787,
    7627897473303434798, 2633542348821978290, 5714662808802647771, 15665352327749112015],
   [8041781683128153420, 13172746155287932121, 6611413049825339003, 9076340683484238287,
    12295614633782496176, 11414162089917699064, 13658053783093222498, 16216654076620539925,
    17782598995085730495, 18362903988365127450, 11596821426909532371, 13499421621575350806,
    15256312875610823807, 1558929959921731273, 17567284148478398281, 4615393526180652407]]

def internalDiagonal : List Nat :=
  [1, 2, 18446744073709551555, 3,
   18446744073709551554, 4, 18446744073709551553, 18446744073709551552,
   6, 18446744073709551551, 7, 18446744073709551550,
   8, 18446744073709551549, 9, 18446744073709551548]

def constantAt (rows : List (List Nat)) (round lane : Nat) : F :=
  ofNat ((rows.getD round []).getD lane 0)

def sbox (x : F) : F :=
  let x2 := x * x
  let x4 := x2 * x2
  x4 * x2 * x

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
