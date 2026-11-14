import NightstreamFPrime.Spec.Algebra

/-! The width-16 Poseidon2 instance over `fieldModulus` (`decisions/poseidon2-p64-instance.md`).
`scripts/poseidon2/derive.py` derives the round constants and the internal diagonal and writes them
to `scripts/poseidon2/params.json`. The literals below are that file's width-16 instance, as
canonical residues. The layers follow plonky3 `p3-poseidon2 0.5.3`. -/

namespace NightstreamFPrime.Spec.Poseidon2

def width : Nat := 16
def rate : Nat := 12
def digestLen : Nat := 4
def halfFullRounds : Nat := 4
def partialRounds : Nat := 22

abbrev State := List F

def ofNat (n : Nat) : F := ⟨n % fieldModulus, Nat.mod_lt _ (by decide)⟩

def initialConstants : List (List Nat) :=
  [[12636804980880967846, 16736532232107165500, 4836694764141735874, 4639536440338654302,
    15582121682392026614, 9230548520109342161, 6512674115442529763, 15623341923167353858,
    1672953842270242786, 11188453102193896200, 7564734281139603135, 7054913528056871010,
    1830995645382265875, 17768031537859440065, 14218342480096191686, 6340347240413876462],
   [11772294118312790256, 3542380241280516415, 5513546723254740663, 138615007802823419,
    5647627248142650917, 10078658854564970516, 4162310934199293441, 10985814646094360523,
    5508986491302979157, 12804574419876305617, 16809457995335845009, 10083824294548122199,
    7091151999514037588, 2813780550927568854, 1505070123659706260, 15705593091285152329],
   [16346592572221230289, 5917330695164472137, 10183339126308925573, 12014797672663882253,
    9279946522805269125, 15657808947398157350, 8560220515539453181, 5866161307479662148,
    8262117129289574491, 4491087910169738163, 14325169830526276512, 10135308560825186894,
    5228070396315423026, 13336376921187837712, 2455568362154084154, 990495920709174079],
   [17055595709904449881, 15873439004876081665, 15766459368044885739, 11857706734817241399,
    16051049577993660372, 7992040049950974682, 13545545069786182126, 14358843605013273411,
    11318775708734409385, 3710328625670043923, 15408102934752366886, 7830543074676164584,
    16038198064567788883, 9063004124435584042, 17444701252005907991, 18180745309129606155]]

def internalConstants : List Nat :=
  [248129226802411505, 13743530486470815716, 5900121121239796588,
   11804998099291340661, 8786293660148775452, 6824681443996726608,
   15227704540163158519, 8367187098120952855, 16003535748082887266,
   13931382347329436468, 11864186209076043354, 10908150843957377599,
   15309098979312698196, 7406323311710872823, 16565046105017585879,
   18034087188624120274, 12595505982365297677, 14640114969567447657,
   13247398736086271596, 212740773279947709, 278419132071503090,
   8991094185839820623]

def terminalConstants : List (List Nat) :=
  [[13299466096494452311, 12769097280806939949, 12980794758999292759, 12485718232377110686,
    13508754101685200609, 5975847699931589519, 3809916015033307623, 17141578096760893684,
    2149922062342650665, 16251838837471430008, 1764164040917344789, 12746115122662023609,
    4260080277820885026, 10788845032470832763, 14353832261393182355, 4756649606211735290],
   [4154961343765510954, 15940164003352720504, 11766889230861649991, 6257087493022936357,
    4778990173478170575, 18415619616893099691, 12597246528008666129, 14211568158000976339,
    4767115697445068110, 12570480850519731912, 4665827349589571727, 4774912573617945815,
    186562210816135733, 14008215308940752777, 15013698295759921959, 11939373852127404977],
   [8839746612011749247, 7109953674418109544, 15161625799949739060, 10476369374187095481,
    14819833511683759954, 536638751129487753, 17509512622715781686, 9414118451560315968,
    2391520536790699897, 13866177019205733727, 14844228349995260905, 11938404480018823340,
    4752806354978272429, 4329905090612702030, 10502326865601337258, 5633833814161874090],
   [11635821469199330134, 1037445992391313205, 156723973773465148, 8936286655370992570,
    14312401287329186061, 13241755096870787599, 12326958112711116003, 8544402462410608246,
    8740390829074643229, 11244854067100363510, 4533544813859712987, 18401498054319815042,
    6772245856459278760, 7992845694003848145, 1563664802427926635, 13262978740168500374]]

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
