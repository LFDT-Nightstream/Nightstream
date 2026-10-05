import NightstreamFPrime.Spec.Algebra

/-!
Owns the reference Poseidon2 permutation and sponge hash of the protocol:
width 16, rate 12, capacity 4, S-box `x⁷`, 4 initial + 22 partial + 4 terminal
rounds over Goldilocks, with the plonky3 external layer (`M₄` on each block,
then the circulant sum) and the internal layer `diag(v) + J`. The round-constant
table is protocol authority; Rust is checked against it by test vectors.

Capacity 4 (256 bits) and a 4-word digest give 128-bit collision security.
Plonky3 `poseidon2_round_numbers_128` checks 8 full and 22 partial rounds for
width 16 with `x⁷` over Goldilocks.

Provenance: constants are `neo_ccs::crypto::poseidon2_goldilocks::round_constants()`
(plonky3 `new_from_rng_128` draw order from the fixed ChaCha8 seed); the
permutation itself is written fresh from the plonky3 `p3-poseidon2 0.5.3` /
`p3-goldilocks 0.5.3` definitions.
-/

namespace NightstreamFPrime.Spec.Poseidon2

def width : Nat := 16
def rate : Nat := 12
def digestLen : Nat := 4
def halfFullRounds : Nat := 4
def partialRounds : Nat := 22

/-- Permutation state: exactly `width` lanes, materialized so that executable
evaluation is linear in the round count. -/
abbrev State := List F

def ofNat (n : Nat) : F := ⟨n % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

/-- Round constants for the four initial full rounds, sixteen lanes each. -/
def initialConstants : List (List Nat) :=
  [[15504881536434223753, 2212164856944708396, 1885257220781225929, 17531637481572944510,
    16769640728293682348, 445908668462176974, 1308472042479836079, 17465001500823438575,
    1922033642430128704, 2657514617275794404, 17238706657248448792, 7348277157222259646,
    10777112892842897939, 1771261721914735482, 9409693344407549465, 16619731096074499912],
   [1922036059108268922, 2681686362645798986, 12432722052283819565, 2826979200512189741,
    5080805286413226676, 16827966425431695029, 9196241087337510154, 2350771591198563053,
    2989012136977041732, 4359939046747977080, 16089932437481530267, 6601984573273403484,
    13005272261058756234, 17128237926164276121, 8240789415616872849, 8676316357341090631],
   [16452552554259143025, 17874550554210084887, 3031715677034868367, 18215520516675091549,
    18186005068527139405, 11138995707668647102, 15098195648006184282, 2025927025270509469,
    9957669227203243937, 11554336633716867616, 9729067570563846225, 4239770196713589268,
    4390607796152185292, 17647511975646925721, 7671337049037340193, 4209452938403606590],
   [6593973666654839090, 8390781086037206386, 7324343054784993307, 17780748563735894140,
    15974082699116886783, 13213371256836887512, 7312926934405385057, 10393853239698468203,
    2710107888698774842, 2801523468128575786, 15894340394120906162, 13510783799941644149,
    7917164295139071913, 13839801071899888959, 6672989303670154677, 4519956214037211385]]

/-- Round constants for the twenty-two partial rounds: lane zero only. -/
def internalConstants : List Nat :=
  [3785078240232899765, 13753534232687663059, 15164579152346391244,
   7188840087678967607, 14466302407008998516, 13002203018489042439,
   6216879054540921071, 8639253411382064186, 4983986534873630152,
   2915380551046574730, 7837744426713061492, 10524728973359902812,
   9778451922853050075, 13310295482970625397, 10551647698332907197,
   8509293821221348719, 4319821126274335035, 1417436523646501240,
   447570613036226838, 16267896682506996530, 14710998805405468930,
   3269000544927978862]

/-- Round constants for the four terminal full rounds, sixteen lanes each. -/
def terminalConstants : List (List Nat) :=
  [[7482194551502142718, 3471957803411196592, 8846669050136897522, 4431017908497072775,
    14382646627736292998, 15636596632746594248, 14521990061611210983, 4351091752509404379,
    14119848206371842921, 528205008764728916, 15379406877060454284, 13572057177474709483,
    780214424511389757, 10591233664360718633, 1849508423779478786, 7345390174439848870],
   [14580881241235634775, 8777273265976228774, 1758781345554053863, 9701442189086298420,
    15685565327448534444, 5672331717709479627, 7675233227955155107, 8852669876726984824,
    1218164705289579190, 13224810758441726241, 557024023478380004, 3923346290699247117,
    4196774554581694822, 16262909137268628555, 6531098975686849205, 538070144030448988],
   [16157559630818414765, 3330859574359708906, 13312877616183741059, 15699706004066187344,
    2181468677625794151, 12293285838251430515, 17109377825740910727, 11746958123598489878,
    7654965179475073269, 15178922343313110770, 14240408894833620294, 4224192993509995210,
    13093043512401634422, 16636225261759530156, 13489384640167770266, 8105602514957866176],
   [13910460326211973254, 13010277363854955001, 8570865802160232388, 14830753997593808291,
    16178721091989175194, 10926358020058189153, 8413180834413067310, 1124528750616792490,
    16054199595598201491, 729673474029476808, 1545919217216143455, 15484244716222357662,
    9149791094276206087, 3342519128264714984, 14246551315881547762, 6145097356981870399]]

/-- Diagonal `v` of the internal layer `diag(v) + J`
(`MATRIX_DIAG_16_GOLDILOCKS`: −2, 1, 2, 1/2, 3, 4, −1/2, −3, −4, 1/2³, 1/2⁴,
1/2⁵, −1/2³, −1/2⁴, −1/2⁵, 1/2³²). -/
def internalDiagonal : List Nat :=
  [18446744069414584319, 1, 2,
   9223372034707292161, 3, 4,
   9223372034707292160, 18446744069414584318, 18446744069414584317,
   16140901060737761281, 17293822565076172801, 17870283317245378561,
   2305843008676823040, 1152921504338411520, 576460752169205760,
   18446744065119617026]

def constantAt (rows : List (List Nat)) (round lane : Nat) : F :=
  ofNat ((rows.getD round []).getD lane 0)

/-- `x⁷`. -/
def sbox (x : F) : F :=
  let x2 := x * x
  let x4 := x2 * x2
  x4 * x2 * x

/-- `M₄ = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]` on one 4-lane block. -/
def mat4 : List F → List F
  | [x0, x1, x2, x3] =>
    [2 * x0 + 3 * x1 + x2 + x3,
     x0 + 2 * x1 + 3 * x2 + x3,
     x0 + x1 + 2 * x2 + 3 * x3,
     3 * x0 + x1 + x2 + 2 * x3]
  | other => other

/-- External linear layer: `M₄` on each four-lane block, then every lane
adds the sum of the block lanes congruent to it mod 4. -/
def externalLayer (s : State) : State :=
  let blocks := (List.range (width / 4)).flatMap fun block =>
    mat4 ((s.drop (4 * block)).take 4)
  (List.range width).map fun i =>
    blocks.getD i 0 +
      ((List.range (width / 4)).map fun block =>
        blocks.getD (4 * block + i % 4) 0).foldl (· + ·) 0

/-- Internal linear layer: `s_i ← v_i · s_i + Σ_j s_j`. -/
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

/-- The Poseidon2 permutation. -/
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

/-- Every permutation output has exactly `width` lanes. -/
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

/-- Absorbing an all-zero block into a full-width state is a plain
permutation. -/
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

/-- Tail-recursive absorption of the first `count` rate-sized blocks.

Unlike the indexed reference expression in `hash`, this function drops only
the next fixed-size block. Its executable cost is linear in the input length.
-/
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

/-- Sponge hash: absorb `rate`-sized chunks, add 1 to lane 0, permute, squeeze
the first `digestLen` lanes. Matches
`neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash`. -/
def hash (input : List F) : List F :=
  let chunks := (List.range ((input.length + rate - 1) / rate)).map
    (fun c => (input.drop (c * rate)).take rate)
  let absorbed := chunks.foldl absorbBlock zeroState
  let padded := permute ((List.range width).map fun i =>
    if i = 0 then absorbed.getD 0 0 + 1 else absorbed.getD i 0)
  padded.take digestLen

/-- Linear executable form of `hash`. The theorem below keeps `hash` as the
semantic authority and installs this definition only for compiled execution.
-/
def hashFast (input : List F) : List F :=
  let blockCount := (input.length + rate - 1) / rate
  let absorbed := absorbBlocksFast blockCount zeroState input
  let padded := permute ((List.range width).map fun i =>
    if i = 0 then absorbed.getD 0 0 + 1 else absorbed.getD i 0)
  padded.take digestLen

@[csimp] theorem hash_eq_hashFast : @hash = @hashFast := by
  funext input
  dsimp only [hash, hashFast]
  rw [absorbBlocksFast_eq_indexed]

theorem constant_table_shape :
    initialConstants.length = halfFullRounds ∧
    terminalConstants.length = halfFullRounds ∧
    internalConstants.length = partialRounds ∧
    internalDiagonal.length = width ∧
    (∀ row ∈ initialConstants, row.length = width) ∧
    (∀ row ∈ terminalConstants, row.length = width) := by
  decide

end NightstreamFPrime.Spec.Poseidon2
