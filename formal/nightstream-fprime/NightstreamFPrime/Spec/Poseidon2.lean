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

/-- Matches `neo_ccs::crypto::poseidon2::poseidon2_hash`. -/
def hash (input : List F) : List F :=
  let chunks := (List.range ((input.length + rate - 1) / rate)).map
    (fun c => (input.drop (c * rate)).take rate)
  let absorbed := chunks.foldl absorbBlock zeroState
  let padded := permute ((List.range width).map fun i =>
    if i = 0 then absorbed.getD 0 0 + 1 else absorbed.getD i 0)
  padded.take digestLen

def hashFast (input : List F) : List F :=
  let blockCount := (input.length + rate - 1) / rate
  let absorbed := absorbBlocksFast blockCount zeroState input
  let padded := permute ((List.range width).map fun i =>
    if i = 0 then absorbed.getD 0 0 + 1 else absorbed.getD i 0)
  padded.take digestLen

theorem hash_length (input : List F) : (hash input).length = digestLen := by
  unfold hash
  dsimp only
  rw [List.length_take, permute_length]
  decide

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

/-- The internal matrix is `J + diag(D)` with the `D` of `decisions/poseidon2-p64-instance.md`. -/
theorem internalDiagonal_signed :
    internalDiagonal.map ofNat = [1, 2, -2, 3, -3, 4, -4, -5, 6, -6, 7, -7, 8, -8, 9, -9] := by
  decide

end NightstreamFPrime.Spec.Poseidon2
