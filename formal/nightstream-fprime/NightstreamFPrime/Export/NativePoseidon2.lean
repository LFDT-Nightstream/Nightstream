import NightstreamFPrime.Export.NativePoseidon2RoundCore

namespace NightstreamFPrime.Export.NativePoseidon2

open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open Fin.CommRing

namespace State64

private def initialConstant0 : State64 :=
  ⟨12636804980880967846, 16736532232107165500, 4836694764141735874, 4639536440338654302,
    15582121682392026614, 9230548520109342161, 6512674115442529763, 15623341923167353858,
    1672953842270242786, 11188453102193896200, 7564734281139603135, 7054913528056871010,
    1830995645382265875, 17768031537859440065, 14218342480096191686, 6340347240413876462, by decide⟩
private def initialConstant1 : State64 :=
  ⟨11772294118312790256, 3542380241280516415, 5513546723254740663, 138615007802823419,
    5647627248142650917, 10078658854564970516, 4162310934199293441, 10985814646094360523,
    5508986491302979157, 12804574419876305617, 16809457995335845009, 10083824294548122199,
    7091151999514037588, 2813780550927568854, 1505070123659706260, 15705593091285152329, by decide⟩
private def initialConstant2 : State64 :=
  ⟨16346592572221230289, 5917330695164472137, 10183339126308925573, 12014797672663882253,
    9279946522805269125, 15657808947398157350, 8560220515539453181, 5866161307479662148,
    8262117129289574491, 4491087910169738163, 14325169830526276512, 10135308560825186894,
    5228070396315423026, 13336376921187837712, 2455568362154084154, 990495920709174079, by decide⟩
private def initialConstant3 : State64 :=
  ⟨17055595709904449881, 15873439004876081665, 15766459368044885739, 11857706734817241399,
    16051049577993660372, 7992040049950974682, 13545545069786182126, 14358843605013273411,
    11318775708734409385, 3710328625670043923, 15408102934752366886, 7830543074676164584,
    16038198064567788883, 9063004124435584042, 17444701252005907991, 18180745309129606155, by decide⟩

private def partialConstant00 : UInt64 := 248129226802411505
private def partialConstant01 : UInt64 := 13743530486470815716
private def partialConstant02 : UInt64 := 5900121121239796588
private def partialConstant03 : UInt64 := 11804998099291340661
private def partialConstant04 : UInt64 := 8786293660148775452
private def partialConstant05 : UInt64 := 6824681443996726608
private def partialConstant06 : UInt64 := 15227704540163158519
private def partialConstant07 : UInt64 := 8367187098120952855
private def partialConstant08 : UInt64 := 16003535748082887266
private def partialConstant09 : UInt64 := 13931382347329436468
private def partialConstant10 : UInt64 := 11864186209076043354
private def partialConstant11 : UInt64 := 10908150843957377599
private def partialConstant12 : UInt64 := 15309098979312698196
private def partialConstant13 : UInt64 := 7406323311710872823
private def partialConstant14 : UInt64 := 16565046105017585879
private def partialConstant15 : UInt64 := 18034087188624120274
private def partialConstant16 : UInt64 := 12595505982365297677
private def partialConstant17 : UInt64 := 14640114969567447657
private def partialConstant18 : UInt64 := 13247398736086271596
private def partialConstant19 : UInt64 := 212740773279947709
private def partialConstant20 : UInt64 := 278419132071503090
private def partialConstant21 : UInt64 := 8991094185839820623

private def terminalConstant0 : State64 :=
  ⟨13299466096494452311, 12769097280806939949, 12980794758999292759, 12485718232377110686,
    13508754101685200609, 5975847699931589519, 3809916015033307623, 17141578096760893684,
    2149922062342650665, 16251838837471430008, 1764164040917344789, 12746115122662023609,
    4260080277820885026, 10788845032470832763, 14353832261393182355, 4756649606211735290, by decide⟩
private def terminalConstant1 : State64 :=
  ⟨4154961343765510954, 15940164003352720504, 11766889230861649991, 6257087493022936357,
    4778990173478170575, 18415619616893099691, 12597246528008666129, 14211568158000976339,
    4767115697445068110, 12570480850519731912, 4665827349589571727, 4774912573617945815,
    186562210816135733, 14008215308940752777, 15013698295759921959, 11939373852127404977, by decide⟩
private def terminalConstant2 : State64 :=
  ⟨8839746612011749247, 7109953674418109544, 15161625799949739060, 10476369374187095481,
    14819833511683759954, 536638751129487753, 17509512622715781686, 9414118451560315968,
    2391520536790699897, 13866177019205733727, 14844228349995260905, 11938404480018823340,
    4752806354978272429, 4329905090612702030, 10502326865601337258, 5633833814161874090, by decide⟩
private def terminalConstant3 : State64 :=
  ⟨11635821469199330134, 1037445992391313205, 156723973773465148, 8936286655370992570,
    14312401287329186061, 13241755096870787599, 12326958112711116003, 8544402462410608246,
    8740390829074643229, 11244854067100363510, 4533544813859712987, 18401498054319815042,
    6772245856459278760, 7992845694003848145, 1563664802427926635, 13262978740168500374, by decide⟩

private def FullConstantMatch (constants : State64) (rows : List (List Nat))
    (round : Nat) : Prop :=
  constants.x0.denote = Poseidon2.constantAt rows round 0 ∧
  constants.x1.denote = Poseidon2.constantAt rows round 1 ∧
  constants.x2.denote = Poseidon2.constantAt rows round 2 ∧
  constants.x3.denote = Poseidon2.constantAt rows round 3 ∧
  constants.x4.denote = Poseidon2.constantAt rows round 4 ∧
  constants.x5.denote = Poseidon2.constantAt rows round 5 ∧
  constants.x6.denote = Poseidon2.constantAt rows round 6 ∧
  constants.x7.denote = Poseidon2.constantAt rows round 7 ∧
  constants.x8.denote = Poseidon2.constantAt rows round 8 ∧
  constants.x9.denote = Poseidon2.constantAt rows round 9 ∧
  constants.x10.denote = Poseidon2.constantAt rows round 10 ∧
  constants.x11.denote = Poseidon2.constantAt rows round 11 ∧
  constants.x12.denote = Poseidon2.constantAt rows round 12 ∧
  constants.x13.denote = Poseidon2.constantAt rows round 13 ∧
  constants.x14.denote = Poseidon2.constantAt rows round 14 ∧
  constants.x15.denote = Poseidon2.constantAt rows round 15

private theorem fullRound64_denote_at (rows : List (List Nat)) (round : Nat)
    (constants state : State64)
    (constantMatch : FullConstantMatch constants rows round) :
    (fullRound64 state constants).denote = Poseidon2.fullRound rows round state.denote := by
  rcases constantMatch with ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩
  rw [fullRound64_denote]
  unfold Poseidon2.fullRound
  apply congrArg Poseidon2.externalLayer
  simp [Poseidon2.width, denote, List.range_succ, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15]

private theorem partialRound64_denote_at (round : Nat) (constant : UInt64)
    (constantCanonical : constant.toNat < fieldModulus)
    (constantMatch : constant.denote =
      Poseidon2.ofNat (Poseidon2.internalConstants.getD round 0))
    (state : State64) :
    (partialRound64 state constant).denote = Poseidon2.partialRound round state.denote := by
  rw [partialRound64_denote constant constantCanonical state, constantMatch]
  unfold Poseidon2.partialRound
  apply congrArg Poseidon2.internalLayer
  simp [Poseidon2.width, denote, List.range_succ]

private theorem initialRound0_denote (s : State64) :
    (fullRound64 s initialConstant0).denote = Poseidon2.fullRound Poseidon2.initialConstants 0 s.denote :=
  fullRound64_denote_at _ 0 _ s (by unfold FullConstantMatch; decide)
private theorem initialRound1_denote (s : State64) :
    (fullRound64 s initialConstant1).denote = Poseidon2.fullRound Poseidon2.initialConstants 1 s.denote :=
  fullRound64_denote_at _ 1 _ s (by unfold FullConstantMatch; decide)
private theorem initialRound2_denote (s : State64) :
    (fullRound64 s initialConstant2).denote = Poseidon2.fullRound Poseidon2.initialConstants 2 s.denote :=
  fullRound64_denote_at _ 2 _ s (by unfold FullConstantMatch; decide)
private theorem initialRound3_denote (s : State64) :
    (fullRound64 s initialConstant3).denote = Poseidon2.fullRound Poseidon2.initialConstants 3 s.denote :=
  fullRound64_denote_at _ 3 _ s (by unfold FullConstantMatch; decide)

private theorem partialRound00_denote (s : State64) :
    (partialRound64 s partialConstant00).denote = Poseidon2.partialRound 0 s.denote :=
  partialRound64_denote_at 0 _ (by decide) (by decide) s
private theorem partialRound01_denote (s : State64) :
    (partialRound64 s partialConstant01).denote = Poseidon2.partialRound 1 s.denote :=
  partialRound64_denote_at 1 _ (by decide) (by decide) s
private theorem partialRound02_denote (s : State64) :
    (partialRound64 s partialConstant02).denote = Poseidon2.partialRound 2 s.denote :=
  partialRound64_denote_at 2 _ (by decide) (by decide) s
private theorem partialRound03_denote (s : State64) :
    (partialRound64 s partialConstant03).denote = Poseidon2.partialRound 3 s.denote :=
  partialRound64_denote_at 3 _ (by decide) (by decide) s
private theorem partialRound04_denote (s : State64) :
    (partialRound64 s partialConstant04).denote = Poseidon2.partialRound 4 s.denote :=
  partialRound64_denote_at 4 _ (by decide) (by decide) s
private theorem partialRound05_denote (s : State64) :
    (partialRound64 s partialConstant05).denote = Poseidon2.partialRound 5 s.denote :=
  partialRound64_denote_at 5 _ (by decide) (by decide) s
private theorem partialRound06_denote (s : State64) :
    (partialRound64 s partialConstant06).denote = Poseidon2.partialRound 6 s.denote :=
  partialRound64_denote_at 6 _ (by decide) (by decide) s
private theorem partialRound07_denote (s : State64) :
    (partialRound64 s partialConstant07).denote = Poseidon2.partialRound 7 s.denote :=
  partialRound64_denote_at 7 _ (by decide) (by decide) s
private theorem partialRound08_denote (s : State64) :
    (partialRound64 s partialConstant08).denote = Poseidon2.partialRound 8 s.denote :=
  partialRound64_denote_at 8 _ (by decide) (by decide) s
private theorem partialRound09_denote (s : State64) :
    (partialRound64 s partialConstant09).denote = Poseidon2.partialRound 9 s.denote :=
  partialRound64_denote_at 9 _ (by decide) (by decide) s
private theorem partialRound10_denote (s : State64) :
    (partialRound64 s partialConstant10).denote = Poseidon2.partialRound 10 s.denote :=
  partialRound64_denote_at 10 _ (by decide) (by decide) s
private theorem partialRound11_denote (s : State64) :
    (partialRound64 s partialConstant11).denote = Poseidon2.partialRound 11 s.denote :=
  partialRound64_denote_at 11 _ (by decide) (by decide) s
private theorem partialRound12_denote (s : State64) :
    (partialRound64 s partialConstant12).denote = Poseidon2.partialRound 12 s.denote :=
  partialRound64_denote_at 12 _ (by decide) (by decide) s
private theorem partialRound13_denote (s : State64) :
    (partialRound64 s partialConstant13).denote = Poseidon2.partialRound 13 s.denote :=
  partialRound64_denote_at 13 _ (by decide) (by decide) s
private theorem partialRound14_denote (s : State64) :
    (partialRound64 s partialConstant14).denote = Poseidon2.partialRound 14 s.denote :=
  partialRound64_denote_at 14 _ (by decide) (by decide) s
private theorem partialRound15_denote (s : State64) :
    (partialRound64 s partialConstant15).denote = Poseidon2.partialRound 15 s.denote :=
  partialRound64_denote_at 15 _ (by decide) (by decide) s
private theorem partialRound16_denote (s : State64) :
    (partialRound64 s partialConstant16).denote = Poseidon2.partialRound 16 s.denote :=
  partialRound64_denote_at 16 _ (by decide) (by decide) s
private theorem partialRound17_denote (s : State64) :
    (partialRound64 s partialConstant17).denote = Poseidon2.partialRound 17 s.denote :=
  partialRound64_denote_at 17 _ (by decide) (by decide) s
private theorem partialRound18_denote (s : State64) :
    (partialRound64 s partialConstant18).denote = Poseidon2.partialRound 18 s.denote :=
  partialRound64_denote_at 18 _ (by decide) (by decide) s
private theorem partialRound19_denote (s : State64) :
    (partialRound64 s partialConstant19).denote = Poseidon2.partialRound 19 s.denote :=
  partialRound64_denote_at 19 _ (by decide) (by decide) s
private theorem partialRound20_denote (s : State64) :
    (partialRound64 s partialConstant20).denote = Poseidon2.partialRound 20 s.denote :=
  partialRound64_denote_at 20 _ (by decide) (by decide) s
private theorem partialRound21_denote (s : State64) :
    (partialRound64 s partialConstant21).denote = Poseidon2.partialRound 21 s.denote :=
  partialRound64_denote_at 21 _ (by decide) (by decide) s

private theorem terminalRound0_denote (s : State64) :
    (fullRound64 s terminalConstant0).denote = Poseidon2.fullRound Poseidon2.terminalConstants 0 s.denote :=
  fullRound64_denote_at _ 0 _ s (by unfold FullConstantMatch; decide)
private theorem terminalRound1_denote (s : State64) :
    (fullRound64 s terminalConstant1).denote = Poseidon2.fullRound Poseidon2.terminalConstants 1 s.denote :=
  fullRound64_denote_at _ 1 _ s (by unfold FullConstantMatch; decide)
private theorem terminalRound2_denote (s : State64) :
    (fullRound64 s terminalConstant2).denote = Poseidon2.fullRound Poseidon2.terminalConstants 2 s.denote :=
  fullRound64_denote_at _ 2 _ s (by unfold FullConstantMatch; decide)
private theorem terminalRound3_denote (s : State64) :
    (fullRound64 s terminalConstant3).denote = Poseidon2.fullRound Poseidon2.terminalConstants 3 s.denote :=
  fullRound64_denote_at _ 3 _ s (by unfold FullConstantMatch; decide)

@[inline] private def initialRounds64 (state : State64) : State64 :=
  let state := fullRound64 state initialConstant0
  let state := fullRound64 state initialConstant1
  let state := fullRound64 state initialConstant2
  fullRound64 state initialConstant3

@[inline] private def partialRounds64 (state : State64) : State64 :=
  let state := partialRound64 state partialConstant00
  let state := partialRound64 state partialConstant01
  let state := partialRound64 state partialConstant02
  let state := partialRound64 state partialConstant03
  let state := partialRound64 state partialConstant04
  let state := partialRound64 state partialConstant05
  let state := partialRound64 state partialConstant06
  let state := partialRound64 state partialConstant07
  let state := partialRound64 state partialConstant08
  let state := partialRound64 state partialConstant09
  let state := partialRound64 state partialConstant10
  let state := partialRound64 state partialConstant11
  let state := partialRound64 state partialConstant12
  let state := partialRound64 state partialConstant13
  let state := partialRound64 state partialConstant14
  let state := partialRound64 state partialConstant15
  let state := partialRound64 state partialConstant16
  let state := partialRound64 state partialConstant17
  let state := partialRound64 state partialConstant18
  let state := partialRound64 state partialConstant19
  let state := partialRound64 state partialConstant20
  partialRound64 state partialConstant21

@[inline] private def terminalRounds64 (state : State64) : State64 :=
  let state := fullRound64 state terminalConstant0
  let state := fullRound64 state terminalConstant1
  let state := fullRound64 state terminalConstant2
  fullRound64 state terminalConstant3

private theorem initialRounds64_denote (state : State64) :
    (initialRounds64 state).denote =
      Poseidon2.rounds (Poseidon2.fullRound Poseidon2.initialConstants)
        Poseidon2.halfFullRounds state.denote := by
  simp only [initialRounds64]
  rw [initialRound3_denote, initialRound2_denote, initialRound1_denote,
    initialRound0_denote]
  rfl

private theorem partialRounds64_denote (state : State64) :
    (partialRounds64 state).denote =
      Poseidon2.rounds Poseidon2.partialRound Poseidon2.partialRounds state.denote := by
  simp only [partialRounds64]
  rw [partialRound21_denote, partialRound20_denote, partialRound19_denote,
    partialRound18_denote, partialRound17_denote, partialRound16_denote,
    partialRound15_denote, partialRound14_denote, partialRound13_denote,
    partialRound12_denote, partialRound11_denote, partialRound10_denote,
    partialRound09_denote, partialRound08_denote, partialRound07_denote,
    partialRound06_denote, partialRound05_denote, partialRound04_denote,
    partialRound03_denote, partialRound02_denote, partialRound01_denote,
    partialRound00_denote]
  rfl

private theorem terminalRounds64_denote (state : State64) :
    (terminalRounds64 state).denote =
      Poseidon2.rounds (Poseidon2.fullRound Poseidon2.terminalConstants)
        Poseidon2.halfFullRounds state.denote := by
  simp only [terminalRounds64]
  rw [terminalRound3_denote, terminalRound2_denote, terminalRound1_denote,
    terminalRound0_denote]
  rfl

@[noinline] def permute64 (state : State64) : State64 :=
  terminalRounds64 (partialRounds64 (initialRounds64 (externalLayer64 state)))

theorem permute64_denote (state : State64) :
    (permute64 state).denote = Poseidon2.permute state.denote := by
  rw [permute64, terminalRounds64_denote, partialRounds64_denote,
    initialRounds64_denote, externalLayer64_denote]
  rfl

end State64

@[inline] def ofNat64 (value : Nat) : UInt64 :=
  if value < UInt64.size then
    if value < fieldModulus then UInt64.ofNat value
    else UInt64.ofNat (value - fieldModulus)
  else UInt64.ofNat (value % fieldModulus)

private theorem ofNat64_toNat (value : Nat) :
    (ofNat64 value).toNat = value % fieldModulus := by
  simp only [ofNat64]
  split <;> rename_i sizeBranch
  · split <;> rename_i modulusBranch
    · rw [UInt64.toNat_ofNat_of_lt' sizeBranch,
        Nat.mod_eq_of_lt modulusBranch]
    · have modulusLe : fieldModulus ≤ value := by omega
      have differenceSize : value - fieldModulus < UInt64.size := by omega
      have differenceModulus : value - fieldModulus < fieldModulus := by
        have sizeLtTwice : UInt64.size < 2 * fieldModulus := by decide
        omega
      rw [UInt64.toNat_ofNat_of_lt' differenceSize,
        Nat.mod_eq_sub_mod modulusLe, Nat.mod_eq_of_lt differenceModulus]
  · have residueModulus := Nat.mod_lt value (by decide : 0 < fieldModulus)
    have residueSize : value % fieldModulus < UInt64.size :=
      Nat.lt_trans residueModulus (by decide)
    rw [UInt64.toNat_ofNat_of_lt' residueSize]

theorem ofNat64_canonical (value : Nat) :
    (ofNat64 value).toNat < fieldModulus := by
  rw [ofNat64_toNat]
  exact Nat.mod_lt _ (by decide)

@[simp] theorem ofNat64_denote (value : Nat) :
    (ofNat64 value).denote = Poseidon2.ofNat value := by
  apply Fin.ext
  simp [UInt64.denote, Poseidon2.ofNat, ofNat64_toNat]

private theorem limbBase_eq_radix : Package.limbBase = radix := rfl
private theorem uint64Size_eq_radixSquare : UInt64.size = radix * radix := by decide

private theorem fastLow_denote (value : Nat) (bound : value < UInt64.size) :
    (low64 (UInt64.ofNat value)).denote =
      Poseidon2.ofNat (value % Package.limbBase) := by
  apply Fin.ext
  simp [UInt64.denote, Poseidon2.ofNat, low64_toNat,
    UInt64.toNat_ofNat_of_lt' bound, limbBase_eq_radix]

private theorem fastMid_denote (value : Nat) (bound : value < UInt64.size) :
    (high64 (UInt64.ofNat value)).denote =
      Poseidon2.ofNat ((value / Package.limbBase) % Package.limbBase) := by
  have quotientBound : value / radix < radix := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0 < radix)).2
    simpa [uint64Size_eq_radixSquare] using bound
  apply Fin.ext
  simp [UInt64.denote, Poseidon2.ofNat, high64_toNat,
    UInt64.toNat_ofNat_of_lt' bound, limbBase_eq_radix]
  rw [Nat.mod_eq_of_lt quotientBound]

private theorem fastCarry_denote (value : Nat) (bound : value < UInt64.size) :
    (0 : UInt64).denote =
      Poseidon2.ofNat (value / (Package.limbBase * Package.limbBase)) := by
  have valueBound : value < Package.limbBase * Package.limbBase := by
    simpa [limbBase_eq_radix, uint64Size_eq_radixSquare] using bound
  rw [Nat.div_eq_of_lt valueBound]
  decide

private theorem getD_canonical (words : List UInt64)
    (canonical : ∀ word ∈ words, word.toNat < fieldModulus) (index : Nat) :
    (words.getD index 0).toNat < fieldModulus := by
  rw [List.getD_eq_getElem?_getD]
  cases member : words[index]? with
  | none => decide
  | some word => exact canonical word (List.mem_of_getElem? member)

private theorem getElem?_getD_denote (words : List UInt64) (index : Nat) :
    (Option.map UInt64.denote words[index]?).getD 0 =
      (words[index]?.getD 0).denote := by
  cases words[index]? with
  | none => decide
  | some word => rfl

@[inline] def absorbWords64 (state : State64) (words : List UInt64)
    (canonical : ∀ word ∈ words, word.toNat < fieldModulus) : State64 :=
  State64.permute64 {
    x0 := add64 state.x0 (words.getD 0 0)
    x1 := add64 state.x1 (words.getD 1 0)
    x2 := add64 state.x2 (words.getD 2 0)
    x3 := add64 state.x3 (words.getD 3 0)
    x4 := add64 state.x4 (words.getD 4 0)
    x5 := add64 state.x5 (words.getD 5 0)
    x6 := add64 state.x6 (words.getD 6 0)
    x7 := add64 state.x7 (words.getD 7 0)
    x8 := add64 state.x8 (words.getD 8 0)
    x9 := add64 state.x9 (words.getD 9 0)
    x10 := add64 state.x10 (words.getD 10 0)
    x11 := add64 state.x11 (words.getD 11 0)
    x12 := add64 state.x12 (words.getD 12 0)
    x13 := add64 state.x13 (words.getD 13 0)
    x14 := add64 state.x14 (words.getD 14 0)
    x15 := add64 state.x15 (words.getD 15 0)
    canonical := ⟨add64_canonical _ _ state.c0 (getD_canonical words canonical 0),
      add64_canonical _ _ state.c1 (getD_canonical words canonical 1),
      add64_canonical _ _ state.c2 (getD_canonical words canonical 2),
      add64_canonical _ _ state.c3 (getD_canonical words canonical 3),
      add64_canonical _ _ state.c4 (getD_canonical words canonical 4),
      add64_canonical _ _ state.c5 (getD_canonical words canonical 5),
      add64_canonical _ _ state.c6 (getD_canonical words canonical 6),
      add64_canonical _ _ state.c7 (getD_canonical words canonical 7),
      add64_canonical _ _ state.c8 (getD_canonical words canonical 8),
      add64_canonical _ _ state.c9 (getD_canonical words canonical 9),
      add64_canonical _ _ state.c10 (getD_canonical words canonical 10),
      add64_canonical _ _ state.c11 (getD_canonical words canonical 11),
      add64_canonical _ _ state.c12 (getD_canonical words canonical 12),
      add64_canonical _ _ state.c13 (getD_canonical words canonical 13),
      add64_canonical _ _ state.c14 (getD_canonical words canonical 14),
      add64_canonical _ _ state.c15 (getD_canonical words canonical 15)⟩ }

@[simp] theorem absorbWords64_denote (state : State64) (words : List UInt64)
    (canonical : ∀ word ∈ words, word.toNat < fieldModulus) :
    (absorbWords64 state words canonical).denote =
      Poseidon2.absorbBlock state.denote (words.map UInt64.denote) := by
  rw [absorbWords64, State64.permute64_denote]
  unfold Poseidon2.absorbBlock
  apply congrArg Poseidon2.permute
  simp only [State64.denote]
  rw [add64_denote _ _ state.c0 (getD_canonical words canonical 0),
    add64_denote _ _ state.c1 (getD_canonical words canonical 1),
    add64_denote _ _ state.c2 (getD_canonical words canonical 2),
    add64_denote _ _ state.c3 (getD_canonical words canonical 3),
    add64_denote _ _ state.c4 (getD_canonical words canonical 4),
    add64_denote _ _ state.c5 (getD_canonical words canonical 5),
    add64_denote _ _ state.c6 (getD_canonical words canonical 6),
    add64_denote _ _ state.c7 (getD_canonical words canonical 7),
    add64_denote _ _ state.c8 (getD_canonical words canonical 8),
    add64_denote _ _ state.c9 (getD_canonical words canonical 9),
    add64_denote _ _ state.c10 (getD_canonical words canonical 10),
    add64_denote _ _ state.c11 (getD_canonical words canonical 11),
    add64_denote _ _ state.c12 (getD_canonical words canonical 12),
    add64_denote _ _ state.c13 (getD_canonical words canonical 13),
    add64_denote _ _ state.c14 (getD_canonical words canonical 14),
    add64_denote _ _ state.c15 (getD_canonical words canonical 15)]
  simp [Poseidon2.width, List.range_succ, getElem?_getD_denote]

@[inline] def nodeWords64 (node : StreamingIdentity.Node) : List UInt64 :=
  if node.value < UInt64.size then
    let word := UInt64.ofNat node.value
    [ofNat64 node.tag, low64 word, high64 word, 0]
  else
    [ofNat64 node.tag, ofNat64 (node.value % Package.limbBase),
      ofNat64 ((node.value / Package.limbBase) % Package.limbBase),
      ofNat64 (node.value / (Package.limbBase * Package.limbBase))]

theorem nodeWords64_canonical (node : StreamingIdentity.Node) :
    ∀ word ∈ nodeWords64 node, word.toNat < fieldModulus := by
  unfold nodeWords64
  split
  · simp only [List.mem_cons, List.not_mem_nil, or_false]
    rintro word (rfl | rfl | rfl | rfl)
    · exact ofNat64_canonical _
    · exact Nat.lt_trans (low64_bound _) (by decide)
    · exact Nat.lt_trans (high64_bound _) (by decide)
    · decide
  · simp only [List.mem_cons, List.not_mem_nil, or_false]
    rintro word (rfl | rfl | rfl | rfl) <;> exact ofNat64_canonical _

@[simp] theorem nodeWords64_length (node : StreamingIdentity.Node) :
    (nodeWords64 node).length = 4 := by
  unfold nodeWords64
  split <;> rfl

theorem nodeWords64_denote (node : StreamingIdentity.Node) :
    (nodeWords64 node).map UInt64.denote = node.words := by
  rcases node with ⟨tag, value⟩
  unfold nodeWords64
  split <;> rename_i valueBranch
  · simp [StreamingIdentity.Node.words, fastLow_denote _ valueBranch,
      fastMid_denote _ valueBranch, fastCarry_denote _ valueBranch]
  · simp [StreamingIdentity.Node.words]

structure HashState64 where
  sponge : State64
  pending : List UInt64
  pendingCanonical : ∀ word ∈ pending, word.toNat < fieldModulus

def HashState64.denote (state : HashState64) : StreamingIdentity.HashState where
  sponge := state.sponge.denote
  pending := state.pending.map UInt64.denote

@[noinline] def pushNode64 (state : HashState64)
    (node : StreamingIdentity.Node) : HashState64 :=
  let buffer := state.pending ++ nodeWords64 node
  have bufferCanonical : ∀ word ∈ buffer, word.toNat < fieldModulus := by
    intro word member
    rcases List.mem_append.mp member with pending | fresh
    · exact state.pendingCanonical word pending
    · exact nodeWords64_canonical node word fresh
  if Poseidon2.rate ≤ buffer.length then
    { sponge := absorbWords64 state.sponge (buffer.take Poseidon2.rate)
        (fun word member => bufferCanonical word (List.mem_of_mem_take member))
      pending := buffer.drop Poseidon2.rate
      pendingCanonical := fun word member =>
        bufferCanonical word (List.mem_of_mem_drop member) }
  else
    { sponge := state.sponge
      pending := buffer
      pendingCanonical := bufferCanonical }

theorem pushNode64_denote (state : HashState64)
    (node : StreamingIdentity.Node) :
    (pushNode64 state node).denote =
      StreamingIdentity.pushNode state.denote node := by
  have words := StreamingIdentity.Node.words_length node
  by_cases full : Poseidon2.rate ≤ state.pending.length + 4
  · simp [pushNode64, StreamingIdentity.pushNode, HashState64.denote, full,
      List.map_take, List.map_drop, nodeWords64_denote, words]
  · simp [pushNode64, StreamingIdentity.pushNode, HashState64.denote, full,
      nodeWords64_denote, words]

private def identityBlock0 : List UInt64 :=
  [78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47]
private def identityBlock1 : List UInt64 :=
  [70, 80, 114, 105, 109, 101, 47, 112, 97, 99, 107, 97]

def initialState64 : HashState64 where
  sponge := absorbWords64 (absorbWords64 State64.zero identityBlock0 (by decide))
    identityBlock1 (by decide)
  pending := [103, 101, 47, 118, 50]
  pendingCanonical := by decide

theorem initialState64_denote :
    initialState64.denote = StreamingIdentity.initialState := by
  simp only [initialState64, HashState64.denote, StreamingIdentity.initialState,
    StreamingIdentity.prefixState, StreamingIdentity.HashState.mk.injEq,
    absorbWords64_denote]
  constructor
  · have z : State64.zero.denote = Poseidon2.zeroState := by decide
    have d0 : identityBlock0.map UInt64.denote =
        [(78 : F), 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47] := by
      decide
    have d1 : identityBlock1.map UInt64.denote =
        [(70 : F), 80, 114, 105, 109, 101, 47, 112, 97, 99, 107, 97] := by
      decide
    rw [z, d0, d1]
    rfl
  · decide

@[inline] private def pad64 (state : State64) : State64 :=
  { state with
    x0 := add64 state.x0 1
    canonical := ⟨add64_canonical _ _ state.c0 (by decide), state.c1, state.c2, state.c3, state.c4, state.c5, state.c6, state.c7,
      state.c8, state.c9, state.c10, state.c11, state.c12, state.c13, state.c14, state.c15⟩ }

private theorem pad64_denote (state : State64) :
    (pad64 state).denote = (List.range Poseidon2.width).map fun lane =>
      if lane = 0 then state.denote.getD 0 0 + 1 else state.denote.getD lane 0 := by
  simp only [pad64, State64.denote]
  rw [add64_denote _ _ state.c0 (by decide)]
  have oneDenote : (1 : UInt64).denote = (1 : F) := by decide
  rw [oneDenote]
  simp [Poseidon2.width, List.range_succ]

structure Digest64 where
  x0 : UInt64
  x1 : UInt64
  x2 : UInt64
  x3 : UInt64

def Digest64.denote (digest : Digest64) : List F :=
  [digest.x0.denote, digest.x1.denote, digest.x2.denote, digest.x3.denote]

private def finalState64 (state : HashState64) : State64 :=
  State64.permute64 (pad64
    (absorbWords64 state.sponge state.pending state.pendingCanonical))

private theorem finalState64_denote (state : HashState64) :
    (finalState64 state).denote =
      let absorbed := Poseidon2.absorbBlock state.sponge.denote
        (state.pending.map UInt64.denote)
      Poseidon2.permute ((List.range Poseidon2.width).map fun lane =>
        if lane = 0 then absorbed.getD 0 0 + 1 else absorbed.getD lane 0) := by
  rw [finalState64, State64.permute64_denote, pad64_denote,
    absorbWords64_denote]

@[inline] def finalize64 (state : HashState64) : Digest64 :=
  let padded := finalState64 state
  ⟨padded.x0, padded.x1, padded.x2, padded.x3⟩

theorem finalize64_denote (state : HashState64) :
    (finalize64 state).denote = StreamingIdentity.finalize state.denote := by
  have permutation := congrArg (List.take Poseidon2.digestLen)
    (finalState64_denote state)
  simpa [finalize64, Digest64.denote, HashState64.denote,
    StreamingIdentity.finalize, Poseidon2.digestLen, State64.denote]
    using permutation

end NightstreamFPrime.Export.NativePoseidon2
