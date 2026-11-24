import NightstreamFPrime.Export.NativePoseidon2RoundCore

namespace NightstreamFPrime.Export.NativePoseidon2

open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open Fin.CommRing

namespace State64

private def initialConstant0 : State64 :=
  ⟨1579613653969377219, 15509510893087893340, 10090715174060125222, 5625716564419252202,
    3006005019077469174, 18314693207014427912, 10170571510627764565, 2027625550790675754,
    3983470257916202094, 3423470109396435354, 3450488264035752368, 3151070045406026687,
    13462781804006123550, 13288575772684627216, 13745549378090937523, 7780139165529418388, by decide⟩
private def initialConstant1 : State64 :=
  ⟨11699769881923825909, 4349260652212695338, 8228348357294662898, 12037145371429260559,
    7743383997471469184, 7842006993218978420, 18369471830829996810, 10157815501749214163,
    11128005290408113568, 12272917402811871099, 106820976709770637, 6178857985076893256,
    14036787039817180193, 1659458828486927362, 15938183887757779615, 10060334973073197601, by decide⟩
private def initialConstant2 : State64 :=
  ⟨12437259172286625042, 3143835545170851304, 14361575850007152549, 11810209965918630788,
    17667700303515849362, 13451932870729425780, 17496644301504593329, 1778669202480014256,
    11124488067381312952, 14273583775934080755, 13257379154210125715, 15182093233826554989,
    5343815959848316783, 5696193414571279556, 6089787229781325813, 3717595218375990594, by decide⟩
private def initialConstant3 : State64 :=
  ⟨12355842859741995537, 14160921043125059199, 13682913322779819091, 12343504462597575239,
    14249643564311509087, 15849758561085927402, 12186564686511703262, 690520822232718983,
    684200944081655198, 6008958449718781068, 3098079696536837500, 12083506578988585239,
    6489163969976034646, 1186411428667285248, 11192474480021157091, 17095617549195818058, by decide⟩

private def partialConstant00 : UInt64 := 2949845317987848448
private def partialConstant01 : UInt64 := 6935557324168256069
private def partialConstant02 : UInt64 := 2055390472619727748
private def partialConstant03 : UInt64 := 12399309956582731760
private def partialConstant04 : UInt64 := 3088058410657255303
private def partialConstant05 : UInt64 := 8790959064194804235
private def partialConstant06 : UInt64 := 4582664136559626757
private def partialConstant07 : UInt64 := 14821962908959377092
private def partialConstant08 : UInt64 := 2002685046256118388
private def partialConstant09 : UInt64 := 15689593639089632368
private def partialConstant10 : UInt64 := 10512046881800463697
private def partialConstant11 : UInt64 := 11973738501993105496
private def partialConstant12 : UInt64 := 13009070935614847359
private def partialConstant13 : UInt64 := 12908033183314664096
private def partialConstant14 : UInt64 := 9530849385991780809
private def partialConstant15 : UInt64 := 17528314872631229800
private def partialConstant16 : UInt64 := 17869527288639567155
private def partialConstant17 : UInt64 := 7829055113315023688
private def partialConstant18 : UInt64 := 3256047469251174543
private def partialConstant19 : UInt64 := 3015723851705964382
private def partialConstant20 : UInt64 := 5177282273995529875
private def partialConstant21 : UInt64 := 16035152896984012190

private def terminalConstant0 : State64 :=
  ⟨15766898731627159621, 6714370142978148338, 5906664861467821209, 2197120128337974909,
    9400421283898153864, 16740473708727660723, 7287335615015370599, 12399217337488717232,
    12655834778591227242, 7516917516498792203, 17291898461659961460, 3483661079725019706,
    18038566855851505564, 6166154478921479976, 8824538844032254328, 17138720803294251067, by decide⟩
private def terminalConstant1 : State64 :=
  ⟨4158286797471309733, 1589482450994236426, 4836112672767737407, 6108317238076822824,
    11174425919404088598, 2755155124184744585, 13202523778257173956, 6643835433125335951,
    2748965361441888477, 14297205492001640506, 12595977886586773319, 257624298448087449,
    2744326799038764281, 9097808034015176417, 5523099006222423939, 7256201493271212720, by decide⟩
private def terminalConstant2 : State64 :=
  ⟨9738621014989498753, 237412858668409460, 6405904454922718574, 16343835012250148340,
    9342245617122491936, 824922755340087111, 5398764608665671277, 16030622638137836774,
    4921899302518525668, 10676667276121395674, 16304445044992359788, 587812931475326787,
    7627897473303434798, 2633542348821978290, 5714662808802647771, 15665352327749112015, by decide⟩
private def terminalConstant3 : State64 :=
  ⟨8041781683128153420, 13172746155287932121, 6611413049825339003, 9076340683484238287,
    12295614633782496176, 11414162089917699064, 13658053783093222498, 16216654076620539925,
    17782598995085730495, 18362903988365127450, 11596821426909532371, 13499421621575350806,
    15256312875610823807, 1558929959921731273, 17567284148478398281, 4615393526180652407, by decide⟩

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

private theorem canonicalWords (words : List UInt64)
    (checked : words.all (fun word => decide (word.toNat < fieldModulus)) = true) :
    ∀ word ∈ words, word.toNat < fieldModulus := by
  simpa only [List.all_eq_true, decide_eq_true_eq] using checked

def initialState64 : HashState64 where
  sponge := absorbWords64
    (absorbWords64 State64.zero identityBlock0 (canonicalWords _ (by decide +kernel)))
    identityBlock1 (canonicalWords _ (by decide +kernel))
  pending := [103, 101, 47, 118, 50]
  pendingCanonical := canonicalWords _ (by decide +kernel)

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

structure Digest64 where
  x0 : UInt64
  x1 : UInt64
  x2 : UInt64
  x3 : UInt64

def Digest64.denote (digest : Digest64) : List F :=
  [digest.x0.denote, digest.x1.denote, digest.x2.denote, digest.x3.denote]

/-- The padding block of `StreamingIdentity.finalize`: the pending words, then the word 1. -/
private def finalState64 (state : HashState64) : State64 :=
  absorbWords64 state.sponge (state.pending ++ [1]) (fun word member => by
    rcases List.mem_append.mp member with pending | one
    · exact state.pendingCanonical word pending
    · rw [List.mem_singleton.mp one]
      decide)

private theorem finalState64_denote (state : HashState64) :
    (finalState64 state).denote =
      Poseidon2.absorbBlock state.sponge.denote (state.pending.map UInt64.denote ++ [1]) := by
  have oneDenote : (1 : UInt64).denote = (1 : F) := by decide
  rw [finalState64, absorbWords64_denote, List.map_append, List.map_singleton, oneDenote]

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
