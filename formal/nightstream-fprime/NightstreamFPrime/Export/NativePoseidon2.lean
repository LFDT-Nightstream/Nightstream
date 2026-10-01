import NightstreamFPrime.Export.NativePoseidon2RoundCore

/-!
Owns native Poseidon2 constants, the exact 4/22/4 schedule, and the streaming
sponge. Fixed-width Goldilocks arithmetic and round operations belong to
`NativeGoldilocks` and `NativePoseidon2RoundCore`.
-/

namespace NightstreamFPrime.Export.NativePoseidon2

open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open Fin.CommRing

namespace State64

private def initialConstant0 : State64 :=
  ⟨15504881536434223753, 2212164856944708396, 1885257220781225929, 17531637481572944510,
    16769640728293682348, 445908668462176974, 1308472042479836079, 17465001500823438575,
    1922033642430128704, 2657514617275794404, 17238706657248448792, 7348277157222259646,
    10777112892842897939, 1771261721914735482, 9409693344407549465, 16619731096074499912, by decide⟩
private def initialConstant1 : State64 :=
  ⟨1922036059108268922, 2681686362645798986, 12432722052283819565, 2826979200512189741,
    5080805286413226676, 16827966425431695029, 9196241087337510154, 2350771591198563053,
    2989012136977041732, 4359939046747977080, 16089932437481530267, 6601984573273403484,
    13005272261058756234, 17128237926164276121, 8240789415616872849, 8676316357341090631, by decide⟩
private def initialConstant2 : State64 :=
  ⟨16452552554259143025, 17874550554210084887, 3031715677034868367, 18215520516675091549,
    18186005068527139405, 11138995707668647102, 15098195648006184282, 2025927025270509469,
    9957669227203243937, 11554336633716867616, 9729067570563846225, 4239770196713589268,
    4390607796152185292, 17647511975646925721, 7671337049037340193, 4209452938403606590, by decide⟩
private def initialConstant3 : State64 :=
  ⟨6593973666654839090, 8390781086037206386, 7324343054784993307, 17780748563735894140,
    15974082699116886783, 13213371256836887512, 7312926934405385057, 10393853239698468203,
    2710107888698774842, 2801523468128575786, 15894340394120906162, 13510783799941644149,
    7917164295139071913, 13839801071899888959, 6672989303670154677, 4519956214037211385, by decide⟩

private def partialConstant00 : UInt64 := 3785078240232899765
private def partialConstant01 : UInt64 := 13753534232687663059
private def partialConstant02 : UInt64 := 15164579152346391244
private def partialConstant03 : UInt64 := 7188840087678967607
private def partialConstant04 : UInt64 := 14466302407008998516
private def partialConstant05 : UInt64 := 13002203018489042439
private def partialConstant06 : UInt64 := 6216879054540921071
private def partialConstant07 : UInt64 := 8639253411382064186
private def partialConstant08 : UInt64 := 4983986534873630152
private def partialConstant09 : UInt64 := 2915380551046574730
private def partialConstant10 : UInt64 := 7837744426713061492
private def partialConstant11 : UInt64 := 10524728973359902812
private def partialConstant12 : UInt64 := 9778451922853050075
private def partialConstant13 : UInt64 := 13310295482970625397
private def partialConstant14 : UInt64 := 10551647698332907197
private def partialConstant15 : UInt64 := 8509293821221348719
private def partialConstant16 : UInt64 := 4319821126274335035
private def partialConstant17 : UInt64 := 1417436523646501240
private def partialConstant18 : UInt64 := 447570613036226838
private def partialConstant19 : UInt64 := 16267896682506996530
private def partialConstant20 : UInt64 := 14710998805405468930
private def partialConstant21 : UInt64 := 3269000544927978862

private def terminalConstant0 : State64 :=
  ⟨7482194551502142718, 3471957803411196592, 8846669050136897522, 4431017908497072775,
    14382646627736292998, 15636596632746594248, 14521990061611210983, 4351091752509404379,
    14119848206371842921, 528205008764728916, 15379406877060454284, 13572057177474709483,
    780214424511389757, 10591233664360718633, 1849508423779478786, 7345390174439848870, by decide⟩
private def terminalConstant1 : State64 :=
  ⟨14580881241235634775, 8777273265976228774, 1758781345554053863, 9701442189086298420,
    15685565327448534444, 5672331717709479627, 7675233227955155107, 8852669876726984824,
    1218164705289579190, 13224810758441726241, 557024023478380004, 3923346290699247117,
    4196774554581694822, 16262909137268628555, 6531098975686849205, 538070144030448988, by decide⟩
private def terminalConstant2 : State64 :=
  ⟨16157559630818414765, 3330859574359708906, 13312877616183741059, 15699706004066187344,
    2181468677625794151, 12293285838251430515, 17109377825740910727, 11746958123598489878,
    7654965179475073269, 15178922343313110770, 14240408894833620294, 4224192993509995210,
    13093043512401634422, 16636225261759530156, 13489384640167770266, 8105602514957866176, by decide⟩
private def terminalConstant3 : State64 :=
  ⟨13910460326211973254, 13010277363854955001, 8570865802160232388, 14830753997593808291,
    16178721091989175194, 10926358020058189153, 8413180834413067310, 1124528750616792490,
    16054199595598201491, 729673474029476808, 1545919217216143455, 15484244716222357662,
    9149791094276206087, 3342519128264714984, 14246551315881547762, 6145097356981870399, by decide⟩

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
    (constantCanonical : constant.toNat < goldilocksModulus)
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

/-- Fixed 4/22/4 Poseidon2 permutation over sixteen machine-word lanes. -/
@[noinline] def permute64 (state : State64) : State64 :=
  terminalRounds64 (partialRounds64 (initialRounds64 (externalLayer64 state)))

theorem permute64_denote (state : State64) :
    (permute64 state).denote = Poseidon2.permute state.denote := by
  rw [permute64, terminalRounds64_denote, partialRounds64_denote,
    initialRounds64_denote, externalLayer64_denote]
  rfl

end State64

/-! ## Total native sponge bridge -/

/-- Canonical machine representative of an arbitrary natural field value.
The common `Nat < 2^64` path needs at most one Goldilocks subtraction. -/
@[inline] def ofNat64 (value : Nat) : UInt64 :=
  if value < UInt64.size then
    if value < goldilocksModulus then UInt64.ofNat value
    else UInt64.ofNat (value - goldilocksModulus)
  else UInt64.ofNat (value % goldilocksModulus)

private theorem ofNat64_toNat (value : Nat) :
    (ofNat64 value).toNat = value % goldilocksModulus := by
  simp only [ofNat64]
  split <;> rename_i sizeBranch
  · split <;> rename_i modulusBranch
    · rw [UInt64.toNat_ofNat_of_lt' sizeBranch,
        Nat.mod_eq_of_lt modulusBranch]
    · have modulusLe : goldilocksModulus ≤ value := by omega
      have differenceSize : value - goldilocksModulus < UInt64.size := by omega
      have differenceModulus : value - goldilocksModulus < goldilocksModulus := by
        have sizeLtTwice : UInt64.size < 2 * goldilocksModulus := by decide
        omega
      rw [UInt64.toNat_ofNat_of_lt' differenceSize,
        Nat.mod_eq_sub_mod modulusLe, Nat.mod_eq_of_lt differenceModulus]
  · have residueModulus := Nat.mod_lt value (by decide : 0 < goldilocksModulus)
    have residueSize : value % goldilocksModulus < UInt64.size :=
      Nat.lt_trans residueModulus (by decide)
    rw [UInt64.toNat_ofNat_of_lt' residueSize]

theorem ofNat64_canonical (value : Nat) :
    (ofNat64 value).toNat < goldilocksModulus := by
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
    (canonical : ∀ word ∈ words, word.toNat < goldilocksModulus) (index : Nat) :
    (words.getD index 0).toNat < goldilocksModulus := by
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

/-- Add up to one rate block of canonical words to the sponge lanes and
permute. Missing words are zero. -/
@[inline] def absorbWords64 (state : State64) (words : List UInt64)
    (canonical : ∀ word ∈ words, word.toNat < goldilocksModulus) : State64 :=
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
    (canonical : ∀ word ∈ words, word.toNat < goldilocksModulus) :
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

/-- The four canonical machine words of one node. The common `value < 2^64`
path splits the value into two 32-bit limbs and a zero high limb. -/
@[inline] def nodeWords64 (node : StreamingIdentity.Node) : List UInt64 :=
  if node.value < UInt64.size then
    let word := UInt64.ofNat node.value
    [ofNat64 node.tag, low64 word, high64 word, 0]
  else
    [ofNat64 node.tag, ofNat64 (node.value % Package.limbBase),
      ofNat64 ((node.value / Package.limbBase) % Package.limbBase),
      ofNat64 (node.value / (Package.limbBase * Package.limbBase))]

theorem nodeWords64_canonical (node : StreamingIdentity.Node) :
    ∀ word ∈ nodeWords64 node, word.toNat < goldilocksModulus := by
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

/-- Native streaming sponge and its pending words. -/
structure HashState64 where
  sponge : State64
  pending : List UInt64
  pendingCanonical : ∀ word ∈ pending, word.toNat < goldilocksModulus

def HashState64.denote (state : HashState64) : StreamingIdentity.HashState where
  sponge := state.sponge.denote
  pending := state.pending.map UInt64.denote

/-- Absorb one canonical streaming node without constructing field values. -/
@[noinline] def pushNode64 (state : HashState64)
    (node : StreamingIdentity.Node) : HashState64 :=
  let buffer := state.pending ++ nodeWords64 node
  have bufferCanonical : ∀ word ∈ buffer, word.toNat < goldilocksModulus := by
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

/-- Two identity-domain blocks; the last five domain words stay pending. -/
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

/-- Four machine words returned by the native squeeze. -/
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

/-- Final pending-word absorption, pad permutation, and four-word squeeze. -/
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
