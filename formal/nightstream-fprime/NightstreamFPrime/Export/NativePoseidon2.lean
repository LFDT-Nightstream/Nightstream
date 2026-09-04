import NightstreamFPrime.Export.NativePoseidon2RoundCore

/-!
Owns native Poseidon2 constants, the exact 4/22/4 schedule, and the streaming
sponge. Fixed-width Goldilocks arithmetic and round operations belong to
`NativePoseidon2RoundCore`.
-/

namespace NightstreamFPrime.Export.NativePoseidon2

open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open Fin.CommRing

namespace State64

@[inline] private def constantState (x0 x1 x2 x3 x4 x5 x6 x7 : UInt64)
    (canonical :
      x0.toNat < goldilocksModulus ∧ x1.toNat < goldilocksModulus ∧
      x2.toNat < goldilocksModulus ∧ x3.toNat < goldilocksModulus ∧
      x4.toNat < goldilocksModulus ∧ x5.toNat < goldilocksModulus ∧
      x6.toNat < goldilocksModulus ∧ x7.toNat < goldilocksModulus) : State64 :=
  ⟨x0, x1, x2, x3, x4, x5, x6, x7, canonical⟩

private def initialConstant0 : State64 := constantState
  15504881536434223753 2212164856944708396 1885257220781225929 17531637481572944510
  16769640728293682348 445908668462176974 1308472042479836079 17465001500823438575 (by decide)
private def initialConstant1 : State64 := constantState
  1922033642430128704 2657514617275794404 17238706657248448792 7348277157222259646
  10777112892842897939 1771261721914735482 9409693344407549465 16619731096074499912 (by decide)
private def initialConstant2 : State64 := constantState
  1922036059108268922 2681686362645798986 12432722052283819565 2826979200512189741
  5080805286413226676 16827966425431695029 9196241087337510154 2350771591198563053 (by decide)
private def initialConstant3 : State64 := constantState
  2989012136977041732 4359939046747977080 16089932437481530267 6601984573273403484
  13005272261058756234 17128237926164276121 8240789415616872849 8676316357341090631 (by decide)

private def partialConstant00 : UInt64 := 7482194551502142718
private def partialConstant01 : UInt64 := 3471957803411196592
private def partialConstant02 : UInt64 := 8846669050136897522
private def partialConstant03 : UInt64 := 4431017908497072775
private def partialConstant04 : UInt64 := 14382646627736292998
private def partialConstant05 : UInt64 := 15636596632746594248
private def partialConstant06 : UInt64 := 14521990061611210983
private def partialConstant07 : UInt64 := 4351091752509404379
private def partialConstant08 : UInt64 := 14119848206371842921
private def partialConstant09 : UInt64 := 528205008764728916
private def partialConstant10 : UInt64 := 15379406877060454284
private def partialConstant11 : UInt64 := 13572057177474709483
private def partialConstant12 : UInt64 := 780214424511389757
private def partialConstant13 : UInt64 := 10591233664360718633
private def partialConstant14 : UInt64 := 1849508423779478786
private def partialConstant15 : UInt64 := 7345390174439848870
private def partialConstant16 : UInt64 := 14580881241235634775
private def partialConstant17 : UInt64 := 8777273265976228774
private def partialConstant18 : UInt64 := 1758781345554053863
private def partialConstant19 : UInt64 := 9701442189086298420
private def partialConstant20 : UInt64 := 15685565327448534444
private def partialConstant21 : UInt64 := 5672331717709479627

private def terminalConstant0 : State64 := constantState
  16452552554259143025 17874550554210084887 3031715677034868367 18215520516675091549
  18186005068527139405 11138995707668647102 15098195648006184282 2025927025270509469 (by decide)
private def terminalConstant1 : State64 := constantState
  9957669227203243937 11554336633716867616 9729067570563846225 4239770196713589268
  4390607796152185292 17647511975646925721 7671337049037340193 4209452938403606590 (by decide)
private def terminalConstant2 : State64 := constantState
  6593973666654839090 8390781086037206386 7324343054784993307 17780748563735894140
  15974082699116886783 13213371256836887512 7312926934405385057 10393853239698468203 (by decide)
private def terminalConstant3 : State64 := constantState
  2710107888698774842 2801523468128575786 15894340394120906162 13510783799941644149
  7917164295139071913 13839801071899888959 6672989303670154677 4519956214037211385 (by decide)

private def FullConstantMatch (constants : State64) (rows : List (List Nat))
    (round : Nat) : Prop :=
  constants.x0.denote = Poseidon2.constantAt rows round 0 ∧
  constants.x1.denote = Poseidon2.constantAt rows round 1 ∧
  constants.x2.denote = Poseidon2.constantAt rows round 2 ∧
  constants.x3.denote = Poseidon2.constantAt rows round 3 ∧
  constants.x4.denote = Poseidon2.constantAt rows round 4 ∧
  constants.x5.denote = Poseidon2.constantAt rows round 5 ∧
  constants.x6.denote = Poseidon2.constantAt rows round 6 ∧
  constants.x7.denote = Poseidon2.constantAt rows round 7

private theorem fullRound64_denote_at (rows : List (List Nat)) (round : Nat)
    (constants state : State64)
    (constantMatch : FullConstantMatch constants rows round) :
    (fullRound64 state constants).denote = Poseidon2.fullRound rows round state.denote := by
  rcases constantMatch with ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩
  rw [fullRound64_denote]
  unfold Poseidon2.fullRound
  apply congrArg Poseidon2.externalLayer
  apply List.ext_get
  · simp [Poseidon2.width]
  · intro index leftLt rightLt
    have indexLt : index < 8 := by simpa using leftLt
    interval_cases index <;>
      simp [Poseidon2.width, denote, h0, h1, h2, h3, h4, h5, h6, h7]

private theorem partialRound64_denote_at (round : Nat) (constant : UInt64)
    (constantCanonical : constant.toNat < goldilocksModulus)
    (constantMatch : constant.denote =
      Poseidon2.ofNat (Poseidon2.internalConstants.getD round 0))
    (state : State64) :
    (partialRound64 state constant).denote = Poseidon2.partialRound round state.denote := by
  rw [partialRound64_denote constant constantCanonical state, constantMatch]
  unfold Poseidon2.partialRound
  apply congrArg Poseidon2.internalLayer
  apply List.ext_get
  · simp [Poseidon2.width, denote]
  · intro index leftLt rightLt
    have indexLt : index < 8 := by simpa [denote] using leftLt
    interval_cases index <;> simp [Poseidon2.width, denote]

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

/-- Fixed 4/22/4 Poseidon2 permutation over eight machine-word lanes. -/
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

/-- Absorb one canonical native rate block. -/
@[inline] def absorbBlock64 (state : State64) (b0 b1 b2 b3 : UInt64)
    (c0 : b0.toNat < goldilocksModulus) (c1 : b1.toNat < goldilocksModulus)
    (c2 : b2.toNat < goldilocksModulus) (c3 : b3.toNat < goldilocksModulus) : State64 :=
  State64.permute64 {
    x0 := add64 state.x0 b0
    x1 := add64 state.x1 b1
    x2 := add64 state.x2 b2
    x3 := add64 state.x3 b3
    x4 := state.x4
    x5 := state.x5
    x6 := state.x6
    x7 := state.x7
    canonical := ⟨add64_canonical _ _ state.c0 c0,
      add64_canonical _ _ state.c1 c1, add64_canonical _ _ state.c2 c2,
      add64_canonical _ _ state.c3 c3, state.c4, state.c5, state.c6, state.c7⟩ }

@[simp] theorem absorbBlock64_denote (state : State64)
    (b0 b1 b2 b3 : UInt64) (c0 : b0.toNat < goldilocksModulus)
    (c1 : b1.toNat < goldilocksModulus) (c2 : b2.toNat < goldilocksModulus)
    (c3 : b3.toNat < goldilocksModulus) :
    (absorbBlock64 state b0 b1 b2 b3 c0 c1 c2 c3).denote =
      Poseidon2.absorbBlock state.denote
        [b0.denote, b1.denote, b2.denote, b3.denote] := by
  rw [absorbBlock64, State64.permute64_denote]
  unfold Poseidon2.absorbBlock
  simp only [State64.denote]
  rw [add64_denote _ _ state.c0 c0, add64_denote _ _ state.c1 c1,
    add64_denote _ _ state.c2 c2, add64_denote _ _ state.c3 c3]
  apply congrArg Poseidon2.permute
  apply List.ext_get
  · simp [Poseidon2.width]
  · intro index leftLt rightLt
    have indexLt : index < 8 := by simpa using leftLt
    interval_cases index <;> simp [Poseidon2.width]

/-- Native streaming sponge and the carried fourth word. -/
structure HashState64 where
  sponge : State64
  carry : UInt64
  carryCanonical : carry.toNat < goldilocksModulus

def HashState64.denote (state : HashState64) : StreamingIdentity.HashState where
  sponge := state.sponge.denote
  carry := state.carry.denote

/-- Absorb one canonical streaming node without constructing field values. -/
@[noinline] def pushNode64 (state : HashState64)
    (node : StreamingIdentity.Node) : HashState64 :=
  let tag := ofNat64 node.tag
  if node.value < UInt64.size then
    let word := UInt64.ofNat node.value
    { sponge := absorbBlock64 state.sponge state.carry tag (low64 word) (high64 word)
        state.carryCanonical (ofNat64_canonical _)
          (Nat.lt_trans (low64_bound _) (by decide))
          (Nat.lt_trans (high64_bound _) (by decide))
      carry := 0
      carryCanonical := by decide }
  else
    let low := ofNat64 (node.value % Package.limbBase)
    let mid := ofNat64 ((node.value / Package.limbBase) % Package.limbBase)
    let high := ofNat64 (node.value / (Package.limbBase * Package.limbBase))
    { sponge := absorbBlock64 state.sponge state.carry tag low mid
        state.carryCanonical (ofNat64_canonical _) (ofNat64_canonical _)
          (ofNat64_canonical _)
      carry := high
      carryCanonical := ofNat64_canonical _ }

theorem pushNode64_denote (state : HashState64)
    (node : StreamingIdentity.Node) :
    (pushNode64 state node).denote =
      StreamingIdentity.pushNode state.denote node := by
  rcases node with ⟨tag, value⟩
  simp only [pushNode64]
  split <;> rename_i valueBranch
  · simp [HashState64.denote, StreamingIdentity.pushNode,
      StreamingIdentity.Node.block, StreamingIdentity.Node.nextCarry,
      StreamingIdentity.Node.words, fastLow_denote _ valueBranch,
      fastMid_denote _ valueBranch, fastCarry_denote _ valueBranch]
  · simp [HashState64.denote, StreamingIdentity.pushNode,
      StreamingIdentity.Node.block, StreamingIdentity.Node.nextCarry,
      StreamingIdentity.Node.words]

/-- Initial seven domain blocks and the carried twenty-ninth domain word. -/
private def domainBlock0 (s : State64) := absorbBlock64 s 78 105 103 104
  (by decide) (by decide) (by decide) (by decide)
private def domainBlock1 (s : State64) := absorbBlock64 s 116 115 116 114
  (by decide) (by decide) (by decide) (by decide)
private def domainBlock2 (s : State64) := absorbBlock64 s 101 97 109 47
  (by decide) (by decide) (by decide) (by decide)
private def domainBlock3 (s : State64) := absorbBlock64 s 70 80 114 105
  (by decide) (by decide) (by decide) (by decide)
private def domainBlock4 (s : State64) := absorbBlock64 s 109 101 47 112
  (by decide) (by decide) (by decide) (by decide)
private def domainBlock5 (s : State64) := absorbBlock64 s 97 99 107 97
  (by decide) (by decide) (by decide) (by decide)
private def domainBlock6 (s : State64) := absorbBlock64 s 103 101 47 118
  (by decide) (by decide) (by decide) (by decide)

def initialState64 : HashState64 where
  sponge := domainBlock6 (domainBlock5 (domainBlock4 (domainBlock3
    (domainBlock2 (domainBlock1 (domainBlock0 State64.zero))))))
  carry := 50
  carryCanonical := by decide

theorem initialState64_denote :
    initialState64.denote = StreamingIdentity.initialState := by
  simp only [initialState64, HashState64.denote, StreamingIdentity.initialState,
    StreamingIdentity.HashState.mk.injEq]
  constructor
  · rw [domainBlock6, absorbBlock64_denote, domainBlock5, absorbBlock64_denote,
      domainBlock4, absorbBlock64_denote, domainBlock3, absorbBlock64_denote,
      domainBlock2, absorbBlock64_denote, domainBlock1, absorbBlock64_denote,
      domainBlock0, absorbBlock64_denote]
    have z : State64.zero.denote = Poseidon2.zeroState := by decide
    have d0 : [(78 : UInt64).denote, (105 : UInt64).denote,
        (103 : UInt64).denote, (104 : UInt64).denote] = [(78 : F), 105, 103, 104] := by decide
    have d1 : [(116 : UInt64).denote, (115 : UInt64).denote,
        (116 : UInt64).denote, (114 : UInt64).denote] = [(116 : F), 115, 116, 114] := by decide
    have d2 : [(101 : UInt64).denote, (97 : UInt64).denote,
        (109 : UInt64).denote, (47 : UInt64).denote] = [(101 : F), 97, 109, 47] := by decide
    have d3 : [(70 : UInt64).denote, (80 : UInt64).denote,
        (114 : UInt64).denote, (105 : UInt64).denote] = [(70 : F), 80, 114, 105] := by decide
    have d4 : [(109 : UInt64).denote, (101 : UInt64).denote,
        (47 : UInt64).denote, (112 : UInt64).denote] = [(109 : F), 101, 47, 112] := by decide
    have d5 : [(97 : UInt64).denote, (99 : UInt64).denote,
        (107 : UInt64).denote, (97 : UInt64).denote] = [(97 : F), 99, 107, 97] := by decide
    have d6 : [(103 : UInt64).denote, (101 : UInt64).denote,
        (47 : UInt64).denote, (118 : UInt64).denote] = [(103 : F), 101, 47, 118] := by decide
    rw [z, d0, d1, d2, d3, d4, d5, d6]
    rfl
  · decide

@[inline] private def pad64 (state : State64) : State64 where
  x0 := add64 state.x0 1
  x1 := state.x1
  x2 := state.x2
  x3 := state.x3
  x4 := state.x4
  x5 := state.x5
  x6 := state.x6
  x7 := state.x7
  canonical := ⟨add64_canonical _ _ state.c0 (by decide), state.c1, state.c2,
    state.c3, state.c4, state.c5, state.c6, state.c7⟩

private theorem pad64_denote (state : State64) :
    (pad64 state).denote = (List.range Poseidon2.width).map fun lane =>
      if lane = 0 then state.denote.getD 0 0 + 1 else state.denote.getD lane 0 := by
  simp only [pad64, State64.denote]
  rw [add64_denote _ _ state.c0 (by decide)]
  have oneDenote : (1 : UInt64).denote = (1 : F) := by decide
  rw [oneDenote]
  apply List.ext_get
  · simp [Poseidon2.width]
  · intro index leftLt rightLt
    have indexLt : index < 8 := by simpa using leftLt
    interval_cases index <;>
      simp [Poseidon2.width]

/-- Four machine words returned by the native squeeze. -/
structure Digest64 where
  x0 : UInt64
  x1 : UInt64
  x2 : UInt64
  x3 : UInt64

def Digest64.denote (digest : Digest64) : List F :=
  [digest.x0.denote, digest.x1.denote, digest.x2.denote, digest.x3.denote]

private def absorbCarry64 (state : HashState64) : State64 :=
  absorbBlock64 state.sponge state.carry 0 0 0 state.carryCanonical
    (by decide) (by decide) (by decide)

private theorem absorbCarry64_denote (state : HashState64) :
    (absorbCarry64 state).denote =
      Poseidon2.absorbBlock state.sponge.denote [state.carry.denote] := by
  rw [absorbCarry64, absorbBlock64_denote]
  unfold Poseidon2.absorbBlock
  apply congrArg Poseidon2.permute
  apply List.ext_get
  · simp [Poseidon2.width]
  · intro index leftLt rightLt
    have indexLt : index < 8 := by simpa [State64.denote] using leftLt
    interval_cases index <;>
      simp [Poseidon2.width, State64.denote, UInt64.denote, Poseidon2.ofNat]

private def finalState64 (state : HashState64) : State64 :=
  State64.permute64 (pad64 (absorbCarry64 state))

private theorem finalState64_denote (state : HashState64) :
    (finalState64 state).denote =
      let absorbed := Poseidon2.absorbBlock state.sponge.denote [state.carry.denote]
      Poseidon2.permute ((List.range Poseidon2.width).map fun lane =>
        if lane = 0 then absorbed.getD 0 0 + 1 else absorbed.getD lane 0) := by
  rw [finalState64, State64.permute64_denote, pad64_denote,
    absorbCarry64_denote]

/-- Final carry absorption, pad permutation, and four-word squeeze. -/
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
