import NightstreamFPrime.Export.NativePoseidon2RoundCore

namespace NightstreamFPrime.Export.NativePoseidon2

open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open Fin.CommRing

namespace State64

private def initialConstant0 : State64 :=
  ⟨17625877419914948618, 7043359134273770980, 8324781800510000080, 6444119057261225485,
    17717541571092793106, 3986291954004622034, 690229812634116840, 12411915661887220387,
    6363654443139256914, 12105449363599063910, 3497590068843640987, 4222579949428206412,
    9871869144392188158, 6608483792230178088, 1361840775413654458, 8678242877107847347, by decide⟩
private def initialConstant1 : State64 :=
  ⟨10884132779455466827, 2140996594960535156, 6875127883761382716, 2527369230034983186,
    13363996391325744878, 8922005884211686156, 7168778978332361576, 856389527476709175,
    737755066540146832, 13556839930373010819, 10871866087908657800, 9591461885764716448,
    12537801773373699941, 3260007846858055737, 13363583202969484245, 4218638171078850311, by decide⟩
private def initialConstant2 : State64 :=
  ⟨7865032564572175278, 836093766369726329, 12527643047030536902, 1194545980657409763,
    12619589965719950681, 4280088488896281756, 15089781684263861043, 15828500408936832218,
    6419605478123467032, 2370996673254346830, 8789802724387314278, 14988192665513313034,
    4090662135633155679, 8312064199888872985, 13543758618702640312, 16933079866928560805, by decide⟩
private def initialConstant3 : State64 :=
  ⟨14849793775896549215, 5585003427041122008, 4039188622500476663, 8473588348029641541,
    14983429621133257453, 10213932717772184020, 14126063598418757144, 9149271627910874493,
    10568511515541199945, 8217860339194175559, 17410842106438810972, 9293755649055318357,
    4035347364804663719, 680576525761979526, 3925491009028150975, 161661792350051539, by decide⟩

private def partialConstant00 : UInt64 := 1025323479709822526
private def partialConstant01 : UInt64 := 6111066563283471619
private def partialConstant02 : UInt64 := 11883645335560589882
private def partialConstant03 : UInt64 := 5173392838145480224
private def partialConstant04 : UInt64 := 12046738740207396666
private def partialConstant05 : UInt64 := 11342877365108905038
private def partialConstant06 : UInt64 := 9954226806870194281
private def partialConstant07 : UInt64 := 284817437666484542
private def partialConstant08 : UInt64 := 7855238300370871153
private def partialConstant09 : UInt64 := 18340977697387037787
private def partialConstant10 : UInt64 := 12536785651043586535
private def partialConstant11 : UInt64 := 16922285383409970114
private def partialConstant12 : UInt64 := 6304755170918535876
private def partialConstant13 : UInt64 := 11920214520673410344
private def partialConstant14 : UInt64 := 15734762412768547841
private def partialConstant15 : UInt64 := 244322478157940423
private def partialConstant16 : UInt64 := 15954689516362665467
private def partialConstant17 : UInt64 := 17014463839077801966
private def partialConstant18 : UInt64 := 7540924580042546453
private def partialConstant19 : UInt64 := 8685971293749213383
private def partialConstant20 : UInt64 := 11638412499393676635
private def partialConstant21 : UInt64 := 9220075213657617084
private def partialConstant22 : UInt64 := 11350647073967129101
private def partialConstant23 : UInt64 := 10320484671217327025
private def partialConstant24 : UInt64 := 945321669835615161
private def partialConstant25 : UInt64 := 13689403322670363205
private def partialConstant26 : UInt64 := 9683142702532056693
private def partialConstant27 : UInt64 := 5152317403353772282
private def partialConstant28 : UInt64 := 12964243777216042008
private def partialConstant29 : UInt64 := 2572560616624767066
private def partialConstant30 : UInt64 := 5182608262343950270
private def partialConstant31 : UInt64 := 16474692687110834850
private def partialConstant32 : UInt64 := 743293326911585472
private def partialConstant33 : UInt64 := 1616451609802447080
private def partialConstant34 : UInt64 := 13870644991934200425
private def partialConstant35 : UInt64 := 6825873398398074331
private def partialConstant36 : UInt64 := 8937369057000642915
private def partialConstant37 : UInt64 := 13389634706674966780
private def partialConstant38 : UInt64 := 15982711626548638113
private def partialConstant39 : UInt64 := 12255367991153439320
private def partialConstant40 : UInt64 := 792159564585665664
private def partialConstant41 : UInt64 := 6095908108145386095
private def partialConstant42 : UInt64 := 13648899865260407704

private def terminalConstant0 : State64 :=
  ⟨56971392948291674, 6529889566834057128, 12351691752271381888, 17534431371875637294,
    6215672481047689667, 12917561096631389626, 8117077045244781265, 11392705445391554246,
    7980869059539648603, 8342592727398070895, 12598706612339968649, 14640463469680921020,
    2631864508405068938, 6769287453291648466, 10437499808121220125, 17140876581450999214, by decide⟩
private def terminalConstant1 : State64 :=
  ⟨13871899794127076084, 2976067163615152067, 17978719630500811223, 1697476385485035771,
    4624025223533863106, 16863109693143641675, 11259424642272530908, 12021139148564917949,
    9937415099130005641, 13718025083052700970, 6859334593920279903, 11330696051914832680,
    7582076360953008071, 8878013572368147473, 2039859730827428652, 7936695803456373494, by decide⟩
private def terminalConstant2 : State64 :=
  ⟨11613291057305099669, 551761998716281552, 7542914500287710423, 8812857942695644507,
    1690102753676828056, 15311849568606065067, 8830487934816023360, 6755221570880506368,
    6171890655123760836, 12545410498335876764, 9618969152146346251, 17122735410931870227,
    16462569480944052768, 6003165932848932655, 7670196776380585973, 4049877194394998303, by decide⟩
private def terminalConstant3 : State64 :=
  ⟨17998679892441632083, 2244367866900113196, 9247992278828546609, 9597834897550131057,
    10546074231789401477, 4261339652036653761, 13818713594235094297, 2019448978527439247,
    14018620292311326206, 3735807046906337575, 12052117347272897982, 7812511896467295333,
    9683662180518309794, 15034282915069102610, 9372870626443950502, 15048534190854756183, by decide⟩

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
private theorem partialRound22_denote (s : State64) :
    (partialRound64 s partialConstant22).denote = Poseidon2.partialRound 22 s.denote :=
  partialRound64_denote_at 22 _ (by decide) (by decide) s
private theorem partialRound23_denote (s : State64) :
    (partialRound64 s partialConstant23).denote = Poseidon2.partialRound 23 s.denote :=
  partialRound64_denote_at 23 _ (by decide) (by decide) s
private theorem partialRound24_denote (s : State64) :
    (partialRound64 s partialConstant24).denote = Poseidon2.partialRound 24 s.denote :=
  partialRound64_denote_at 24 _ (by decide) (by decide) s
private theorem partialRound25_denote (s : State64) :
    (partialRound64 s partialConstant25).denote = Poseidon2.partialRound 25 s.denote :=
  partialRound64_denote_at 25 _ (by decide) (by decide) s
private theorem partialRound26_denote (s : State64) :
    (partialRound64 s partialConstant26).denote = Poseidon2.partialRound 26 s.denote :=
  partialRound64_denote_at 26 _ (by decide) (by decide) s
private theorem partialRound27_denote (s : State64) :
    (partialRound64 s partialConstant27).denote = Poseidon2.partialRound 27 s.denote :=
  partialRound64_denote_at 27 _ (by decide) (by decide) s
private theorem partialRound28_denote (s : State64) :
    (partialRound64 s partialConstant28).denote = Poseidon2.partialRound 28 s.denote :=
  partialRound64_denote_at 28 _ (by decide) (by decide) s
private theorem partialRound29_denote (s : State64) :
    (partialRound64 s partialConstant29).denote = Poseidon2.partialRound 29 s.denote :=
  partialRound64_denote_at 29 _ (by decide) (by decide) s
private theorem partialRound30_denote (s : State64) :
    (partialRound64 s partialConstant30).denote = Poseidon2.partialRound 30 s.denote :=
  partialRound64_denote_at 30 _ (by decide) (by decide) s
private theorem partialRound31_denote (s : State64) :
    (partialRound64 s partialConstant31).denote = Poseidon2.partialRound 31 s.denote :=
  partialRound64_denote_at 31 _ (by decide) (by decide) s
private theorem partialRound32_denote (s : State64) :
    (partialRound64 s partialConstant32).denote = Poseidon2.partialRound 32 s.denote :=
  partialRound64_denote_at 32 _ (by decide) (by decide) s
private theorem partialRound33_denote (s : State64) :
    (partialRound64 s partialConstant33).denote = Poseidon2.partialRound 33 s.denote :=
  partialRound64_denote_at 33 _ (by decide) (by decide) s
private theorem partialRound34_denote (s : State64) :
    (partialRound64 s partialConstant34).denote = Poseidon2.partialRound 34 s.denote :=
  partialRound64_denote_at 34 _ (by decide) (by decide) s
private theorem partialRound35_denote (s : State64) :
    (partialRound64 s partialConstant35).denote = Poseidon2.partialRound 35 s.denote :=
  partialRound64_denote_at 35 _ (by decide) (by decide) s
private theorem partialRound36_denote (s : State64) :
    (partialRound64 s partialConstant36).denote = Poseidon2.partialRound 36 s.denote :=
  partialRound64_denote_at 36 _ (by decide) (by decide) s
private theorem partialRound37_denote (s : State64) :
    (partialRound64 s partialConstant37).denote = Poseidon2.partialRound 37 s.denote :=
  partialRound64_denote_at 37 _ (by decide) (by decide) s
private theorem partialRound38_denote (s : State64) :
    (partialRound64 s partialConstant38).denote = Poseidon2.partialRound 38 s.denote :=
  partialRound64_denote_at 38 _ (by decide) (by decide) s
private theorem partialRound39_denote (s : State64) :
    (partialRound64 s partialConstant39).denote = Poseidon2.partialRound 39 s.denote :=
  partialRound64_denote_at 39 _ (by decide) (by decide) s
private theorem partialRound40_denote (s : State64) :
    (partialRound64 s partialConstant40).denote = Poseidon2.partialRound 40 s.denote :=
  partialRound64_denote_at 40 _ (by decide) (by decide) s
private theorem partialRound41_denote (s : State64) :
    (partialRound64 s partialConstant41).denote = Poseidon2.partialRound 41 s.denote :=
  partialRound64_denote_at 41 _ (by decide) (by decide) s
private theorem partialRound42_denote (s : State64) :
    (partialRound64 s partialConstant42).denote = Poseidon2.partialRound 42 s.denote :=
  partialRound64_denote_at 42 _ (by decide) (by decide) s

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
  let state := partialRound64 state partialConstant21
  let state := partialRound64 state partialConstant22
  let state := partialRound64 state partialConstant23
  let state := partialRound64 state partialConstant24
  let state := partialRound64 state partialConstant25
  let state := partialRound64 state partialConstant26
  let state := partialRound64 state partialConstant27
  let state := partialRound64 state partialConstant28
  let state := partialRound64 state partialConstant29
  let state := partialRound64 state partialConstant30
  let state := partialRound64 state partialConstant31
  let state := partialRound64 state partialConstant32
  let state := partialRound64 state partialConstant33
  let state := partialRound64 state partialConstant34
  let state := partialRound64 state partialConstant35
  let state := partialRound64 state partialConstant36
  let state := partialRound64 state partialConstant37
  let state := partialRound64 state partialConstant38
  let state := partialRound64 state partialConstant39
  let state := partialRound64 state partialConstant40
  let state := partialRound64 state partialConstant41
  partialRound64 state partialConstant42

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
  rw [partialRound42_denote, partialRound41_denote, partialRound40_denote,
    partialRound39_denote, partialRound38_denote, partialRound37_denote, partialRound36_denote,
    partialRound35_denote, partialRound34_denote, partialRound33_denote, partialRound32_denote,
    partialRound31_denote, partialRound30_denote, partialRound29_denote, partialRound28_denote,
    partialRound27_denote, partialRound26_denote, partialRound25_denote, partialRound24_denote,
    partialRound23_denote, partialRound22_denote, partialRound21_denote, partialRound20_denote,
    partialRound19_denote, partialRound18_denote, partialRound17_denote, partialRound16_denote,
    partialRound15_denote, partialRound14_denote, partialRound13_denote, partialRound12_denote,
    partialRound11_denote, partialRound10_denote, partialRound09_denote, partialRound08_denote,
    partialRound07_denote, partialRound06_denote, partialRound05_denote, partialRound04_denote,
    partialRound03_denote, partialRound02_denote, partialRound01_denote, partialRound00_denote]
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
