import Nightstream.Implementation.R1CS.Core.TranscriptCertificate

/-! Generated exact recursive F-prime transcript owner. Hashes below are drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveTranscriptArtifact

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576

def rangeSha256 : String := "fc3f12b219bc321190c5111dae423624d5afe1d725eafb53b61ef3c3c89a792c"
def rowStart : Nat := 17497
def rowEnd : Nat := 30759
def rowCount : Nat := 13262

structure ContextColumns where
  vkFsDigest : List Nat
  piCcsHeader : List Nat
  chunkCount : Nat
  stepCount : Nat
  z0 : List Nat
  zi : List Nat
  pc : Nat
  semanticState : List Nat
  accumulatorDigest : List Nat
  publicTrace : List Nat
  initialSemanticState : List Nat
  nextChunkDigest : List Nat
deriving DecidableEq, Repr

def contextColumns : ContextColumns :=
  { vkFsDigest := [10834, 10835, 10836, 10837]
    piCcsHeader := [10838, 10839, 10840, 10841]
    chunkCount := 10842
    stepCount := 10843
    z0 := [10844, 10845, 10846, 10847]
    zi := [10848, 10849, 10850, 10851]
    pc := 10833
    semanticState := [10852, 10853, 10854, 10855]
    accumulatorDigest := [10856, 10857, 10858, 10859]
    publicTrace := [10860, 10861, 10862, 10863]
    initialSemanticState := [20, 21, 22, 23]
    nextChunkDigest := [10864, 10865, 10866, 10867] }

def freshPublicColumns : List (List Nat) := [[31266, 31267, 31268, 31269, 31270, 31271, 31272, 31273, 31274, 31275, 31276, 31277, 31278, 31279, 31280, 31281, 31282, 31283, 31284, 31285, 31286, 31287, 31288, 31289, 31290, 31291, 31292, 31293, 31294, 31295, 31296, 31297, 31298, 31299, 31300, 31301, 31302, 31303, 31304, 31305, 31306, 31307, 31308, 31309, 31310, 31311, 31312, 31313, 31314, 31315, 31316, 31317, 31318, 31319, 31320, 31321, 31322, 31323, 31324, 31325, 31326, 31327, 31328, 31329, 31330, 31331, 31332, 31333, 31334, 31335, 31336, 31337, 31338, 31339, 31340, 31341, 31342, 31343, 31344, 31345, 31346, 31347, 31348, 31349, 31350, 31351, 31352, 31353, 31354, 31355, 31356, 31357, 31358, 31359, 31360, 31361, 31362, 31363, 31364, 31365, 31366, 31367, 31368, 31369, 31370, 31371, 31372, 31373, 31374, 31375, 31376, 31377, 31378, 31379, 31380, 31381, 31382, 31383, 31384, 31385, 31386, 31387, 31388, 31389, 31390, 31391, 31392, 31393, 31394, 31395, 31396, 31397, 31398, 31399, 31400, 31401, 31402, 31403, 31404, 31405, 31406, 31407, 31408, 31409, 31410, 31411, 31412, 31413, 31414, 31415, 31416, 31417, 31418, 31419, 31420, 31421, 31422, 31423, 31424, 31425, 31426, 31427, 31428, 31429, 31430, 31431, 31432, 31433, 31434, 31435, 31436, 31437, 31438, 31439, 31440, 31441, 31442, 31443, 31444, 31445, 31446, 31447, 31448, 31449, 31450, 31451, 31452, 31453, 31454, 31455, 31456, 31457, 31458, 31459, 31460, 31461, 31462, 31463, 31464, 31465, 31466, 31467, 31468, 31469, 31470, 31471, 31472, 31473, 31474, 31475, 31476, 31477, 31478, 31479, 31480, 31481, 31482, 31483, 31484, 31485, 31486, 31487, 31488, 31489, 31490, 31491, 31492, 31493, 31494, 31495, 31496, 31497, 31498, 31499, 31500, 31501, 31502, 31503, 31504, 31505, 31506, 31507, 31508, 31509, 31510, 31511, 31512, 31513, 31514, 31515, 31516, 31517, 31518, 31519, 31520, 31521, 31522]]


def constantPins : List (Nat × Nat) := [(17030, 13352904953114469), (17031, 12662), (17032, 5588068993763335165), (17033, 8915949836372482089), (17034, 17173225159890609573), (17035, 7949739509201708592), (17036, 14220403973771685868), (17037, 3390541613068238826), (17038, 13), (17039, 28549272306736998), (17040, 126883524736559), (17641, 4), (18242, 21), (18243, 28549272306736998), (18244, 32478900775383087), (18845, 32199629150185567), (18846, 4), (19447, 22), (19448, 28549272306736998), (20049, 26858244947862319), (20050, 29659826228653923), (20051, 110), (20052, 1), (20653, 21), (20654, 28549272306736998), (20655, 27970959037461295), (21256, 31078106134377839), (21257, 1), (21858, 11), (21859, 28549272306736998), (21860, 811563567), (21861, 4), (23062, 14), (23063, 28549272306736998), (23064, 31078105948846639), (23065, 4), (24266, 10), (24267, 28549272306736998), (24268, 6516783), (24269, 1), (24870, 25), (24871, 28549272306736998), (24872, 32772462024684335), (24873, 32758250078167913), (25474, 1852399461), (25475, 4), (26076, 21), (26077, 28549272306736998), (26678, 29665233406746927), (26679, 31078106134701415), (26680, 4), (27881, 23), (27882, 28549272306736998), (27883, 27981936923602991), (27884, 26851600298570847), (27885, 28265), (28486, 4), (29087, 20), (29088, 28549272306736998), (29089, 26858244947862319), (29690, 128038971337060), (29691, 4)]

def call0 : Poseidon2Call.Call :=
  { rowStart := 11, rowEnd := 611, inputColumns := [17030, 17031, 17038, 17039, 17034, 17035, 17036, 17037], firstAllocatedColumn := 17041 }

def call1 : Poseidon2Call.Call :=
  { rowStart := 612, rowEnd := 1212, inputColumns := [17040, 17641, 10834, 10835, 17637, 17638, 17639, 17640], firstAllocatedColumn := 17642 }

def call2 : Poseidon2Call.Call :=
  { rowStart := 1215, rowEnd := 1815, inputColumns := [10836, 10837, 18242, 18243, 18238, 18239, 18240, 18241], firstAllocatedColumn := 18245 }

def call3 : Poseidon2Call.Call :=
  { rowStart := 1817, rowEnd := 2417, inputColumns := [18244, 18845, 18846, 10838, 18841, 18842, 18843, 18844], firstAllocatedColumn := 18847 }

def call4 : Poseidon2Call.Call :=
  { rowStart := 2419, rowEnd := 3019, inputColumns := [10839, 10840, 10841, 19447, 19443, 19444, 19445, 19446], firstAllocatedColumn := 19449 }

def call5 : Poseidon2Call.Call :=
  { rowStart := 3023, rowEnd := 3623, inputColumns := [19448, 20049, 20050, 20051, 20045, 20046, 20047, 20048], firstAllocatedColumn := 20053 }

def call6 : Poseidon2Call.Call :=
  { rowStart := 3626, rowEnd := 4226, inputColumns := [20052, 10842, 20653, 20654, 20649, 20650, 20651, 20652], firstAllocatedColumn := 20656 }

def call7 : Poseidon2Call.Call :=
  { rowStart := 4228, rowEnd := 4828, inputColumns := [20655, 21256, 21257, 10843, 21252, 21253, 21254, 21255], firstAllocatedColumn := 21258 }

def call8 : Poseidon2Call.Call :=
  { rowStart := 4832, rowEnd := 5432, inputColumns := [21858, 21859, 21860, 21861, 21854, 21855, 21856, 21857], firstAllocatedColumn := 21862 }

def call9 : Poseidon2Call.Call :=
  { rowStart := 5432, rowEnd := 6032, inputColumns := [10844, 10845, 10846, 10847, 22458, 22459, 22460, 22461], firstAllocatedColumn := 22462 }

def call10 : Poseidon2Call.Call :=
  { rowStart := 6036, rowEnd := 6636, inputColumns := [23062, 23063, 23064, 23065, 23058, 23059, 23060, 23061], firstAllocatedColumn := 23066 }

def call11 : Poseidon2Call.Call :=
  { rowStart := 6636, rowEnd := 7236, inputColumns := [10848, 10849, 10850, 10851, 23662, 23663, 23664, 23665], firstAllocatedColumn := 23666 }

def call12 : Poseidon2Call.Call :=
  { rowStart := 7240, rowEnd := 7840, inputColumns := [24266, 24267, 24268, 24269, 24262, 24263, 24264, 24265], firstAllocatedColumn := 24270 }

def call13 : Poseidon2Call.Call :=
  { rowStart := 7844, rowEnd := 8444, inputColumns := [10833, 24870, 24871, 24872, 24866, 24867, 24868, 24869], firstAllocatedColumn := 24874 }

def call14 : Poseidon2Call.Call :=
  { rowStart := 8446, rowEnd := 9046, inputColumns := [24873, 25474, 25475, 10852, 25470, 25471, 25472, 25473], firstAllocatedColumn := 25476 }

def call15 : Poseidon2Call.Call :=
  { rowStart := 9048, rowEnd := 9648, inputColumns := [10853, 10854, 10855, 26076, 26072, 26073, 26074, 26075], firstAllocatedColumn := 26078 }

def call16 : Poseidon2Call.Call :=
  { rowStart := 9651, rowEnd := 10251, inputColumns := [26077, 26678, 26679, 26680, 26674, 26675, 26676, 26677], firstAllocatedColumn := 26681 }

def call17 : Poseidon2Call.Call :=
  { rowStart := 10251, rowEnd := 10851, inputColumns := [10856, 10857, 10858, 10859, 27277, 27278, 27279, 27280], firstAllocatedColumn := 27281 }

def call18 : Poseidon2Call.Call :=
  { rowStart := 10856, rowEnd := 11456, inputColumns := [27881, 27882, 27883, 27884, 27877, 27878, 27879, 27880], firstAllocatedColumn := 27886 }

def call19 : Poseidon2Call.Call :=
  { rowStart := 11457, rowEnd := 12057, inputColumns := [27885, 28486, 10860, 10861, 28482, 28483, 28484, 28485], firstAllocatedColumn := 28487 }

def call20 : Poseidon2Call.Call :=
  { rowStart := 12060, rowEnd := 12660, inputColumns := [10862, 10863, 29087, 29088, 29083, 29084, 29085, 29086], firstAllocatedColumn := 29090 }

def call21 : Poseidon2Call.Call :=
  { rowStart := 12662, rowEnd := 13262, inputColumns := [29089, 29690, 29691, 10864, 29686, 29687, 29688, 29689], firstAllocatedColumn := 29692 }

def calls : List Poseidon2Call.Call := [call0, call1, call2, call3, call4, call5, call6, call7, call8, call9, call10, call11, call12, call13, call14, call15, call16, call17, call18, call19, call20, call21]

def trace : TranscriptCertificate.Trace := ⟨constantPins, calls⟩

def schedule : List TranscriptCertificate.PieceRef :=
  [.pin 0,
   .pin 1,
   .pin 2,
   .pin 3,
   .pin 4,
   .pin 5,
   .pin 6,
   .pin 7,
   .pin 8,
   .pin 9,
   .pin 10,
   .call 0,
   .pin 11,
   .call 1,
   .pin 12,
   .pin 13,
   .pin 14,
   .call 2,
   .pin 15,
   .pin 16,
   .call 3,
   .pin 17,
   .pin 18,
   .call 4,
   .pin 19,
   .pin 20,
   .pin 21,
   .pin 22,
   .call 5,
   .pin 23,
   .pin 24,
   .pin 25,
   .call 6,
   .pin 26,
   .pin 27,
   .call 7,
   .pin 28,
   .pin 29,
   .pin 30,
   .pin 31,
   .call 8,
   .call 9,
   .pin 32,
   .pin 33,
   .pin 34,
   .pin 35,
   .call 10,
   .call 11,
   .pin 36,
   .pin 37,
   .pin 38,
   .pin 39,
   .call 12,
   .pin 40,
   .pin 41,
   .pin 42,
   .pin 43,
   .call 13,
   .pin 44,
   .pin 45,
   .call 14,
   .pin 46,
   .pin 47,
   .call 15,
   .pin 48,
   .pin 49,
   .pin 50,
   .call 16,
   .call 17,
   .pin 51,
   .pin 52,
   .pin 53,
   .pin 54,
   .pin 55,
   .call 18,
   .pin 56,
   .call 19,
   .pin 57,
   .pin 58,
   .pin 59,
   .call 20,
   .pin 60,
   .pin 61,
   .call 21]

def rowPieces : List (List Row) :=
  schedule.map fun piece => piece.rows trace

def ownerRows : List Row := trace.orderedRows schedule

theorem ownerRows_length : ownerRows.length = rowCount := by native_decide

def pinIndicesBoundedCheck : Bool :=
  schedule.all fun piece =>
    match piece with
    | .pin index => decide (index < trace.pins.length)
    | .call _ => true

def callIndicesBoundedCheck : Bool :=
  schedule.all fun piece =>
    match piece with
    | .pin _ => true
    | .call index => decide (index < trace.calls.length)

def everyPinScheduledCheck : Bool :=
  (List.range trace.pins.length).all fun index =>
    decide (.pin index ∈ schedule)

def everyCallScheduledCheck : Bool :=
  (List.range trace.calls.length).all fun index =>
    decide (.call index ∈ schedule)

theorem pinIndicesBounded_checked : pinIndicesBoundedCheck = true := by native_decide
theorem callIndicesBounded_checked : callIndicesBoundedCheck = true := by native_decide
theorem everyPinScheduled_checked : everyPinScheduledCheck = true := by native_decide
theorem everyCallScheduled_checked : everyCallScheduledCheck = true := by native_decide

theorem traceValid : trace.OrderedValid schedule ownerRows where
  pinIndicesBounded := by
    intro index member
    exact of_decide_eq_true
      ((List.all_eq_true.mp pinIndicesBounded_checked) (.pin index) member)
  callIndicesBounded := by
    intro index member
    exact of_decide_eq_true
      ((List.all_eq_true.mp callIndicesBounded_checked) (.call index) member)
  everyPinScheduled := by
    intro index indexLt
    exact of_decide_eq_true
      ((List.all_eq_true.mp everyPinScheduled_checked) index
        (List.mem_range.mpr indexLt))
  everyCallScheduled := by
    intro index indexLt
    exact of_decide_eq_true
      ((List.all_eq_true.mp everyCallScheduled_checked) index
        (List.mem_range.mpr indexLt))
  pinValuesCanonical := by native_decide
  exactRows := rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveTranscriptArtifact
