/-
Generated file: production combined-NC execution artifact; do not hand-edit.

Owns: the compact direct-terminal raw-WitnessMat projection program and active post-PiDEC source-normalized pins.

Does not own: row satisfaction, commitment binding, semantic acceptance,
security reductions, costs, or permission to remove rows.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.execution` | The generated execution payload named by `Owns` above | computed artifact |
-/

import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Schema

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan

def schemaVersion : Nat := 1
def profileTag : Nat := 1
def pendingProjectionProfileTag : Nat := 1
def pendingProjectionJoinId : Nat := 1
def recursiveSelectorTag : Nat := 2
def radixBase : Nat := 2
def logicalWidth : Nat := 11437038
def packedRows : Nat := 54
def blockCount : Nat := 211797
def blockVariables : Nat := 19
def tensorVariables : Nat := 18
def blockDomainSize : Nat := 524288
def childCount : Nat := 14
def activeLanes : Nat := 54
def paddedLanes : Nat := 64
def virtualZeroLanes : Nat := 10
def witnessEntriesPerChild : Nat := 11437038
def witnessFamilyEntries : Nat := 160118532
def tensorMultiplications : Nat := 262143
def tensorRows : Nat := 1310715
def projectionProductRows : Nat := 22874076
def finalScaleMultiplications : Nat := 54
def finalScaleRows : Nat := 270
def derivedColumns : Nat := 24185061
def terminalRows : Nat := 108
def totalRows : Nat := 24185169
def tensorRelativeRowStart : Nat := 0
def projectionProductRelativeRowStart : Nat := 1310715
def terminalRelativeRowStart : Nat := 24185061
def tensorFirstColumnAfterWitnessFamily : Nat := 160118532
def projectionProductFirstColumnAfterWitnessFamily : Nat := 161429247
def finalScaleFirstColumnAfterWitnessFamily : Nat := 184303323
def factorFinalRound : Bool := true
def factoredVariable : Option Nat := some 18

/-- Row-major `FinalWitnessWires` offset: lane first, then packed block. -/
def witnessOffset (lane block : Nat) : Nat := lane * blockCount + block
def childWitnessRelativeColumn (child lane block : Nat) : Nat :=
child * witnessEntriesPerChild + witnessOffset lane block
def tensorRoundMulCount (round : Nat) : Nat := Nat.min blockCount (2 ^ round)
def tensorMulOrdinal (round parent : Nat) : Nat :=
(List.range round).foldl (fun count prior => count + tensorRoundMulCount prior) 0 + parent
def tensorMulFirstColumnAfterWitnessFamily (round parent : Nat) : Nat :=
tensorFirstColumnAfterWitnessFamily + 5 * tensorMulOrdinal round parent
def tensorMulOutputColumnsAfterWitnessFamily (round parent : Nat) : Nat × Nat :=
(tensorMulFirstColumnAfterWitnessFamily round parent + 3,
tensorMulFirstColumnAfterWitnessFamily round parent + 4)
def projectionProductRelativeRow (lane block limb : Nat) : Nat :=
projectionProductRelativeRowStart + 2 * (lane * blockCount + block) + limb
def projectionProductColumnAfterWitnessFamily (lane block limb : Nat) : Nat :=
projectionProductFirstColumnAfterWitnessFamily + 2 * (lane * blockCount + block) + limb
def finalScaleRelativeRow (lane definition : Nat) : Nat :=
projectionProductRelativeRowStart + projectionProductRows + 5 * lane + definition
def terminalRelativeRow (lane limb : Nat) : Nat :=
terminalRelativeRowStart + 2 * lane + limb

def tensorRoundMulCounts : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
def tensorRoundHighCounts : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 80725]
def tensorRoundRelativeRowStarts : List Nat := [0, 5, 15, 35, 75, 155, 315, 635, 1275, 2555, 5115, 10235, 20475, 40955, 81915, 163835, 327675, 655355]
def childWitnessRelativeBases : List Nat := [0, 11437038, 22874076, 34311114, 45748152, 57185190, 68622228, 80059266, 91496304, 102933342, 114370380, 125807418, 137244456, 148681494]
/-- Post-PiDEC source-normalized pins; not terminal absolute columns. -/
def pendingOldBlockSourceNormalizedColumns : List (Nat × Nat) := [(53626, 53627), (53628, 53629), (53630, 53631), (53632, 53633), (53634, 53635), (53636, 53637), (53638, 53639), (53640, 53641), (53642, 53643), (53644, 53645), (53646, 53647), (53648, 53649), (53650, 53651), (53652, 53653), (53654, 53655), (53656, 53657), (53658, 53659), (53660, 53661), (53662, 53663)]
/-- Post-PiDEC source-normalized pins; not terminal absolute columns. -/
def pendingParentYZcolSourceNormalizedColumns : List (Nat × Nat) := [(53664, 53665), (53666, 53667), (53668, 53669), (53670, 53671), (53672, 53673), (53674, 53675), (53676, 53677), (53678, 53679), (53680, 53681), (53682, 53683), (53684, 53685), (53686, 53687), (53688, 53689), (53690, 53691), (53692, 53693), (53694, 53695), (53696, 53697), (53698, 53699), (53700, 53701), (53702, 53703), (53704, 53705), (53706, 53707), (53708, 53709), (53710, 53711), (53712, 53713), (53714, 53715), (53716, 53717), (53718, 53719), (53720, 53721), (53722, 53723), (53724, 53725), (53726, 53727), (53728, 53729), (53730, 53731), (53732, 53733), (53734, 53735), (53736, 53737), (53738, 53739), (53740, 53741), (53742, 53743), (53744, 53745), (53746, 53747), (53748, 53749), (53750, 53751), (53752, 53753), (53754, 53755), (53756, 53757), (53758, 53759), (53760, 53761), (53762, 53763), (53764, 53765), (53766, 53767), (53768, 53769), (53770, 53771)]
def recursiveSelectorColumns : List Nat := [270, 271, 272]
def recursiveSelectorValues : List Nat := [0, 0, 1]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan
