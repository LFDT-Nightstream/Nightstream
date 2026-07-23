import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Certificates

/-!
Exact root certificate for the compact steady-recursive private decoder.

The four registry decisions inspect only bounded proof-free context headers. The
payload facts are the separately checked chunk theorems from `Certificates`;
this file combines those theorems in the kernel and never expands the
10,997,106-column decoder.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder

namespace EG

export Generated
  (schemaVersion sourceStart sourceStop finalStart finalStop templatesPerChunk
    recordsPerChunk aliasConsumersPerChunk templateCount templateAtomCount
    callCount batchCount openingGroupCount aliasLinkCount aliasConsumerCount
    rootSummary batches templateChunkContexts callChunkContexts
    openingGroupShardContexts aliasLinkChunkContexts aliasConsumerChunkContexts
    templateChunks callChunks openingGroupShards aliasLinkChunks
    aliasConsumerChunks)

end EG

set_option maxRecDepth 4096

theorem generatedHeader_exact :
    EG.schemaVersion = 1 ∧
    EG.sourceStart = 257 ∧
    EG.sourceStop = 10997363 ∧
    EG.finalStart = 311 ∧
    EG.finalStop = 10340178 ∧
    EG.templatesPerChunk = 8 ∧
    EG.recordsPerChunk = 256 ∧
    EG.aliasConsumersPerChunk = 128 ∧
    EG.templateCount = 555 ∧
    EG.templateAtomCount = 17739 ∧
    EG.callCount = 3549 ∧
    EG.batchCount = 12 ∧
    EG.openingGroupCount = 1176 ∧
    EG.aliasLinkCount = 1505 ∧
    EG.aliasConsumerCount = 1493 ∧
    EG.rootSummary.sourceColumns = EG.sourceStop - EG.sourceStart ∧
    EG.rootSummary.freshCoordinates = EG.finalStop - EG.finalStart := by
  native_decide

theorem generatedTemplateRegistryContext_exact :
    templateRegistryContextValid EG.templateChunkContexts EG.templateChunks
      EG.templateCount EG.templateAtomCount = true := by
  native_decide

theorem generatedCallRegistryContext_exact :
    callRegistryContextValid EG.callChunkContexts EG.callChunks
      EG.callCount EG.sourceStart EG.sourceStop EG.finalStart EG.finalStop = true := by
  native_decide

theorem generatedOpeningRegistryContext_exact :
    openingRegistryContextValid EG.batches EG.openingGroupShardContexts
      EG.openingGroupShards = true := by
  native_decide

theorem generatedAliasLinkRegistryContext_exact :
    aliasLinkRegistryContextValid EG.aliasLinkChunkContexts EG.aliasLinkChunks
      EG.aliasLinkCount = true := by
  native_decide

theorem generatedAliasConsumerRegistryContext_exact :
    aliasConsumerRegistryContextValid EG.aliasConsumerChunkContexts
      EG.aliasConsumerChunks EG.aliasConsumerCount EG.aliasLinkCount = true := by
  native_decide

private theorem fin70_cases {predicate : Fin 70 → Prop}
    (case0 : predicate 0)
    (case1 : predicate 1)
    (case2 : predicate 2)
    (case3 : predicate 3)
    (case4 : predicate 4)
    (case5 : predicate 5)
    (case6 : predicate 6)
    (case7 : predicate 7)
    (case8 : predicate 8)
    (case9 : predicate 9)
    (case10 : predicate 10)
    (case11 : predicate 11)
    (case12 : predicate 12)
    (case13 : predicate 13)
    (case14 : predicate 14)
    (case15 : predicate 15)
    (case16 : predicate 16)
    (case17 : predicate 17)
    (case18 : predicate 18)
    (case19 : predicate 19)
    (case20 : predicate 20)
    (case21 : predicate 21)
    (case22 : predicate 22)
    (case23 : predicate 23)
    (case24 : predicate 24)
    (case25 : predicate 25)
    (case26 : predicate 26)
    (case27 : predicate 27)
    (case28 : predicate 28)
    (case29 : predicate 29)
    (case30 : predicate 30)
    (case31 : predicate 31)
    (case32 : predicate 32)
    (case33 : predicate 33)
    (case34 : predicate 34)
    (case35 : predicate 35)
    (case36 : predicate 36)
    (case37 : predicate 37)
    (case38 : predicate 38)
    (case39 : predicate 39)
    (case40 : predicate 40)
    (case41 : predicate 41)
    (case42 : predicate 42)
    (case43 : predicate 43)
    (case44 : predicate 44)
    (case45 : predicate 45)
    (case46 : predicate 46)
    (case47 : predicate 47)
    (case48 : predicate 48)
    (case49 : predicate 49)
    (case50 : predicate 50)
    (case51 : predicate 51)
    (case52 : predicate 52)
    (case53 : predicate 53)
    (case54 : predicate 54)
    (case55 : predicate 55)
    (case56 : predicate 56)
    (case57 : predicate 57)
    (case58 : predicate 58)
    (case59 : predicate 59)
    (case60 : predicate 60)
    (case61 : predicate 61)
    (case62 : predicate 62)
    (case63 : predicate 63)
    (case64 : predicate 64)
    (case65 : predicate 65)
    (case66 : predicate 66)
    (case67 : predicate 67)
    (case68 : predicate 68)
    (case69 : predicate 69) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro index
  refine Fin.cases case5 ?_ index
  intro index
  refine Fin.cases case6 ?_ index
  intro index
  refine Fin.cases case7 ?_ index
  intro index
  refine Fin.cases case8 ?_ index
  intro index
  refine Fin.cases case9 ?_ index
  intro index
  refine Fin.cases case10 ?_ index
  intro index
  refine Fin.cases case11 ?_ index
  intro index
  refine Fin.cases case12 ?_ index
  intro index
  refine Fin.cases case13 ?_ index
  intro index
  refine Fin.cases case14 ?_ index
  intro index
  refine Fin.cases case15 ?_ index
  intro index
  refine Fin.cases case16 ?_ index
  intro index
  refine Fin.cases case17 ?_ index
  intro index
  refine Fin.cases case18 ?_ index
  intro index
  refine Fin.cases case19 ?_ index
  intro index
  refine Fin.cases case20 ?_ index
  intro index
  refine Fin.cases case21 ?_ index
  intro index
  refine Fin.cases case22 ?_ index
  intro index
  refine Fin.cases case23 ?_ index
  intro index
  refine Fin.cases case24 ?_ index
  intro index
  refine Fin.cases case25 ?_ index
  intro index
  refine Fin.cases case26 ?_ index
  intro index
  refine Fin.cases case27 ?_ index
  intro index
  refine Fin.cases case28 ?_ index
  intro index
  refine Fin.cases case29 ?_ index
  intro index
  refine Fin.cases case30 ?_ index
  intro index
  refine Fin.cases case31 ?_ index
  intro index
  refine Fin.cases case32 ?_ index
  intro index
  refine Fin.cases case33 ?_ index
  intro index
  refine Fin.cases case34 ?_ index
  intro index
  refine Fin.cases case35 ?_ index
  intro index
  refine Fin.cases case36 ?_ index
  intro index
  refine Fin.cases case37 ?_ index
  intro index
  refine Fin.cases case38 ?_ index
  intro index
  refine Fin.cases case39 ?_ index
  intro index
  refine Fin.cases case40 ?_ index
  intro index
  refine Fin.cases case41 ?_ index
  intro index
  refine Fin.cases case42 ?_ index
  intro index
  refine Fin.cases case43 ?_ index
  intro index
  refine Fin.cases case44 ?_ index
  intro index
  refine Fin.cases case45 ?_ index
  intro index
  refine Fin.cases case46 ?_ index
  intro index
  refine Fin.cases case47 ?_ index
  intro index
  refine Fin.cases case48 ?_ index
  intro index
  refine Fin.cases case49 ?_ index
  intro index
  refine Fin.cases case50 ?_ index
  intro index
  refine Fin.cases case51 ?_ index
  intro index
  refine Fin.cases case52 ?_ index
  intro index
  refine Fin.cases case53 ?_ index
  intro index
  refine Fin.cases case54 ?_ index
  intro index
  refine Fin.cases case55 ?_ index
  intro index
  refine Fin.cases case56 ?_ index
  intro index
  refine Fin.cases case57 ?_ index
  intro index
  refine Fin.cases case58 ?_ index
  intro index
  refine Fin.cases case59 ?_ index
  intro index
  refine Fin.cases case60 ?_ index
  intro index
  refine Fin.cases case61 ?_ index
  intro index
  refine Fin.cases case62 ?_ index
  intro index
  refine Fin.cases case63 ?_ index
  intro index
  refine Fin.cases case64 ?_ index
  intro index
  refine Fin.cases case65 ?_ index
  intro index
  refine Fin.cases case66 ?_ index
  intro index
  refine Fin.cases case67 ?_ index
  intro index
  refine Fin.cases case68 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = (0 : Fin 1) := Fin.ext valueZero
  subst index
  exact case69

private theorem fin14_cases {predicate : Fin 14 → Prop}
    (case0 : predicate 0)
    (case1 : predicate 1)
    (case2 : predicate 2)
    (case3 : predicate 3)
    (case4 : predicate 4)
    (case5 : predicate 5)
    (case6 : predicate 6)
    (case7 : predicate 7)
    (case8 : predicate 8)
    (case9 : predicate 9)
    (case10 : predicate 10)
    (case11 : predicate 11)
    (case12 : predicate 12)
    (case13 : predicate 13) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro index
  refine Fin.cases case5 ?_ index
  intro index
  refine Fin.cases case6 ?_ index
  intro index
  refine Fin.cases case7 ?_ index
  intro index
  refine Fin.cases case8 ?_ index
  intro index
  refine Fin.cases case9 ?_ index
  intro index
  refine Fin.cases case10 ?_ index
  intro index
  refine Fin.cases case11 ?_ index
  intro index
  refine Fin.cases case12 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = (0 : Fin 1) := Fin.ext valueZero
  subst index
  exact case13

private theorem fin15_cases {predicate : Fin 15 → Prop}
    (case0 : predicate 0)
    (case1 : predicate 1)
    (case2 : predicate 2)
    (case3 : predicate 3)
    (case4 : predicate 4)
    (case5 : predicate 5)
    (case6 : predicate 6)
    (case7 : predicate 7)
    (case8 : predicate 8)
    (case9 : predicate 9)
    (case10 : predicate 10)
    (case11 : predicate 11)
    (case12 : predicate 12)
    (case13 : predicate 13)
    (case14 : predicate 14) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro index
  refine Fin.cases case5 ?_ index
  intro index
  refine Fin.cases case6 ?_ index
  intro index
  refine Fin.cases case7 ?_ index
  intro index
  refine Fin.cases case8 ?_ index
  intro index
  refine Fin.cases case9 ?_ index
  intro index
  refine Fin.cases case10 ?_ index
  intro index
  refine Fin.cases case11 ?_ index
  intro index
  refine Fin.cases case12 ?_ index
  intro index
  refine Fin.cases case13 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = (0 : Fin 1) := Fin.ext valueZero
  subst index
  exact case14

private theorem fin6_cases {predicate : Fin 6 → Prop}
    (case0 : predicate 0)
    (case1 : predicate 1)
    (case2 : predicate 2)
    (case3 : predicate 3)
    (case4 : predicate 4)
    (case5 : predicate 5) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = (0 : Fin 1) := Fin.ext valueZero
  subst index
  exact case5

private theorem fin12_cases {predicate : Fin 12 → Prop}
    (case0 : predicate 0) (case1 : predicate 1)
    (case2 : predicate 2) (case3 : predicate 3)
    (case4 : predicate 4) (case5 : predicate 5)
    (case6 : predicate 6) (case7 : predicate 7)
    (case8 : predicate 8) (case9 : predicate 9)
    (case10 : predicate 10) (case11 : predicate 11) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro index
  refine Fin.cases case5 ?_ index
  intro index
  refine Fin.cases case6 ?_ index
  intro index
  refine Fin.cases case7 ?_ index
  intro index
  refine Fin.cases case8 ?_ index
  intro index
  refine Fin.cases case9 ?_ index
  intro index
  refine Fin.cases case10 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = (0 : Fin 1) := Fin.ext valueZero
  subst index
  exact case11

theorem generatedTemplateChunks_exact :
    ∀ index : Fin 70,
      templateChunkAtValid EG.batches EG.templateChunkContexts
        EG.templateChunks index.val = true :=
  fin70_cases
    templateChunk0_valid
    templateChunk1_valid
    templateChunk2_valid
    templateChunk3_valid
    templateChunk4_valid
    templateChunk5_valid
    templateChunk6_valid
    templateChunk7_valid
    templateChunk8_valid
    templateChunk9_valid
    templateChunk10_valid
    templateChunk11_valid
    templateChunk12_valid
    templateChunk13_valid
    templateChunk14_valid
    templateChunk15_valid
    templateChunk16_valid
    templateChunk17_valid
    templateChunk18_valid
    templateChunk19_valid
    templateChunk20_valid
    templateChunk21_valid
    templateChunk22_valid
    templateChunk23_valid
    templateChunk24_valid
    templateChunk25_valid
    templateChunk26_valid
    templateChunk27_valid
    templateChunk28_valid
    templateChunk29_valid
    templateChunk30_valid
    templateChunk31_valid
    templateChunk32_valid
    templateChunk33_valid
    templateChunk34_valid
    templateChunk35_valid
    templateChunk36_valid
    templateChunk37_valid
    templateChunk38_valid
    templateChunk39_valid
    templateChunk40_valid
    templateChunk41_valid
    templateChunk42_valid
    templateChunk43_valid
    templateChunk44_valid
    templateChunk45_valid
    templateChunk46_valid
    templateChunk47_valid
    templateChunk48_valid
    templateChunk49_valid
    templateChunk50_valid
    templateChunk51_valid
    templateChunk52_valid
    templateChunk53_valid
    templateChunk54_valid
    templateChunk55_valid
    templateChunk56_valid
    templateChunk57_valid
    templateChunk58_valid
    templateChunk59_valid
    templateChunk60_valid
    templateChunk61_valid
    templateChunk62_valid
    templateChunk63_valid
    templateChunk64_valid
    templateChunk65_valid
    templateChunk66_valid
    templateChunk67_valid
    templateChunk68_valid
    templateChunk69_valid

theorem generatedCallChunks_exact :
    ∀ index : Fin 14,
      callChunkAtValid EG.templateChunks EG.callChunkContexts
        EG.callChunks index.val = true :=
  fin14_cases
    callChunk0_valid
    callChunk1_valid
    callChunk2_valid
    callChunk3_valid
    callChunk4_valid
    callChunk5_valid
    callChunk6_valid
    callChunk7_valid
    callChunk8_valid
    callChunk9_valid
    callChunk10_valid
    callChunk11_valid
    callChunk12_valid
    callChunk13_valid

theorem generatedOpeningGroupShards_exact :
    ∀ index : Fin 15,
      openingGroupShardAtValid EG.openingGroupShardContexts
        EG.openingGroupShards index.val = true :=
  fin15_cases
    openingGroupShard0_valid
    openingGroupShard1_valid
    openingGroupShard2_valid
    openingGroupShard3_valid
    openingGroupShard4_valid
    openingGroupShard5_valid
    openingGroupShard6_valid
    openingGroupShard7_valid
    openingGroupShard8_valid
    openingGroupShard9_valid
    openingGroupShard10_valid
    openingGroupShard11_valid
    openingGroupShard12_valid
    openingGroupShard13_valid
    openingGroupShard14_valid

theorem generatedAliasLinkChunks_exact :
    ∀ index : Fin 6,
      aliasLinkChunkAtValid EG.batches EG.templateChunks EG.callChunks
        EG.openingGroupShards EG.aliasLinkChunkContexts
        EG.aliasLinkChunks index.val = true :=
  fin6_cases
    aliasLinkChunk0_valid
    aliasLinkChunk1_valid
    aliasLinkChunk2_valid
    aliasLinkChunk3_valid
    aliasLinkChunk4_valid
    aliasLinkChunk5_valid

theorem generatedAliasConsumerChunks_exact :
    ∀ index : Fin 12,
      aliasConsumerChunkAtValid EG.batches EG.templateChunks EG.callChunks
        EG.openingGroupShards EG.aliasLinkChunks EG.aliasConsumerChunkContexts
        EG.aliasConsumerChunks index.val = true :=
  fin12_cases
    aliasConsumerChunk0_valid
    aliasConsumerChunk1_valid
    aliasConsumerChunk2_valid
    aliasConsumerChunk3_valid
    aliasConsumerChunk4_valid
    aliasConsumerChunk5_valid
    aliasConsumerChunk6_valid
    aliasConsumerChunk7_valid
    aliasConsumerChunk8_valid
    aliasConsumerChunk9_valid
    aliasConsumerChunk10_valid
    aliasConsumerChunk11_valid

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder
