import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Templates.Part0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Templates.Part1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Templates.Part2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Templates.Part3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Templates.Part4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Templates.Part5
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Templates.Part6
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Calls.Part0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Calls.Part1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.Calls.Part2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.OpeningGroups.Part0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.OpeningGroups.Part1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.OpeningGroups.Part2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.AliasLinks.Part0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.AliasLinks.Part1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.AliasConsumers.Part0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.AliasConsumers.Part1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated.AliasConsumers.Part2

/-! Generated file: complete compact private-decoder registry. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated

def schemaVersion : Nat := 1
def sourceStart : Nat := 257
def sourceStop : Nat := 10997363
def finalStart : Nat := 311
def finalStop : Nat := 10340178
def templatesPerChunk : Nat := 8
def recordsPerChunk : Nat := 256
def aliasConsumersPerChunk : Nat := 128
def templateCount : Nat := 555
def templateAtomCount : Nat := 17739
def callCount : Nat := 3549
def batchCount : Nat := 12
def openingGroupCount : Nat := 1176
def aliasLinkCount : Nat := 1505
def aliasConsumerCount : Nat := 1493
def rootSummary : RawSummary := { sourceColumns := 10997106, freshCoordinates := 10339867, census := { eliminated := 3963194, unit := 6863364, balanced := 170295, binary := 253, decompositionAliases := 3459864, equalityAliases := 1760, equalityAliasSavings := 61920, retainedCoordinatesBeforeAliases := 13861651, centeredColumns := 3454708 } }

def templateChunkContexts : List (RawTemplateChunkContext) := [
  { templateStart := 0, templateStop := 8, atomCount := 256 }
,   { templateStart := 8, templateStop := 16, atomCount := 256 }
,   { templateStart := 16, templateStop := 24, atomCount := 256 }
,   { templateStart := 24, templateStop := 32, atomCount := 256 }
,   { templateStart := 32, templateStop := 40, atomCount := 256 }
,   { templateStart := 40, templateStop := 48, atomCount := 256 }
,   { templateStart := 48, templateStop := 56, atomCount := 256 }
,   { templateStart := 56, templateStop := 64, atomCount := 256 }
,   { templateStart := 64, templateStop := 72, atomCount := 256 }
,   { templateStart := 72, templateStop := 80, atomCount := 256 }
,   { templateStart := 80, templateStop := 88, atomCount := 256 }
,   { templateStart := 88, templateStop := 96, atomCount := 256 }
,   { templateStart := 96, templateStop := 104, atomCount := 256 }
,   { templateStart := 104, templateStop := 112, atomCount := 256 }
,   { templateStart := 112, templateStop := 120, atomCount := 256 }
,   { templateStart := 120, templateStop := 128, atomCount := 256 }
,   { templateStart := 128, templateStop := 136, atomCount := 256 }
,   { templateStart := 136, templateStop := 144, atomCount := 256 }
,   { templateStart := 144, templateStop := 152, atomCount := 256 }
,   { templateStart := 152, templateStop := 160, atomCount := 256 }
,   { templateStart := 160, templateStop := 168, atomCount := 256 }
,   { templateStart := 168, templateStop := 176, atomCount := 256 }
,   { templateStart := 176, templateStop := 184, atomCount := 256 }
,   { templateStart := 184, templateStop := 192, atomCount := 256 }
,   { templateStart := 192, templateStop := 200, atomCount := 256 }
,   { templateStart := 200, templateStop := 208, atomCount := 256 }
,   { templateStart := 208, templateStop := 216, atomCount := 256 }
,   { templateStart := 216, templateStop := 224, atomCount := 256 }
,   { templateStart := 224, templateStop := 232, atomCount := 256 }
,   { templateStart := 232, templateStop := 240, atomCount := 256 }
,   { templateStart := 240, templateStop := 248, atomCount := 256 }
,   { templateStart := 248, templateStop := 256, atomCount := 256 }
,   { templateStart := 256, templateStop := 264, atomCount := 256 }
,   { templateStart := 264, templateStop := 272, atomCount := 256 }
,   { templateStart := 272, templateStop := 280, atomCount := 256 }
,   { templateStart := 280, templateStop := 288, atomCount := 256 }
,   { templateStart := 288, templateStop := 296, atomCount := 256 }
,   { templateStart := 296, templateStop := 304, atomCount := 256 }
,   { templateStart := 304, templateStop := 312, atomCount := 256 }
,   { templateStart := 312, templateStop := 320, atomCount := 256 }
,   { templateStart := 320, templateStop := 328, atomCount := 256 }
,   { templateStart := 328, templateStop := 336, atomCount := 256 }
,   { templateStart := 336, templateStop := 344, atomCount := 256 }
,   { templateStart := 344, templateStop := 352, atomCount := 256 }
,   { templateStart := 352, templateStop := 360, atomCount := 256 }
,   { templateStart := 360, templateStop := 368, atomCount := 256 }
,   { templateStart := 368, templateStop := 376, atomCount := 256 }
,   { templateStart := 376, templateStop := 384, atomCount := 256 }
,   { templateStart := 384, templateStop := 392, atomCount := 256 }
,   { templateStart := 392, templateStop := 400, atomCount := 256 }
,   { templateStart := 400, templateStop := 408, atomCount := 256 }
,   { templateStart := 408, templateStop := 416, atomCount := 256 }
,   { templateStart := 416, templateStop := 424, atomCount := 256 }
,   { templateStart := 424, templateStop := 432, atomCount := 256 }
,   { templateStart := 432, templateStop := 440, atomCount := 256 }
,   { templateStart := 440, templateStop := 448, atomCount := 256 }
,   { templateStart := 448, templateStop := 456, atomCount := 256 }
,   { templateStart := 456, templateStop := 464, atomCount := 256 }
,   { templateStart := 464, templateStop := 472, atomCount := 256 }
,   { templateStart := 472, templateStop := 480, atomCount := 256 }
,   { templateStart := 480, templateStop := 488, atomCount := 256 }
,   { templateStart := 488, templateStop := 496, atomCount := 256 }
,   { templateStart := 496, templateStop := 504, atomCount := 256 }
,   { templateStart := 504, templateStop := 512, atomCount := 256 }
,   { templateStart := 512, templateStop := 520, atomCount := 256 }
,   { templateStart := 520, templateStop := 528, atomCount := 256 }
,   { templateStart := 528, templateStop := 536, atomCount := 256 }
,   { templateStart := 536, templateStop := 544, atomCount := 256 }
,   { templateStart := 544, templateStop := 552, atomCount := 256 }
,   { templateStart := 552, templateStop := 555, atomCount := 75 }
]

def callChunkContexts : List (RawCallChunkContext) := [
  { callStart := 0, callStop := 256, sourceStart := 257, sourceStop := 169518, finalStart := 311, finalStop := 2313755 }
,   { callStart := 256, callStop := 512, sourceStart := 169518, sourceStop := 3764279, finalStart := 2313755, finalStop := 3664135 }
,   { callStart := 512, callStop := 768, sourceStart := 3764279, sourceStop := 3792111, finalStart := 3664135, finalStop := 3839779 }
,   { callStart := 768, callStop := 1024, sourceStart := 3792111, sourceStop := 3879061, finalStart := 3839779, finalStop := 4050192 }
,   { callStart := 1024, callStop := 1280, sourceStart := 3879061, sourceStop := 3907096, finalStart := 4050192, finalStop := 4217759 }
,   { callStart := 1280, callStop := 1536, sourceStart := 3907096, sourceStop := 3935144, finalStart := 4217759, finalStop := 4385285 }
,   { callStart := 1536, callStop := 1792, sourceStart := 3935144, sourceStop := 3963198, finalStart := 4385285, finalStop := 4552606 }
,   { callStart := 1792, callStop := 2048, sourceStart := 3963198, sourceStop := 3991291, finalStart := 4552606, finalStop := 4731038 }
,   { callStart := 2048, callStop := 2304, sourceStart := 3991291, sourceStop := 4081011, finalStart := 4731038, finalStop := 4943541 }
,   { callStart := 2304, callStop := 2560, sourceStart := 4081011, sourceStop := 6939797, finalStart := 4943541, finalStop := 6222525 }
,   { callStart := 2560, callStop := 2816, sourceStart := 6939797, sourceStop := 6983134, finalStart := 6222525, finalStop := 6798066 }
,   { callStart := 2816, callStop := 3072, sourceStart := 6983134, sourceStop := 7025596, finalStart := 6798066, finalStop := 7325694 }
,   { callStart := 3072, callStop := 3328, sourceStart := 7025596, sourceStop := 7696066, finalStart := 7325694, finalStop := 9117065 }
,   { callStart := 3328, callStop := 3549, sourceStart := 7696066, sourceStop := 10997363, finalStart := 9117065, finalStop := 10340178 }
]

def openingGroupShardContexts : List (RawOpeningGroupShardContext) := [
  { batch := 0, groupStart := 0, groupStop := 7, openingStart := 0, openingStop := 497, directStart := 0, directStop := 256 }
,   { batch := 1, groupStart := 0, groupStop := 1, openingStart := 0, openingStop := 108, directStart := 0, directStop := 0 }
,   { batch := 2, groupStart := 0, groupStop := 39, openingStart := 0, openingStop := 2302, directStart := 0, directStop := 0 }
,   { batch := 3, groupStart := 0, groupStop := 1, openingStart := 0, openingStop := 108, directStart := 0, directStop := 0 }
,   { batch := 4, groupStart := 0, groupStop := 256, openingStart := 0, openingStop := 25106, directStart := 0, directStop := 0 }
,   { batch := 4, groupStart := 256, groupStop := 272, openingStart := 25106, openingStop := 26711, directStart := 0, directStop := 0 }
,   { batch := 5, groupStart := 0, groupStop := 1, openingStart := 0, openingStop := 108, directStart := 0, directStop := 0 }
,   { batch := 6, groupStart := 0, groupStop := 256, openingStart := 0, openingStop := 14050, directStart := 0, directStop := 0 }
,   { batch := 6, groupStart := 256, groupStop := 420, openingStart := 14050, openingStop := 23033, directStart := 0, directStop := 0 }
,   { batch := 7, groupStart := 0, groupStop := 1, openingStart := 0, openingStop := 108, directStart := 0, directStop := 0 }
,   { batch := 8, groupStart := 0, groupStop := 142, openingStart := 0, openingStop := 4454, directStart := 0, directStop := 0 }
,   { batch := 9, groupStart := 0, groupStop := 1, openingStart := 0, openingStop := 108, directStart := 0, directStop := 0 }
,   { batch := 10, groupStart := 0, groupStop := 256, openingStart := 0, openingStop := 25106, directStart := 0, directStop := 0 }
,   { batch := 10, groupStart := 256, groupStop := 290, openingStart := 25106, openingStop := 26603, directStart := 0, directStop := 0 }
,   { batch := 11, groupStart := 0, groupStop := 1, openingStart := 0, openingStop := 108, directStart := 0, directStop := 0 }
]

def aliasLinkChunkContexts : List (RawAliasLinkChunkContext) := [
  { linkStart := 0, linkStop := 256 }
,   { linkStart := 256, linkStop := 512 }
,   { linkStart := 512, linkStop := 768 }
,   { linkStart := 768, linkStop := 1024 }
,   { linkStart := 1024, linkStop := 1280 }
,   { linkStart := 1280, linkStop := 1505 }
]

def aliasConsumerChunkContexts : List (RawAliasConsumerChunkContext) := [
  { consumerStart := 0, consumerStop := 128, linkStart := 0, linkStop := 140 }
,   { consumerStart := 128, consumerStop := 256, linkStart := 140, linkStop := 268 }
,   { consumerStart := 256, consumerStop := 384, linkStart := 268, linkStop := 396 }
,   { consumerStart := 384, consumerStop := 512, linkStart := 396, linkStop := 524 }
,   { consumerStart := 512, consumerStop := 640, linkStart := 524, linkStop := 652 }
,   { consumerStart := 640, consumerStop := 768, linkStart := 652, linkStop := 780 }
,   { consumerStart := 768, consumerStop := 896, linkStart := 780, linkStop := 908 }
,   { consumerStart := 896, consumerStop := 1024, linkStart := 908, linkStop := 1036 }
,   { consumerStart := 1024, consumerStop := 1152, linkStart := 1036, linkStop := 1164 }
,   { consumerStart := 1152, consumerStop := 1280, linkStart := 1164, linkStop := 1292 }
,   { consumerStart := 1280, consumerStop := 1408, linkStart := 1292, linkStop := 1420 }
,   { consumerStart := 1408, consumerStop := 1493, linkStart := 1420, linkStop := 1505 }
]

def templateChunks : List (List RawTemplate) := [
  TemplatePart0.chunk0
,   TemplatePart0.chunk1
,   TemplatePart0.chunk2
,   TemplatePart0.chunk3
,   TemplatePart0.chunk4
,   TemplatePart0.chunk5
,   TemplatePart0.chunk6
,   TemplatePart0.chunk7
,   TemplatePart0.chunk8
,   TemplatePart0.chunk9
,   TemplatePart1.chunk10
,   TemplatePart1.chunk11
,   TemplatePart1.chunk12
,   TemplatePart1.chunk13
,   TemplatePart1.chunk14
,   TemplatePart1.chunk15
,   TemplatePart1.chunk16
,   TemplatePart1.chunk17
,   TemplatePart1.chunk18
,   TemplatePart1.chunk19
,   TemplatePart2.chunk20
,   TemplatePart2.chunk21
,   TemplatePart2.chunk22
,   TemplatePart2.chunk23
,   TemplatePart2.chunk24
,   TemplatePart2.chunk25
,   TemplatePart2.chunk26
,   TemplatePart2.chunk27
,   TemplatePart2.chunk28
,   TemplatePart2.chunk29
,   TemplatePart3.chunk30
,   TemplatePart3.chunk31
,   TemplatePart3.chunk32
,   TemplatePart3.chunk33
,   TemplatePart3.chunk34
,   TemplatePart3.chunk35
,   TemplatePart3.chunk36
,   TemplatePart3.chunk37
,   TemplatePart3.chunk38
,   TemplatePart3.chunk39
,   TemplatePart4.chunk40
,   TemplatePart4.chunk41
,   TemplatePart4.chunk42
,   TemplatePart4.chunk43
,   TemplatePart4.chunk44
,   TemplatePart4.chunk45
,   TemplatePart4.chunk46
,   TemplatePart4.chunk47
,   TemplatePart4.chunk48
,   TemplatePart4.chunk49
,   TemplatePart5.chunk50
,   TemplatePart5.chunk51
,   TemplatePart5.chunk52
,   TemplatePart5.chunk53
,   TemplatePart5.chunk54
,   TemplatePart5.chunk55
,   TemplatePart5.chunk56
,   TemplatePart5.chunk57
,   TemplatePart5.chunk58
,   TemplatePart5.chunk59
,   TemplatePart6.chunk60
,   TemplatePart6.chunk61
,   TemplatePart6.chunk62
,   TemplatePart6.chunk63
,   TemplatePart6.chunk64
,   TemplatePart6.chunk65
,   TemplatePart6.chunk66
,   TemplatePart6.chunk67
,   TemplatePart6.chunk68
,   TemplatePart6.chunk69
]

def callChunks : List (List RawCall) := [
  CallPart0.chunk0
,   CallPart0.chunk1
,   CallPart0.chunk2
,   CallPart0.chunk3
,   CallPart0.chunk4
,   CallPart1.chunk5
,   CallPart1.chunk6
,   CallPart1.chunk7
,   CallPart1.chunk8
,   CallPart1.chunk9
,   CallPart2.chunk10
,   CallPart2.chunk11
,   CallPart2.chunk12
,   CallPart2.chunk13
]

def openingGroupShards : List (List RawOpeningGroup) := [
  OpeningGroupPart0.chunk0
,   OpeningGroupPart0.chunk1
,   OpeningGroupPart0.chunk2
,   OpeningGroupPart0.chunk3
,   OpeningGroupPart0.chunk4
,   OpeningGroupPart1.chunk5
,   OpeningGroupPart1.chunk6
,   OpeningGroupPart1.chunk7
,   OpeningGroupPart1.chunk8
,   OpeningGroupPart1.chunk9
,   OpeningGroupPart2.chunk10
,   OpeningGroupPart2.chunk11
,   OpeningGroupPart2.chunk12
,   OpeningGroupPart2.chunk13
,   OpeningGroupPart2.chunk14
]

def aliasLinkChunks : List (List RawAliasLink) := [
  AliasLinkPart0.chunk0
,   AliasLinkPart0.chunk1
,   AliasLinkPart0.chunk2
,   AliasLinkPart0.chunk3
,   AliasLinkPart0.chunk4
,   AliasLinkPart1.chunk5
]

def aliasConsumerChunks : List (List RawAliasConsumer) := [
  AliasConsumerPart0.chunk0
,   AliasConsumerPart0.chunk1
,   AliasConsumerPart0.chunk2
,   AliasConsumerPart0.chunk3
,   AliasConsumerPart0.chunk4
,   AliasConsumerPart1.chunk5
,   AliasConsumerPart1.chunk6
,   AliasConsumerPart1.chunk7
,   AliasConsumerPart1.chunk8
,   AliasConsumerPart1.chunk9
,   AliasConsumerPart2.chunk10
,   AliasConsumerPart2.chunk11
]

def batches : List (RawBatch) := [
  { sourceStart := 92723, sourceEnd := 153467, inputBinding := true, commitmentFields := 108, openings := 497, directOpenings := 256, groupShardStart := 0, groupShardCount := 1 }
,   { sourceStart := 153467, sourceEnd := 166699, inputBinding := false, commitmentFields := 54, openings := 108, directOpenings := 0, groupShardStart := 1, groupShardCount := 1 }
,   { sourceStart := 177011, sourceEnd := 457965, inputBinding := true, commitmentFields := 108, openings := 2302, directOpenings := 0, groupShardStart := 2, groupShardCount := 1 }
,   { sourceStart := 457965, sourceEnd := 471197, inputBinding := false, commitmentFields := 54, openings := 108, directOpenings := 0, groupShardStart := 3, groupShardCount := 1 }
,   { sourceStart := 487528, sourceEnd := 3746380, inputBinding := true, commitmentFields := 108, openings := 26711, directOpenings := 0, groupShardStart := 4, groupShardCount := 2 }
,   { sourceStart := 3746380, sourceEnd := 3759612, inputBinding := false, commitmentFields := 54, openings := 108, directOpenings := 0, groupShardStart := 6, groupShardCount := 1 }
,   { sourceStart := 4082284, sourceEnd := 6892420, inputBinding := true, commitmentFields := 108, openings := 23033, directOpenings := 0, groupShardStart := 7, groupShardCount := 2 }
,   { sourceStart := 6892420, sourceEnd := 6905652, inputBinding := false, commitmentFields := 54, openings := 108, directOpenings := 0, groupShardStart := 9, groupShardCount := 1 }
,   { sourceStart := 7074162, sourceEnd := 7617660, inputBinding := true, commitmentFields := 108, openings := 4454, directOpenings := 0, groupShardStart := 10, groupShardCount := 1 }
,   { sourceStart := 7617660, sourceEnd := 7630892, inputBinding := false, commitmentFields := 54, openings := 108, directOpenings := 0, groupShardStart := 11, groupShardCount := 1 }
,   { sourceStart := 7722751, sourceEnd := 10968427, inputBinding := true, commitmentFields := 108, openings := 26603, directOpenings := 0, groupShardStart := 12, groupShardCount := 2 }
,   { sourceStart := 10968427, sourceEnd := 10981659, inputBinding := false, commitmentFields := 54, openings := 108, directOpenings := 0, groupShardStart := 14, groupShardCount := 1 }
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated
