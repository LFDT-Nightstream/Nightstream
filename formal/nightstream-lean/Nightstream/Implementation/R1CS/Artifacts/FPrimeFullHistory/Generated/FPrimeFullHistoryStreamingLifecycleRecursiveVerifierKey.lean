import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKeySchema

/-! Generated exact source-to-final provenance for the recursive lifecycle verifier-key stage.

This is a compact leaf of the monolithic reference compiler audit. It is not the final phased profile.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey.Artifact

def artifactSha256 : String := "bbedccb76a9e57f42a76ccfb1eaf2a44e2abdd376e584d040b089f6bde7b3892"

def sourceRuns : List SourceRun := [
    { sourceRows := { start := 30664206, stop := 30664207 }, disposition := "LinearDefinition(SelectiveRewriteId(401988))", emittedStart := none },
    { sourceRows := { start := 30664207, stop := 30664208 }, disposition := "LinearDefinition(SelectiveRewriteId(401989))", emittedStart := none },
    { sourceRows := { start := 30664208, stop := 30664209 }, disposition := "LinearDefinition(SelectiveRewriteId(401990))", emittedStart := none },
    { sourceRows := { start := 30664209, stop := 30664210 }, disposition := "LinearDefinition(SelectiveRewriteId(401991))", emittedStart := none },
    { sourceRows := { start := 30664210, stop := 30664211 }, disposition := "LinearDefinition(SelectiveRewriteId(401992))", emittedStart := none },
    { sourceRows := { start := 30664211, stop := 30664212 }, disposition := "LinearDefinition(SelectiveRewriteId(401993))", emittedStart := none },
    { sourceRows := { start := 30664212, stop := 30664213 }, disposition := "LinearDefinition(SelectiveRewriteId(401994))", emittedStart := none },
    { sourceRows := { start := 30664213, stop := 30664214 }, disposition := "LinearDefinition(SelectiveRewriteId(401995))", emittedStart := none },
    { sourceRows := { start := 30664214, stop := 30664215 }, disposition := "LinearDefinition(SelectiveRewriteId(401996))", emittedStart := none },
    { sourceRows := { start := 30664215, stop := 30664216 }, disposition := "LinearDefinition(SelectiveRewriteId(401997))", emittedStart := none },
    { sourceRows := { start := 30664216, stop := 30664217 }, disposition := "LinearDefinition(SelectiveRewriteId(401998))", emittedStart := none },
    { sourceRows := { start := 30664217, stop := 30664218 }, disposition := "LinearDefinition(SelectiveRewriteId(401999))", emittedStart := none },
    { sourceRows := { start := 30664218, stop := 30664219 }, disposition := "LinearDefinition(SelectiveRewriteId(402000))", emittedStart := none },
    { sourceRows := { start := 30664219, stop := 30664220 }, disposition := "LinearDefinition(SelectiveRewriteId(402001))", emittedStart := none },
    { sourceRows := { start := 30664220, stop := 30664221 }, disposition := "LinearDefinition(SelectiveRewriteId(402002))", emittedStart := none },
    { sourceRows := { start := 30664221, stop := 30664222 }, disposition := "LinearDefinition(SelectiveRewriteId(402003))", emittedStart := none },
    { sourceRows := { start := 30664222, stop := 30664223 }, disposition := "LinearDefinition(SelectiveRewriteId(402004))", emittedStart := none },
    { sourceRows := { start := 30664223, stop := 30664224 }, disposition := "LinearDefinition(SelectiveRewriteId(402005))", emittedStart := none },
    { sourceRows := { start := 30664224, stop := 30664225 }, disposition := "LinearDefinition(SelectiveRewriteId(402006))", emittedStart := none },
    { sourceRows := { start := 30664225, stop := 30664226 }, disposition := "LinearDefinition(SelectiveRewriteId(402007))", emittedStart := none },
    { sourceRows := { start := 30664226, stop := 30664227 }, disposition := "LinearDefinition(SelectiveRewriteId(402008))", emittedStart := none },
    { sourceRows := { start := 30664227, stop := 30664228 }, disposition := "LinearDefinition(SelectiveRewriteId(402009))", emittedStart := none },
    { sourceRows := { start := 30664228, stop := 30664229 }, disposition := "LinearDefinition(SelectiveRewriteId(402010))", emittedStart := none },
    { sourceRows := { start := 30664229, stop := 30664230 }, disposition := "LinearDefinition(SelectiveRewriteId(402011))", emittedStart := none },
    { sourceRows := { start := 30664230, stop := 30664231 }, disposition := "LinearDefinition(SelectiveRewriteId(402012))", emittedStart := none },
    { sourceRows := { start := 30664231, stop := 30664232 }, disposition := "LinearDefinition(SelectiveRewriteId(402013))", emittedStart := none },
    { sourceRows := { start := 30664232, stop := 30664832 }, disposition := "Poseidon2(SelectiveRewriteId(9594))", emittedStart := none },
    { sourceRows := { start := 30664832, stop := 30664833 }, disposition := "LinearDefinition(SelectiveRewriteId(402014))", emittedStart := none },
    { sourceRows := { start := 30664833, stop := 30664834 }, disposition := "LinearDefinition(SelectiveRewriteId(402015))", emittedStart := none },
    { sourceRows := { start := 30664834, stop := 30664835 }, disposition := "LinearDefinition(SelectiveRewriteId(402016))", emittedStart := none },
    { sourceRows := { start := 30664835, stop := 30664836 }, disposition := "LinearDefinition(SelectiveRewriteId(402017))", emittedStart := none },
    { sourceRows := { start := 30664836, stop := 30665436 }, disposition := "Poseidon2(SelectiveRewriteId(9595))", emittedStart := none },
    { sourceRows := { start := 30665436, stop := 30665437 }, disposition := "LinearDefinition(SelectiveRewriteId(402018))", emittedStart := none },
    { sourceRows := { start := 30665437, stop := 30665438 }, disposition := "LinearDefinition(SelectiveRewriteId(402019))", emittedStart := none },
    { sourceRows := { start := 30665438, stop := 30665439 }, disposition := "LinearDefinition(SelectiveRewriteId(402020))", emittedStart := none },
    { sourceRows := { start := 30665439, stop := 30665440 }, disposition := "LinearDefinition(SelectiveRewriteId(402021))", emittedStart := none },
    { sourceRows := { start := 30665440, stop := 30666040 }, disposition := "Poseidon2(SelectiveRewriteId(9596))", emittedStart := none },
    { sourceRows := { start := 30666040, stop := 30666041 }, disposition := "LinearDefinition(SelectiveRewriteId(402022))", emittedStart := none },
    { sourceRows := { start := 30666041, stop := 30666042 }, disposition := "LinearDefinition(SelectiveRewriteId(402023))", emittedStart := none },
    { sourceRows := { start := 30666042, stop := 30666043 }, disposition := "LinearDefinition(SelectiveRewriteId(402024))", emittedStart := none },
    { sourceRows := { start := 30666043, stop := 30666044 }, disposition := "LinearDefinition(SelectiveRewriteId(402025))", emittedStart := none },
    { sourceRows := { start := 30666044, stop := 30666644 }, disposition := "Poseidon2(SelectiveRewriteId(9597))", emittedStart := none },
    { sourceRows := { start := 30666644, stop := 30666645 }, disposition := "LinearDefinition(SelectiveRewriteId(402026))", emittedStart := none },
    { sourceRows := { start := 30666645, stop := 30666646 }, disposition := "LinearDefinition(SelectiveRewriteId(402027))", emittedStart := none },
    { sourceRows := { start := 30666646, stop := 30666647 }, disposition := "LinearDefinition(SelectiveRewriteId(402028))", emittedStart := none },
    { sourceRows := { start := 30666647, stop := 30666648 }, disposition := "LinearDefinition(SelectiveRewriteId(402029))", emittedStart := none },
    { sourceRows := { start := 30666648, stop := 30667248 }, disposition := "Poseidon2(SelectiveRewriteId(9598))", emittedStart := none },
    { sourceRows := { start := 30667248, stop := 30667249 }, disposition := "LinearDefinition(SelectiveRewriteId(402030))", emittedStart := none },
    { sourceRows := { start := 30667249, stop := 30667250 }, disposition := "LinearDefinition(SelectiveRewriteId(402031))", emittedStart := none },
    { sourceRows := { start := 30667250, stop := 30667251 }, disposition := "LinearDefinition(SelectiveRewriteId(402032))", emittedStart := none },
    { sourceRows := { start := 30667251, stop := 30667252 }, disposition := "LinearDefinition(SelectiveRewriteId(402033))", emittedStart := none },
    { sourceRows := { start := 30667252, stop := 30667852 }, disposition := "Poseidon2(SelectiveRewriteId(9599))", emittedStart := none },
    { sourceRows := { start := 30667852, stop := 30667853 }, disposition := "LinearDefinition(SelectiveRewriteId(402034))", emittedStart := none },
    { sourceRows := { start := 30667853, stop := 30667854 }, disposition := "LinearDefinition(SelectiveRewriteId(402035))", emittedStart := none },
    { sourceRows := { start := 30667854, stop := 30667855 }, disposition := "LinearDefinition(SelectiveRewriteId(402036))", emittedStart := none },
    { sourceRows := { start := 30667855, stop := 30667856 }, disposition := "LinearDefinition(SelectiveRewriteId(402037))", emittedStart := none },
    { sourceRows := { start := 30667856, stop := 30668456 }, disposition := "Poseidon2(SelectiveRewriteId(9600))", emittedStart := none },
    { sourceRows := { start := 30668456, stop := 30668457 }, disposition := "LinearDefinition(SelectiveRewriteId(402038))", emittedStart := none },
    { sourceRows := { start := 30668457, stop := 30668458 }, disposition := "LinearDefinition(SelectiveRewriteId(402039))", emittedStart := none },
    { sourceRows := { start := 30668458, stop := 30668459 }, disposition := "LinearDefinition(SelectiveRewriteId(402040))", emittedStart := none },
    { sourceRows := { start := 30668459, stop := 30668460 }, disposition := "LinearDefinition(SelectiveRewriteId(402041))", emittedStart := none },
    { sourceRows := { start := 30668460, stop := 30669060 }, disposition := "Poseidon2(SelectiveRewriteId(9601))", emittedStart := none },
    { sourceRows := { start := 30669060, stop := 30669061 }, disposition := "LinearDefinition(SelectiveRewriteId(402042))", emittedStart := none },
    { sourceRows := { start := 30669061, stop := 30669062 }, disposition := "LinearDefinition(SelectiveRewriteId(402043))", emittedStart := none },
    { sourceRows := { start := 30669062, stop := 30669063 }, disposition := "LinearDefinition(SelectiveRewriteId(402044))", emittedStart := none },
    { sourceRows := { start := 30669063, stop := 30669064 }, disposition := "LinearDefinition(SelectiveRewriteId(402045))", emittedStart := none },
    { sourceRows := { start := 30669064, stop := 30669664 }, disposition := "Poseidon2(SelectiveRewriteId(9602))", emittedStart := none },
    { sourceRows := { start := 30669664, stop := 30669665 }, disposition := "LinearDefinition(SelectiveRewriteId(402046))", emittedStart := none },
    { sourceRows := { start := 30669665, stop := 30670265 }, disposition := "Poseidon2(SelectiveRewriteId(9603))", emittedStart := none },
    { sourceRows := { start := 30670265, stop := 30670266 }, disposition := "LinearDefinition(SelectiveRewriteId(402047))", emittedStart := none },
    { sourceRows := { start := 30670266, stop := 30670866 }, disposition := "Poseidon2(SelectiveRewriteId(9604))", emittedStart := none },
    { sourceRows := { start := 30670866, stop := 30670867 }, disposition := "LinearDefinition(SelectiveRewriteId(402048))", emittedStart := none },
    { sourceRows := { start := 30670867, stop := 30670868 }, disposition := "LinearDefinition(SelectiveRewriteId(402049))", emittedStart := none },
    { sourceRows := { start := 30670868, stop := 30670869 }, disposition := "LinearDefinition(SelectiveRewriteId(402050))", emittedStart := none },
    { sourceRows := { start := 30670869, stop := 30670870 }, disposition := "LinearDefinition(SelectiveRewriteId(402051))", emittedStart := none },
    { sourceRows := { start := 30670870, stop := 30670871 }, disposition := "LinearDefinition(SelectiveRewriteId(402052))", emittedStart := none },
    { sourceRows := { start := 30670871, stop := 30670872 }, disposition := "LinearDefinition(SelectiveRewriteId(402053))", emittedStart := none },
    { sourceRows := { start := 30670872, stop := 30670873 }, disposition := "LinearDefinition(SelectiveRewriteId(402054))", emittedStart := none },
    { sourceRows := { start := 30670873, stop := 30670874 }, disposition := "LinearDefinition(SelectiveRewriteId(402055))", emittedStart := none },
    { sourceRows := { start := 30670874, stop := 30670875 }, disposition := "LinearDefinition(SelectiveRewriteId(402056))", emittedStart := none },
    { sourceRows := { start := 30670875, stop := 30670876 }, disposition := "LinearDefinition(SelectiveRewriteId(402057))", emittedStart := none },
    { sourceRows := { start := 30670876, stop := 30670877 }, disposition := "LinearDefinition(SelectiveRewriteId(402058))", emittedStart := none },
    { sourceRows := { start := 30670877, stop := 30670878 }, disposition := "LinearDefinition(SelectiveRewriteId(402059))", emittedStart := none },
    { sourceRows := { start := 30670878, stop := 30670879 }, disposition := "LinearDefinition(SelectiveRewriteId(402060))", emittedStart := none },
    { sourceRows := { start := 30670879, stop := 30670880 }, disposition := "LinearDefinition(SelectiveRewriteId(402061))", emittedStart := none },
    { sourceRows := { start := 30670880, stop := 30671480 }, disposition := "Poseidon2(SelectiveRewriteId(9605))", emittedStart := none },
    { sourceRows := { start := 30671480, stop := 30671481 }, disposition := "LinearDefinition(SelectiveRewriteId(402062))", emittedStart := none },
    { sourceRows := { start := 30671481, stop := 30671482 }, disposition := "LinearDefinition(SelectiveRewriteId(402063))", emittedStart := none },
    { sourceRows := { start := 30671482, stop := 30671483 }, disposition := "LinearDefinition(SelectiveRewriteId(402064))", emittedStart := none },
    { sourceRows := { start := 30671483, stop := 30671484 }, disposition := "LinearDefinition(SelectiveRewriteId(402065))", emittedStart := none },
    { sourceRows := { start := 30671484, stop := 30672084 }, disposition := "Poseidon2(SelectiveRewriteId(9606))", emittedStart := none },
    { sourceRows := { start := 30672084, stop := 30672085 }, disposition := "LinearDefinition(SelectiveRewriteId(402066))", emittedStart := none },
    { sourceRows := { start := 30672085, stop := 30672086 }, disposition := "LinearDefinition(SelectiveRewriteId(402067))", emittedStart := none },
    { sourceRows := { start := 30672086, stop := 30672087 }, disposition := "LinearDefinition(SelectiveRewriteId(402068))", emittedStart := none },
    { sourceRows := { start := 30672087, stop := 30672088 }, disposition := "LinearDefinition(SelectiveRewriteId(402069))", emittedStart := none },
    { sourceRows := { start := 30672088, stop := 30672688 }, disposition := "Poseidon2(SelectiveRewriteId(9607))", emittedStart := none },
    { sourceRows := { start := 30672688, stop := 30672689 }, disposition := "LinearDefinition(SelectiveRewriteId(402070))", emittedStart := none },
    { sourceRows := { start := 30672689, stop := 30673289 }, disposition := "Poseidon2(SelectiveRewriteId(9608))", emittedStart := none },
    { sourceRows := { start := 30673289, stop := 30673290 }, disposition := "LinearDefinition(SelectiveRewriteId(402071))", emittedStart := none },
    { sourceRows := { start := 30673290, stop := 30673890 }, disposition := "Poseidon2(SelectiveRewriteId(9609))", emittedStart := none },
    { sourceRows := { start := 30673890, stop := 30673894 }, disposition := "Retained", emittedStart := some 5008413 },
    { sourceRows := { start := 30673894, stop := 30673895 }, disposition := "LinearDefinition(SelectiveRewriteId(402072))", emittedStart := none },
    { sourceRows := { start := 30673895, stop := 30673896 }, disposition := "LinearDefinition(SelectiveRewriteId(402073))", emittedStart := none },
    { sourceRows := { start := 30673896, stop := 30673897 }, disposition := "LinearDefinition(SelectiveRewriteId(402074))", emittedStart := none },
    { sourceRows := { start := 30673897, stop := 30673898 }, disposition := "LinearDefinition(SelectiveRewriteId(402075))", emittedStart := none },
    { sourceRows := { start := 30673898, stop := 30673899 }, disposition := "LinearDefinition(SelectiveRewriteId(402076))", emittedStart := none },
    { sourceRows := { start := 30673899, stop := 30673900 }, disposition := "LinearDefinition(SelectiveRewriteId(402077))", emittedStart := none },
    { sourceRows := { start := 30673900, stop := 30673901 }, disposition := "LinearDefinition(SelectiveRewriteId(402078))", emittedStart := none },
    { sourceRows := { start := 30673901, stop := 30673902 }, disposition := "LinearDefinition(SelectiveRewriteId(402079))", emittedStart := none },
    { sourceRows := { start := 30673902, stop := 30673903 }, disposition := "LinearDefinition(SelectiveRewriteId(402080))", emittedStart := none },
    { sourceRows := { start := 30673903, stop := 30673904 }, disposition := "LinearDefinition(SelectiveRewriteId(402081))", emittedStart := none },
    { sourceRows := { start := 30673904, stop := 30673905 }, disposition := "LinearDefinition(SelectiveRewriteId(402082))", emittedStart := none },
    { sourceRows := { start := 30673905, stop := 30673906 }, disposition := "LinearDefinition(SelectiveRewriteId(402083))", emittedStart := none },
    { sourceRows := { start := 30673906, stop := 30673907 }, disposition := "LinearDefinition(SelectiveRewriteId(402084))", emittedStart := none },
    { sourceRows := { start := 30673907, stop := 30674507 }, disposition := "Poseidon2(SelectiveRewriteId(9610))", emittedStart := none },
    { sourceRows := { start := 30674507, stop := 30674508 }, disposition := "LinearDefinition(SelectiveRewriteId(402085))", emittedStart := none },
    { sourceRows := { start := 30674508, stop := 30674509 }, disposition := "LinearDefinition(SelectiveRewriteId(402086))", emittedStart := none },
    { sourceRows := { start := 30674509, stop := 30674510 }, disposition := "LinearDefinition(SelectiveRewriteId(402087))", emittedStart := none },
    { sourceRows := { start := 30674510, stop := 30674511 }, disposition := "LinearDefinition(SelectiveRewriteId(402088))", emittedStart := none },
    { sourceRows := { start := 30674511, stop := 30675111 }, disposition := "Poseidon2(SelectiveRewriteId(9611))", emittedStart := none },
    { sourceRows := { start := 30675111, stop := 30675112 }, disposition := "LinearDefinition(SelectiveRewriteId(402089))", emittedStart := none },
    { sourceRows := { start := 30675112, stop := 30675113 }, disposition := "LinearDefinition(SelectiveRewriteId(402090))", emittedStart := none },
    { sourceRows := { start := 30675113, stop := 30675114 }, disposition := "LinearDefinition(SelectiveRewriteId(402091))", emittedStart := none },
    { sourceRows := { start := 30675114, stop := 30675115 }, disposition := "LinearDefinition(SelectiveRewriteId(402092))", emittedStart := none },
    { sourceRows := { start := 30675115, stop := 30675715 }, disposition := "Poseidon2(SelectiveRewriteId(9612))", emittedStart := none },
    { sourceRows := { start := 30675715, stop := 30675716 }, disposition := "LinearDefinition(SelectiveRewriteId(402093))", emittedStart := none },
    { sourceRows := { start := 30675716, stop := 30676316 }, disposition := "Poseidon2(SelectiveRewriteId(9613))", emittedStart := none },
    { sourceRows := { start := 30676316, stop := 30676320 }, disposition := "Retained", emittedStart := some 5008417 },
    { sourceRows := { start := 30676320, stop := 30676321 }, disposition := "LinearDefinition(SelectiveRewriteId(402094))", emittedStart := none },
    { sourceRows := { start := 30676321, stop := 30676322 }, disposition := "LinearDefinition(SelectiveRewriteId(402095))", emittedStart := none },
    { sourceRows := { start := 30676322, stop := 30676323 }, disposition := "LinearDefinition(SelectiveRewriteId(402096))", emittedStart := none },
    { sourceRows := { start := 30676323, stop := 30676324 }, disposition := "LinearDefinition(SelectiveRewriteId(402097))", emittedStart := none },
  ]

def finalRuns : List FinalRun := [
    { family := "Retained", rows := { start := 5008413, stop := 5008417 }, rewriteId := none },
    { family := "Retained", rows := { start := 5008417, stop := 5008421 }, rewriteId := none },
    { family := "Poseidon2", rows := { start := 5094521, stop := 5094607 }, rewriteId := some 9594 },
    { family := "Poseidon2", rows := { start := 5094607, stop := 5094693 }, rewriteId := some 9595 },
    { family := "Poseidon2", rows := { start := 5094693, stop := 5094779 }, rewriteId := some 9596 },
    { family := "Poseidon2", rows := { start := 5094779, stop := 5094865 }, rewriteId := some 9597 },
    { family := "Poseidon2", rows := { start := 5094865, stop := 5094951 }, rewriteId := some 9598 },
    { family := "Poseidon2", rows := { start := 5094951, stop := 5095037 }, rewriteId := some 9599 },
    { family := "Poseidon2", rows := { start := 5095037, stop := 5095123 }, rewriteId := some 9600 },
    { family := "Poseidon2", rows := { start := 5095123, stop := 5095209 }, rewriteId := some 9601 },
    { family := "Poseidon2", rows := { start := 5095209, stop := 5095295 }, rewriteId := some 9602 },
    { family := "Poseidon2", rows := { start := 5095295, stop := 5095381 }, rewriteId := some 9603 },
    { family := "Poseidon2", rows := { start := 5095381, stop := 5095467 }, rewriteId := some 9604 },
    { family := "Poseidon2", rows := { start := 5095467, stop := 5095553 }, rewriteId := some 9605 },
    { family := "Poseidon2", rows := { start := 5095553, stop := 5095639 }, rewriteId := some 9606 },
    { family := "Poseidon2", rows := { start := 5095639, stop := 5095725 }, rewriteId := some 9607 },
    { family := "Poseidon2", rows := { start := 5095725, stop := 5095811 }, rewriteId := some 9608 },
    { family := "Poseidon2", rows := { start := 5095811, stop := 5095897 }, rewriteId := some 9609 },
    { family := "Poseidon2", rows := { start := 5095897, stop := 5095983 }, rewriteId := some 9610 },
    { family := "Poseidon2", rows := { start := 5095983, stop := 5096069 }, rewriteId := some 9611 },
    { family := "Poseidon2", rows := { start := 5096069, stop := 5096155 }, rewriteId := some 9612 },
    { family := "Poseidon2", rows := { start := 5096155, stop := 5096241 }, rewriteId := some 9613 },
  ]

def rawArtifact : RawArtifact :=
  { schemaVersion := 2,
    profileId := "nightstream/goldilocks/streaming-lifecycle-selective/v1",
    sourceArtifactIdentity := "rust:nightstream/streaming-lifecycle-recursive/source-rows/v1",
    finalArtifactIdentity := "rust:nightstream/streaming-lifecycle-selective/final-rows/v1",
    stagePath := "fprime.recursive.verifier_key", occurrence := 11882,
    sourceRows := { start := 30664206, stop := 30676324 }, sourceColumns := { start := 30388263, stop := 30400381 },
    structureDigestColumns := { start := 30388263, stop := 30388267 },
    ajtaiPpDigestColumns := { start := 30388267, stop := 30388271 },
    initialSemanticStateDigestColumns := { start := 30388271, stop := 30388275 },
    baseVerifierKeyHash := { sourceRows := { start := 30664206, stop := 30670866 }, recipe := { constantValues := [23, 30521782141150574, 31069335676202596, 13356207430137391, 13430, 1, 4294967295, 81, 54, 18, 1073741824, 0, 2, 16, 65536, 0, 216, 2, 114, 649, 0], constantStartColumn := 30388275, localColumns := [30388263, 30388264, 30388265, 30388266, 646, 647, 648, 649, 30388267, 30388268, 30388269, 30388270, 30388271, 30388272, 30388273, 30388274], payloadColumns := [], orderedInputColumns := [30388275, 30388276, 30388277, 30388278, 30388279, 30388263, 30388264, 30388265, 30388266, 646, 647, 648, 649, 30388267, 30388268, 30388269, 30388270, 30388280, 30388281, 30388282, 30388283, 30388284, 30388285, 30388286, 30388287, 30388288, 30388289, 30388290, 30388291, 30388292, 30388293, 30388294, 30388295, 30388271, 30388272, 30388273, 30388274], outputColumns := [30394927, 30394928, 30394929, 30394930] } },
    policyVerifierKeyHash := { sourceRows := { start := 30670866, stop := 30673890 }, recipe := { constantValues := [30, 30521782141150574, 31069335676202596, 26867006312248879, 13362791782838128, 12662, 1, 1, 1], constantStartColumn := 30394935, localColumns := [30394927, 30394928, 30394929, 30394930], payloadColumns := [], orderedInputColumns := [30394935, 30394936, 30394937, 30394938, 30394939, 30394940, 30394927, 30394928, 30394929, 30394930, 30394941, 30394942, 30394943], outputColumns := [30397951, 30397952, 30397953, 30397954] } },
    policyDigestBinding := { sourceRows := { start := 30673890, stop := 30673894 }, leftColumns := [642, 643, 644, 645], rightColumns := [30397951, 30397952, 30397953, 30397954] },
    initialBoundaryHash := { sourceRows := { start := 30673894, stop := 30676316 }, recipe := { constantValues := [34, 30521782141150574, 31069335676202596, 27419021446900015, 28268948330012524, 55483184018017, 649, 0], constantStartColumn := 30397959, localColumns := [30388263, 30388264, 30388265, 30388266], payloadColumns := [], orderedInputColumns := [30397959, 30397960, 30397961, 30397962, 30397963, 30397964, 30388263, 30388264, 30388265, 30388266, 30397965, 30397966], outputColumns := [30400373, 30400374, 30400375, 30400376] } },
    initialBoundaryBinding := { sourceRows := { start := 30676316, stop := 30676320 }, leftColumns := [652, 653, 654, 655], rightColumns := [30400373, 30400374, 30400375, 30400376] },
    publicTraceBinding := { sourceRows := { start := 30676320, stop := 30676324 }, leftColumns := [668, 669, 670, 671], rightColumns := [656, 657, 658, 659] },
    finalRowCount := 10306243,
    sourceRuns := sourceRuns,
    finalRuns := finalRuns }

theorem sourceRuns_cover : SourceRunChain 30664206 sourceRuns 30676324 :=
by
  unfold sourceRuns
  exact SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.nil 30676324))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

theorem finalRuns_inside : FinalRunsWithin 10306243 finalRuns :=
by
  unfold finalRuns
  exact FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.nil))))))))))))))))))))))

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey
