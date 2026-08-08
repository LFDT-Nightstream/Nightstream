import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiCcs
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PoseidonTraces
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiRlcInput
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiRlcCombined
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiDecChildren0
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiDecChildren1
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiDecChildren2
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiDecChildren3
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiDecChildren4
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiDecChildren5
import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated.PiDecChildren6

/-! GENERATED FILE - assembled deterministic production NIFS receipt. -/

namespace Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated

open Nightstream.Implementation.Rust.NifsProductionGolden

def piDecChildren : List RawClaim :=
  piDecChildren0 ++ piDecChildren1 ++ piDecChildren2 ++ piDecChildren3 ++
    piDecChildren4 ++ piDecChildren5 ++ piDecChildren6

def receipt : ProductionReceipt :=
  { relationId := relationId
    relationMatrices := relationMatrices
    fixtureAssignment := fixtureAssignment
    piCcsStatement := piCcsStatement
    piCcsProof := piCcsProof
    poseidonPermutationTraces := poseidonPermutationTraces
    piCcsPermutationCount := piCcsPermutationCount
    rhoStartPermutationCount := rhoStartPermutationCount
    piCcsOutputsDigest := piCcsOutputsDigest
    rhoStart := rhoStart
    piRlcInputs := piRlcInputs
    piRlcCombined := piRlcCombined
    piDecChildren := piDecChildren
    canonicalNifsProofByteCount := canonicalNifsProofByteCount }

end Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden.Generated
