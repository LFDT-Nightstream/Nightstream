import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TranscriptRows

set_option maxRecDepth 100000

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptRows

example : ProductPoseidon2.construction3DomainFields.length = 36 := by
  decide

example : ProductPoseidon2.verifierChallengeLabelFields.length = 20 := by
  decide

example : (ProductPoseidon2.candidateFields
    ⟨0, by decide⟩ ⟨0, by decide⟩ ⟨0, by decide⟩).length = 2 := by
  decide
