import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.PostPiCcsBridge
import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.SamplerResponseSound

/-! Regression surface for the exact V2 full-field PiRLC sampler rows. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductPiRlcSamplerRows

#check Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptCursor.afterFullOutput_absorbed
#check Nightstream.Implementation.NebulaV2.ProductPiRlcPostPiCcsBridge.rows_imply_candidate_exact
#check Nightstream.Implementation.NebulaV2.ProductPiRlcCandidateClassificationSound.all_candidates_sound
#check Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedSound.sound
#check Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchSound.sampleCoefficient_eq_some_output
#check Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.sampler_available
#check Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.samplerSucceeded_eq_true
#check Nightstream.Implementation.NebulaV2.ProductPiRlcSamplerResponseSound.output_eq_scalarResponse

example :
    Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows.aggregateRowCount =
      1710720 :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows.aggregateRowCount_eq

example :
    Nightstream.Implementation.NebulaV2.ProductPiRlcCandidateClassificationRows.aggregateRowCount =
      216270 :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcCandidateClassificationRows.aggregateRowCount_eq

example :
    Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount =
      7290 :=
  Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount_eq

end tests.NebulaV2ProductPiRlcSamplerRows
