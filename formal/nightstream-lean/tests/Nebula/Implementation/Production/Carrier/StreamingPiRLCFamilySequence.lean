import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilySequence

/-! Regression surface for the complete production PiRLC family sequence. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcFamilySequence

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence

#check AcceptedRun
#check FamilyFinishRelation
#check AcceptedRun.final_cursor
#check AcceptedRun.challenges_eq_authoritative
#check AcceptedRun.output_eq_authoritative
#check AcceptedRun.residual_prefix_exact
#check AcceptedRun.concreteCompleteResidualRun
#check AcceptedRun.start_finish_recovers_inputs_or_failure
#check AcceptedRun.outputs_exact_or_failure

end tests.NebulaProductionStreamingPiRlcFamilySequence
