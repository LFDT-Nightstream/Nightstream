import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputResidual

/-! Regression surface for the production PiRLC input residual. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcInputResidual

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual

#check coordinateWitness_eq
#check familyMaskedWitness_sum
#check familyBindings_sum
#check phaseWitness_eq_familyMaskedWitness
#check phaseBinding_eq_familyBinding
#check phaseBindings_sum
#check ResidualTransition
#check CompleteResidualRun
#check honest_completeResidualRun
#check complete_zero_residual_recovers_inputs_or_failure

end tests.NebulaProductionStreamingPiRlcInputResidual
