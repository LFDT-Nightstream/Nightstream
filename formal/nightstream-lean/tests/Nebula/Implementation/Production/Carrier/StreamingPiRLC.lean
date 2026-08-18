import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLC

/-! Regression surface for bounded-family PiRLC semantics. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlc

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlc

#check sourceSchedule_length
#check sourceSchedule_covers
#check sourceSchedule_nodup
#check sourceInputFields_length
#check familyInputFrame_length
#check inputFrame_length
#check run_cursor
#check run_binding
#check run_familySchedule_cursor
#check run_familySchedule_binding
#check replay_eq_authoritative_or_collision
#check local_rows_imply_combineOne
#check outputBundle_eq_combineBundles
#check outputPublic_eq_combinePublicInputs
#check outputEvaluation_eq_combineEvaluationFamily
#check typedOutput_exact
#check familyCount_eq
#check persistentFields_length
#check perFamilyVisibleFieldCount_eq
#check perFamilyAuxiliaryColumnCount_eq
#check perFamilyArithmeticRowCount_eq
#check perFamilyAlgebraFieldCount_eq
#check totalStreamingArithmeticRowCount_eq

end tests.NebulaProductionStreamingPiRlc
