import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution

/-! Focused interface regression for deterministic causal paper `Pi_CCS`. -/

namespace tests.PiCcsPaperJointStrongExecution

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution

#check Context
#check History
#check Strategy
#check PrefixExecution
#check execute
#check execute_history_induction
#check execute_history_challenges_eq_roundPoint
#check execute_probe_coins
#check execute_rounds_eq_history
#check Execution
#check attachWitness
#check AmbientSuccess
#check acceptedCheck
#check acceptedCheck_eq_true_iff
#check ambientCheck
#check ambientCheck_eq_true_iff
#check SourceExtracted
#check MixingFailure
#check SumCheckFailure
#check acceptedPrefix_extracts_fixedWitness_or_badEvent
#check ambientSuccess_implies_source_or_badEvent

end tests.PiCcsPaperJointStrongExecution
