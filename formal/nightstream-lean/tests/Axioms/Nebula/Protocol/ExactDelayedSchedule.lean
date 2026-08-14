import Nightstream.Protocol.Nebula.ExactDelayedSchedule
import tests.Axioms.Support

set_option pp.universes true in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedSchedule.Schedule.receipt_claims_exact

set_option pp.universes true in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedSchedule.Schedule.every_receipt_accepted

set_option pp.universes true in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedSchedule.Schedule.complete_index_schedule
