import Nightstream.Protocol.NebulaV2.ExactDelayedSchedule
import tests.Axioms.Support

set_option pp.universes true in
#audit_axioms Nightstream.Protocol.NebulaV2.ExactDelayedSchedule.Schedule.receipt_claims_exact

set_option pp.universes true in
#audit_axioms Nightstream.Protocol.NebulaV2.ExactDelayedSchedule.Schedule.every_receipt_accepted

set_option pp.universes true in
#audit_axioms Nightstream.Protocol.NebulaV2.ExactDelayedSchedule.Schedule.complete_index_schedule
