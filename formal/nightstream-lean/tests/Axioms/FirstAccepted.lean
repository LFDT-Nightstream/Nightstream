import Nightstream.SuperNeo.Sampling.FirstAccepted
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the independent first-accepted sampler semantics.
-/

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.mem_firstAccepted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.mem_firstAccepted

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.bounded_success_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.bounded_success_exact

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_eq_none_iff_shortfall' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_eq_none_iff_shortfall

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_append_of_success' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_append_of_success

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.ReferenceSemantics.output_unique' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.ReferenceSemantics.output_unique

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.ReferenceExecution.agrees_with_bounded_prefix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.ReferenceExecution.agrees_with_bounded_prefix

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.ReferenceExecution.exists_of_bounded_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.ReferenceExecution.exists_of_bounded_success

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_eq_some_iff_referenceExecution_within' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_eq_some_iff_referenceExecution_within

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.ReferenceExecution.consumedPrefix_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.ReferenceExecution.consumedPrefix_eq

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.bounded_success_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.bounded_success_length

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.boundedCursor_eq_some_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.boundedCursor_eq_some_iff

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.BoundedExecution.exists_of_bounded_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.BoundedExecution.exists_of_bounded_success

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.BoundedExecution.output_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.BoundedExecution.output_length

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_eq_some_iff_boundedExecution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample_eq_some_iff_boundedExecution

/-- info: 'Nightstream.SuperNeo.Sampling.FirstAccepted.getElem?_firstAccepted_eq_symbol_of_prefix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Sampling.FirstAccepted.getElem?_firstAccepted_eq_symbol_of_prefix
