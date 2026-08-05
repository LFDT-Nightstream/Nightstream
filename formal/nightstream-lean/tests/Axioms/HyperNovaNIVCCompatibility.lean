import Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec
import tests.Axioms.Support

/-! Fail-closed dependency guards for corrected HyperNova Definition 12. -/

/-- info: 'Nightstream.HyperNova.NIVCCompatibility.Codec.encode_injective' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.NIVCCompatibility.Codec.encode_injective

/-- info: 'Nightstream.HyperNova.NIVCCompatibility.Codec.fixedWidthInjective_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.NIVCCompatibility.Codec.fixedWidthInjective_canonical

/-- info: 'Nightstream.HyperNova.NIVCCompatibility.CompilerLayout.capacities_of_fits' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.NIVCCompatibility.CompilerLayout.capacities_of_fits

/-- info: 'Nightstream.HyperNova.NIVCCompatibility.CompilerLayout.columns_fit_row_domain' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.NIVCCompatibility.CompilerLayout.columns_fit_row_domain

/-- info: 'Nightstream.HyperNova.NIVCCompatibility.StatementIdentifierScheme.eq_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.NIVCCompatibility.StatementIdentifierScheme.eq_or_collision

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec.toNivcCodec_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec.toNivcCodec_canonical

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec.toTotalNivcCodec_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec.toTotalNivcCodec_canonical
