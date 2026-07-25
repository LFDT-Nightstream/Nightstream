import tests.FPrimeProductionHashCallBoundary
import tests.Axioms.Support

/-!
Fail-closed guards for the totalized production hash wrapper boundary.
-/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.absentCurrentPreimage_not_aligned' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.absentCurrentPreimage_not_aligned

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.paperHash_absentCurrent' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.paperHash_absentCurrent

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.paperHash_eq_none_of_not_aligned' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.paperHash_eq_none_of_not_aligned

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.paperHash_eq_none_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.paperHash_eq_none_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.absentCurrent_encoding_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.absentCurrent_encoding_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.paperHash_encoding_eq_absent_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.paperHash_encoding_eq_absent_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.alignedCurrent_encoding_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.alignedCurrent_encoding_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.no_nonoptionalCoreRefines' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary.no_nonoptionalCoreRefines
