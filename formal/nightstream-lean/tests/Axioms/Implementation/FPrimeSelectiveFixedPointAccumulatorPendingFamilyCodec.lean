import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator
import tests.Axioms.Support

/-! Fail-closed dependency gate for the pending-family accumulator codec. -/

/-- info: 'Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.encodeCarrier_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.encodeCarrier_injective

/-- info: 'Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.semanticPayload_columnPoint_eq_of_encode_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.semanticPayload_columnPoint_eq_of_encode_eq

/-- info: 'Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.encodeWithPrefix_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.encodeWithPrefix_injective

/-- info: 'Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.encodeProductionCarrier_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.encodeProductionCarrier_injective

/-- info: 'Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.production_field_count' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.production_field_count

/-- info: 'Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.pendingFamily_field_saving' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Accumulator.PendingFamilyCodec.pendingFamily_field_saving
