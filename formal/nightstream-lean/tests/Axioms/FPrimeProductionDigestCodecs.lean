import tests.FPrimeProductionDigestCodecs
import tests.Axioms.Support

/-!
Fail-closed guards for the production digest codecs and compact fixed-one
`encodeInstance` affine map.
-/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.digestCodec_encode_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.digestCodec_encode_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.digestCodec_roundtrip' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.digestCodec_roundtrip

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.optionalDigestCodec_encode_none' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.optionalDigestCodec_encode_none

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.optionalDigestCodec_encode_some' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.optionalDigestCodec_encode_some

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.optionalDigestCodec_roundtrip' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.optionalDigestCodec_roundtrip

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.adapterEncodedCodec_roundtrip' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.adapterEncodedCodec_roundtrip

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.encodeInstance_coordinates_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs.encodeInstance_coordinates_exact
