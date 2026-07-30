import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for reuse of the selected paper
`Pi_CCS` transcript point by the following `Pi_RLC` occurrence.
-/

namespace NightstreamTests.Axioms.FPrimePaperNifsPiRlcPointBinding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.squeezeK_realizesColumns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.squeezeK_realizesColumns

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.replay_point_realizesColumns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.replay_point_realizesColumns

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.decoded_realizedColumns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.decoded_realizedColumns

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.ofFn_pointAt_eq_point' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.ofFn_pointAt_eq_point

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.decodePointColumns_eq_piCcsPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding.decodePointColumns_eq_piCcsPoint

end NightstreamTests.Axioms.FPrimePaperNifsPiRlcPointBinding
