import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the selected paper
`Pi_RLC` public-source binding.
-/

namespace NightstreamTests.Axioms.FPrimePaperNifsPiRlcSourceBinding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding.FrameViews.inputOpening_eq_piCcsOutputProjection' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding.FrameViews.inputOpening_eq_piCcsOutputProjection

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding.Placement.decoded_freshColumns_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding.Placement.decoded_freshColumns_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding.Placement.decoded_runningColumns_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding.Placement.decoded_runningColumns_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding.Placement.decoded_inputColumns_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding.Placement.decoded_inputColumns_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame.decodedPiRlcInput_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame.decodedPiRlcInput_eq

end NightstreamTests.Axioms.FPrimePaperNifsPiRlcSourceBinding
