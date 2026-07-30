import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the selected paper
`Pi_CCS`-sampler-`Pi_RLC` quotient handoff.
-/

namespace NightstreamTests.Axioms.FPrimePaperNifsPiRlcQuotientHandoff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.challengeColumns_values' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.challengeColumns_values

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.decoded_selectedPrior_eq_samplerFinal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.decoded_selectedPrior_eq_samplerFinal

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.samplerChallengesBound_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.samplerChallengesBound_of_rows

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.samplerBinding_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.samplerBinding_of_rows

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.equations_or_transcriptBadRoot_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff.equations_or_transcriptBadRoot_of_rows

end NightstreamTests.Axioms.FPrimePaperNifsPiRlcQuotientHandoff
