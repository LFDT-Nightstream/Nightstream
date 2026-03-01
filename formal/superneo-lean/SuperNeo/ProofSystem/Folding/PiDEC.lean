import SuperNeo.PiDEC
import SuperNeo.ProofSystem.Types

namespace SuperNeo.ProofSystem.Folding

abbrev PiDECAssumptions (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.PiDECAssumptions ctx

abbrev FinalStatement (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.piDECKnowledgeStatement ctx

/-- Proof-system wrapper: final theorem for `Π_DEC`. -/
theorem final_of_assumption
  {ctx : SuperNeo.ProtocolTargetContext}
  (h : PiDECAssumptions ctx)
  (hAccepted : ∃ tr : SuperNeo.SumCheckTranscript,
      SuperNeo.SumCheckAccepted (SuperNeo.sumcheckInstanceOfContext ctx) tr) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_assumptions h hAccepted

end SuperNeo.ProofSystem.Folding
