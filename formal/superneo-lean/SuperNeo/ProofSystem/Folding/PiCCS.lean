import SuperNeo.PiCCS
import SuperNeo.ProofSystem.Types

namespace SuperNeo.ProofSystem.Folding

abbrev PiCCSAssumptions (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.PiCCSAssumptions ctx

abbrev StrongStatement (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.piCCSStrongStatement ctx

/-- Proof-system wrapper: strong relation theorem for `Π_CCS`. -/
theorem soundness_relations
  {ctx : SuperNeo.ProtocolTargetContext}
  (h : PiCCSAssumptions ctx)
  (hAccepted : ∃ tr : SuperNeo.SumCheckTranscript,
      SuperNeo.SumCheckAccepted (SuperNeo.sumcheckInstanceOfContext ctx) tr) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_assumptions h hAccepted

end SuperNeo.ProofSystem.Folding
