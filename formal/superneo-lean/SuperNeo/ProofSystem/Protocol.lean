import SuperNeo.ProtocolTheorem
import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Security
import SuperNeo.ProofSystem.SumCheck
import SuperNeo.ProofSystem.Folding

namespace SuperNeo.ProofSystem

abbrev FinalTheoremAssumptions (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.FinalTheoremAssumptions ctx

abbrev FinalCompletenessStatement
  (ctx : SuperNeo.ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) :=
  SuperNeo.FinalCompletenessStatement ctx hA

abbrev FinalKnowledgeSoundnessStatement
  (ctx : SuperNeo.ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) :=
  SuperNeo.FinalKnowledgeSoundnessStatement ctx hA

abbrev FinalTheoremShape
  (ctx : SuperNeo.ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) :=
  SuperNeo.FinalTheoremShape ctx hA

/-- Canonical proof-system final theorem shape constructor. -/
theorem finalTheoremShape_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  FinalTheoremShape ctx hA := by
  exact SuperNeo.finalTheoremShape_of_assumptions hA

end SuperNeo.ProofSystem
