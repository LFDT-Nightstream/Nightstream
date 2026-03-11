import SuperNeo.PiDEC
import SuperNeo.ProofSystem.Types

namespace SuperNeo.ProofSystem.Folding

abbrev PiDECAssumptions (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.PiDECAssumptions ctx

abbrev FinalStatement (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.piDECKnowledgeStatement ctx

/--
Proof-system wrapper: final theorem for `Π_DEC` from an explicit Section 7.1
proof-system CE realization.
-/
theorem final_of_section71_ce
  {ctx : SuperNeo.ProtocolTargetContext}
  (hReal : SuperNeo.ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_section71_ce hReal hCE

/--
Proof-system wrapper: final theorem for `Π_DEC` from one concrete Section 7.1
theorem instance.
-/
theorem final_of_section71Provider
  {ctx : SuperNeo.ProtocolTargetContext}
  (hProvider : SuperNeo.ProtocolSection71Provider ctx) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_section71Provider hProvider

/--
Proof-system wrapper: final relation theorem for `Π_DEC` from one generic
Section 7.1 proof-system theorem instance plus compact specialization.
-/
theorem final_of_section71Specialization
  {ctx : SuperNeo.ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : SuperNeo.ProtocolSection71Specialization ctx hInst) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_section71Specialization hInst hSpec

/--
Proof-system wrapper: final theorem for `Π_DEC` from one theorem-native Section
7.1 setup.
-/
theorem final_of_section71Setup
  {ctx : SuperNeo.ProtocolTargetContext}
  (hSetup : SuperNeo.ProtocolSection71Setup ctx) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_section71Setup hSetup

/--
Proof-system wrapper: final theorem for `Π_DEC` from one paper-faithful
Section 7.1 theorem instance.
-/
theorem final_of_section71TheoremInstance
  {ctx : SuperNeo.ProtocolTargetContext}
  (hInst : SuperNeo.ProtocolSection71TheoremInstance ctx) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_section71TheoremInstance hInst

/--
Proof-system wrapper: final theorem for `Π_DEC` from one theorem-native Section
7.1 context object.
-/
theorem final_of_section71Context
  (hCtx : SuperNeo.ProtocolSection71Context) :
  FinalStatement hCtx.target := by
  exact SuperNeo.piDEC_of_section71Context hCtx

/--
Proof-system wrapper: final theorem for `Π_DEC` from one protocol-side Section
7.5 target-data owner and one accepted transition witness.
-/
theorem final_of_protocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext}
  (hTarget : SuperNeo.ProtocolTargetData ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_protocolTargetData hTarget hWitness

/-- Proof-system wrapper: final theorem for `Π_DEC` from CCS relation and a SumCheck witness. -/
theorem final_of_ccsRelation
  {ctx : SuperNeo.ProtocolTargetContext}
  (hCCS : SuperNeo.ccsRelation ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_ccsRelation hCCS hWitness

/-- Proof-system wrapper: final theorem for `Π_DEC`. -/
theorem final_of_assumption
  {ctx : SuperNeo.ProtocolTargetContext}
  (h : PiDECAssumptions ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  FinalStatement ctx := by
  exact SuperNeo.piDEC_of_assumptions h hWitness

end SuperNeo.ProofSystem.Folding
