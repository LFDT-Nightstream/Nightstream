import SuperNeo.PiCCS
import SuperNeo.ProofSystem.Types

namespace SuperNeo.ProofSystem.Folding

abbrev PiCCSAssumptions (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.PiCCSAssumptions ctx

abbrev StrongStatement (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.piCCSStrongStatement ctx

/--
Proof-system wrapper: strong relation theorem for `Π_CCS` from an explicit
Section 7.1 proof-system CE realization.
-/
theorem soundness_relations_of_section71_ce
  {ctx : SuperNeo.ProtocolTargetContext}
  (hReal : SuperNeo.ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_section71_ce hReal hCE

/--
Proof-system wrapper: strong relation theorem for `Π_CCS` from one concrete
Section 7.1 theorem instance.
-/
theorem soundness_relations_of_section71Provider
  {ctx : SuperNeo.ProtocolTargetContext}
  (hProvider : SuperNeo.ProtocolSection71Provider ctx) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_section71Provider hProvider

/--
Proof-system wrapper: strong relation theorem for `Π_CCS` from one generic
Section 7.1 proof-system theorem instance plus compact specialization.
-/
theorem soundness_relations_of_section71Specialization
  {ctx : SuperNeo.ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : SuperNeo.ProtocolSection71Specialization ctx hInst) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_section71Specialization hInst hSpec

/--
Proof-system wrapper: strong relation theorem for `Π_CCS` from one theorem-
native Section 7.1 setup.
-/
theorem soundness_relations_of_section71Setup
  {ctx : SuperNeo.ProtocolTargetContext}
  (hSetup : SuperNeo.ProtocolSection71Setup ctx) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_section71Setup hSetup

/--
Proof-system wrapper: strong relation theorem for `Π_CCS` from one paper-
faithful Section 7.1 theorem instance.
-/
theorem soundness_relations_of_section71TheoremInstance
  {ctx : SuperNeo.ProtocolTargetContext}
  (hInst : SuperNeo.ProtocolSection71TheoremInstance ctx) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_section71TheoremInstance hInst

/--
Proof-system wrapper: strong relation theorem for `Π_CCS` from one theorem-
native Section 7.1 context object.
-/
theorem soundness_relations_of_section71Context
  (hCtx : SuperNeo.ProtocolSection71Context) :
  StrongStatement hCtx.target := by
  exact SuperNeo.piCCSStrong_of_section71Context hCtx

/--
Proof-system wrapper: strong relation theorem for `Π_CCS` from one protocol-
side Section 7.5 target-data owner and one accepted transition witness.
-/
theorem soundness_relations_of_protocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext}
  (hTarget : SuperNeo.ProtocolTargetData ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_protocolTargetData hTarget hWitness

/-- Proof-system wrapper: strong relation theorem for `Π_CCS` from CCS relation and a SumCheck witness. -/
theorem soundness_relations_of_ccsRelation
  {ctx : SuperNeo.ProtocolTargetContext}
  (hCCS : SuperNeo.ccsRelation ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_ccsRelation hCCS hWitness

/-- Proof-system wrapper: strong relation theorem for `Π_CCS`. -/
theorem soundness_relations
  {ctx : SuperNeo.ProtocolTargetContext}
  (h : PiCCSAssumptions ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  StrongStatement ctx := by
  exact SuperNeo.piCCSStrong_of_assumptions h hWitness

end SuperNeo.ProofSystem.Folding
