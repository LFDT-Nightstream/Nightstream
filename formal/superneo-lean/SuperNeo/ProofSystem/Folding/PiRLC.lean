import SuperNeo.PiRLC
import SuperNeo.ProofSystem.Types

namespace SuperNeo.ProofSystem.Folding

abbrev PiRLCAssumptions (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.PiRLCAssumptions ctx

abbrev WeakStatement (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.piRLCWeakStatement ctx

/--
Proof-system wrapper: weak relation theorem for `Π_RLC` from an explicit
Section 7.1 proof-system CE realization.
-/
theorem weak_relaxed_of_section71_ce
  {ctx : SuperNeo.ProtocolTargetContext}
  (hReal : SuperNeo.ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_section71_ce hReal hCE

/--
Proof-system wrapper: weak relation theorem for `Π_RLC` from one concrete
Section 7.1 theorem instance.
-/
theorem weak_relaxed_of_section71Provider
  {ctx : SuperNeo.ProtocolTargetContext}
  (hProvider : SuperNeo.ProtocolSection71Provider ctx) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_section71Provider hProvider

/--
Proof-system wrapper: weak relation theorem for `Π_RLC` from one generic
Section 7.1 proof-system theorem instance plus compact specialization.
-/
theorem weak_relaxed_of_section71Specialization
  {ctx : SuperNeo.ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : SuperNeo.ProtocolSection71Specialization ctx hInst) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_section71Specialization hInst hSpec

/--
Proof-system wrapper: weak relation theorem for `Π_RLC` from one theorem-native
Section 7.1 setup.
-/
theorem weak_relaxed_of_section71Setup
  {ctx : SuperNeo.ProtocolTargetContext}
  (hSetup : SuperNeo.ProtocolSection71Setup ctx) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_section71Setup hSetup

/--
Proof-system wrapper: weak relation theorem for `Π_RLC` from one paper-
faithful Section 7.1 theorem instance.
-/
theorem weak_relaxed_of_section71TheoremInstance
  {ctx : SuperNeo.ProtocolTargetContext}
  (hInst : SuperNeo.ProtocolSection71TheoremInstance ctx) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_section71TheoremInstance hInst

/--
Proof-system wrapper: weak relation theorem for `Π_RLC` from one theorem-
native Section 7.1 context object.
-/
theorem weak_relaxed_of_section71Context
  (hCtx : SuperNeo.ProtocolSection71Context) :
  WeakStatement hCtx.target := by
  exact SuperNeo.piRLCWeak_of_section71Context hCtx

/--
Proof-system wrapper: weak relation theorem for `Π_RLC` from one protocol-side
Section 7.5 target-data owner and one accepted transition witness.
-/
theorem weak_relaxed_of_protocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext}
  (hTarget : SuperNeo.ProtocolTargetData ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_protocolTargetData hTarget hWitness

/-- Proof-system wrapper: weak relation theorem for `Π_RLC` from CCS relation and a SumCheck witness. -/
theorem weak_relaxed_of_ccsRelation
  {ctx : SuperNeo.ProtocolTargetContext}
  (hCCS : SuperNeo.ccsRelation ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_ccsRelation hCCS hWitness

/-- Proof-system wrapper: weak relation theorem for `Π_RLC`. -/
theorem weak_relaxed
  {ctx : SuperNeo.ProtocolTargetContext}
  (h : PiRLCAssumptions ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_assumptions h hWitness

end SuperNeo.ProofSystem.Folding
