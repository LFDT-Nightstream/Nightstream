import DirectCcsFPrime.DirectStageSemantics
import SuperNeo.FoldingProtocol.ProtocolSection71Context

/-!
Adapters from theorem-native SuperNeo stage contexts to direct CCS F' stages.

The direct parent-only path should reuse the SuperNeo Section 7.1 proof
surface. A direct stage package therefore supplies the computed stage data plus
the exact theorem-native SuperNeo contexts for those computed objects; this
module derives the direct `ReusedStageAuthority` fields from `ceRelation`.
-/

namespace DirectCcsFPrime

namespace DirectStageSuperNeoReuse

/-- Package an existing compact `ceRelation` as direct reused stage authority. -/
def reusedStageAuthority_of_ceRelation
    {ctx : SuperNeo.ProtocolTargetContext}
    (h : SuperNeo.ceRelation ctx) :
    SuperNeoBridge.ReusedStageAuthority ctx where
  ceRelation := h

/-- Reuse a theorem-native SuperNeo Section 7.1 context as direct stage authority. -/
def reusedStageAuthority_of_section71Context
    (h : SuperNeo.ProtocolSection71Context) :
    SuperNeoBridge.ReusedStageAuthority h.target :=
  reusedStageAuthority_of_ceRelation
    (SuperNeo.ProtocolSection71Context.ceRelation h)

/--
Contextual direct stage computations whose computed SuperNeo contexts are
backed by theorem-native Section 7.1 contexts.

The equality fields are the exact alignment obligations: the context recovered
from the upstream SuperNeo package must be the context carried by the direct
`Pi_CCS` output or computed for the direct `Pi_RLC` parent source.
-/
structure Section71ContextualStageComputations
    (Digest : Type)
    (n : Nat) where
  computePiCCS :
    Nat →
      DirectTerminalSoundness.AccHandle Digest n →
      DirectStageSemantics.ContextualPiCCSOut
  computePiCCS_step :
    ∀ i prior,
      (computePiCCS i prior).step = i
  computePiRLC :
    Nat →
      DirectStageSemantics.ContextualPiCCSOut →
      DigestParentBinding.Source Digest
  piRLCContext :
    DirectStageSemantics.ContextualPiCCSOut →
      DigestParentBinding.Source Digest →
      SuperNeo.ProtocolTargetContext
  piCCSSection71 :
    ∀ _i _prior,
      SuperNeo.ProtocolSection71Context
  piCCSSection71_target :
    ∀ i prior,
      (piCCSSection71 i prior).target =
        (computePiCCS i prior).ctx
  piRLCSection71 :
    ∀ _i _out,
      SuperNeo.ProtocolSection71Context
  piRLCSection71_target :
    ∀ i out,
      (piRLCSection71 i out).target =
        piRLCContext out (computePiRLC i out)

namespace Section71ContextualStageComputations

/--
Forget the Section 7.1 owner object into the direct contextual reused-stage
package consumed by the parent-only production theorem.
-/
def toContextualReused
    {Digest : Type}
    {n : Nat}
    (stage : Section71ContextualStageComputations Digest n) :
    DirectStageSemantics.ContextualReusedStageComputations Digest n where
  computePiCCS := stage.computePiCCS
  computePiCCS_step := stage.computePiCCS_step
  computePiRLC := stage.computePiRLC
  piRLCContext := stage.piRLCContext
  piCCSAuthority := by
    intro i prior
    apply reusedStageAuthority_of_ceRelation
    have hRel :
        SuperNeo.ceRelation ((stage.piCCSSection71 i prior).target) :=
      SuperNeo.ProtocolSection71Context.ceRelation
        (stage.piCCSSection71 i prior)
    simpa [stage.piCCSSection71_target i prior] using hRel
  piRLCAuthority := by
    intro i out
    apply reusedStageAuthority_of_ceRelation
    have hRel :
        SuperNeo.ceRelation ((stage.piRLCSection71 i out).target) :=
      SuperNeo.ProtocolSection71Context.ceRelation
        (stage.piRLCSection71 i out)
    simpa [stage.piRLCSection71_target i out] using hRel

/-- The computed `Pi_CCS` context satisfies the imported strong statement. -/
theorem piCCSStrong_of_compute
    {Digest : Type}
    {n : Nat}
    (stage : Section71ContextualStageComputations Digest n)
    (i : Nat)
    (prior : DirectTerminalSoundness.AccHandle Digest n) :
    SuperNeo.PiCCSInterface.piCCSStrongStatement
      (stage.computePiCCS i prior).ctx :=
  SuperNeoBridge.ReusedStageAuthority.piCCSStrong
    ((stage.toContextualReused).piCCSAuthority i prior)

/-- The computed `Pi_RLC` context satisfies the imported weak statement. -/
theorem piRLCWeak_of_compute
    {Digest : Type}
    {n : Nat}
    (stage : Section71ContextualStageComputations Digest n)
    (i : Nat)
    (out : DirectStageSemantics.ContextualPiCCSOut) :
    SuperNeo.PiRLCInterface.piRLCWeakStatement
      (stage.piRLCContext out (stage.computePiRLC i out)) :=
  SuperNeoBridge.ReusedStageAuthority.piRLCWeak
    ((stage.toContextualReused).piRLCAuthority i out)

/-- The computed `Pi_CCS` context also exposes the imported DEC knowledge surface. -/
theorem piDECKnowledge_of_piCCS_compute
    {Digest : Type}
    {n : Nat}
    (stage : Section71ContextualStageComputations Digest n)
    (i : Nat)
    (prior : DirectTerminalSoundness.AccHandle Digest n) :
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (stage.computePiCCS i prior).ctx :=
  SuperNeoBridge.ReusedStageAuthority.piDECKnowledge
    ((stage.toContextualReused).piCCSAuthority i prior)

/-- The computed `Pi_RLC` context also exposes the imported DEC knowledge surface. -/
theorem piDECKnowledge_of_piRLC_compute
    {Digest : Type}
    {n : Nat}
    (stage : Section71ContextualStageComputations Digest n)
    (i : Nat)
    (out : DirectStageSemantics.ContextualPiCCSOut) :
    SuperNeo.PiDECInterface.piDECKnowledgeStatement
      (stage.piRLCContext out (stage.computePiRLC i out)) :=
  SuperNeoBridge.ReusedStageAuthority.piDECKnowledge
    ((stage.toContextualReused).piRLCAuthority i out)

end Section71ContextualStageComputations

end DirectStageSuperNeoReuse

end DirectCcsFPrime
