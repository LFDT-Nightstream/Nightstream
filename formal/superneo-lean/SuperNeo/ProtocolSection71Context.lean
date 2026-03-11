import SuperNeo.ProtocolRelations

/-!
Single-object theorem-native owner for one compact Section 7.1 protocol
instance.
-/

namespace SuperNeo

/--
One compact protocol target together with one specialized paper-faithful
Section 7.1 theorem instance.

This is the smallest explicit upstream owner once the actual Definition-14 data
and its specialization back to the compact `ProtocolTargetContext` have been
constructed. It removes the need for downstream consumers to thread `ctx` and a
separate `ProtocolSection71TheoremInstance ctx` in parallel.
-/
structure ProtocolSection71Context where
  target : ProtocolTargetContext
  theoremInstance : ProtocolSection71TheoremInstance target

namespace ProtocolSection71Context

/-- Recover the compact CCS relation from one Section 7.1 context. -/
theorem ccsRelation
  (h : ProtocolSection71Context) :
  SuperNeo.ccsRelation h.target :=
  h.theoremInstance.ccsRelation

/-- Recover the compact CE relation from one Section 7.1 context. -/
theorem ceRelation
  (h : ProtocolSection71Context) :
  SuperNeo.ceRelation h.target :=
  h.theoremInstance.ceRelation

/-- Recover the specialized Definition-14 realization boundary. -/
def realization
  (h : ProtocolSection71Context) :
  ProtocolSection71Realization h.target :=
  h.theoremInstance.realization

/-- Recover the generic Section 7.1 setup package. -/
def setup
  (h : ProtocolSection71Context) :
  ProtocolSection71Setup h.target :=
  h.theoremInstance.setup

/-- Recover the concrete Section 7.1 provider bundle. -/
def provider
  (h : ProtocolSection71Context) :
  ProtocolSection71Provider h.target :=
  h.theoremInstance.provider

/-- Recover the compact challenge-set from one Section 7.1 context. -/
theorem challengeSet_eq_cset
  (h : ProtocolSection71Context) :
  h.theoremInstance.params.challengeSet = h.target.cset :=
  h.theoremInstance.challengeSet_eq_cset

/-- Recover that CCS and CE share one commitment inside the Section 7.1 context. -/
theorem sharedCommitment_eq
  (h : ProtocolSection71Context) :
  h.theoremInstance.ccsStatement.commitment =
    h.theoremInstance.ceStatement.commitment :=
  h.theoremInstance.sharedCommitment_eq

/-- Recover that CCS and CE share one public input inside the Section 7.1 context. -/
theorem sharedPublicInput_eq
  (h : ProtocolSection71Context) :
  h.theoremInstance.ccsStatement.publicInput =
    h.theoremInstance.ceStatement.publicInput :=
  h.theoremInstance.sharedPublicInput_eq

/-- Recover that CE uses the CCS full vector `[x, w]` inside the Section 7.1 context. -/
theorem ceAssignment_eq_fullVector
  (h : ProtocolSection71Context) :
  h.theoremInstance.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.theoremInstance.ccsStatement h.theoremInstance.ccsWitness :=
  h.theoremInstance.ceAssignment_eq_fullVector

end ProtocolSection71Context

/--
Derive compact `ccsRelation` directly from one theorem-native Section 7.1
context.
-/
theorem ccsRelation_of_section71Context
  (h : ProtocolSection71Context) :
  ccsRelation h.target :=
  h.ccsRelation

/--
Derive compact `ceRelation` directly from one theorem-native Section 7.1
context.
-/
theorem ceRelation_of_section71Context
  (h : ProtocolSection71Context) :
  ceRelation h.target :=
  h.ceRelation

end SuperNeo
