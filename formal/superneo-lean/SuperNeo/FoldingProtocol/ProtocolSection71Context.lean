import SuperNeo.FoldingProtocol.ProtocolRelations

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

end ProtocolSection71Context

end SuperNeo
