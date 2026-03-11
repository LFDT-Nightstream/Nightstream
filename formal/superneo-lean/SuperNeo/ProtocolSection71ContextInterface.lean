import SuperNeo.ProtocolSection71Context

/-!
Contract interface for `SuperNeo.ProtocolSection71Context`.

Spec: `specs/ProtocolSection71Context.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
-/

namespace SuperNeo

namespace ProtocolSection71ContextInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProtocolSection71Context"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String :=
  ["§7.1 Structure / CCS / CE", "§7.2 Global Reduction Parameters"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "ProtocolSection71Context"
  , "ProtocolSection71Context.ccsRelation"
  , "ProtocolSection71Context.ceRelation"
  , "ProtocolSection71Context.realization"
  , "ProtocolSection71Context.setup"
  , "ProtocolSection71Context.provider"
  , "ProtocolSection71Context.challengeSet_eq_cset"
  , "ProtocolSection71Context.sharedCommitment_eq"
  , "ProtocolSection71Context.sharedPublicInput_eq"
  , "ProtocolSection71Context.ceAssignment_eq_fullVector"
  , "ccsRelation_of_section71Context"
  , "ceRelation_of_section71Context"
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/--
[Role: Theorem-Target] One compact protocol target together with one
specialized paper-faithful Section 7.1 theorem instance.
-/
abbrev ProtocolSection71Context := SuperNeo.ProtocolSection71Context

/-- [Role: Theorem-Target] Recover compact `ccsRelation` from one Section 7.1 context. -/
theorem ProtocolSection71Context_ccsRelation
  (h : ProtocolSection71Context) :
  SuperNeo.ccsRelation h.target :=
  SuperNeo.ProtocolSection71Context.ccsRelation h

/-- [Role: Theorem-Target] Recover compact `ceRelation` from one Section 7.1 context. -/
theorem ProtocolSection71Context_ceRelation
  (h : ProtocolSection71Context) :
  SuperNeo.ceRelation h.target :=
  SuperNeo.ProtocolSection71Context.ceRelation h

/-- [Role: Theorem-Target] Recover the realized Definition-14 boundary from one Section 7.1 context. -/
def ProtocolSection71Context_realization
  (h : ProtocolSection71Context) :
  ProtocolSection71Realization h.target :=
  SuperNeo.ProtocolSection71Context.realization h

/-- [Role: Theorem-Target] Recover the packaged generic Section 7.1 setup from one Section 7.1 context. -/
def ProtocolSection71Context_setup
  (h : ProtocolSection71Context) :
  ProtocolSection71Setup h.target :=
  SuperNeo.ProtocolSection71Context.setup h

/-- [Role: Theorem-Target] Recover the concrete Section 7.1 provider from one Section 7.1 context. -/
def ProtocolSection71Context_provider
  (h : ProtocolSection71Context) :
  ProtocolSection71Provider h.target :=
  SuperNeo.ProtocolSection71Context.provider h

/-- [Role: Theorem-Target] Recover the compact challenge-set from one Section 7.1 context. -/
theorem ProtocolSection71Context_challengeSet_eq_cset
  (h : ProtocolSection71Context) :
  h.theoremInstance.params.challengeSet = h.target.cset :=
  SuperNeo.ProtocolSection71Context.challengeSet_eq_cset h

/-- [Role: Theorem-Target] Recover that CCS and CE share one commitment in the Section 7.1 context. -/
theorem ProtocolSection71Context_sharedCommitment_eq
  (h : ProtocolSection71Context) :
  h.theoremInstance.ccsStatement.commitment =
    h.theoremInstance.ceStatement.commitment :=
  SuperNeo.ProtocolSection71Context.sharedCommitment_eq h

/-- [Role: Theorem-Target] Recover that CCS and CE share one public input in the Section 7.1 context. -/
theorem ProtocolSection71Context_sharedPublicInput_eq
  (h : ProtocolSection71Context) :
  h.theoremInstance.ccsStatement.publicInput =
    h.theoremInstance.ceStatement.publicInput :=
  SuperNeo.ProtocolSection71Context.sharedPublicInput_eq h

/-- [Role: Theorem-Target] Recover that CE uses the CCS full vector `[x, w]` in the Section 7.1 context. -/
theorem ProtocolSection71Context_ceAssignment_eq_fullVector
  (h : ProtocolSection71Context) :
  h.theoremInstance.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.theoremInstance.ccsStatement h.theoremInstance.ccsWitness :=
  SuperNeo.ProtocolSection71Context.ceAssignment_eq_fullVector h

/-- [Role: Theorem-Target] Derive compact `ccsRelation` directly from one Section 7.1 context. -/
theorem ccsRelation_of_section71Context
  (h : ProtocolSection71Context) :
  SuperNeo.ccsRelation h.target :=
  SuperNeo.ccsRelation_of_section71Context h

/-- [Role: Theorem-Target] Derive compact `ceRelation` directly from one Section 7.1 context. -/
theorem ceRelation_of_section71Context
  (h : ProtocolSection71Context) :
  SuperNeo.ceRelation h.target :=
  SuperNeo.ceRelation_of_section71Context h

end ProtocolSection71ContextInterface

end SuperNeo
