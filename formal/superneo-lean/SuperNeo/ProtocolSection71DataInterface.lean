import SuperNeo.ProtocolSection71Data

/-!
Contract interface for `SuperNeo.ProtocolSection71Data`.

Spec: `specs/ProtocolSection71Data.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
-/

namespace SuperNeo

namespace ProtocolSection71DataInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProtocolSection71Data"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String :=
  ["§7.1 Structure / CCS / CE", "§7.2 Global Reduction Parameters"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "ProtocolSection71Data"
  , "ProtocolSection71Data.params"
  , "ProtocolSection71Data.challengeSet_eq_cset"
  , "ProtocolSection71Data.sharedCommitment_eq"
  , "ProtocolSection71Data.sharedPublicInput_eq"
  , "ProtocolSection71Data.ceAssignment_eq_fullVector"
  , "ProtocolSection71Data.theoremInstance"
  , "ProtocolSection71Data.context"
  , "ProtocolSection71Data.ccsRelation"
  , "ProtocolSection71Data.ceRelation"
  , "ccsRelation_of_section71Data"
  , "ceRelation_of_section71Data"
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- [Role: Theorem-Target] Explicit protocol-side Definition-14 owner specialized to one compact target context. -/
abbrev ProtocolSection71Data := SuperNeo.ProtocolSection71Data

/-- [Role: Theorem-Target] Recover the shared Definition-14 parameter package. -/
def ProtocolSection71Data_params
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams h.Commitment :=
  SuperNeo.ProtocolSection71Data.params h

/-- [Role: Theorem-Target] Recover the compact challenge-set equality. -/
theorem ProtocolSection71Data_challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  h.challengeSet = ctx.cset :=
  SuperNeo.ProtocolSection71Data.challengeSet_eq_cset h

/-- [Role: Theorem-Target] Recover that CCS and CE share one commitment. -/
theorem ProtocolSection71Data_sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  SuperNeo.ProtocolSection71Data.sharedCommitment_eq h

/-- [Role: Theorem-Target] Recover that CCS and CE share one public input. -/
theorem ProtocolSection71Data_sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  SuperNeo.ProtocolSection71Data.sharedPublicInput_eq h

/-- [Role: Theorem-Target] Recover that CE uses the CCS full vector `[x, w]`. -/
theorem ProtocolSection71Data_ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness :=
  SuperNeo.ProtocolSection71Data.ceAssignment_eq_fullVector h

/-- [Role: Theorem-Target] Build the compact protocol theorem instance. -/
def ProtocolSection71Data_theoremInstance
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  ProtocolSection71TheoremInstance ctx :=
  SuperNeo.ProtocolSection71Data.theoremInstance h

/-- [Role: Theorem-Target] Build the single-object compact Section 7.1 context owner. -/
def ProtocolSection71Data_context
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  ProtocolSection71Context :=
  SuperNeo.ProtocolSection71Data.context h

/-- [Role: Theorem-Target] Derive compact `ccsRelation` from protocol-side Section 7.1 data. -/
theorem ProtocolSection71Data_ccsRelation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  SuperNeo.ccsRelation ctx :=
  SuperNeo.ProtocolSection71Data.ccsRelation h

/-- [Role: Theorem-Target] Derive compact `ceRelation` from protocol-side Section 7.1 data. -/
theorem ProtocolSection71Data_ceRelation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  SuperNeo.ceRelation ctx :=
  SuperNeo.ProtocolSection71Data.ceRelation h

/-- [Role: Theorem-Target] Derive compact `ccsRelation` directly from one protocol-side Section 7.1 data owner. -/
theorem ccsRelation_of_section71Data
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  SuperNeo.ccsRelation ctx :=
  SuperNeo.ccsRelation_of_section71Data h

/-- [Role: Theorem-Target] Derive compact `ceRelation` directly from one protocol-side Section 7.1 data owner. -/
theorem ceRelation_of_section71Data
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  SuperNeo.ceRelation ctx :=
  SuperNeo.ceRelation_of_section71Data h

end ProtocolSection71DataInterface

end SuperNeo
