import SuperNeo.ProtocolSection71Context

/-!
Protocol-side Definition-14 data owner for one compact Section 7.1 theorem
instance.
-/

namespace SuperNeo

/--
Explicit protocol-side Section 7.1 data specialized to one compact target
context.

This keeps the paper's Definition-14 ingredients visible as fields rather than
hiding them inside a prebuilt theorem-instance package.
-/
structure ProtocolSection71Data (ctx : ProtocolTargetContext) where
  Commitment : Type
  challengeSet : Array Coeffs
  commitMap :
    SuperNeo.ProofSystem.ConstraintSystem.CommitmentMap Commitment
  inputProjector :
    SuperNeo.ProofSystem.ConstraintSystem.InputProjector
  shape :
    SuperNeo.ProofSystem.ConstraintSystem.CCSStructure
  normBound : Nat
  ccsStatement :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Statement Commitment
  ccsWitness :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Witness
  ceStatement :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment
  ceWitness :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Witness
  challengeSet_eq : challengeSet = ctx.cset
  sharedCommitment :
    ccsStatement.commitment = ceStatement.commitment
  sharedPublicInput :
    ccsStatement.publicInput = ceStatement.publicInput
  sharedAssignment :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
        ccsStatement ccsWitness =
      ceWitness.assignment
  ccsHolds :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
        { challengeSet := challengeSet
          commitMap := commitMap
          inputProjector := inputProjector
          shape := shape }
        normBound)
      ccsStatement
      ccsWitness
  ceHolds :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
        { challengeSet := challengeSet
          commitMap := commitMap
          inputProjector := inputProjector
          shape := shape }
        normBound)
      ceStatement
      ceWitness
  ccsHolds_from_relation :
    ccsRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
          { challengeSet := challengeSet
            commitMap := commitMap
            inputProjector := inputProjector
            shape := shape }
          normBound)
        ccsStatement
        ccsWitness
  ccsRelation_of_holds :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
          { challengeSet := challengeSet
            commitMap := commitMap
            inputProjector := inputProjector
            shape := shape }
          normBound)
        ccsStatement
        ccsWitness →
      ccsRelation ctx
  ceHolds_from_relation :
    ceRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
          { challengeSet := challengeSet
            commitMap := commitMap
            inputProjector := inputProjector
            shape := shape }
          normBound)
        ceStatement
        ceWitness
  ceRelation_of_holds :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
          { challengeSet := challengeSet
            commitMap := commitMap
            inputProjector := inputProjector
            shape := shape }
          normBound)
        ceStatement
        ceWitness →
      ceRelation ctx

namespace ProtocolSection71Data

/-- Shared Definition-14 parameter package induced by one protocol-side data owner. -/
def params
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams h.Commitment :=
  { challengeSet := h.challengeSet
    commitMap := h.commitMap
    inputProjector := h.inputProjector
    shape := h.shape }

/-- Recover the compact challenge-set from one protocol-side Section 7.1 package. -/
theorem challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  h.challengeSet = ctx.cset :=
  h.challengeSet_eq

/-- Recover that CCS and CE share one commitment. -/
theorem sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  h.sharedCommitment

/-- Recover that CCS and CE share one public input. -/
theorem sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  h.sharedPublicInput

/-- Recover that CE uses the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness := by
  simpa using h.sharedAssignment.symm

/-- Recover the compact protocol-theorem Section 7.1 theorem instance. -/
def theoremInstance
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  ProtocolSection71TheoremInstance ctx where
  Commitment := h.Commitment
  params := h.params
  normBound := h.normBound
  ccsStatement := h.ccsStatement
  ccsWitness := h.ccsWitness
  ceStatement := h.ceStatement
  ceWitness := h.ceWitness
  challengeSet_eq := h.challengeSet_eq
  sharedCommitment := h.sharedCommitment
  sharedPublicInput := h.sharedPublicInput
  sharedAssignment := h.sharedAssignment
  ccsHolds := by
    simpa [params] using h.ccsHolds
  ceHolds := by
    simpa [params] using h.ceHolds
  ccsHolds_from_relation := by
    intro hCCS
    simpa [params] using h.ccsHolds_from_relation hCCS
  ccsRelation_of_holds := by
    intro hCCS
    exact h.ccsRelation_of_holds (by simpa [params] using hCCS)
  ceHolds_from_relation := by
    intro hCE
    simpa [params] using h.ceHolds_from_relation hCE
  ceRelation_of_holds := by
    intro hCE
    exact h.ceRelation_of_holds (by simpa [params] using hCE)

/-- Recover the single-object compact theorem-native Section 7.1 owner. -/
def context
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  ProtocolSection71Context :=
  { target := ctx
    theoremInstance := h.theoremInstance }

/-- One protocol-side Section 7.1 data owner yields compact `ccsRelation`. -/
theorem ccsRelation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  SuperNeo.ccsRelation ctx :=
  h.theoremInstance.ccsRelation

/-- One protocol-side Section 7.1 data owner yields compact `ceRelation`. -/
theorem ceRelation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  SuperNeo.ceRelation ctx :=
  h.theoremInstance.ceRelation

end ProtocolSection71Data

/-- Derive compact `ccsRelation` directly from one protocol-side Section 7.1 data owner. -/
theorem ccsRelation_of_section71Data
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  ccsRelation ctx :=
  h.ccsRelation

/-- Derive compact `ceRelation` directly from one protocol-side Section 7.1 data owner. -/
theorem ceRelation_of_section71Data
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Data ctx) :
  ceRelation ctx :=
  h.ceRelation

end SuperNeo
