import OpeningConvergence.Basic
import SuperNeo.ProofSystem.Lattice
import SuperNeo.ProofSystem.ConstraintSystem.CCSInterface

/-!
# Module 5: SuperNeoBoundary — Interface

Owns the concrete instantiation of `AjtaiPCSBoundary` against the current
proved SuperNeo lattice / CE surfaces.

This is the paper-faithful end of the local opening-convergence package for
the current base-field CE boundary: one reduced claim is satisfied exactly
when there exists a SuperNeo CE witness for the claim statement derived from
the opened object, point, and payload. The full repo-wide end state still
needs the extension-field bridge above this boundary.

## Spec
See `specs/SuperNeoBoundary.spec.md`
-/

namespace OpeningConvergence.SuperNeoBoundary

abbrev F := SuperNeo.F
abbrev Coeffs := SuperNeo.Coeffs
abbrev Commitment := SuperNeo.ProofSystem.Commitment
abbrev CE := SuperNeo.ProofSystem.ConstraintSystem.CCSInterface.CE Commitment
abbrev CEStatement := SuperNeo.ProofSystem.ConstraintSystem.CCSInterface.CEStatement Commitment
abbrev CEWitness := SuperNeo.ProofSystem.ConstraintSystem.CCSInterface.CEWitness
def CEHolds (ce : CE) (stmt : CEStatement) (wit : CEWitness) : Prop :=
  SuperNeo.ProofSystem.ConstraintSystem.CCSInterface.CEHolds ce stmt wit

/-- Convert one point in `F^ell` to the SuperNeo `Coeffs` carrier used by CE statements. -/
def pointToCoeffs {ell : Nat} (point : Fin ell → F) : Coeffs :=
  Array.ofFn point

@[simp] theorem pointToCoeffs_size {ell : Nat} (point : Fin ell → F) :
    (pointToCoeffs point).size = ell := by
  simp [pointToCoeffs]

/-- Convert one packed-column evaluation into the coefficient-vector carrier used
    by SuperNeo CE statements. -/
def packedColumnToCoeffs (eval : PackedColumnEval F) : Coeffs :=
  Array.ofFn eval.coeffs

@[simp] theorem packedColumnToCoeffs_size (eval : PackedColumnEval F) :
    (packedColumnToCoeffs eval).size = AJTAI_D := by
  simp [packedColumnToCoeffs]

/-- Convert the payload-side packed-column evaluations into the CE statement's
    evaluation array. -/
def payloadToEvaluations (payload : FamilyEvalPayload F) : Array Coeffs :=
  Array.ofFn fun j => packedColumnToCoeffs (payload.columnEvals j)

@[simp] theorem payloadToEvaluations_size (payload : FamilyEvalPayload F) :
    (payloadToEvaluations payload).size = packedColumnCount payload.schema := by
  simp [payloadToEvaluations]

/-- The theorem-facing SuperNeo data needed to interpret one `OpenedObjectId`
    as one concrete CE opening target. -/
structure OpenedObject where
  id : OpenedObjectId
  schema : FamilySchema
  commitment : Commitment
  publicInput : Coeffs
  ce : CE
  rowDomainLogSize : Nat
  arity_eq : ce.shape.arity = rowDomainLogSize

/-- One explicit registry mapping theorem-facing object identities to the
    concrete SuperNeo opening targets they denote. -/
structure Registry where
  lookup : OpenedObjectId → Option OpenedObject
  selfKey :
    ∀ {id obj}, lookup id = some obj → obj.id = id

/-- Build the CE statement corresponding to one convergence claim. -/
def claimStatement
    (obj : OpenedObject)
    {ell : Nat}
    (point : Fin ell → F)
    (payload : FamilyEvalPayload F) : CEStatement where
  commitment := obj.commitment
  publicInput := obj.publicInput
  point := pointToCoeffs point
  evaluations := payloadToEvaluations payload

/-- Concrete Ajtai PCS boundary for `OpeningConvergence` instantiated from
    SuperNeo's CE relation: a claim verifies iff the registry resolves the
    opened object and there exists a CE witness for the derived claim statement. -/
def boundary (registry : Registry) : AjtaiPCSBoundary F where
  verify := fun {ell} objectId point payload =>
    ∃ obj : OpenedObject,
      ∃ wit : CEWitness,
        registry.lookup objectId = some obj ∧
        obj.schema = payload.schema ∧
        obj.rowDomainLogSize = ell ∧
        CEHolds obj.ce (claimStatement obj point payload) wit

/-- Registry soundness: any successful lookup returns the object keyed by the
    requested identity. -/
theorem boundary_lookup_self
    (registry : Registry)
    {id : OpenedObjectId}
    {obj : OpenedObject}
    (hLookup : registry.lookup id = some obj) :
    obj.id = id :=
  registry.selfKey hLookup

/-- Local claim-satisfaction predicate induced by one concrete SuperNeo
    registry-backed PCS boundary. -/
abbrev ClaimSatisfied
    (registry : Registry)
    {ell : Nat}
    (claim : FamilyEvalClaim F ell) : Prop :=
  (boundary registry).verify claim.openedObject claim.point claim.payload

end OpeningConvergence.SuperNeoBoundary
