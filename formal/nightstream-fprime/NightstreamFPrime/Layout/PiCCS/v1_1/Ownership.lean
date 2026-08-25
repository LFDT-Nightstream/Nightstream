import NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS Steps 1--5.
Obligation: Give every row family and physical column in the sole PiCCS
layout one explicit owner.

The parent adds no boundary rows or logical copy columns. Shared values remain
the same symbolic expressions used by `Formal.opsAt`. R1CS multiplication
intermediates occupy the one interval after all logical child intervals.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Ownership

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Mathematical owner of one PiCCS child row family. -/
inductive ChildOwner where
  | statementBinding
  | statementAbsorption
  | challengeDerivation
  | roundTranscript
  | initialClaim
  | sumcheckChain
  | eval_K
  | eval_A
  | ccsTerminal
  | normTerminal
  | finalIdentity
  | outputBinding
deriving Repr, DecidableEq

def childOrder : List ChildOwner :=
  [.statementBinding, .statementAbsorption, .challengeDerivation,
    .roundTranscript, .initialClaim, .sumcheckChain, .eval_K, .eval_A,
    .ccsTerminal, .normTerminal, .finalIdentity, .outputBinding]

theorem childOrder_length : childOrder.length = 12 := by
  rfl

theorem childConstraintLists_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    (childConstraintLists relation interface offset).length = 12 := by
  rfl

inductive RowRole where
  | directRecipe
  | multiplication (ordinal : Nat)
  | assertion
deriving Repr, DecidableEq

structure RowOwner where
  child : ChildOwner
  logicalRow : Nat
  role : RowRole
deriving Repr, DecidableEq

def ownersFor (child : ChildOwner) : Nat → List Expr → List RowOwner
  | _, [] => []
  | logicalRow, expression :: rest =>
      let current :=
        match R1CS.directConstraint expression with
        | some _ => [⟨child, logicalRow, .directRecipe⟩]
        | none =>
            (List.range (R1CS.mulCount expression)).map (fun ordinal =>
              ⟨child, logicalRow, .multiplication ordinal⟩) ++
            [⟨child, logicalRow, .assertion⟩]
      current ++ ownersFor child (logicalRow + 1) rest

theorem ownersFor_length (child : ChildOwner) (start : Nat)
    (constraints : List Expr) :
    (ownersFor child start constraints).length =
      R1CS.totalRowCount constraints := by
  induction constraints generalizing start with
  | nil => rfl
  | cons expression rest ih =>
      cases result : R1CS.directConstraint expression with
      | none =>
          simp [ownersFor, R1CS.totalRowCount, R1CS.constraintRowCount,
            result, ih]
          omega
      | some direct =>
          simp [ownersFor, R1CS.totalRowCount, R1CS.constraintRowCount,
            result, ih]
          omega

def ownersForChildren : List ChildOwner → List (List Expr) → List RowOwner
  | child :: children, constraints :: rest =>
      ownersFor child 0 constraints ++ ownersForChildren children rest
  | _, _ => []

theorem ownersForChildren_length (children : List ChildOwner)
    (constraintLists : List (List Expr))
    (sameLength : children.length = constraintLists.length) :
    (ownersForChildren children constraintLists).length =
      (constraintLists.map R1CS.totalRowCount).sum := by
  induction children generalizing constraintLists with
  | nil =>
      cases constraintLists with
      | nil => rfl
      | cons constraints rest => simp at sameLength
  | cons child children ih =>
      cases constraintLists with
      | nil => simp at sameLength
      | cons constraints rest =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp only [ownersForChildren, List.length_append,
            ownersFor_length, List.map_cons, List.sum_cons]
          rw [ih rest sameLength]

def rowOwners
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List RowOwner :=
  ownersForChildren childOrder
    (childConstraintLists relation interface offset)

theorem rowOwners_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    (rowOwners relation interface offset).length =
      physicalRowCount relation interface offset := by
  calc
    _ = ((childConstraintLists relation interface offset).map
          R1CS.totalRowCount).sum :=
      ownersForChildren_length childOrder _ (by
        rw [childOrder_length, childConstraintLists_length])
    _ = (physicalRowDeltas relation interface offset).sum := rfl
    _ = R1CS.totalRowCount
          (logicalConstraints relation interface offset) :=
      (totalRowCount_eq_deltas relation interface offset).symm
    _ = _ := (physicalRowCount_eq relation interface offset).symm

/-- Total owner function for the complete physical row list. -/
def rowOwner
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (row : Fin (physicalRowCount relation interface offset)) :
    RowOwner :=
  (rowOwners relation interface offset).get
    ⟨row.val, by rw [rowOwners_length]; exact row.isLt⟩

def ownedRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List (RowOwner × R1CS.Row) :=
  (rowOwners relation interface offset).zip
    (physicalRows relation interface offset)

theorem ownedRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    (ownedRows relation interface offset).length =
      physicalRowCount relation interface offset := by
  rw [ownedRows, List.length_zip, rowOwners_length]
  simp [physicalRowCount, physicalRows, R1CS.LoweringPlan.rowCount]

/-- Exact child-private deltas in `Formal.opsAt` order. -/
def logicalPrivateDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Nat :=
  (Formal.opsAt relation interface offset).map Op.localLength

theorem logicalPrivateDeltas_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    (logicalPrivateDeltas relation interface offset).length = 12 := by
  simp [logicalPrivateDeltas, Formal.opsAt]

/-- Parent assembly contributes no logical copy columns. -/
theorem noBoundaryColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    logicalColumnCount relation interface offset =
      offset + (logicalPrivateDeltas relation interface offset).sum := by
  rw [logicalColumnCount_eq relation interface offset, Formal.main_ops]
  rfl

/-- Parent assembly contributes no logical copy rows. -/
theorem noBoundaryRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    logicalConstraints relation interface offset =
      orderedConstraints relation interface offset :=
  logicalConstraints_eq_ordered relation interface offset

inductive ColumnOwner where
  | external (index : Nat)
  | statementBinding (index : Nat)
  | statementAbsorption (index : Nat)
  | challengeDerivation (index : Nat)
  | roundTranscript (index : Nat)
  | initialClaim (index : Nat)
  | sumcheckChain (index : Nat)
  | eval_K (index : Nat)
  | eval_A (index : Nat)
  | ccsTerminal (index : Nat)
  | normTerminal (index : Nat)
  | finalIdentity (index : Nat)
  | outputBinding (index : Nat)
  | r1csIntermediate (index : Nat)
deriving Repr, DecidableEq

/-- Every physical column has one owner. Zero-length child intervals are
retained in the type so the ownership vocabulary matches all 12 leaves. -/
def columnOwner
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (column : Fin (physicalColumnCount relation interface offset)) :
    ColumnOwner :=
  if column.val < offset then
    .external column.val
  else if column.val < Formal.statementAbsorptionOffset interface offset then
    .statementBinding (column.val - offset)
  else if column.val < Formal.challengeOffset interface offset then
    .statementAbsorption
      (column.val - Formal.statementAbsorptionOffset interface offset)
  else if column.val < Formal.roundTranscriptOffset interface offset then
    .challengeDerivation
      (column.val - Formal.challengeOffset interface offset)
  else if column.val < Formal.initialClaimOffset interface offset then
    .roundTranscript
      (column.val - Formal.roundTranscriptOffset interface offset)
  else if column.val < Formal.sumcheckOffset interface offset then
    .initialClaim (column.val - Formal.initialClaimOffset interface offset)
  else if column.val < Formal.evalKOffset interface offset then
    .sumcheckChain (column.val - Formal.sumcheckOffset interface offset)
  else if column.val < Formal.evalAOffset interface offset then
    .eval_K (column.val - Formal.evalKOffset interface offset)
  else if column.val < Formal.ccsOffset interface offset then
    .eval_A (column.val - Formal.evalAOffset interface offset)
  else if column.val < Formal.normOffset relation interface offset then
    .ccsTerminal (column.val - Formal.ccsOffset interface offset)
  else if column.val < Formal.finalIdentityOffset relation interface offset then
    .normTerminal (column.val - Formal.normOffset relation interface offset)
  else if column.val < Formal.outputBindingOffset relation interface offset then
    .finalIdentity
      (column.val - Formal.finalIdentityOffset relation interface offset)
  else if column.val < logicalColumnCount relation interface offset then
    .outputBinding
      (column.val - Formal.outputBindingOffset relation interface offset)
  else
    .r1csIntermediate
      (column.val - logicalColumnCount relation interface offset)

theorem columnOwner_unique
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (column : Fin (physicalColumnCount relation interface offset))
    {first second : ColumnOwner}
    (firstEq : columnOwner relation interface offset column = first)
    (secondEq : columnOwner relation interface offset column = second) :
    first = second := by
  rw [← firstEq, secondEq]

end NightstreamFPrime.Layout.PiCCS.v1_1.Ownership
