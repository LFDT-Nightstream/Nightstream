import NightstreamFPrime.Layout.PiDEC.v1_1.Preservation

/-!
Owns the total row and column maps for the exact PiDEC v1_1 layout.

Every lowered row names its logical child and row role. Every physical column
is external, one of the 54 public-split sign cells, or an R1CS intermediate.
The parent adds no copy row or boundary column.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.Ownership

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

inductive ChildOwner where
  | inputBinding
  | publicInputSplit
  | commitment
  | eval_K
  | eval_A
  | outputBinding
deriving Repr, DecidableEq

def childOrder : List ChildOwner :=
  [.inputBinding, .publicInputSplit, .commitment, .eval_K, .eval_A,
    .outputBinding]

theorem childOrder_length : childOrder.length = 6 := by
  rfl

theorem childConstraintLists_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    (childConstraintLists relation interface offset).length = 6 := by
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
  | cons expression rest inductionHypothesis =>
      cases result : R1CS.directConstraint expression with
      | none =>
          simp [ownersFor, R1CS.totalRowCount, R1CS.constraintRowCount,
            result, inductionHypothesis]
          omega
      | some direct =>
          simp [ownersFor, R1CS.totalRowCount, R1CS.constraintRowCount,
            result, inductionHypothesis]
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
  | cons child children inductionHypothesis =>
      cases constraintLists with
      | nil => simp at sameLength
      | cons constraints rest =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp only [ownersForChildren, List.length_append,
            ownersFor_length, List.map_cons, List.sum_cons]
          rw [inductionHypothesis rest sameLength]

def rowOwners
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List RowOwner :=
  ownersForChildren childOrder
    (childConstraintLists relation interface offset)

theorem rowOwners_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
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

theorem rowOwners_length_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    (rowOwners relation interface offset).length = 7128 := by
  rw [rowOwners_length,
    physicalRowCount_eq_production relation interface offset inputs]

def rowOwner
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (row : Fin (physicalRowCount relation interface offset)) :
    RowOwner :=
  (rowOwners relation interface offset).get
    ⟨row.val, by rw [rowOwners_length]; exact row.isLt⟩

def ownedRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List (RowOwner × R1CS.Row) :=
  (rowOwners relation interface offset).zip
    (physicalRows relation interface offset)

theorem ownedRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    (ownedRows relation interface offset).length =
      physicalRowCount relation interface offset := by
  rw [ownedRows, List.length_zip, rowOwners_length, physicalRows_length]
  simp

/-- The parent inserts no copy or boundary row between its six children. -/
theorem noBoundaryRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    logicalConstraints relation interface offset =
      orderedConstraints relation interface offset :=
  logicalConstraints_eq_ordered relation interface offset

/-- The parent's exact logical interval contains only the 54 sign cells. -/
theorem noBoundaryColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    logicalColumnCount relation interface offset = offset + 54 :=
  logicalColumnCount_eq_production relation interface offset

inductive ColumnOwner where
  | external (index : Nat)
  | publicInputSplit (index : Nat)
  | r1csIntermediate (index : Nat)
deriving Repr, DecidableEq

/-- Total owner function for every external, logical, and R1CS-fresh column. -/
def columnOwner
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat)
    (column : Fin (physicalColumnCount relation interface offset)) :
    ColumnOwner :=
  if column.val < offset then
    .external column.val
  else if column.val < logicalColumnCount relation interface offset then
    .publicInputSplit (column.val - offset)
  else
    .r1csIntermediate
      (column.val - logicalColumnCount relation interface offset)

theorem columnOwner_external
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (column : Fin (physicalColumnCount relation interface offset))
    (below : column.val < offset) :
    columnOwner relation interface offset column = .external column.val := by
  unfold columnOwner
  rw [if_pos below]

theorem columnOwner_publicInputSplit
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (column : Fin (physicalColumnCount relation interface offset))
    (atOrAbove : offset ≤ column.val)
    (below : column.val < logicalColumnCount relation interface offset) :
    columnOwner relation interface offset column =
      .publicInputSplit (column.val - offset) := by
  unfold columnOwner
  rw [if_neg (Nat.not_lt.mpr atOrAbove), if_pos below]

theorem columnOwner_r1csIntermediate
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (column : Fin (physicalColumnCount relation interface offset))
    (atOrAbove : logicalColumnCount relation interface offset ≤ column.val) :
    columnOwner relation interface offset column =
      .r1csIntermediate
        (column.val - logicalColumnCount relation interface offset) := by
  have phaseAbove : offset ≤ column.val := by
    have logicalAbove : offset ≤ logicalColumnCount relation interface offset := by
      rw [logicalColumnCount_eq_production]
      omega
    exact Nat.le_trans logicalAbove atOrAbove
  unfold columnOwner
  rw [if_neg (Nat.not_lt.mpr phaseAbove),
    if_neg (Nat.not_lt.mpr atOrAbove)]

theorem columnOwner_unique
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (column : Fin (physicalColumnCount relation interface offset))
    {first second : ColumnOwner}
    (firstEq : columnOwner relation interface offset column = first)
    (secondEq : columnOwner relation interface offset column = second) :
    first = second := by
  rw [← firstEq, ← secondEq]

end NightstreamFPrime.Layout.PiDEC.v1_1.Ownership
