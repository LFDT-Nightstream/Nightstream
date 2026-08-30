import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation

/-!
Owns every physical row and column of the Stage 1 running transition.
The transition is parent wiring: one flag-binding family, one indexed mux
family, and four base-state equality rows. No file boundary or copy family
exists.
-/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.Ownership

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
open NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

inductive ConstraintOwner where
  | binding
  | mux (word : Nat)
  | baseState (word : Nat)
deriving Repr, DecidableEq

def constraintOwner : Nat → ConstraintOwner
  | 0 => .binding
  | logicalRow + 1 =>
      if logicalRow < RunningTransition.exactWordCount then
        .mux logicalRow
      else
        .baseState (logicalRow - RunningTransition.exactWordCount)

inductive RowRole where
  | directRecipe
  | multiplication (ordinal : Nat)
  | assertion
deriving Repr, DecidableEq

structure RowOwner where
  constraint : ConstraintOwner
  logicalRow : Nat
  role : RowRole
deriving Repr, DecidableEq

def ownersFor : Nat → List Expr → List RowOwner
  | _, [] => []
  | logicalRow, expression :: rest =>
      let current :=
        match R1CS.directConstraint expression with
        | some _ => [⟨constraintOwner logicalRow, logicalRow,
            .directRecipe⟩]
        | none =>
            (List.range (R1CS.mulCount expression)).map (fun ordinal =>
              ⟨constraintOwner logicalRow, logicalRow,
                .multiplication ordinal⟩) ++
            [⟨constraintOwner logicalRow, logicalRow, .assertion⟩]
      current ++ ownersFor (logicalRow + 1) rest

theorem ownersFor_length (start : Nat) (constraints : List Expr) :
    (ownersFor start constraints).length =
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

def rowOwners
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List RowOwner :=
  ownersFor 0 (logicalConstraints logicalWidth publicFits)

theorem rowOwners_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (rowOwners logicalWidth publicFits).length =
      physicalRowCount logicalWidth publicFits := by
  calc
    _ = R1CS.totalRowCount (logicalConstraints logicalWidth publicFits) :=
      ownersFor_length _ _
    _ = 321303 := totalRowCount_eq relation
    _ = _ := (physicalRowCount_eq relation).symm

theorem rowOwners_length_production
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (rowOwners logicalWidth publicFits).length = 321303 := by
  rw [rowOwners_length relation, physicalRowCount_eq relation]

def rowOwner
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (row : Fin (physicalRowCount logicalWidth publicFits)) : RowOwner :=
  (rowOwners logicalWidth publicFits).get
    ⟨row.val, by rw [rowOwners_length relation]; exact row.isLt⟩

def ownedRows
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (_relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List (RowOwner × R1CS.Row) :=
  (rowOwners logicalWidth publicFits).zip
    (physicalRows logicalWidth publicFits)

theorem ownedRows_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (ownedRows relation).length =
      physicalRowCount logicalWidth publicFits := by
  rw [ownedRows, List.length_zip, rowOwners_length relation,
    physicalRows_length]
  simp

/-- The transition adds no copy or file-boundary row. -/
theorem noBoundaryRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    logicalConstraints logicalWidth publicFits =
      RunningTransition.bindingConstraint (interface logicalWidth publicFits)
          phaseOffset ::
        (RunningTransition.muxConstraints (interface logicalWidth publicFits)
            phaseOffset ++
          RunningTransition.baseStateConstraints
            (interface logicalWidth publicFits) phaseOffset) := by
  rw [logicalConstraints_eq]
  rfl

inductive ColumnOwner where
  | external (index : Nat)
  | inverseHint
  | r1csIntermediate (index : Nat)
deriving Repr, DecidableEq

def columnOwner
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (column : Fin (physicalColumnCount logicalWidth publicFits)) :
    ColumnOwner :=
  if column.val < phaseOffset then
    .external column.val
  else if column.val < logicalColumnCount then
    .inverseHint
  else
    .r1csIntermediate (column.val - logicalColumnCount)

theorem columnOwner_external
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (column : Fin (physicalColumnCount logicalWidth publicFits))
    (below : column.val < phaseOffset) :
    columnOwner logicalWidth publicFits column = .external column.val := by
  unfold columnOwner
  rw [if_pos below]

theorem columnOwner_inverseHint
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (column : Fin (physicalColumnCount logicalWidth publicFits))
    (atOrAbove : phaseOffset ≤ column.val)
    (below : column.val < logicalColumnCount) :
    columnOwner logicalWidth publicFits column = .inverseHint := by
  unfold columnOwner
  rw [if_neg (Nat.not_lt.mpr atOrAbove), if_pos below]

theorem columnOwner_r1csIntermediate
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (column : Fin (physicalColumnCount logicalWidth publicFits))
    (atOrAbove : logicalColumnCount ≤ column.val) :
    columnOwner logicalWidth publicFits column =
      .r1csIntermediate (column.val - logicalColumnCount) := by
  have phaseAbove : phaseOffset ≤ column.val := by
    have logicalAbove : phaseOffset ≤ logicalColumnCount := by
      unfold logicalColumnCount
      omega
    exact Nat.le_trans logicalAbove atOrAbove
  unfold columnOwner
  rw [if_neg (Nat.not_lt.mpr phaseAbove),
    if_neg (Nat.not_lt.mpr atOrAbove)]

theorem columnOwner_unique
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (column : Fin (physicalColumnCount logicalWidth publicFits))
    {first second : ColumnOwner}
    (firstEq : columnOwner logicalWidth publicFits column = first)
    (secondEq : columnOwner logicalWidth publicFits column = second) :
    first = second := by
  rw [← firstEq, ← secondEq]

end NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.Ownership
