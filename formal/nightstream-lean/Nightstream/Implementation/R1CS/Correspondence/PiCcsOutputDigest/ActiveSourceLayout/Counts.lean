import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout

/-!
Authority-family counts for the active `Pi_CCS` output role tree.

Assurance tier: model-level representation correspondence.

Owns: counting the actual typed roles by authority owner; closed forms at
every source/vector branch; and exact reconciliation to the complete
serializer length.

Does not own: R1CS rows or columns, physical source bindings, `y_zcol`
authority, SIS/Poseidon2, costs, or row removal.

Emits constraints: no.

| Tree branch | Verifier shape | `y_ring` output | `y_zcol` output |
|---|---:|---:|---:|
| outer header | `8` | `0` | `0` |
| one source header and widths | `10 + matrixCount` | `0` | `0` |
| one source payload | `0` | `matrixCount * 2D` | `2D` |
| complete message | `8 + sources * (10 + matrices)` | `sources * matrices * 2D` | `sources * 2D` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Count roles by their actual `inputOwner` classification. -/
def countOwned
    {shape : SemanticShape}
    (owner : InputOwner) : List (SourceRole shape) -> Nat
  | [] => 0
  | role :: roles =>
      (if inputOwner role = owner then 1 else 0) + countOwned owner roles

/-- Exact semantic field count owned by one authority class. This is not an
R1CS row or column count. -/
def ownerFieldCount (shape : SemanticShape) (owner : InputOwner) : Nat :=
  countOwned owner (sourceRoles shape)

@[simp] theorem countOwned_append
    {shape : SemanticShape}
    (owner : InputOwner)
    (left right : List (SourceRole shape)) :
    countOwned owner (left ++ right) =
      countOwned owner left + countOwned owner right := by
  induction left with
  | nil => simp [countOwned]
  | cons role left inductionHypothesis =>
      simp [countOwned, inductionHypothesis, Nat.add_assoc]

private theorem countOwned_flatten
    {shape : SemanticShape}
    (owner : InputOwner)
    (blocks : List (List (SourceRole shape))) :
    countOwned owner blocks.flatten =
      (blocks.map (countOwned owner)).sum := by
  induction blocks with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp [countOwned_append, inductionHypothesis]

private theorem sum_ofFn_const (count value : Nat) :
    (List.ofFn fun _ : Fin count => value).sum = count * value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      simp only [List.sum_cons, inductionHypothesis, Nat.succ_mul]
      omega

private theorem countOwned_flattenOfFn_const
    {shape : SemanticShape}
    {count value : Nat}
    (owner : InputOwner)
    (blocks : Fin count -> List (SourceRole shape))
    (blockCount : forall index, countOwned owner (blocks index) = value) :
    countOwned owner (List.ofFn blocks).flatten = count * value := by
  rw [countOwned_flatten, List.map_ofFn]
  have counts :
      List.ofFn (countOwned owner ∘ blocks) =
        List.ofFn (fun _ : Fin count => value) := by
    apply congrArg List.ofFn
    funext index
    exact blockCount index
  rw [counts, sum_ofFn_const]

private theorem yRingLimbCounts
    {shape : SemanticShape}
    (owner : InputOwner)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) :
    countOwned owner
        (twoLimbRoles fun lane limb =>
          SourceRole.yRingLimb source matrix lane limb) =
      if owner = .yRingOutput then 2 * ringDegree else 0 := by
  rw [twoLimbRoles]
  cases owner with
  | verifierShape =>
      simpa [countOwned, inputOwner] using
        countOwned_flattenOfFn_const .verifierShape
          (fun lane : Fin ringDegree =>
            [SourceRole.yRingLimb source matrix lane .c0,
             SourceRole.yRingLimb source matrix lane .c1])
          (value := 0) (by intro; rfl)
  | yRingOutput =>
      simpa [countOwned, inputOwner, Nat.mul_comm] using
        countOwned_flattenOfFn_const .yRingOutput
          (fun lane : Fin ringDegree =>
            [SourceRole.yRingLimb source matrix lane .c0,
             SourceRole.yRingLimb source matrix lane .c1])
          (value := 2) (by intro; rfl)
  | yZcolOutput =>
      simpa [countOwned, inputOwner] using
        countOwned_flattenOfFn_const .yZcolOutput
          (fun lane : Fin ringDegree =>
            [SourceRole.yRingLimb source matrix lane .c0,
             SourceRole.yRingLimb source matrix lane .c1])
          (value := 0) (by intro; rfl)

private theorem yZcolLimbCounts
    {shape : SemanticShape}
    (owner : InputOwner)
    (source : Fin shape.sourceCount) :
    countOwned owner
        (twoLimbRoles fun lane limb =>
          SourceRole.yZcolLimb source lane limb) =
      if owner = .yZcolOutput then 2 * ringDegree else 0 := by
  rw [twoLimbRoles]
  cases owner with
  | verifierShape =>
      simpa [countOwned, inputOwner] using
        countOwned_flattenOfFn_const .verifierShape
          (fun lane : Fin ringDegree =>
            [SourceRole.yZcolLimb source lane .c0,
             SourceRole.yZcolLimb source lane .c1])
          (value := 0) (by intro; rfl)
  | yRingOutput =>
      simpa [countOwned, inputOwner] using
        countOwned_flattenOfFn_const .yRingOutput
          (fun lane : Fin ringDegree =>
            [SourceRole.yZcolLimb source lane .c0,
             SourceRole.yZcolLimb source lane .c1])
          (value := 0) (by intro; rfl)
  | yZcolOutput =>
      simpa [countOwned, inputOwner, Nat.mul_comm] using
        countOwned_flattenOfFn_const .yZcolOutput
          (fun lane : Fin ringDegree =>
            [SourceRole.yZcolLimb source lane .c0,
             SourceRole.yZcolLimb source lane .c1])
          (value := 2) (by intro; rfl)

private theorem yRingVectorCounts
    {shape : SemanticShape}
    (owner : InputOwner)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) :
    countOwned owner (yRingVectorRoles source matrix) =
      match owner with
      | .verifierShape => 1
      | .yRingOutput => 2 * ringDegree
      | .yZcolOutput => 0 := by
  cases owner <;>
    simp [yRingVectorRoles, countOwned, inputOwner,
      yRingLimbCounts]

private theorem yRingCounts
    {shape : SemanticShape}
    (owner : InputOwner)
    (source : Fin shape.sourceCount) :
    countOwned owner (yRingRoles source) =
      match owner with
      | .verifierShape => shape.matrixCount
      | .yRingOutput => shape.matrixCount * (2 * ringDegree)
      | .yZcolOutput => 0 := by
  rw [yRingRoles]
  cases owner with
  | verifierShape =>
      simpa using countOwned_flattenOfFn_const .verifierShape
        (fun matrix : Fin shape.matrixCount =>
          yRingVectorRoles source matrix)
        (value := 1) (yRingVectorCounts .verifierShape source)
  | yRingOutput =>
      simpa using countOwned_flattenOfFn_const .yRingOutput
        (fun matrix : Fin shape.matrixCount =>
          yRingVectorRoles source matrix)
        (value := 2 * ringDegree) (yRingVectorCounts .yRingOutput source)
  | yZcolOutput =>
      simpa using countOwned_flattenOfFn_const .yZcolOutput
        (fun matrix : Fin shape.matrixCount =>
          yRingVectorRoles source matrix)
        (value := 0) (yRingVectorCounts .yZcolOutput source)

private theorem yZcolCounts
    {shape : SemanticShape}
    (owner : InputOwner)
    (source : Fin shape.sourceCount) :
    countOwned owner (yZcolRoles source) =
      match owner with
      | .verifierShape => 1
      | .yRingOutput => 0
      | .yZcolOutput => 2 * ringDegree := by
  cases owner <;>
    simp [yZcolRoles, countOwned, inputOwner, yZcolLimbCounts]

private theorem sourceHeaderCounts
    {shape : SemanticShape}
    (owner : InputOwner)
    (source : Fin shape.sourceCount) :
    countOwned owner (sourceHeaderRoles source) =
      if owner = .verifierShape then 9 else 0 := by
  rw [sourceHeaderRoles, countOwned_append]
  cases owner with
  | verifierShape =>
      simp [countOwned, inputOwner]
  | yRingOutput =>
      simp [countOwned, inputOwner]
  | yZcolOutput =>
      simp [countOwned, inputOwner]

private theorem sourceBlockCounts
    {shape : SemanticShape}
    (owner : InputOwner)
    (source : Fin shape.sourceCount) :
    countOwned owner (sourceBlockRoles source) =
      match owner with
      | .verifierShape => 10 + shape.matrixCount
      | .yRingOutput => shape.matrixCount * (2 * ringDegree)
      | .yZcolOutput => 2 * ringDegree := by
  cases owner <;>
    simp [sourceBlockRoles, countOwned_append, sourceHeaderCounts,
      yRingCounts, yZcolCounts] <;>
    omega

private theorem outerHeaderCounts
    (shape : SemanticShape)
    (owner : InputOwner) :
    countOwned owner (outerHeaderRoles shape) =
      if owner = .verifierShape then 8 else 0 := by
  rw [outerHeaderRoles, countOwned_append]
  cases owner with
  | verifierShape =>
      simp [countOwned, inputOwner]
  | yRingOutput =>
      simp [countOwned, inputOwner]
  | yZcolOutput =>
      simp [countOwned, inputOwner]

theorem ownerFieldCount_verifierShape (shape : SemanticShape) :
    ownerFieldCount shape .verifierShape =
      8 + shape.sourceCount * (10 + shape.matrixCount) := by
  rw [ownerFieldCount, sourceRoles, countOwned_append,
    outerHeaderCounts]
  rw [countOwned_flattenOfFn_const .verifierShape
    (fun source : Fin shape.sourceCount => sourceBlockRoles source)
    (value := 10 + shape.matrixCount)
    (sourceBlockCounts .verifierShape)]
  simp

theorem ownerFieldCount_yRingOutput (shape : SemanticShape) :
    ownerFieldCount shape .yRingOutput =
      shape.sourceCount * shape.matrixCount * (2 * ringDegree) := by
  rw [ownerFieldCount, sourceRoles, countOwned_append,
    outerHeaderCounts]
  rw [countOwned_flattenOfFn_const .yRingOutput
    (fun source : Fin shape.sourceCount => sourceBlockRoles source)
    (value := shape.matrixCount * (2 * ringDegree))
    (sourceBlockCounts .yRingOutput)]
  simp [Nat.mul_assoc]

theorem ownerFieldCount_yZcolOutput (shape : SemanticShape) :
    ownerFieldCount shape .yZcolOutput =
      shape.sourceCount * (2 * ringDegree) := by
  rw [ownerFieldCount, sourceRoles, countOwned_append,
    outerHeaderCounts]
  rw [countOwned_flattenOfFn_const .yZcolOutput
    (fun source : Fin shape.sourceCount => sourceBlockRoles source)
    (value := 2 * ringDegree)
    (sourceBlockCounts .yZcolOutput)]
  simp

private theorem countOwned_partition
    {shape : SemanticShape}
    (roles : List (SourceRole shape)) :
    countOwned .verifierShape roles +
        countOwned .yRingOutput roles +
        countOwned .yZcolOutput roles = roles.length := by
  induction roles with
  | nil => rfl
  | cons role roles inductionHypothesis =>
      cases ownerEq : inputOwner role <;>
        simp [countOwned, ownerEq] <;>
        omega

/-- The three authority families partition the actual role tree and exactly
reconcile to the complete semantic field count. -/
theorem ownerFieldCounts_reconcile (shape : SemanticShape) :
    ownerFieldCount shape .verifierShape +
        ownerFieldCount shape .yRingOutput +
        ownerFieldCount shape .yZcolOutput =
      ActiveSemantics.fieldCount shape := by
  rw [ownerFieldCount, ownerFieldCount, ownerFieldCount,
    countOwned_partition, sourceRoles_length]

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout
