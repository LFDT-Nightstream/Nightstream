import NightstreamFPrime.Circuit.Sequence

/-! Preserve the exact constructed environment when appending an opaque
child. The child still supplies its own scope, row, and agreement proofs. -/

namespace NightstreamFPrime.Circuit.Sequence

theorem child_rows (env : Env) (operations : List Op) (rows : holdsFlat env operations)
    (name : String) (child : FormalCircuit) (start : Nat)
    (member : childOp name child start ∈ operations) :
    holdsFlat env (Circuit.ops child.main start) := by
  intro expression inside
  apply rows expression
  exact List.mem_flatMap.mpr ⟨childOp name child start, member, inside⟩

theorem appendBuiltAt_current
    {initial : Env} {base : Nat} (before : Prefix initial base)
    (name : String) (child : FormalCircuit) (start : Nat)
    (startEq : base + localLength before.operations = start)
    (childScope : ∀ expression ∈ flatConstraints (Circuit.ops child.main start),
      expression.VarsBelow (start + localLength (Circuit.ops child.main start)))
    (after : Env)
    (childAgrees : AgreesOutside before.current after start (localLength (Circuit.ops child.main start)))
    (childRows : holdsFlat after (Circuit.ops child.main start)) :
    ∃ completed : Prefix initial base,
      completed.operations = before.operations ++ [childOp name child start] ∧
      base + localLength completed.operations = start + localLength (Circuit.ops child.main start) ∧
      PreservesPrefix before completed ∧
      completed.current = after := by
  let operation := childOp name child start
  have length : operation.localLength = localLength (Circuit.ops child.main start) :=
    childOp_localLength name child start
  have constraints : operation.flatConstraints = flatConstraints (Circuit.ops child.main start) := rfl
  have beforeRows : ConstraintsHold after (flatConstraints before.operations) := by
    apply constraintsHold_of_agree_below before.current after _ _ before.scope
      (fun index below => childAgrees index (Or.inl (by omega))) before.rows
  let completed : Prefix initial base := {
    current := after
    operations := before.operations ++ [operation]
    agrees := by
      have combined := before.agrees.append (by simpa only [startEq] using childAgrees)
      simpa only [localLength_append, localLength_singleton, length] using combined
    scope := by
      intro expression member
      rw [flatConstraints_append, flatConstraints_singleton] at member
      rcases List.mem_append.mp member with earlier | added
      · apply Expr.VarsBelow.mono _ (before.scope expression earlier)
        rw [localLength_append, localLength_singleton]
        omega
      · rw [constraints] at added
        apply Expr.VarsBelow.mono _ (childScope expression added)
        rw [localLength_append, localLength_singleton, length]
        omega
    rows := by
      change ConstraintsHold after (flatConstraints (before.operations ++ [operation]))
      rw [flatConstraints_append, flatConstraints_singleton, constraints]
      exact (constraintsHold_append after _ _).mpr ⟨beforeRows, childRows⟩ }
  refine ⟨completed, rfl, ?_, ?_, rfl⟩
  · change base + localLength (before.operations ++ [operation]) = _
    rw [localLength_append, localLength_singleton, length]
    omega
  · constructor
    · simp only [completed, localLength_append]
      omega
    · intro index below
      exact childAgrees index (Or.inl (by omega))

end NightstreamFPrime.Circuit.Sequence
