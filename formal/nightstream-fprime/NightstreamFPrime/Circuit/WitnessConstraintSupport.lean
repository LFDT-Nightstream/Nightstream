import NightstreamFPrime.Circuit.WitnessSupport

/-! Transfers constraint support to actual reads for arithmetic witness owners. -/

namespace NightstreamFPrime.Circuit

/-- This property must be proved by the owner of each opaque child. Arbitrary
subcircuits and hinted batches do not satisfy it by construction. -/
def WitnessesFromConstraints (operations : List Op) : Prop :=
  ∀ allowed : Nat → Prop,
    (∀ expression ∈ flatConstraints operations, expression.VarsSatisfy allowed) →
    ∀ batch ∈ witnesses operations, batch.ReadsSatisfy allowed

private theorem recipes_supported (start : Nat) (recipes : List Expr)
    (allowed : Nat → Prop)
    (supported : ∀ expression ∈ recipeConstraints start recipes,
      expression.VarsSatisfy allowed) :
    ∀ recipe ∈ recipes, recipe.VarsSatisfy allowed := by
  induction recipes generalizing start with
  | nil => simp
  | cons recipe rest inductionHypothesis =>
      intro current member
      rcases List.mem_cons.mp member with rfl | member
      · exact (supported (Expr.var start - current) (by simp [recipeConstraints])).2.2
      · exact inductionHypothesis (start + 1)
          (fun expression member => supported expression (by simp [recipeConstraints, member]))
          current member

theorem WitnessesFromConstraints.arithmetic (start : Nat) (recipes : List Expr) :
    WitnessesFromConstraints [Op.witness (WitnessBatch.arithmetic start recipes)] := by
  intro allowed supported batch member
  simp only [witnesses, List.flatMap_cons, List.flatMap_nil, Op.witnesses,
    List.append_nil, List.mem_singleton] at member
  subst batch
  rw [WitnessBatch.readsSatisfy_arithmetic]
  exact recipes_supported start recipes allowed (by
    simpa [flatConstraints, Op.flatConstraints, WitnessBatch.arithmetic] using supported)

theorem WitnessesFromConstraints.assertions (expressions : List Expr) :
    WitnessesFromConstraints (expressions.map Op.assertZero) := by
  intro allowed supported batch member
  simp [witnesses, List.flatMap_map, Op.witnesses] at member

theorem WitnessesFromConstraints.append (left right : List Op)
    (leftSupported : WitnessesFromConstraints left)
    (rightSupported : WitnessesFromConstraints right) :
    WitnessesFromConstraints (left ++ right) := by
  intro allowed supported batch member
  simp only [witnesses, List.flatMap_append, List.mem_append] at member
  rcases member with member | member
  · exact leftSupported allowed
      (fun expression member => supported expression (by simp [member])) batch member
  · exact rightSupported allowed
      (fun expression member => supported expression (by simp [member])) batch member

theorem WitnessesFromConstraints.call (circuit : FormalCircuit) (name : String)
    (offset : Nat) (supported : WitnessesFromConstraints (Circuit.ops circuit.main offset)) :
    WitnessesFromConstraints [.subcircuit (circuit.asSubcircuit name offset)] := by
  simpa [WitnessesFromConstraints, flatConstraints, witnesses, Op.flatConstraints,
    Op.witnesses, FormalCircuit.asSubcircuit] using supported

end NightstreamFPrime.Circuit
