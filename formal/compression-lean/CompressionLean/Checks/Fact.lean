import CompressionLean.Spec.Checks

/-!
Owns the proof that every fact of a check without a failure holds
(`all_of_firstFailure`). The shape of a check (`Fact`, `firstFailure`) is in
`Spec/Checks.lean`.
-/

namespace CompressionLean.Checks

open CompressionLean.Spec

/-- Every fact of the list holds, as one conjunction in list order. -/
def Fact.All : List Fact → Prop
  | [] => True
  | f :: facts => f.holds ∧ Fact.All facts

theorem Fact.all_of_forall : ∀ {facts : List Fact}, (∀ f ∈ facts, f.holds) → Fact.All facts
  | [], _ => trivial
  | _ :: _, each => ⟨each _ List.mem_cons_self, Fact.all_of_forall fun f member => each f (List.mem_cons_of_mem _ member)⟩

/-- Every fact of a list without a failure holds. -/
theorem all_of_firstFailure {facts : List Fact} (pass : firstFailure facts = none) : Fact.All facts :=
  Fact.all_of_forall fun f member => by
    simpa using List.find?_eq_none.mp (Option.map_eq_none_iff.mp pass) f member

end CompressionLean.Checks
