/-!
Owns the shape of a check that an executable runs: a list of named decidable
facts (`Fact`), the name of the first that fails (`firstFailure`), and that
every fact of a list without a failure holds (`all_of_firstFailure`).
-/

namespace CompressionLean.Checks

/-- A fact of a check: its name and a decidable proposition. -/
structure Fact where
  name : String
  holds : Prop
  [decidable : Decidable holds]

attribute [instance] Fact.decidable

/-- The name of the first fact that fails. -/
def firstFailure (facts : List Fact) : Option String :=
  (facts.find? fun f => !decide f.holds).map (·.name)

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
