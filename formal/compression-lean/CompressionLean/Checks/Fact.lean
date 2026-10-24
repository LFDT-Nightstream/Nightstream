/-!
Owns the shape of a check that an executable runs: a list of named decidable
facts (`Fact`), the name of the first that fails (`firstFailure`), and that
every fact of a list without a failure holds (`holds_of_firstFailure`).
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

/-- Every fact of a list without a failure holds. -/
theorem holds_of_firstFailure {facts : List Fact} (pass : firstFailure facts = none) : ∀ f ∈ facts, f.holds :=
  fun f member => by
    simpa using List.find?_eq_none.mp (Option.map_eq_none_iff.mp pass) f member

end CompressionLean.Checks
