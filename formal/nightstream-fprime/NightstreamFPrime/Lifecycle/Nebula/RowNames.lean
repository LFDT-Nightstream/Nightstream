import NightstreamFPrime.Lifecycle.Nebula.MemoryCircuit

/-! Owns the spec names of the memory application's assertion constraints, as
(name, count) pairs in constraint order. Conformance tests use them to name the
check that a rejected step fails. Each pair comes from the `Checks` group that
also defines the rows, so a name labels its own rows. They carry no protocol
authority. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec.Nebula

/-- The names of `checks` with their row counts. -/
def checkNames (checks : Checks) : List (String × ℕ) :=
  checks.map fun check => (check.1, check.2.length)

theorem checkNames_count (checks : Checks) :
    ((checkNames checks).map Prod.snd).sum = (joinChecks checks).length := by
  simp [checkNames, joinChecks_length, Function.comp_def]

theorem checkNames_flatMap_count {α : Type} (l : List α) (f : α → Checks) :
    ((l.flatMap fun a => checkNames (f a)).map Prod.snd).sum =
      (l.flatMap fun a => joinChecks (f a)).length := by
  induction l with
  | nil => rfl
  | cons a rest ih =>
    simp only [List.flatMap_cons, List.map_append, List.sum_append, List.length_append,
      checkNames_count, ih]

namespace Rows

variable (p : Plan)

/-- The names of `polyRows`, in row order. -/
def names (app : PortApp p) : List (String × ℕ) :=
  checkNames (bitChecks p) ++ checkNames (armChecks p) ++
    (List.finRange p.bOps).flatMap (fun j => checkNames (slotChecks p j)) ++
    (List.finRange p.bScan).flatMap (fun j => checkNames (scanChecks p j)) ++
    checkNames (closeChecks p) ++ [("application", app.rows.length)]

end Rows

/-- The names of `MemoryApp.assertions`, in constraint order. -/
def MemoryApp.assertionNames (p : Plan) (app : PortApp p) (i : MemoryApp.AppInterface p app)
    (offset : ℕ) : List (String × ℕ) :=
  checkNames (MemoryApp.hashChecks p app i offset) ++ Rows.names p app

end NightstreamFPrime.Lifecycle.Nebula

namespace NightstreamFPrime.Lifecycle.Nebula.Rows

open NightstreamFPrime.Spec.Nebula

/-- The names cover the polynomial rows exactly. -/
theorem names_count (p : Plan) (app : PortApp p) :
    ((names p app).map Prod.snd).sum = (polyRows p app).length := by
  simp only [names, polyRows, memoryRows, appRows, List.map_append, List.sum_append,
    List.length_append, checkNames_count, checkNames_flatMap_count, List.length_map, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  rfl

end NightstreamFPrime.Lifecycle.Nebula.Rows

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec.Nebula

/-- The names cover the assertions exactly. -/
theorem assertionNames_count (p : Plan) (app : PortApp p) (i : AppInterface p app) (offset : ℕ) :
    ((assertionNames p app i offset).map Prod.snd).sum = (assertions p app i offset).length := by
  simp only [assertionNames, assertions, List.map_append, List.sum_append, List.length_append,
    checkNames_count, Rows.names_count, List.length_map]

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
