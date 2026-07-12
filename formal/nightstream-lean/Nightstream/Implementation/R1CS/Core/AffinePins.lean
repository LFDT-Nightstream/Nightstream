import Nightstream.Implementation.R1CS.Core.LinearOutputs

/-!
Contract: universal semantics for mixed zero and equality pin row families.

Several production verifier phases consist only of `x = 0` and `x = y`
assertions. Generated artifacts store a compact run schedule and exact rows;
the theorems here derive the corresponding equalities in both directions.
-/

namespace Nightstream.Implementation.R1CS.AffinePins

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

inductive Pin where
  | zero (column : Nat)
  | constant (column value : Nat)
  | equal (left right : Nat)
deriving DecidableEq, Repr

def Pin.check : Pin → LinearOutputs.Check
  | .zero column => ⟨column, [], .forward⟩
  | .constant column value => ⟨column, [(0, value)], .forward⟩
  | .equal left right => ⟨left, [(right, 1)], .forward⟩

def Pin.row (pin : Pin) : Row := pin.check.row

def Pin.Holds (assignment : Nat → Nat) : Pin → Prop
  | .zero column => assignment column = 0
  | .constant column value => assignment column = value
  | .equal left right => assignment left = assignment right

def Pin.Canonical : Pin → Prop
  | .zero _ => True
  | .constant _ value => 0 < value ∧ value < goldilocksP
  | .equal _ _ => True

def PinsCanonical (pins : List Pin) : Prop :=
  ∀ pin ∈ pins, pin.Canonical

instance (pin : Pin) : Decidable pin.Canonical := by
  cases pin <;> simp [Pin.Canonical] <;> infer_instance

instance (pins : List Pin) : Decidable (PinsCanonical pins) := by
  unfold PinsCanonical
  infer_instance

inductive Run where
  | zero (start step count : Nat)
  | constant (columnStart columnStep valueStart valueStep count : Nat)
  | equal (leftStart rightStart leftStep rightStep count : Nat)
deriving DecidableEq, Repr

def Run.pins : Run → List Pin
  | .zero start step count =>
      (List.range count).map fun offset => .zero (start + offset * step)
  | .constant columnStart columnStep valueStart valueStep count =>
      (List.range count).map fun offset =>
        .constant (columnStart + offset * columnStep)
          (valueStart + offset * valueStep)
  | .equal leftStart rightStart leftStep rightStep count =>
      (List.range count).map fun offset =>
        .equal (leftStart + offset * leftStep)
          (rightStart + offset * rightStep)

def Run.count : Run → Nat
  | .zero _ _ count => count
  | .constant _ _ _ _ count => count
  | .equal _ _ _ _ count => count

def expandRuns (runs : List Run) : List Pin := runs.flatMap Run.pins

def rows (pins : List Pin) : List Row := pins.map Pin.row

theorem Run.pins_length (run : Run) : run.pins.length = run.count := by
  cases run <;> simp [Run.pins, Run.count]

theorem expandRuns_length (runs : List Run) :
    (expandRuns runs).length = (runs.map Run.count).sum := by
  induction runs with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [expandRuns, Run.pins_length]

private theorem canonicalChecks (pins : List Pin)
    (canonical : PinsCanonical pins) :
    LinearOutputs.Canonical (pins.map Pin.check) := by
  intro check member
  rcases List.mem_map.mp member with ⟨pin, pinMember, rfl⟩
  have pinCanonical := canonical pin pinMember
  cases pin <;> simp_all [Pin.check, Pin.Canonical,
    LinearOutputs.Check.Canonical, CanonicalTerms] <;> decide

theorem rows_eq_linearOutputs (pins : List Pin) :
    rows pins = LinearOutputs.rows (pins.map Pin.check) := by
  simp [rows, LinearOutputs.rows, List.map_map, Function.comp_def, Pin.row]

theorem rows_sound
    {pins : List Pin} {assignment : Nat → Nat}
    (pinsCanonical : PinsCanonical pins)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows pins) assignment) :
    ∀ pin ∈ pins, pin.Holds assignment := by
  have equalities := LinearOutputs.rows_sound canonical one
    (canonicalChecks pins pinsCanonical)
    (by simpa [rows_eq_linearOutputs] using satisfies)
  intro pin member
  have equal := equalities pin.check
    (List.mem_map.mpr ⟨pin, member, rfl⟩)
  cases pin with
  | zero column => simpa [Pin.check, Pin.Holds, LinearOutputs.Check.expected,
      lcEval] using equal
  | constant column value =>
      have valueCanonical := (pinsCanonical (.constant column value) member).2
      simpa [Pin.check, Pin.Holds, LinearOutputs.Check.expected, lcEval, one,
        Nat.mod_eq_of_lt valueCanonical] using equal
  | equal left right =>
      simpa [Pin.check, Pin.Holds, LinearOutputs.Check.expected, lcEval,
        Nat.mod_eq_of_lt (canonical right)] using equal

theorem rows_complete
    {pins : List Pin} {assignment : Nat → Nat}
    (pinsCanonical : PinsCanonical pins)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : ∀ pin ∈ pins, pin.Holds assignment) :
    Satisfies (rows pins) assignment := by
  rw [rows_eq_linearOutputs]
  apply LinearOutputs.rows_complete canonical one
    (canonicalChecks pins pinsCanonical)
  intro check member
  rcases List.mem_map.mp member with ⟨pin, pinMember, rfl⟩
  have pinHolds := holds pin pinMember
  cases pin with
  | zero column =>
      simpa [Pin.check, Pin.Holds, LinearOutputs.Check.expected, lcEval]
        using pinHolds
  | constant column value =>
      have valueCanonical := (pinsCanonical (.constant column value) pinMember).2
      simpa [Pin.check, Pin.Holds, LinearOutputs.Check.expected, lcEval, one,
        Nat.mod_eq_of_lt valueCanonical] using pinHolds
  | equal left right =>
      simpa [Pin.check, Pin.Holds, LinearOutputs.Check.expected, lcEval,
        Nat.mod_eq_of_lt (canonical right)] using pinHolds

end Nightstream.Implementation.R1CS.AffinePins
