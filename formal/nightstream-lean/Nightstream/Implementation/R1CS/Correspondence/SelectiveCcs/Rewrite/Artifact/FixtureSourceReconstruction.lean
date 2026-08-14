import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FixtureRefinement
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceRowSemantics
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.KaratsubaDotProduct

/-!
Contract: exact source-witness reconstruction for the deterministic
grouped-product rewrite fixture.

Assurance tier: artifact-checked same-assignment refinement.

Owns: the exact 33 generated source rows, their retained-column boundary,
the canonical 30 temporary values, and satisfaction of every source row from
the three direct product-sum equations.

Does not own: derivation of those three equations from the six final matrix
rows, low-norm validity, selector authority, production-family coverage, or
permission to remove a production row or coordinate.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureSourceReconstruction

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowSemantics
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.KaratsubaDotProduct

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_comm⟩

/-- Exact retained source columns used by each of the six products. -/
def lowLeftColumn : Fin 6 → Nat
  | ⟨0, _⟩ => 15
  | ⟨1, _⟩ => 5
  | ⟨2, _⟩ => 7
  | ⟨3, _⟩ => 9
  | ⟨4, _⟩ => 11
  | ⟨5, _⟩ => 13

def highLeftColumn : Fin 6 → Nat
  | ⟨0, _⟩ => 4
  | ⟨1, _⟩ => 6
  | ⟨2, _⟩ => 8
  | ⟨3, _⟩ => 10
  | ⟨4, _⟩ => 12
  | ⟨5, _⟩ => 14

def lowRightColumn : Fin 6 → Nat
  | ⟨0, _⟩ => 16
  | ⟨1, _⟩ => 18
  | ⟨2, _⟩ => 20
  | ⟨3, _⟩ => 22
  | ⟨4, _⟩ => 24
  | ⟨5, _⟩ => 26

def highRightColumn : Fin 6 → Nat
  | ⟨0, _⟩ => 17
  | ⟨1, _⟩ => 19
  | ⟨2, _⟩ => 21
  | ⟨3, _⟩ => 23
  | ⟨4, _⟩ => 25
  | ⟨5, _⟩ => 27

/-- Read the semantic boundary from retained source values. -/
def boundary (retained : Nat → F) : Boundary where
  lowLeft := fun index => retained (lowLeftColumn index)
  highLeft := fun index => retained (highLeftColumn index)
  lowRight := fun index => retained (lowRightColumn index)
  highRight := fun index => retained (highRightColumn index)
  highHighSum := retained 58
  lowOutput := retained 59
  highOutput := retained 60

/-- Insert the canonical thirty temporary values into the retained source
assignment. Every other source column keeps its retained value. -/
def reconstructedAssignment (retained : Nat → F) : Nat → F
  | 28 => (materialize (boundary retained)).lowLow 0
  | 29 => (materialize (boundary retained)).highHigh 0
  | 30 => (materialize (boundary retained)).cross 0
  | 31 => (materialize (boundary retained)).lowTerm 0
  | 32 => (materialize (boundary retained)).highTerm 0
  | 33 => (materialize (boundary retained)).lowLow 1
  | 34 => (materialize (boundary retained)).highHigh 1
  | 35 => (materialize (boundary retained)).cross 1
  | 36 => (materialize (boundary retained)).lowTerm 1
  | 37 => (materialize (boundary retained)).highTerm 1
  | 38 => (materialize (boundary retained)).lowLow 2
  | 39 => (materialize (boundary retained)).highHigh 2
  | 40 => (materialize (boundary retained)).cross 2
  | 41 => (materialize (boundary retained)).lowTerm 2
  | 42 => (materialize (boundary retained)).highTerm 2
  | 43 => (materialize (boundary retained)).lowLow 3
  | 44 => (materialize (boundary retained)).highHigh 3
  | 45 => (materialize (boundary retained)).cross 3
  | 46 => (materialize (boundary retained)).lowTerm 3
  | 47 => (materialize (boundary retained)).highTerm 3
  | 48 => (materialize (boundary retained)).lowLow 4
  | 49 => (materialize (boundary retained)).highHigh 4
  | 50 => (materialize (boundary retained)).cross 4
  | 51 => (materialize (boundary retained)).lowTerm 4
  | 52 => (materialize (boundary retained)).highTerm 4
  | 53 => (materialize (boundary retained)).lowLow 5
  | 54 => (materialize (boundary retained)).highHigh 5
  | 55 => (materialize (boundary retained)).cross 5
  | 56 => (materialize (boundary retained)).lowTerm 5
  | 57 => (materialize (boundary retained)).highTerm 5
  | column => retained column

theorem generated_source_rows_length : rawSourceRows.length = 33 := by
  decide

private theorem modulusMinusOne :
    residue 18446744069414584320 = (-1 : F) := by
  decide

private theorem modulusMinusSeven :
    residue 18446744069414584314 = (-7 : F) := by
  decide

private theorem lowTermIdentity (lowLow highHigh : F) :
    -lowLow + (-(7 * highHigh) + (lowLow + 7 * highHigh)) = 0 := by
  have lowCancel : -lowLow + lowLow = 0 :=
    Lean.Grind.Fin.neg_add_cancel lowLow
  have highCancel : -(7 * highHigh) + 7 * highHigh = 0 :=
    Lean.Grind.Fin.neg_add_cancel (7 * highHigh)
  calc
    -lowLow + (-(7 * highHigh) + (lowLow + 7 * highHigh)) =
        (-lowLow + lowLow) + (-(7 * highHigh) + 7 * highHigh) := by
      ac_rfl
    _ = 0 := by rw [lowCancel, highCancel, Fin.zero_add]

private theorem highTermIdentity (lowLow highHigh cross : F) :
    lowLow + (highHigh + (-cross +
        (cross + -lowLow + -highHigh))) = 0 := by
  have lowCancel : -lowLow + lowLow = 0 :=
    Lean.Grind.Fin.neg_add_cancel lowLow
  have highCancel : -highHigh + highHigh = 0 :=
    Lean.Grind.Fin.neg_add_cancel highHigh
  have crossCancel : -cross + cross = 0 :=
    Lean.Grind.Fin.neg_add_cancel cross
  calc
    lowLow + (highHigh + (-cross + (cross + -lowLow + -highHigh))) =
        (-lowLow + lowLow) + (-highHigh + highHigh) +
          (-cross + cross) := by
      ac_rfl
    _ = 0 := by
      rw [lowCancel, highCancel, crossCancel, Fin.zero_add, Fin.add_zero]

private theorem negatedSixIdentity (value : Fin 6 → F) :
    -value 0 + (-value 1 + (-value 2 + (-value 3 +
        (-value 4 + (-value 5 + sumSix value))))) = 0 := by
  have cancel0 : -value 0 + value 0 = 0 :=
    Lean.Grind.Fin.neg_add_cancel (value 0)
  have cancel1 : -value 1 + value 1 = 0 :=
    Lean.Grind.Fin.neg_add_cancel (value 1)
  have cancel2 : -value 2 + value 2 = 0 :=
    Lean.Grind.Fin.neg_add_cancel (value 2)
  have cancel3 : -value 3 + value 3 = 0 :=
    Lean.Grind.Fin.neg_add_cancel (value 3)
  have cancel4 : -value 4 + value 4 = 0 :=
    Lean.Grind.Fin.neg_add_cancel (value 4)
  have cancel5 : -value 5 + value 5 = 0 :=
    Lean.Grind.Fin.neg_add_cancel (value 5)
  simp only [sumSix]
  calc
    -value 0 + (-value 1 + (-value 2 + (-value 3 +
        (-value 4 + (-value 5 +
          (value 0 + value 1 + value 2 + value 3 + value 4 + value 5)))))) =
        (-value 0 + value 0) + (-value 1 + value 1) +
          (-value 2 + value 2) + (-value 3 + value 3) +
          (-value 4 + value 4) + (-value 5 + value 5) := by
      ac_rfl
    _ = 0 := by
      rw [cancel0, cancel1, cancel2, cancel3, cancel4, cancel5]
      simp only [Fin.zero_add, Fin.add_zero]

private theorem crossInputOrder
    (lowLeft highLeft lowRight highRight : F) :
    (highLeft + lowLeft) * (lowRight + highRight) =
      (lowLeft + highLeft) * (lowRight + highRight) := by
  rw [Lean.Grind.Fin.add_comm highLeft lowLeft]

/-- Every retained boundary that satisfies the compact semantic relation has
a concrete assignment satisfying all 33 exact generated source rows. -/
theorem generated_source_rows_hold
    (retained : Nat → F) (reduced : ReducedHolds (boundary retained)) :
    ∀ row ∈ rawSourceRows,
      Holds 1 (reconstructedAssignment retained) row := by
  have source := sourceHolds_materialize_of_reduced (boundary retained) reduced
  rcases source with ⟨_, highHighSum, lowOutput, highOutput⟩
  simp only [boundary, materialize, lowLowValue, highHighValue, crossValue,
    lowLeftColumn, highLeftColumn, lowRightColumn, highRightColumn] at highHighSum lowOutput highOutput
  intro row member
  simp only [rawSourceRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl
  all_goals
    simp only [Holds, evalLinearCombination, evalTerms,
      reconstructedAssignment, materialize, boundary,
      lowLowValue, highHighValue, crossValue,
      lowLeftColumn, highLeftColumn, lowRightColumn, highRightColumn,
      residue_zero, residue_one, Fin.zero_mul, Fin.one_mul, Fin.mul_one,
      Fin.zero_add, Fin.add_zero,
      modulusMinusOne, modulusMinusSeven, Lean.Grind.Fin.neg_mul,
      Fin.sub_eq_add_neg]
  all_goals try rfl
  · exact crossInputOrder _ _ _ _
  · exact lowTermIdentity _ _
  · exact highTermIdentity _ _ _
  · exact lowTermIdentity _ _
  · exact highTermIdentity _ _ _
  · exact lowTermIdentity _ _
  · exact highTermIdentity _ _ _
  · exact lowTermIdentity _ _
  · exact highTermIdentity _ _ _
  · exact lowTermIdentity _ _
  · exact highTermIdentity _ _ _
  · exact lowTermIdentity _ _
  · exact highTermIdentity _ _ _
  · rw [highHighSum]
    simpa only [boundary, highHighValue, highLeftColumn, highRightColumn]
      using negatedSixIdentity (highHighValue (boundary retained))
  · rw [lowOutput]
    simpa only [boundary, materialize, lowLowValue, highHighValue,
      lowLeftColumn, highLeftColumn, lowRightColumn, highRightColumn]
      using negatedSixIdentity (materialize (boundary retained)).lowTerm
  · rw [highOutput]
    simpa only [boundary, materialize, lowLowValue, highHighValue, crossValue,
      lowLeftColumn, highLeftColumn, lowRightColumn, highRightColumn,
      Fin.sub_eq_add_neg]
      using negatedSixIdentity (materialize (boundary retained)).highTerm

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureSourceReconstruction
