import NightstreamFPrime.Circuit.VariableSupport

/-!
Owns the exact predicate used to extend caller-selected variable support by
one newly allocated half-open interval.
-/

namespace NightstreamFPrime.Circuit.SupportRange

/-- Existing support plus one local half-open interval. -/
def Extend (allowed : Nat → Prop) (start finish index : Nat) : Prop :=
  allowed index ∨ (start ≤ index ∧ index < finish)

theorem base {allowed : Nat → Prop} {start finish index : Nat}
    (support : allowed index) : Extend allowed start finish index :=
  Or.inl support

theorem interval {allowed : Nat → Prop} {start finish index : Nat}
    (lower : start ≤ index) (upper : index < finish) :
    Extend allowed start finish index :=
  Or.inr ⟨lower, upper⟩

theorem mono_finish {allowed : Nat → Prop} {start middle finish index : Nat}
    (support : Extend allowed start middle index) (middleLe : middle ≤ finish) :
    Extend allowed start finish index := by
  rcases support with support | ⟨lower, upper⟩
  · exact Or.inl support
  · exact Or.inr ⟨lower, Nat.lt_of_lt_of_le upper middleLe⟩

/-- Flatten two adjacent support extensions into their complete interval. -/
theorem flatten {allowed : Nat → Prop}
    {start middle finish index : Nat}
    (startLeMiddle : start ≤ middle) (middleLeFinish : middle ≤ finish)
    (support : Extend (Extend allowed start middle) middle finish index) :
    Extend allowed start finish index := by
  rcases support with (support | ⟨lower, upper⟩) | ⟨lower, upper⟩
  · exact Or.inl support
  · exact Or.inr ⟨lower, Nat.lt_of_lt_of_le upper middleLeFinish⟩
  · exact Or.inr ⟨Nat.le_trans startLeMiddle lower, upper⟩

end NightstreamFPrime.Circuit.SupportRange
