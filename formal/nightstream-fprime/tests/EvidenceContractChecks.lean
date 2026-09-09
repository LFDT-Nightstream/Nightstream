import tests.EvidenceMetadata

namespace LeanGraph.Checks

def Target : Prop := ∀ n : Nat, n = n
def Represents (n : Nat) : Prop := n = n
theorem conditional : ∀ n : Nat, Represents n → n = n := fun _ h => h
theorem direct : Target := fun _ => rfl
theorem assembled : Target := fun n => conditional n rfl

/-- error: closure theorem does not prove the exact registered target -/
#guard_msgs in
#evidence_closed Target by conditional

#evidence_closed Target by direct
#evidence_closed Target by assembled
#audit_axioms conditional
#audit_axioms direct
#audit_axioms assembled

def Meaning : Nat := 2
def MeaningTarget : Prop := Meaning = 2
def SameMeaningTarget : Prop := Meaning = 2
def OtherMeaning : Nat := 3
def ChangedMeaningTarget : Prop := OtherMeaning = 2

run_cmd do
  unless expression (Lean.mkForall `x .default (Lean.mkConst ``Nat) (Lean.mkBVar 0)) ==
      expression (Lean.mkForall `y .default (Lean.mkConst ``Nat) (Lean.mkBVar 0)) do
    throwError "bound variable names changed the meaning representation"

#evidence_export MeaningTarget
#evidence_export ChangedMeaningTarget
#evidence_export assembled

end LeanGraph.Checks
