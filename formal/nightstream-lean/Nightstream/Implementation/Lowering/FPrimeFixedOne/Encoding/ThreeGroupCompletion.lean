import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CrossArmSeparation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.VisibleCompletion

/-!
Contract: generic completion of one always-active occurrence group, one
selected group, and one inactive sibling group.

Owns: preservation and row-satisfaction composition for three already
separated `ArmPlan`s.

Does not own: Step/Terminal occurrence lists, semantic witnesses, branch or
join rows, production artifacts, or concrete codecs.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CompletionSeparation

/-- Complete the selected group, its inactive sibling, and the always-active
group while preserving every visible coordinate and every earlier row. -/
theorem completeThreeGroups
    {parameters : Parameters}
    {profile : Profile parameters}
    {firstActive secondActive : ColumnId}
    (always :
      ArmPlan (signature parameters) (profile.family parameters)
        oneColumn oneColumn)
    (first :
      ArmPlan (signature parameters) (profile.family parameters)
        oneColumn firstActive)
    (second :
      ArmPlan (signature parameters) (profile.family parameters)
        oneColumn secondActive)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment oneColumn = 1)
    (firstActiveOne : assignment firstActive = 1)
    (secondActiveZero : assignment secondActive = 0)
    (alwaysHonest : always.HonestActive assignment)
    (firstHonest : first.HonestActive assignment)
    (secondHonest : second.HonestInactive assignment)
    (alwaysSeparated : always.Separated)
    (firstSeparated : first.Separated)
    (secondSeparated : second.Separated)
    (firstSecond : ArmPlan.PlansSeparated first second)
    (alwaysFirst : ArmPlan.PlansSeparated always first)
    (firstAlways : ArmPlan.PlansSeparated first always)
    (alwaysSecond : ArmPlan.PlansSeparated always second)
    (secondAlways : ArmPlan.PlansSeparated second always)
    (oneVisible : oneColumn ∈ always.visibleIds) :
    ∃ completed : ColumnId -> Field,
      AgreesOn
          (always.visibleIds ++
            (first.visibleIds ++ second.visibleIds))
          assignment completed ∧
        Satisfies always.rows completed ∧
        Satisfies first.rows completed ∧
        Satisfies second.rows completed := by
  rcases first.completeActiveThenInactive second laws assignment
      constantOne firstActiveOne secondActiveZero firstHonest secondHonest
      firstSeparated secondSeparated firstSecond with
    ⟨middle, branchAgrees, branchChanges, branchRows⟩
  have branchTempsAlways :
      IdsDisjoint
        (first.temporaryIds ++ second.temporaryIds)
        always.visibleIds := by
    intro id member visibleMember
    rcases List.mem_append.mp member with firstMember | secondMember
    · exact firstAlways.firstTempsSecondVisible
        id firstMember visibleMember
    · exact secondAlways.firstTempsSecondVisible
        id secondMember visibleMember
  have middleAlwaysAgrees :
      AgreesOn always.visibleIds assignment middle :=
    agreesOn_of_changesOnly branchTempsAlways branchChanges
  have middleOne : middle oneColumn = 1 := by
    rw [middleAlwaysAgrees oneColumn oneVisible, constantOne]
  have middleAlwaysHonest : always.HonestActive middle :=
    ArmPlan.honestActiveOccurrences_of_agrees always.occurrences
      assignment middle middleAlwaysAgrees alwaysHonest
  rcases always.completeActive laws middle middleOne middleOne
      middleAlwaysHonest alwaysSeparated with
    ⟨completed, alwaysAgrees, alwaysChanges, alwaysRows⟩
  have completedFirstProtected :
      AgreesOn (first.visibleIds ++ first.temporaryIds)
        middle completed := by
    apply agreesOn_of_changesOnly
    · intro id temporaryMember protectedMember
      rcases List.mem_append.mp protectedMember with
        visibleMember | firstTemporary
      · exact alwaysFirst.firstTempsSecondVisible
          id temporaryMember visibleMember
      · exact alwaysFirst.firstTempsSecondTemps
          id temporaryMember firstTemporary
    · exact alwaysChanges
  have completedSecondProtected :
      AgreesOn (second.visibleIds ++ second.temporaryIds)
        middle completed := by
    apply agreesOn_of_changesOnly
    · intro id temporaryMember protectedMember
      rcases List.mem_append.mp protectedMember with
        visibleMember | secondTemporary
      · exact alwaysSecond.firstTempsSecondVisible
          id temporaryMember visibleMember
      · exact alwaysSecond.firstTempsSecondTemps
          id temporaryMember secondTemporary
    · exact alwaysChanges
  have splitBranch :=
    (satisfies_append_iff first.rows second.rows middle).1 branchRows
  have completedFirstRows : Satisfies first.rows completed :=
    satisfies_of_agrees first.rows middle completed
      (agreesOn_of_subset (first.rows_support firstSeparated)
        completedFirstProtected) splitBranch.1
  have completedSecondRows : Satisfies second.rows completed :=
    satisfies_of_agrees second.rows middle completed
      (agreesOn_of_subset (second.rows_support secondSeparated)
        completedSecondProtected) splitBranch.2
  have assignmentCompletedAlways :
      AgreesOn always.visibleIds assignment completed :=
    agreesOn_trans middleAlwaysAgrees alwaysAgrees
  have assignmentCompletedFirst :
      AgreesOn first.visibleIds assignment completed := by
    exact agreesOn_trans
      (agreesOn_of_subset
        (fun id member => List.mem_append_left second.visibleIds member)
        branchAgrees)
      (agreesOn_of_subset
        (fun id member =>
          List.mem_append_left first.temporaryIds member)
        completedFirstProtected)
  have assignmentCompletedSecond :
      AgreesOn second.visibleIds assignment completed := by
    exact agreesOn_trans
      (agreesOn_of_subset
        (fun id member => List.mem_append_right first.visibleIds member)
        branchAgrees)
      (agreesOn_of_subset
        (fun id member =>
          List.mem_append_left second.temporaryIds member)
        completedSecondProtected)
  refine ⟨completed, ?_, alwaysRows,
    completedFirstRows, completedSecondRows⟩
  intro id member
  rcases List.mem_append.mp member with alwaysMember | branchMember
  · exact assignmentCompletedAlways id alwaysMember
  · rcases List.mem_append.mp branchMember with
      firstMember | secondMember
    · exact assignmentCompletedFirst id firstMember
    · exact assignmentCompletedSecond id secondMember

end CompletionSeparation

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
