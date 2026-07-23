import Nightstream.Implementation.Lowering.Goldilocks.SelectedBranchSupport

/-!
Contract: physical composition for one top-level typed branch with a
base/recursive pair of sequential primitive receipts.
Owns:
- an ordered occurrence list whose constructors are exactly certified calls,
  Boolean assertions, and literal pins;
- structural semantic facts generated from those occurrences;
- sequential active/inactive soundness and honest completion under explicit
  column-support separation;
- one concrete branch receipt: activation rows, both arm receipts, and the
  coordinate-wise mux receipt.
Does not own: arbitrary block recursion, a caller-supplied acceptance
proposition, a whole-arm opaque call, generated artifacts, Rust behavior, or
an unclassified glue-row escape.
Emits constraints: exactly the concatenation returned by `SelectedBranch.rows`.
Allocations are exactly the concatenation returned by
`SelectedBranch.allocations`.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

/-! ## Structural arm occurrences -/

/-- One exact non-control primitive occurrence inside an arm. -/
inductive ArmOccurrence
    (signature : Signature.{u})
    (family : Family signature.types)
    (one active : ColumnId) : Type (u + 2) where
  | call
      {context : Schema signature.types}
      (call : signature.Call)
      {references :
        Refs signature.types context (signature.callInputs call)}
      (recipe : CallRecipe signature family call)
      (frame : CallFrame family call references)
      (oneExact : frame.one = one)
      (activeExact : frame.active = active) :
      ArmOccurrence signature family one active
  | assertion
      (recipe : BoolAssertRecipe)
      (oneExact : recipe.one = one)
      (activeExact : recipe.active = active) :
      ArmOccurrence signature family one active
  | literal
      {α : Type u}
      (codec : Codec α)
      {layout : Layout}
      (recipe : LiteralPinRecipe codec layout)
      (oneExact : recipe.one = one) :
      ArmOccurrence signature family one active

namespace ArmOccurrence

def rows
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    ArmOccurrence signature family one active -> List OwnedRow
  | .call _ recipe frame _ _ => recipe.rows frame
  | .assertion recipe _ _ => recipe.rows
  | .literal _ recipe _ => recipe.rows

def allocations
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    ArmOccurrence signature family one active -> List OwnedColumn
  | .call _ _ frame _ _ => frame.allocations
  | .assertion _ _ _ => []
  | .literal _ recipe _ => recipe.output.columns

def visibleIds
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    ArmOccurrence signature family one active -> List ColumnId
  | .call _ _ frame _ _ => frame.visibleIds
  | .assertion recipe _ _ =>
      [recipe.one, recipe.active, recipe.condition]
  | .literal _ recipe _ =>
      [one, active] ++ recipe.output.ids

def temporaryIds
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    ArmOccurrence signature family one active -> List ColumnId
  | .call _ _ frame _ _ => frame.temporaries.ids
  | .assertion _ _ _ => []
  | .literal _ _ _ => []

/-- A static call plan is ready only when its physical operands decode. -/
inductive Ready
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    ArmOccurrence signature family one active ->
      (ColumnId -> Field) -> Prop where
  | call
      {context : Schema signature.types}
      {call : signature.Call}
      {references :
        Refs signature.types context (signature.callInputs call)}
      {recipe : CallRecipe signature family call}
      {frame : CallFrame family call references}
      {oneExact : frame.one = one}
      {activeExact : frame.active = active}
      {assignment : ColumnId -> Field}
      (inputs :
        HVec signature.types.Value (signature.callInputs call))
      (decoded : frame.operands.Decodes family assignment inputs) :
      Ready (.call call recipe frame oneExact activeExact) assignment
  | assertion
      {recipe : BoolAssertRecipe}
      {oneExact : recipe.one = one}
      {activeExact : recipe.active = active}
      {assignment : ColumnId -> Field} :
      Ready (.assertion recipe oneExact activeExact) assignment
  | literal
      {α : Type u}
      {layout : Layout}
      {codec : Codec α}
      {recipe : LiteralPinRecipe codec layout}
      {oneExact : recipe.one = one}
      {assignment : ColumnId -> Field}
      (admissible : codec.Admissible recipe.value) :
      Ready (.literal (α := α) codec (layout := layout) recipe oneExact)
        assignment

/-- The exact semantic consequence of an active occurrence.  Call facts
quantify over decoded inputs; runtime values are not stored in the plan. -/
inductive Fact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    ArmOccurrence signature family one active ->
      (ColumnId -> Field) -> Prop where
  | call
      {context : Schema signature.types}
      {call : signature.Call}
      {references :
        Refs signature.types context (signature.callInputs call)}
      {recipe : CallRecipe signature family call}
      {frame : CallFrame family call references}
      {oneExact : frame.one = one}
      {activeExact : frame.active = active}
      {assignment : ColumnId -> Field}
      (evaluates :
        ∀ inputs :
            HVec signature.types.Value (signature.callInputs call),
          frame.operands.Decodes family assignment inputs ->
            ∃ outputs :
                Schema.Values signature.types
                  (signature.callOutputs call),
              signature.callEval call inputs = some outputs ∧
                frame.outputs.Decodes family assignment outputs) :
      Fact (.call call recipe frame oneExact activeExact) assignment
  | assertion
      {recipe : BoolAssertRecipe}
      {oneExact : recipe.one = one}
      {activeExact : recipe.active = active}
      {assignment : ColumnId -> Field}
      (decoded :
        boolCodec.decode [assignment recipe.condition] = some true) :
      Fact (.assertion recipe oneExact activeExact) assignment
  | literal
      {α : Type u}
      {layout : Layout}
      {codec : Codec α}
      {recipe : LiteralPinRecipe codec layout}
      {oneExact : recipe.one = one}
      {assignment : ColumnId -> Field}
      (decoded :
        codec.decode (recipe.output.values assignment) =
          some recipe.value) :
      Fact (.literal (α := α) codec (layout := layout) recipe oneExact)
        assignment

/-- Honest semantic data used to construct active rows. -/
inductive HonestActive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    ArmOccurrence signature family one active ->
      (ColumnId -> Field) -> Prop where
  | call
      {context : Schema signature.types}
      {call : signature.Call}
      {references :
        Refs signature.types context (signature.callInputs call)}
      {recipe : CallRecipe signature family call}
      {frame : CallFrame family call references}
      {oneExact : frame.one = one}
      {activeExact : frame.active = active}
      {assignment : ColumnId -> Field}
      (inputs :
        HVec signature.types.Value (signature.callInputs call))
      (outputs :
        Schema.Values signature.types (signature.callOutputs call))
      (inputsEncoded :
        frame.operands.Encodes family assignment inputs)
      (outputsEncoded :
        frame.outputs.Encodes family assignment outputs)
      (evaluated : signature.callEval call inputs = some outputs) :
      HonestActive (.call call recipe frame oneExact activeExact) assignment
  | assertion
      {recipe : BoolAssertRecipe}
      {oneExact : recipe.one = one}
      {activeExact : recipe.active = active}
      {assignment : ColumnId -> Field}
      (decoded :
        boolCodec.decode [assignment recipe.condition] = some true) :
      HonestActive (.assertion recipe oneExact activeExact) assignment
  | literal
      {α : Type u}
      {layout : Layout}
      {codec : Codec α}
      {recipe : LiteralPinRecipe codec layout}
      {oneExact : recipe.one = one}
      {assignment : ColumnId -> Field}
      (decoded :
        codec.decode (recipe.output.values assignment) =
          some recipe.value) :
      HonestActive
        (.literal (α := α) codec (layout := layout) recipe oneExact)
        assignment

inductive HonestInactive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    ArmOccurrence signature family one active ->
      (ColumnId -> Field) -> Prop where
  | call
      {context : Schema signature.types}
      {call : signature.Call}
      {references :
        Refs signature.types context (signature.callInputs call)}
      {recipe : CallRecipe signature family call}
      {frame : CallFrame family call references}
      {oneExact : frame.one = one}
      {activeExact : frame.active = active}
      {assignment : ColumnId -> Field} :
      HonestInactive (.call call recipe frame oneExact activeExact) assignment
  | assertion
      {recipe : BoolAssertRecipe}
      {oneExact : recipe.one = one}
      {activeExact : recipe.active = active}
      {assignment : ColumnId -> Field} :
      HonestInactive (.assertion recipe oneExact activeExact) assignment
  | literal
      {α : Type u}
      {layout : Layout}
      {codec : Codec α}
      {recipe : LiteralPinRecipe codec layout}
      {oneExact : recipe.one = one}
      {assignment : ColumnId -> Field}
      (decoded :
        codec.decode (recipe.output.values assignment) =
          some recipe.value) :
      HonestInactive
        (.literal (α := α) codec (layout := layout) recipe oneExact)
        assignment

theorem sound
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrence : ArmOccurrence signature family one active)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1)
    (ready : occurrence.Ready assignment)
    (holds : Satisfies occurrence.rows assignment) :
    occurrence.Fact assignment := by
  cases occurrence with
  | call call recipe frame oneExact activeExact =>
      have frameOne : assignment frame.one = 1 := by
        rw [oneExact]
        exact constantOne
      have frameActive : assignment frame.active = 1 := by
        rw [activeExact]
        exact activeOne
      apply Fact.call
      intro inputs decoded
      exact recipe.activeSoundness frame assignment inputs
        frameOne frameActive decoded holds
  | assertion recipe oneExact activeExact =>
      have recipeOne : assignment recipe.one = 1 := by
        rw [oneExact]
        exact constantOne
      have recipeActive : assignment recipe.active = 1 := by
        rw [activeExact]
        exact activeOne
      exact Fact.assertion
        ((recipe.active_iff_decode_true
          laws assignment recipeOne recipeActive).mp holds)
  | literal codec recipe oneExact =>
      have recipeOne : assignment recipe.one = 1 := by
        rw [oneExact]
        exact constantOne
      cases ready with
      | literal admissible =>
          exact Fact.literal
            (recipe.decode_of_satisfies
              assignment recipeOne admissible holds)

theorem completeActive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrence : ArmOccurrence signature family one active)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1)
    (honest : occurrence.HonestActive assignment) :
    ∃ completed : ColumnId -> Field,
      AgreesOn occurrence.visibleIds assignment completed ∧
        ChangesOnly occurrence.temporaryIds assignment completed ∧
        Satisfies occurrence.rows completed := by
  cases occurrence with
  | call call recipe frame oneExact activeExact =>
      have frameOne : assignment frame.one = 1 := by
        rw [oneExact]
        exact constantOne
      have frameActive : assignment frame.active = 1 := by
        rw [activeExact]
        exact activeOne
      cases honest with
      | call inputs outputs inputsEncoded outputsEncoded evaluated =>
          exact recipe.activeHonestCompleteness frame assignment
            inputs outputs frameOne frameActive inputsEncoded
            outputsEncoded evaluated
  | assertion recipe oneExact activeExact =>
      have recipeOne : assignment recipe.one = 1 := by
        rw [oneExact]
        exact constantOne
      have recipeActive : assignment recipe.active = 1 := by
        rw [activeExact]
        exact activeOne
      cases honest with
      | assertion decoded =>
          refine ⟨assignment, agreesOn_refl _ _, ?_, ?_⟩
          · intro _ _
            rfl
          · exact (recipe.active_iff_decode_true
              laws assignment recipeOne recipeActive).mpr decoded
  | literal codec recipe oneExact =>
      have recipeOne : assignment recipe.one = 1 := by
        rw [oneExact]
        exact constantOne
      cases honest with
      | literal decoded =>
          refine ⟨assignment, agreesOn_refl _ _, ?_, ?_⟩
          · intro _ _
            rfl
          · exact recipe.satisfies_of_decode assignment recipeOne decoded

theorem completeInactive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrence : ArmOccurrence signature family one active)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeZero : assignment active = 0)
    (honest : occurrence.HonestInactive assignment) :
    ∃ completed : ColumnId -> Field,
      AgreesOn occurrence.visibleIds assignment completed ∧
        ChangesOnly occurrence.temporaryIds assignment completed ∧
        Satisfies occurrence.rows completed := by
  cases occurrence with
  | call call recipe frame oneExact activeExact =>
      have frameOne : assignment frame.one = 1 := by
        rw [oneExact]
        exact constantOne
      have frameActive : assignment frame.active = 0 := by
        rw [activeExact]
        exact activeZero
      cases honest
      exact recipe.inactiveSatisfiable frame assignment frameOne frameActive
  | assertion recipe oneExact activeExact =>
      have recipeActive : assignment recipe.active = 0 := by
        rw [activeExact]
        exact activeZero
      cases honest
      refine ⟨assignment, agreesOn_refl _ _, ?_, ?_⟩
      · intro _ _
        rfl
      · exact recipe.inactive_complete assignment recipeActive
  | literal codec recipe oneExact =>
      have recipeOne : assignment recipe.one = 1 := by
        rw [oneExact]
        exact constantOne
      cases honest with
      | literal decoded =>
          refine ⟨assignment, agreesOn_refl _ _, ?_, ?_⟩
          · intro _ _
            rfl
          · exact recipe.satisfies_of_decode assignment recipeOne decoded

theorem honestActive_of_agrees
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrence : ArmOccurrence signature family one active)
    (before after : ColumnId -> Field)
    (agrees : AgreesOn occurrence.visibleIds before after)
    (honest : occurrence.HonestActive before) :
    occurrence.HonestActive after := by
  cases occurrence with
  | call call recipe frame oneExact activeExact =>
      cases honest with
      | call inputs outputs inputsEncoded outputsEncoded evaluated =>
          have operandAgrees :
              AgreesOn frame.operands.ids before after := by
            apply agreesOn_of_subset _ agrees
            intro id member
            have contextMember : id ∈ frame.contextBundles.ids := by
              simpa [CallFrame.operands] using
                RefBundles.fromSchema_ids_subset _
                  frame.contextBundles id member
            simp [visibleIds, CallFrame.visibleIds, contextMember]
          have outputAgrees :
              AgreesOn frame.outputs.ids before after := by
            apply agreesOn_of_subset _ agrees
            intro id member
            simp [visibleIds, CallFrame.visibleIds, member]
          exact HonestActive.call inputs outputs
            (frame.operands.encodes_of_agrees family before after inputs
              operandAgrees inputsEncoded)
            (frame.outputs.encodes_of_agrees family before after outputs
              outputAgrees outputsEncoded)
            evaluated
  | assertion recipe oneExact activeExact =>
      cases honest with
      | assertion decoded =>
          have conditionAgrees :
              after recipe.condition = before recipe.condition :=
            agrees recipe.condition (by simp [visibleIds])
          exact HonestActive.assertion
            (by simpa [conditionAgrees] using decoded)
  | literal codec recipe oneExact =>
      cases honest with
      | literal decoded =>
          have outputAgrees :
              AgreesOn recipe.output.ids before after := by
            apply agreesOn_of_subset _ agrees
            intro id member
            simp [visibleIds, member]
          exact HonestActive.literal (by
            rw [recipe.output.values_eq_of_agrees before after outputAgrees]
            exact decoded)

theorem honestInactive_of_agrees
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrence : ArmOccurrence signature family one active)
    (before after : ColumnId -> Field)
    (agrees : AgreesOn occurrence.visibleIds before after)
    (honest : occurrence.HonestInactive before) :
    occurrence.HonestInactive after := by
  cases occurrence with
  | call _ _ _ _ _ =>
      cases honest
      exact HonestInactive.call
  | assertion _ _ _ =>
      cases honest
      exact HonestInactive.assertion
  | literal codec recipe oneExact =>
      cases honest with
      | literal decoded =>
          have outputAgrees :
              AgreesOn recipe.output.ids before after := by
            apply agreesOn_of_subset _ agrees
            intro id member
            simp [visibleIds, member]
          exact HonestInactive.literal (by
            rw [recipe.output.values_eq_of_agrees before after outputAgrees]
            exact decoded)

end ArmOccurrence

structure ArmPlan
    (signature : Signature.{u})
    (family : Family signature.types)
    (one active : ColumnId) where
  occurrences : List (ArmOccurrence signature family one active)

namespace ArmPlan

def ReadyOccurrences
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (assignment : ColumnId -> Field) :
    List (ArmOccurrence signature family one active) -> Prop
  | [] => True
  | head :: tail =>
      head.Ready assignment ∧ ReadyOccurrences assignment tail

def FactsOccurrences
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (assignment : ColumnId -> Field) :
    List (ArmOccurrence signature family one active) -> Prop
  | [] => True
  | head :: tail =>
      head.Fact assignment ∧ FactsOccurrences assignment tail

def HonestActiveOccurrences
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (assignment : ColumnId -> Field) :
    List (ArmOccurrence signature family one active) -> Prop
  | [] => True
  | head :: tail =>
      head.HonestActive assignment ∧
        HonestActiveOccurrences assignment tail

def HonestInactiveOccurrences
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (assignment : ColumnId -> Field) :
    List (ArmOccurrence signature family one active) -> Prop
  | [] => True
  | head :: tail =>
      head.HonestInactive assignment ∧
        HonestInactiveOccurrences assignment tail

def rows
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active) : List OwnedRow :=
  plan.occurrences.flatMap ArmOccurrence.rows

def allocations
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active) : List OwnedColumn :=
  plan.occurrences.flatMap ArmOccurrence.allocations

def visibleIds
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active) : List ColumnId :=
  plan.occurrences.flatMap ArmOccurrence.visibleIds

def temporaryIds
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active) : List ColumnId :=
  plan.occurrences.flatMap ArmOccurrence.temporaryIds

def Ready
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active)
    (assignment : ColumnId -> Field) : Prop :=
  ReadyOccurrences assignment plan.occurrences

/-- Ordered structural conjunction of the exact occurrence facts. -/
def Facts
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active)
    (assignment : ColumnId -> Field) : Prop :=
  FactsOccurrences assignment plan.occurrences

def HonestActive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active)
    (assignment : ColumnId -> Field) : Prop :=
  HonestActiveOccurrences assignment plan.occurrences

def HonestInactive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active)
    (assignment : ColumnId -> Field) : Prop :=
  HonestInactiveOccurrences assignment plan.occurrences

/-- Column-only conditions needed to compose existential call completions.
Each call's temporaries are isolated from every arm-visible coordinate and
from every other occurrence's temporaries; every row mentions only its
occurrence's visible coordinates or its own temporaries. -/
def SeparatedOccurrences
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId} :
    List (ArmOccurrence signature family one active) -> Prop
  | [] => True
  | head :: tail =>
      (∀ id, id ∈ rowsColumns head.rows ->
        id ∈ head.visibleIds ++ head.temporaryIds) ∧
      IdsDisjoint head.temporaryIds head.visibleIds ∧
      IdsDisjoint head.temporaryIds
        (tail.flatMap ArmOccurrence.visibleIds) ∧
      IdsDisjoint head.temporaryIds
        (tail.flatMap ArmOccurrence.temporaryIds) ∧
      IdsDisjoint
        (tail.flatMap ArmOccurrence.temporaryIds) head.visibleIds ∧
      SeparatedOccurrences tail

def Separated
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active) : Prop :=
  SeparatedOccurrences plan.occurrences

theorem rows_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active) :
    plan.rows = plan.occurrences.flatMap ArmOccurrence.rows :=
  rfl

theorem allocations_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active) :
    plan.allocations =
      plan.occurrences.flatMap ArmOccurrence.allocations :=
  rfl

theorem row_count
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active) :
    plan.rows.length =
      (plan.occurrences.map fun occurrence =>
        occurrence.rows.length).sum := by
  change (plan.occurrences.flatMap ArmOccurrence.rows).length =
    (plan.occurrences.map fun occurrence =>
      occurrence.rows.length).sum
  induction plan.occurrences with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [rows, inductionHypothesis]

theorem sound
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1)
    (ready : plan.Ready assignment)
    (holds : Satisfies plan.rows assignment) :
    plan.Facts assignment := by
  change ReadyOccurrences assignment plan.occurrences at ready
  change Satisfies
    (plan.occurrences.flatMap ArmOccurrence.rows) assignment at holds
  change FactsOccurrences assignment plan.occurrences
  revert ready holds
  induction plan.occurrences with
  | nil =>
      intro _ _
      trivial
  | cons head tail inductionHypothesis =>
      intro ready holds
      have split :
          Satisfies head.rows assignment ∧
            Satisfies (tail.flatMap ArmOccurrence.rows) assignment := by
        exact (satisfies_append_iff _ _ _).1 holds
      exact ⟨head.sound laws assignment constantOne activeOne
          ready.1 split.1,
        inductionHypothesis ready.2 split.2⟩

private def HonestMode
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (selected : Bool)
    (occurrence : ArmOccurrence signature family one active)
    (assignment : ColumnId -> Field) : Prop :=
  if selected then
    occurrence.HonestActive assignment
  else
    occurrence.HonestInactive assignment

private theorem honestActiveOccurrences_of_agrees
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrences :
      List (ArmOccurrence signature family one active))
    (before after : ColumnId -> Field)
    (agrees :
      AgreesOn
        (occurrences.flatMap ArmOccurrence.visibleIds)
        before after)
    (honest : HonestActiveOccurrences before occurrences) :
    HonestActiveOccurrences after occurrences := by
  induction occurrences with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      have headAgrees :
          AgreesOn head.visibleIds before after := by
        apply agreesOn_of_subset _ agrees
        intro id member
        simp only [List.flatMap_cons, List.mem_append]
        exact Or.inl member
      have tailAgrees :
          AgreesOn (tail.flatMap ArmOccurrence.visibleIds)
            before after := by
        apply agreesOn_of_subset _ agrees
        intro id member
        simp only [List.flatMap_cons, List.mem_append]
        exact Or.inr member
      exact ⟨head.honestActive_of_agrees
          before after headAgrees honest.1,
        inductionHypothesis tailAgrees honest.2⟩

private theorem honestInactiveOccurrences_of_agrees
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrences :
      List (ArmOccurrence signature family one active))
    (before after : ColumnId -> Field)
    (agrees :
      AgreesOn
        (occurrences.flatMap ArmOccurrence.visibleIds)
        before after)
    (honest : HonestInactiveOccurrences before occurrences) :
    HonestInactiveOccurrences after occurrences := by
  induction occurrences with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      have headAgrees :
          AgreesOn head.visibleIds before after := by
        apply agreesOn_of_subset _ agrees
        intro id member
        simp only [List.flatMap_cons, List.mem_append]
        exact Or.inl member
      have tailAgrees :
          AgreesOn (tail.flatMap ArmOccurrence.visibleIds)
            before after := by
        apply agreesOn_of_subset _ agrees
        intro id member
        simp only [List.flatMap_cons, List.mem_append]
        exact Or.inr member
      exact ⟨head.honestInactive_of_agrees
          before after headAgrees honest.1,
        inductionHypothesis tailAgrees honest.2⟩

private theorem completeOccurrences
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (selected : Bool)
    (occurrences :
      List (ArmOccurrence signature family one active))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeValue :
      assignment active = if selected then 1 else 0)
    (honest :
      if selected then
        HonestActiveOccurrences assignment occurrences
      else
        HonestInactiveOccurrences assignment occurrences)
    (separated : SeparatedOccurrences occurrences) :
    ∃ completed : ColumnId -> Field,
      AgreesOn
          (occurrences.flatMap ArmOccurrence.visibleIds)
          assignment completed ∧
        ChangesOnly
          (occurrences.flatMap ArmOccurrence.temporaryIds)
          assignment completed ∧
        Satisfies
          (occurrences.flatMap ArmOccurrence.rows)
          completed := by
  induction occurrences generalizing assignment with
  | nil =>
      exact ⟨assignment, agreesOn_refl _ _,
        by intro _ _; rfl, trivial⟩
  | cons head tail inductionHypothesis =>
      rcases separated with
        ⟨headSupport, headTempVisible, headTempTailVisible,
          headTempTailTemps, tailTempsHeadVisible, tailSeparated⟩
      have headHonest :
          HonestMode selected head assignment := by
        cases selected <;> exact honest.1
      have localCompletion :
          ∃ middle : ColumnId -> Field,
            AgreesOn head.visibleIds assignment middle ∧
              ChangesOnly head.temporaryIds assignment middle ∧
              Satisfies head.rows middle := by
        cases selected with
        | false =>
            exact head.completeInactive assignment constantOne
              (by simpa using activeValue) headHonest
        | true =>
            exact head.completeActive laws assignment constantOne
              (by simpa using activeValue) headHonest
      rcases localCompletion with
        ⟨middle, middleVisible, middleChanges, headHolds⟩
      have middleTailVisible :
          AgreesOn
            (tail.flatMap ArmOccurrence.visibleIds)
            assignment middle :=
        agreesOn_of_changesOnly headTempTailVisible middleChanges
      have middleConstantOne : middle one = 1 := by
        have oneVisible : one ∈ head.visibleIds := by
          cases head <;> simp_all [ArmOccurrence.visibleIds,
            CallFrame.visibleIds]
        rw [middleVisible one oneVisible, constantOne]
      have middleActiveValue :
          middle active = if selected then 1 else 0 := by
        have activeVisible : active ∈ head.visibleIds := by
          cases head <;> simp_all [ArmOccurrence.visibleIds,
            CallFrame.visibleIds]
        rw [middleVisible active activeVisible, activeValue]
      have tailHonest :
          if selected then
            HonestActiveOccurrences middle tail
          else
            HonestInactiveOccurrences middle tail := by
        cases selected with
        | false =>
            exact honestInactiveOccurrences_of_agrees
              tail assignment middle middleTailVisible honest.2
        | true =>
            exact honestActiveOccurrences_of_agrees
              tail assignment middle middleTailVisible honest.2
      rcases inductionHypothesis middle middleConstantOne
          middleActiveValue tailHonest tailSeparated with
        ⟨completed, completedTailVisible, completedTailChanges,
          tailHolds⟩
      have completedHeadProtected :
          AgreesOn (head.visibleIds ++ head.temporaryIds)
            middle completed := by
        apply agreesOn_of_changesOnly
        · intro id tailTempMember protectedMember
          rcases List.mem_append.mp protectedMember with
            visibleMember | temporaryMember
          · exact tailTempsHeadVisible id tailTempMember visibleMember
          · exact (idsDisjoint_symm headTempTailTemps)
              id tailTempMember temporaryMember
        · exact completedTailChanges
      have completedHeadRows :
          Satisfies head.rows completed := by
        apply satisfies_of_agrees head.rows middle completed
        · apply agreesOn_of_subset headSupport completedHeadProtected
        · exact headHolds
      refine ⟨completed, ?_, ?_, ?_⟩
      · intro id member
        rcases List.mem_append.mp member with headMember | tailMember
        · rw [completedHeadProtected id
              (List.mem_append_left _ headMember),
            middleVisible id headMember]
        · have tailFinal :
              completed id = middle id :=
            completedTailVisible id tailMember
          have tailMiddle :
              middle id = assignment id :=
            middleTailVisible id tailMember
          exact tailFinal.trans tailMiddle
      · intro id notChanged
        have notHead : id ∉ head.temporaryIds := by
          intro member
          exact notChanged (List.mem_append_left _ member)
        have notTail :
            id ∉ tail.flatMap ArmOccurrence.temporaryIds := by
          intro member
          exact notChanged (List.mem_append_right _ member)
        rw [completedTailChanges id notTail, middleChanges id notHead]
      · exact (satisfies_append_iff head.rows
          (tail.flatMap ArmOccurrence.rows) completed).2
            ⟨completedHeadRows, tailHolds⟩

theorem completeActive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1)
    (honest : plan.HonestActive assignment)
    (separated : plan.Separated) :
    ∃ completed : ColumnId -> Field,
      AgreesOn plan.visibleIds assignment completed ∧
        ChangesOnly plan.temporaryIds assignment completed ∧
        Satisfies plan.rows completed := by
  exact completeOccurrences true plan.occurrences laws assignment
    constantOne (by simpa using activeOne)
    honest separated

theorem completeInactive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeZero : assignment active = 0)
    (honest : plan.HonestInactive assignment)
    (separated : plan.Separated) :
    ∃ completed : ColumnId -> Field,
      AgreesOn plan.visibleIds assignment completed ∧
        ChangesOnly plan.temporaryIds assignment completed ∧
        Satisfies plan.rows completed := by
  exact completeOccurrences false plan.occurrences laws assignment
    constantOne (by simpa using activeZero)
    honest separated

private theorem rowsSupportOccurrences
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (occurrences :
      List (ArmOccurrence signature family one active))
    (separated : SeparatedOccurrences occurrences) :
    ∀ id,
      id ∈ rowsColumns (occurrences.flatMap ArmOccurrence.rows) ->
    id ∈
      (occurrences.flatMap ArmOccurrence.visibleIds) ++
        occurrences.flatMap ArmOccurrence.temporaryIds := by
  revert separated
  induction occurrences with
  | nil =>
      intro _ id member
      simp [rowsColumns] at member
  | cons head tail inductionHypothesis =>
      intro separated
      rcases separated with
        ⟨headSupport, _, _, _, _, tailSeparated⟩
      intro id member
      have splitRows :
          rowsColumns
            (head.rows ++ tail.flatMap ArmOccurrence.rows) =
          rowsColumns head.rows ++
            rowsColumns (tail.flatMap ArmOccurrence.rows) := by
        simp [rowsColumns]
      simp only [List.flatMap_cons] at member ⊢
      rw [splitRows] at member
      rcases List.mem_append.mp member with headMember | tailMember
      · rcases List.mem_append.mp (headSupport id headMember) with
          visibleMember | temporaryMember
        · simp only [List.flatMap_cons, List.mem_append]
          exact Or.inl (Or.inl visibleMember)
        · simp only [List.flatMap_cons, List.mem_append]
          exact Or.inr (Or.inl temporaryMember)
      · have tailSupport :=
          inductionHypothesis tailSeparated id tailMember
        rcases List.mem_append.mp tailSupport with
          visibleMember | temporaryMember
        · simp only [List.flatMap_cons, List.mem_append]
          exact Or.inl (Or.inr visibleMember)
        · simp only [List.flatMap_cons, List.mem_append]
          exact Or.inr (Or.inr temporaryMember)

theorem rows_support
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one active : ColumnId}
    (plan : ArmPlan signature family one active)
    (separated : plan.Separated) :
    ∀ id, id ∈ rowsColumns plan.rows ->
      id ∈ plan.visibleIds ++ plan.temporaryIds :=
  rowsSupportOccurrences plan.occurrences separated

/-- Cross-arm conditions are exclusively about physical column identities. -/
structure PlansSeparated
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one firstActive secondActive : ColumnId}
    (first : ArmPlan signature family one firstActive)
    (second : ArmPlan signature family one secondActive) : Prop where
  firstTempsSecondVisible :
    IdsDisjoint first.temporaryIds second.visibleIds
  secondTempsFirstVisible :
    IdsDisjoint second.temporaryIds first.visibleIds
  firstTempsSecondTemps :
    IdsDisjoint first.temporaryIds second.temporaryIds
  firstTempsControl :
    IdsDisjoint first.temporaryIds [one, secondActive]

/-- Complete one selected arm followed by the inactive arm.  The resulting
assignment changes only the two explicitly listed temporary sets. -/
theorem completeActiveThenInactive
    {signature : Signature.{u}}
    {family : Family signature.types}
    {one firstActive secondActive : ColumnId}
    (first : ArmPlan signature family one firstActive)
    (second : ArmPlan signature family one secondActive)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (firstActiveOne : assignment firstActive = 1)
    (secondActiveZero : assignment secondActive = 0)
    (firstHonest : first.HonestActive assignment)
    (secondHonest : second.HonestInactive assignment)
    (firstSeparated : first.Separated)
    (secondSeparated : second.Separated)
    (cross : PlansSeparated first second) :
    ∃ completed : ColumnId -> Field,
      AgreesOn (first.visibleIds ++ second.visibleIds)
          assignment completed ∧
        ChangesOnly (first.temporaryIds ++ second.temporaryIds)
          assignment completed ∧
        Satisfies (first.rows ++ second.rows) completed := by
  rcases first.completeActive laws assignment constantOne firstActiveOne
      firstHonest firstSeparated with
    ⟨middle, middleFirstVisible, middleFirstChanges, firstHolds⟩
  have middleSecondVisible :
      AgreesOn second.visibleIds assignment middle :=
    agreesOn_of_changesOnly cross.firstTempsSecondVisible middleFirstChanges
  have middleOne : middle one = 1 := by
    have preserved :=
      agreesOn_of_changesOnly cross.firstTempsControl middleFirstChanges
    rw [preserved one (by simp), constantOne]
  have middleSecondActive : middle secondActive = 0 := by
    have preserved :=
      agreesOn_of_changesOnly cross.firstTempsControl middleFirstChanges
    rw [preserved secondActive (by simp), secondActiveZero]
  have middleSecondHonest : second.HonestInactive middle := by
    exact honestInactiveOccurrences_of_agrees second.occurrences
      assignment middle middleSecondVisible secondHonest
  rcases second.completeInactive laws middle middleOne middleSecondActive
      middleSecondHonest secondSeparated with
    ⟨completed, completedSecondVisible, completedSecondChanges,
      secondHolds⟩
  have completedFirstProtected :
      AgreesOn (first.visibleIds ++ first.temporaryIds)
        middle completed := by
    apply agreesOn_of_changesOnly
    · intro id secondTempMember protectedMember
      rcases List.mem_append.mp protectedMember with
        visibleMember | temporaryMember
      · exact cross.secondTempsFirstVisible
          id secondTempMember visibleMember
      · exact (idsDisjoint_symm cross.firstTempsSecondTemps)
          id secondTempMember temporaryMember
    · exact completedSecondChanges
  have completedFirstRows : Satisfies first.rows completed := by
    exact satisfies_of_agrees first.rows middle completed
      (agreesOn_of_subset (first.rows_support firstSeparated)
        completedFirstProtected) firstHolds
  refine ⟨completed, ?_, ?_, ?_⟩
  · intro id member
    rcases List.mem_append.mp member with firstMember | secondMember
    · rw [completedFirstProtected id
          (List.mem_append_left _ firstMember),
        middleFirstVisible id firstMember]
    · rw [completedSecondVisible id secondMember,
        middleSecondVisible id secondMember]
  · intro id outside
    have outsideFirst : id ∉ first.temporaryIds :=
      fun member => outside (List.mem_append_left _ member)
    have outsideSecond : id ∉ second.temporaryIds :=
      fun member => outside (List.mem_append_right _ member)
    rw [completedSecondChanges id outsideSecond,
      middleFirstChanges id outsideFirst]
  · exact (satisfies_append_iff _ _ _).2
      ⟨completedFirstRows, secondHolds⟩

end ArmPlan

/-! ## One concrete top-level branch receipt -/

/-- Exact activation, arm, and join receipts for one top-level branch.
`trueActivation` and `falseActivation` are the only control allocations. -/
structure SelectedBranch
    (signature : Signature.{u})
    (family : Family signature.types)
    (joinedLayout : Layout) where
  activation : BranchActivationRecipe
  trueActivation : OwnedColumn
  falseActivation : OwnedColumn
  trueActivationId :
    trueActivation.id = activation.onTrue
  falseActivationId :
    falseActivation.id = activation.onFalse
  trueActivationOwnership :
    trueActivation.ownership = .auxiliaryColumn
  falseActivationOwnership :
    falseActivation.ownership = .auxiliaryColumn
  onTrue :
    ArmPlan signature family activation.one activation.onTrue
  onFalse :
    ArmPlan signature family activation.one activation.onFalse
  mux : MuxRecipe joinedLayout
  muxSelector : mux.selector = activation.selector
  muxTrueVisible :
    ∀ id, id ∈ mux.onTrue.ids -> id ∈ onTrue.visibleIds
  muxFalseVisible :
    ∀ id, id ∈ mux.onFalse.ids -> id ∈ onFalse.visibleIds

namespace SelectedBranch

def rows
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout) :
    List OwnedRow :=
  branch.activation.rows ++
    (branch.onTrue.rows ++
      (branch.onFalse.rows ++ branch.mux.rows))

def allocations
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout) :
    List OwnedColumn :=
  [branch.trueActivation, branch.falseActivation] ++
    branch.onTrue.allocations ++
    branch.onFalse.allocations ++
    branch.mux.joined.columns

theorem rows_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout) :
    branch.rows =
      branch.activation.rows ++
        (branch.onTrue.rows ++
          (branch.onFalse.rows ++ branch.mux.rows)) :=
  rfl

theorem allocations_exact
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout) :
    branch.allocations =
      [branch.trueActivation, branch.falseActivation] ++
        branch.onTrue.allocations ++
        branch.onFalse.allocations ++
        branch.mux.joined.columns :=
  rfl

theorem row_count
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout) :
    branch.rows.length =
      2 + branch.onTrue.rows.length +
        branch.onFalse.rows.length + joinedLayout.owners.length := by
  simp only [rows, List.length_append, BranchActivationRecipe.row_count,
    MuxRecipe.row_count, Nat.add_assoc]

theorem selected_true_sound
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment branch.activation.one = 1)
    (parentActive : assignment branch.activation.active = 1)
    (selectorDecoded :
      boolCodec.decode [assignment branch.activation.selector] = some true)
    (ready : branch.onTrue.Ready assignment)
    (holds : Satisfies branch.rows assignment) :
    branch.onTrue.Facts assignment ∧
      branch.mux.joined.values assignment =
        branch.mux.onTrue.values assignment := by
  have splitActivation :
      Satisfies branch.activation.rows assignment ∧
        Satisfies
          (branch.onTrue.rows ++
            (branch.onFalse.rows ++ branch.mux.rows))
          assignment :=
    (satisfies_append_iff _ _ _).1 holds
  have activationValues :=
    branch.activation.selected_true_sound assignment constantOne
      selectorDecoded splitActivation.1
  have trueAndRest :=
    (satisfies_append_iff branch.onTrue.rows
      (branch.onFalse.rows ++ branch.mux.rows) assignment).1
      splitActivation.2
  have falseAndMux :=
    (satisfies_append_iff branch.onFalse.rows branch.mux.rows assignment).1
      trueAndRest.2
  have trueActive : assignment branch.activation.onTrue = 1 :=
    activationValues.1.trans parentActive
  have muxSelectorDecoded :
      boolCodec.decode [assignment branch.mux.selector] = some true := by
    rw [branch.muxSelector]
    exact selectorDecoded
  exact ⟨branch.onTrue.sound laws assignment constantOne trueActive
      ready trueAndRest.1,
    branch.mux.selected_true_sound assignment
      muxSelectorDecoded falseAndMux.2⟩

theorem selected_false_sound
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment branch.activation.one = 1)
    (parentActive : assignment branch.activation.active = 1)
    (selectorDecoded :
      boolCodec.decode [assignment branch.activation.selector] = some false)
    (ready : branch.onFalse.Ready assignment)
    (holds : Satisfies branch.rows assignment) :
    branch.onFalse.Facts assignment ∧
      branch.mux.joined.values assignment =
        branch.mux.onFalse.values assignment := by
  have splitActivation :=
    (satisfies_append_iff branch.activation.rows
      (branch.onTrue.rows ++
        (branch.onFalse.rows ++ branch.mux.rows))
      assignment).1 holds
  have activationValues :=
    branch.activation.selected_false_sound assignment constantOne
      selectorDecoded splitActivation.1
  have trueAndRest :=
    (satisfies_append_iff branch.onTrue.rows
      (branch.onFalse.rows ++ branch.mux.rows) assignment).1
      splitActivation.2
  have falseAndMux :=
    (satisfies_append_iff branch.onFalse.rows branch.mux.rows assignment).1
      trueAndRest.2
  have falseActive : assignment branch.activation.onFalse = 1 :=
    activationValues.2.trans parentActive
  have muxSelectorDecoded :
      boolCodec.decode [assignment branch.mux.selector] = some false := by
    rw [branch.muxSelector]
    exact selectorDecoded
  exact ⟨branch.onFalse.sound laws assignment constantOne falseActive
      ready falseAndMux.1,
    branch.mux.selected_false_sound assignment
      muxSelectorDecoded falseAndMux.2⟩

theorem joined_decodes_of_source
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout)
    (assignment : ColumnId -> Field)
    (kind : signature.types.Kind)
    (value : signature.types.Value kind)
    (source : ColumnBundle joinedLayout)
    (joinedEqual :
      branch.mux.joined.values assignment = source.values assignment)
    (sourceDecoded :
      source.Decodes family kind assignment value) :
    branch.mux.joined.Decodes family kind assignment value := by
  unfold ColumnBundle.Decodes at sourceDecoded ⊢
  rw [joinedEqual]
  exact sourceDecoded

/-- Close the exact branch row list from an arm completion.  The only
preservation premise is disjointness between arm temporaries and the concrete
activation/mux row support. -/
theorem rows_of_completed_arms
    {signature : Signature.{u}}
    {family : Family signature.types}
    {joinedLayout : Layout}
    (branch : SelectedBranch signature family joinedLayout)
    (before after : ColumnId -> Field)
    (activationHolds : Satisfies branch.activation.rows before)
    (armHolds :
      Satisfies (branch.onTrue.rows ++ branch.onFalse.rows) after)
    (muxHolds : Satisfies branch.mux.rows before)
    (changes :
      ChangesOnly
        (branch.onTrue.temporaryIds ++ branch.onFalse.temporaryIds)
        before after)
    (fixedRowsSeparated :
      IdsDisjoint
        (branch.onTrue.temporaryIds ++ branch.onFalse.temporaryIds)
        (rowsColumns branch.activation.rows ++
          rowsColumns branch.mux.rows)) :
    Satisfies branch.rows after := by
  have fixedAgrees :
      AgreesOn
        (rowsColumns branch.activation.rows ++
          rowsColumns branch.mux.rows) before after :=
    agreesOn_of_changesOnly fixedRowsSeparated changes
  have activationAgrees :
      AgreesOn (rowsColumns branch.activation.rows) before after := by
    apply agreesOn_of_subset _ fixedAgrees
    intro id member
    exact List.mem_append_left _ member
  have muxAgrees :
      AgreesOn (rowsColumns branch.mux.rows) before after := by
    apply agreesOn_of_subset _ fixedAgrees
    intro id member
    exact List.mem_append_right _ member
  have activationAfter :=
    satisfies_of_agrees branch.activation.rows before after
      activationAgrees activationHolds
  have muxAfter :=
    satisfies_of_agrees branch.mux.rows before after muxAgrees muxHolds
  have armsSplit :=
    (satisfies_append_iff branch.onTrue.rows
      branch.onFalse.rows after).1 armHolds
  exact (satisfies_append_iff _ _ _).2
    ⟨activationAfter,
      (satisfies_append_iff _ _ _).2
        ⟨armsSplit.1,
          (satisfies_append_iff _ _ _).2
            ⟨armsSplit.2, muxAfter⟩⟩⟩

end SelectedBranch

end Nightstream.Implementation.Lowering.Goldilocks
