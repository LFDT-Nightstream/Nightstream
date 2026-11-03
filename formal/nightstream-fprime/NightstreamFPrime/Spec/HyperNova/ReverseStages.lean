import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-! The reverse extractor of HyperNova Lemma 17 (Appendix H.3) as a walk. The IVC adversary
outputs a visit, for example a terminal statement and proof. Stage `0` is the IVC adversary;
stage `j + 1` runs stage `j` and then the extractor that Assumption 1 gives for stage `j`'s NIFS
adversary. That adversary outputs the fold attempt of stage `j`'s visit (`attemptOf`), and the
extractor's answer moves the walk one visit back (`back`). Then for any event that each run
of the walk covers with a failed stage or a collision at a visit (`bad_le`),

  Pr[event] ≤ Σ j < depth, (Pr[collision at visit j] + error(stage j)).

Assumption 1 is a premise in the form of HyperNova Definition 7. `Admitted`, `StageAdmitted` and
`Efficient` are abstract, meant as expected polynomial time. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.HyperNova.ReverseStages

open scoped BigOperators ENNReal

variable {Visit Attempt Answer : Type}

theorem event_ne_top {Sample : Type} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

/-- HyperNova Definition 7 adversary for one fold. -/
structure NifsAdversary (Attempt : Type) where
  Tape : Type
  tape : PMF Tape
  run : Tape → Option Attempt

structure NifsExtractor (Answer : Type) (adversary : NifsAdversary Attempt) where
  Coins : Type
  coins : PMF Coins
  run : adversary.Tape → Coins → Answer

noncomputable def NifsExtractor.law {adversary : NifsAdversary Attempt}
    (extractor : NifsExtractor Answer adversary) : PMF (adversary.Tape × extractor.Coins) :=
  adversary.tape.bind fun tape => extractor.coins.map (tape, ·)

structure IvcAdversary (Visit : Type) where
  Tape : Type
  tape : PMF Tape
  output : Tape → Visit

/-- The reverse extractor after some stages: the IVC adversary's output and the answers so far. -/
structure Stage (Visit Answer : Type) where
  Tape : Type
  tape : PMF Tape
  input : Tape → Visit
  results : Tape → List Answer

section Events

variable (attemptOf : Visit → Option Attempt) (back : Visit → Answer → Visit)
  (Succeeds : Attempt → Prop) (Extracted : Attempt → Answer → Prop)

def ExtractionFails (adversary : NifsAdversary Attempt) (extractor : NifsExtractor Answer adversary)
    (draw : adversary.Tape × extractor.Coins) : Prop :=
  ∃ attempt, adversary.run draw.1 = some attempt ∧ Succeeds attempt ∧
    ¬ Extracted attempt (extractor.run draw.1 draw.2)

/-- HyperNova errata Assumption 1 in Definition 7 form. -/
def Assumption1 (Admitted : NifsAdversary Attempt → Prop)
    (Efficient : (adversary : NifsAdversary Attempt) → NifsExtractor Answer adversary → Prop)
    (error : NifsAdversary Attempt → ℝ) : Prop :=
  ∀ adversary, Admitted adversary → ∃ extractor : NifsExtractor Answer adversary,
    Efficient adversary extractor ∧
      (extractor.law.toOuterMeasure
        {draw | ExtractionFails Succeeds Extracted adversary extractor draw}).toReal ≤
        error adversary

/-- The visit after the answers `results`, one step back per answer. -/
def visitAfter (input : Visit) (results : List Answer) : Visit :=
  results.foldl back input

/-- Stage `j` fails: its visit has an attempt that succeeds, and its answer does not extract. -/
def StageFails (input : Visit) (results : List Answer) (j : Nat) : Prop :=
  ∃ attempt answer, attemptOf (visitAfter back input (results.take j)) = some attempt ∧
    results[j]? = some answer ∧ Succeeds attempt ∧ ¬ Extracted attempt answer

theorem visitAfter_cons (input : Visit) (answer : Answer) (rest : List Answer) (j : Nat) :
    visitAfter back input ((answer :: rest).take (j + 1)) =
      visitAfter back (back input answer) (rest.take j) := by
  simp [visitAfter, List.take_succ_cons]

/-- Stage `j + 1` of a walk is stage `j` of the walk from the prior visit. -/
theorem stageFails_cons (input : Visit) (answer : Answer) (rest : List Answer) (j : Nat) :
    StageFails attemptOf back Succeeds Extracted input (answer :: rest) (j + 1) ↔
      StageFails attemptOf back Succeeds Extracted (back input answer) rest j := by
  simp only [StageFails, visitAfter_cons, List.getElem?_cons_succ]

namespace Stage

/-- The stage's NIFS adversary: the attempt of the stage's current visit. -/
def nifs (stage : Stage Visit Answer) : NifsAdversary Attempt where
  Tape := stage.Tape
  tape := stage.tape
  run tape := attemptOf (visitAfter back (stage.input tape) (stage.results tape))

def start (adversary : IvcAdversary Visit) : Stage Visit Answer where
  Tape := adversary.Tape
  tape := adversary.tape
  input := adversary.output
  results _ := []

@[implicit_reducible]
noncomputable def next (stage : Stage Visit Answer)
    (extractor : NifsExtractor Answer (stage.nifs attemptOf back)) : Stage Visit Answer where
  Tape := stage.Tape × extractor.Coins
  tape := extractor.law
  input draw := stage.input draw.1
  results draw := stage.results draw.1 ++ [extractor.run draw.1 draw.2]

theorem next_marginal (stage : Stage Visit Answer)
    (extractor : NifsExtractor Answer (stage.nifs attemptOf back)) (count : Nat)
    (within : ∀ tape, count ≤ (stage.results tape).length) :
    (stage.next attemptOf back extractor).tape.map (fun draw =>
        ((stage.next attemptOf back extractor).input draw,
          ((stage.next attemptOf back extractor).results draw).take count)) =
      stage.tape.map (fun tape => (stage.input tape, (stage.results tape).take count)) := by
  change (stage.tape.bind fun tape => extractor.coins.map (tape, ·)).map _ = _
  rw [PMF.map_bind, ← PMF.bind_pure_comp]
  congr 1
  funext tape
  calc _ = extractor.coins.map
          (Function.const extractor.Coins (stage.input tape, (stage.results tape).take count)) := by
        rw [PMF.map_comp]
        congr 1
        funext coins
        simp only [Function.comp_apply, Function.const_apply, Stage.next,
          List.take_append_of_le_length (within tape)]
    _ = _ := PMF.map_const _ _

end Stage

/-- HyperNova Lemma 17 composition premise. -/
structure Closed (Admitted : NifsAdversary Attempt → Prop)
    (StageAdmitted : Stage Visit Answer → Prop)
    (Efficient : (adversary : NifsAdversary Attempt) → NifsExtractor Answer adversary → Prop) :
    Prop where
  nifs : ∀ stage, StageAdmitted stage → Admitted (stage.nifs attemptOf back)
  next : ∀ (stage : Stage Visit Answer) (extractor : NifsExtractor Answer (stage.nifs attemptOf back)),
    StageAdmitted stage → Efficient (stage.nifs attemptOf back) extractor →
      StageAdmitted (stage.next attemptOf back extractor)

end Events

variable {attemptOf : Visit → Option Attempt} {back : Visit → Answer → Visit}
  {Succeeds : Attempt → Prop} {Extracted : Attempt → Answer → Prop}
  {Admitted : NifsAdversary Attempt → Prop} {StageAdmitted : Stage Visit Answer → Prop}
  {Efficient : (adversary : NifsAdversary Attempt) → NifsExtractor Answer adversary → Prop}
  {error : NifsAdversary Attempt → ℝ}
  (assumption : Assumption1 Succeeds Extracted Admitted Efficient error)
  (closed : Closed attemptOf back Admitted StageAdmitted Efficient)
  (adversary : IvcAdversary Visit) (admitted : StageAdmitted (Stage.start adversary))

/-- HyperNova Lemma 17's reverse extractor. -/
noncomputable def reverseStages : Nat → {stage : Stage Visit Answer // StageAdmitted stage}
  | 0 => ⟨Stage.start adversary, admitted⟩
  | steps + 1 =>
      ⟨(reverseStages steps).1.next attemptOf back
          (Classical.choose (assumption _ (closed.nifs _ (reverseStages steps).2))),
        closed.next _ _ (reverseStages steps).2
          (Classical.choose_spec (assumption _ (closed.nifs _ (reverseStages steps).2))).1⟩

noncomputable def reverseExtractor (steps : Nat) :
    NifsExtractor Answer
      ((reverseStages assumption closed adversary admitted steps).1.nifs attemptOf back) :=
  Classical.choose
    (assumption _ (closed.nifs _ (reverseStages assumption closed adversary admitted steps).2))

theorem reverseExtractor_error (steps : Nat) :
    ((reverseExtractor assumption closed adversary admitted steps).law.toOuterMeasure
        {draw | ExtractionFails Succeeds Extracted _
          (reverseExtractor assumption closed adversary admitted steps) draw}).toReal ≤
      error ((reverseStages assumption closed adversary admitted steps).1.nifs attemptOf back) :=
  (Classical.choose_spec
    (assumption _ (closed.nifs _ (reverseStages assumption closed adversary admitted steps).2))).2

local notation "stages" => reverseStages assumption closed adversary admitted
local notation "extractors" => reverseExtractor assumption closed adversary admitted

theorem reverseStages_succ (steps : Nat) :
    (stages (steps + 1)).1 = (stages steps).1.next attemptOf back (extractors steps) := rfl

theorem results_length (steps : Nat) (tape : (stages steps).1.Tape) :
    ((stages steps).1.results tape).length = steps := by
  induction steps with
  | zero => rfl
  | succ steps induction =>
      change ((stages steps).1.results tape.1 ++ [(extractors steps).run tape.1 tape.2]).length =
        steps + 1
      rw [List.length_append, induction, List.length_singleton]

theorem reverseStages_marginal (steps extra : Nat) :
    (stages (steps + extra)).1.tape.map (fun tape =>
        ((stages (steps + extra)).1.input tape,
          ((stages (steps + extra)).1.results tape).take steps)) =
      (stages steps).1.tape.map
        (fun tape => ((stages steps).1.input tape, (stages steps).1.results tape)) := by
  induction extra with
  | zero =>
      change (stages steps).1.tape.map (fun tape =>
        ((stages steps).1.input tape, ((stages steps).1.results tape).take steps)) = _
      congr 1
      funext tape
      rw [List.take_of_length_le (results_length assumption closed adversary admitted steps tape).le]
  | succ extra induction =>
      rw [show steps + (extra + 1) = steps + extra + 1 from rfl, reverseStages_succ,
        Stage.next_marginal attemptOf back _ _ steps (fun tape => by rw [results_length]; omega)]
      exact induction

theorem stage_event (steps extra : Nat) (event : Set (Visit × List Answer)) :
    (stages (steps + extra)).1.tape.toOuterMeasure
        {tape | ((stages (steps + extra)).1.input tape,
          ((stages (steps + extra)).1.results tape).take steps) ∈ event} =
      (stages steps).1.tape.toOuterMeasure
        {tape | ((stages steps).1.input tape, (stages steps).1.results tape) ∈ event} := by
  have mapped := congrArg (fun law => PMF.toOuterMeasure law event)
    (reverseStages_marginal assumption closed adversary admitted steps extra)
  simp only [PMF.toOuterMeasure_map_apply] at mapped
  exact mapped

/-- The IVC adversary's law of its output is the reverse extractor's law of its input. -/
theorem start_event (total : Nat) (event : Set Visit) :
    adversary.tape.toOuterMeasure {tape | adversary.output tape ∈ event} =
      (stages total).1.tape.toOuterMeasure {tape | (stages total).1.input tape ∈ event} := by
  have moved := stage_event assumption closed adversary admitted 0 total {pair | pair.1 ∈ event}
  rw [Nat.zero_add] at moved
  exact moved.symm

/-- A supported input of a stage is a supported output of the IVC adversary. -/
theorem input_supported (steps : Nat) (tape : (stages steps).1.Tape)
    (supported : tape ∈ (stages steps).1.tape.support) :
    ∃ origin ∈ adversary.tape.support, adversary.output origin = (stages steps).1.input tape := by
  have mapped := reverseStages_marginal assumption closed adversary admitted 0 steps
  rw [Nat.zero_add] at mapped
  have member : ((stages steps).1.input tape, ((stages steps).1.results tape).take 0) ∈
      ((stages 0).1.tape.map
        (fun tape => ((stages 0).1.input tape, (stages 0).1.results tape))).support := by
    rw [← mapped]
    exact (PMF.mem_support_map_iff _ _ _).mpr ⟨tape, supported, rfl⟩
  obtain ⟨origin, originSupported, same⟩ := (PMF.mem_support_map_iff _ _ _).mp member
  exact ⟨origin, originSupported, congrArg Prod.fst same⟩

/-- Stage `j`'s failure, read on a later stage, is stage `j`'s extraction failure. -/
theorem failure_term (j extra : Nat) :
    (stages (j + 1 + extra)).1.tape.toOuterMeasure
        {tape | StageFails attemptOf back Succeeds Extracted ((stages (j + 1 + extra)).1.input tape)
          ((stages (j + 1 + extra)).1.results tape) j} =
      (extractors j).law.toOuterMeasure
        {draw | ExtractionFails Succeeds Extracted _ (extractors j) draw} := by
  let event : Set (Visit × List Answer) :=
    {pair | StageFails attemptOf back Succeeds Extracted pair.1 pair.2 j}
  have taken : {tape | StageFails attemptOf back Succeeds Extracted
        ((stages (j + 1 + extra)).1.input tape) ((stages (j + 1 + extra)).1.results tape) j} =
      {tape | ((stages (j + 1 + extra)).1.input tape,
        ((stages (j + 1 + extra)).1.results tape).take (j + 1)) ∈ event} := by
    ext tape
    simp only [Set.mem_ofPred_eq, event, StageFails, List.take_take,
      Nat.min_eq_left (Nat.le_succ j), List.getElem?_take_of_lt (Nat.lt_succ_self j)]
  rw [taken, stage_event assumption closed adversary admitted (j + 1) extra event,
    reverseStages_succ]
  congr 1
  ext draw
  have length := results_length assumption closed adversary admitted j draw.1
  simp only [Set.mem_ofPred_eq, event, StageFails, Stage.next, ExtractionFails, Stage.nifs]
  have sameVisit : ((stages j).1.results draw.1 ++ [(extractors j).run draw.1 draw.2]).take j =
      (stages j).1.results draw.1 := by
    rw [List.take_append_of_le_length (by omega), List.take_of_length_le (by omega)]
  have sameAnswer : ((stages j).1.results draw.1 ++ [(extractors j).run draw.1 draw.2])[j]? =
      some ((extractors j).run draw.1 draw.2) := by
    rw [List.getElem?_append_right (by omega), length, Nat.sub_self]
    rfl
  rw [sameVisit, sameAnswer]
  constructor
  · rintro ⟨attempt, answer, attemptEq, answerEq, wins, fails⟩
    cases answerEq
    exact ⟨attempt, attemptEq, wins, fails⟩
  · rintro ⟨attempt, attemptEq, wins, fails⟩
    exact ⟨attempt, _, attemptEq, rfl, wins, fails⟩

/-- A visit event at stage `j`, read on a later stage, is the event on stage `j`'s own visit. -/
theorem visit_term (Collision : Visit → Prop) (j extra : Nat) :
    (stages (j + extra)).1.tape.toOuterMeasure
        {tape | Collision (visitAfter back ((stages (j + extra)).1.input tape)
          (((stages (j + extra)).1.results tape).take j))} =
      (stages j).1.tape.toOuterMeasure
        {tape | Collision (visitAfter back ((stages j).1.input tape) ((stages j).1.results tape))} :=
  stage_event assumption closed adversary admitted j extra
    {pair | Collision (visitAfter back pair.1 pair.2)}

/-- HyperNova Lemma 17: an event that every supported run covers with a collision at a visit or
a failed stage has probability at most the collision probabilities plus the stages' Assumption 1
errors. -/
theorem bad_le (depth : Nat) (Bad : Visit → List Answer → Prop) (Collision : Visit → Prop)
    (covered : ∀ origin ∈ adversary.tape.support, ∀ results : List Answer,
      results.length = depth → Bad (adversary.output origin) results →
        ∃ j : Fin depth,
          Collision (visitAfter back (adversary.output origin) (results.take j.val)) ∨
            StageFails attemptOf back Succeeds Extracted (adversary.output origin) results j.val) :
    ((stages depth).1.tape.toOuterMeasure
        {tape | Bad ((stages depth).1.input tape) ((stages depth).1.results tape)}).toReal ≤
      ∑ j : Fin depth,
        (((stages j.val).1.tape.toOuterMeasure
            {tape | Collision (visitAfter back ((stages j.val).1.input tape)
              ((stages j.val).1.results tape))}).toReal +
          error ((stages j.val).1.nifs attemptOf back)) := by
  let law := (stages depth).1.tape
  let collisions : Fin depth → Set (stages depth).1.Tape := fun j =>
    {tape | Collision (visitAfter back ((stages depth).1.input tape)
      (((stages depth).1.results tape).take j.val))}
  let failures : Fin depth → Set (stages depth).1.Tape := fun j =>
    {tape | StageFails attemptOf back Succeeds Extracted ((stages depth).1.input tape)
      ((stages depth).1.results tape) j.val}
  have coveredTapes : {tape | Bad ((stages depth).1.input tape) ((stages depth).1.results tape)} ∩
      law.support ⊆ ⋃ j, collisions j ∪ failures j := by
    rintro tape ⟨bad, supported⟩
    change Bad ((stages depth).1.input tape) ((stages depth).1.results tape) at bad
    obtain ⟨origin, originSupported, same⟩ :=
      input_supported assumption closed adversary admitted depth tape supported
    rw [← same] at bad
    obtain ⟨j, found⟩ := covered origin originSupported _
      (results_length assumption closed adversary admitted depth tape) bad
    rw [same] at found
    exact Set.mem_iUnion.mpr ⟨j, found⟩
  have collisionEq : ∀ j : Fin depth, law.toOuterMeasure (collisions j) =
      (stages j.val).1.tape.toOuterMeasure
        {tape | Collision (visitAfter back ((stages j.val).1.input tape)
          ((stages j.val).1.results tape))} := by
    intro j
    obtain ⟨extra, total⟩ := Nat.exists_eq_add_of_le j.isLt.le
    have moved := visit_term assumption closed adversary admitted Collision j.val extra
    rw [← total] at moved
    exact moved
  have failureEq : ∀ j : Fin depth, law.toOuterMeasure (failures j) =
      (extractors j.val).law.toOuterMeasure
        {draw | ExtractionFails Succeeds Extracted _ (extractors j.val) draw} := by
    intro j
    obtain ⟨extra, total⟩ := Nat.exists_eq_add_of_le (Nat.succ_le_of_lt j.isLt)
    have moved := failure_term assumption closed adversary admitted j.val extra
    rw [← total] at moved
    exact moved
  have bound : law.toOuterMeasure
      {tape | Bad ((stages depth).1.input tape) ((stages depth).1.results tape)} ≤
      ∑ j : Fin depth, ((stages j.val).1.tape.toOuterMeasure
          {tape | Collision (visitAfter back ((stages j.val).1.input tape)
            ((stages j.val).1.results tape))} +
        (extractors j.val).law.toOuterMeasure
          {draw | ExtractionFails Succeeds Extracted _ (extractors j.val) draw}) := by
    rw [← PMF.toOuterMeasure_apply_inter_support]
    calc _ ≤ law.toOuterMeasure (⋃ j, collisions j ∪ failures j) :=
          MeasureTheory.measure_mono coveredTapes
      _ ≤ ∑ j, law.toOuterMeasure (collisions j ∪ failures j) :=
          MeasureTheory.measure_iUnion_fintype_le _ _
      _ ≤ ∑ j, (law.toOuterMeasure (collisions j) + law.toOuterMeasure (failures j)) :=
          Finset.sum_le_sum fun j _ => MeasureTheory.measure_union_le _ _
      _ = _ := Finset.sum_congr rfl fun j _ => by rw [collisionEq j, failureEq j]
  have finite : ∀ j : Fin depth, (stages j.val).1.tape.toOuterMeasure
        {tape | Collision (visitAfter back ((stages j.val).1.input tape)
          ((stages j.val).1.results tape))} +
      (extractors j.val).law.toOuterMeasure
        {draw | ExtractionFails Succeeds Extracted _ (extractors j.val) draw} ≠ ∞ :=
    fun j => ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩
  have realBound := ENNReal.toReal_mono (ENNReal.sum_ne_top.mpr fun j _ => finite j) bound
  rw [ENNReal.toReal_sum fun j _ => finite j] at realBound
  refine realBound.trans (Finset.sum_le_sum fun j _ => ?_)
  rw [ENNReal.toReal_add (event_ne_top _ _) (event_ne_top _ _)]
  exact add_le_add le_rfl (reverseExtractor_error assumption closed adversary admitted j.val)

end NightstreamFPrime.Spec.HyperNova.ReverseStages
