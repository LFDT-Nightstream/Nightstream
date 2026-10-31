import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthReverse
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-! HyperNova Lemma 17 (Appendix H.3) for one slot of option B, with probabilities. Stage `0`
is the IVC adversary; stage `j + 1` runs stage `j` and then the extractor that Assumption 1 gives
for stage `j`'s NIFS adversary, which attacks the `j`-th fold from the end. Then

  Pr[accepted ∧ the answers do not extract every fold] ≤ Σ j < depth, error(stage_j).

Assumption 1 (HyperNova errata, Definition 7 form) is a premise, as for Stage 1; it is needed
only at output bounds up to `B + cap · T · (b − 1)`, where the relaxed binding of
`chainCollision_to_shortKernel` and the κ = 22 MSIS estimate at that norm apply. The IVC
adversary outputs the slot's history; Stage 1 reads it back from step witnesses and adds a
state-hash collision term, which this file does not model. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthSecurity

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthReverse

variable {Key Claim Fresh Proof ClaimWitness FreshWitness : Type}

theorem event_ne_top {Sample : Type} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

/-- What the IVC adversary outputs for one slot: the history, last fold first, and a witness of
the final claim at `bound`. -/
structure Output (Key Claim Fresh Proof ClaimWitness : Type) where
  final : Running Claim
  back : List (Link Key Claim Fresh Proof)
  bound : Nat
  witness : ClaimWitness

/-- HyperNova Definition 7 adversary for one fold. -/
structure NifsAdversary (Key Claim Fresh Proof ClaimWitness : Type) where
  Tape : Type
  tape : PMF Tape
  run : Tape → Option (Attempt Key Claim Fresh Proof ClaimWitness)

structure NifsExtractor (FreshWitness : Type)
    (adversary : NifsAdversary Key Claim Fresh Proof ClaimWitness) where
  Coins : Type
  coins : PMF Coins
  run : adversary.Tape → Coins → ClaimWitness × FreshWitness

noncomputable def NifsExtractor.law {adversary : NifsAdversary Key Claim Fresh Proof ClaimWitness}
    (extractor : NifsExtractor FreshWitness adversary) : PMF (adversary.Tape × extractor.Coins) :=
  adversary.tape.bind fun tape => extractor.coins.map (tape, ·)

structure IvcAdversary (Key Claim Fresh Proof ClaimWitness : Type) where
  Tape : Type
  tape : PMF Tape
  output : Tape → Output Key Claim Fresh Proof ClaimWitness

structure Stage (Key Claim Fresh Proof ClaimWitness FreshWitness : Type) where
  Tape : Type
  tape : PMF Tape
  input : Tape → Output Key Claim Fresh Proof ClaimWitness
  results : Tape → List (ClaimWitness × FreshWitness)

section Events

variable (p : GlobalParams) (cap : Nat) (folds : Folds Key Claim Fresh Proof)
  (Holds : Nat → Claim → ClaimWitness → Prop) (FreshHolds : Fresh → FreshWitness → Prop)

/-- The verifier accepts the slot's history and the decider accepts the final claim below `B`. -/
def Accepted (output : Output Key Claim Fresh Proof ClaimWitness) : Prop :=
  RunsBack cap folds output.final output.back ∧ Holds output.bound output.final.1 output.witness ∧
    output.bound ≤ p.bigB

def ExtractionFails (adversary : NifsAdversary Key Claim Fresh Proof ClaimWitness)
    (extractor : NifsExtractor FreshWitness adversary) (draw : adversary.Tape × extractor.Coins) :
    Prop :=
  ∃ attempt, adversary.run draw.1 = some attempt ∧ Succeeds p cap folds Holds attempt ∧
    ¬ Extracted p Holds FreshHolds attempt (extractor.run draw.1 draw.2)

/-- HyperNova errata Assumption 1 in Definition 7 form for the folds of option B. -/
def Assumption1 (Admitted : NifsAdversary Key Claim Fresh Proof ClaimWitness → Prop)
    (Efficient : (adversary : NifsAdversary Key Claim Fresh Proof ClaimWitness) →
      NifsExtractor FreshWitness adversary → Prop)
    (error : NifsAdversary Key Claim Fresh Proof ClaimWitness → ℝ) : Prop :=
  ∀ adversary, Admitted adversary → ∃ extractor : NifsExtractor FreshWitness adversary,
    Efficient adversary extractor ∧
      (extractor.law.toOuterMeasure
        {draw | ExtractionFails p cap folds Holds FreshHolds adversary extractor draw}).toReal ≤
        error adversary

/-- Stage `j` of the reverse extractor fails on its fold. -/
def StageFails (output : Output Key Claim Fresh Proof ClaimWitness)
    (results : List (ClaimWitness × FreshWitness)) (j : Nat) : Prop :=
  Fails p cap folds Holds FreshHolds output.bound output.witness output.back results j

/-- The answers extract every fold of the slot's history. -/
def HistoryReturned (output : Output Key Claim Fresh Proof ClaimWitness)
    (results : List (ClaimWitness × FreshWitness)) : Prop :=
  Returned p Holds FreshHolds output.bound output.witness output.back results

namespace Stage

/-- Stage `j`'s NIFS adversary: the `j`-th fold from the end, with stage `j − 1`'s answer. -/
def nifs (stage : Stage Key Claim Fresh Proof ClaimWitness FreshWitness) :
    NifsAdversary Key Claim Fresh Proof ClaimWitness where
  Tape := stage.Tape
  tape := stage.tape
  run tape := (attempts p (stage.input tape).bound (stage.input tape).witness
    (stage.input tape).back (stage.results tape))[(stage.results tape).length]?

def start (adversary : IvcAdversary Key Claim Fresh Proof ClaimWitness) :
    Stage Key Claim Fresh Proof ClaimWitness FreshWitness where
  Tape := adversary.Tape
  tape := adversary.tape
  input := adversary.output
  results _ := []

@[implicit_reducible]
noncomputable def next (stage : Stage Key Claim Fresh Proof ClaimWitness FreshWitness)
    (extractor : NifsExtractor FreshWitness (stage.nifs p)) :
    Stage Key Claim Fresh Proof ClaimWitness FreshWitness where
  Tape := stage.Tape × extractor.Coins
  tape := extractor.law
  input draw := stage.input draw.1
  results draw := stage.results draw.1 ++ [extractor.run draw.1 draw.2]

theorem next_marginal (stage : Stage Key Claim Fresh Proof ClaimWitness FreshWitness)
    (extractor : NifsExtractor FreshWitness (stage.nifs p)) (count : Nat)
    (within : ∀ tape, count ≤ (stage.results tape).length) :
    (stage.next p extractor).tape.map (fun draw =>
        ((stage.next p extractor).input draw, ((stage.next p extractor).results draw).take count)) =
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
structure Closed (Admitted : NifsAdversary Key Claim Fresh Proof ClaimWitness → Prop)
    (StageAdmitted : Stage Key Claim Fresh Proof ClaimWitness FreshWitness → Prop)
    (Efficient : (adversary : NifsAdversary Key Claim Fresh Proof ClaimWitness) →
      NifsExtractor FreshWitness adversary → Prop) : Prop where
  nifs : ∀ stage, StageAdmitted stage → Admitted (stage.nifs p)
  next : ∀ (stage : Stage Key Claim Fresh Proof ClaimWitness FreshWitness)
      (extractor : NifsExtractor FreshWitness (stage.nifs p)),
    StageAdmitted stage → Efficient (stage.nifs p) extractor →
      StageAdmitted (stage.next p extractor)

end Events

variable {p : GlobalParams} {cap : Nat} {folds : Folds Key Claim Fresh Proof}
  {Holds : Nat → Claim → ClaimWitness → Prop} {FreshHolds : Fresh → FreshWitness → Prop}
  {Admitted : NifsAdversary Key Claim Fresh Proof ClaimWitness → Prop}
  {StageAdmitted : Stage Key Claim Fresh Proof ClaimWitness FreshWitness → Prop}
  {Efficient : (adversary : NifsAdversary Key Claim Fresh Proof ClaimWitness) →
    NifsExtractor FreshWitness adversary → Prop}
  {error : NifsAdversary Key Claim Fresh Proof ClaimWitness → ℝ}
  (assumption : Assumption1 p cap folds Holds FreshHolds Admitted Efficient error)
  (closed : Closed p Admitted StageAdmitted Efficient)
  (adversary : IvcAdversary Key Claim Fresh Proof ClaimWitness)
  (admitted : StageAdmitted (Stage.start adversary))

/-- HyperNova Lemma 17. -/
noncomputable def reverseStages :
    Nat → {stage : Stage Key Claim Fresh Proof ClaimWitness FreshWitness // StageAdmitted stage}
  | 0 => ⟨Stage.start adversary, admitted⟩
  | steps + 1 =>
      ⟨(reverseStages steps).1.next p
          (Classical.choose (assumption _ (closed.nifs _ (reverseStages steps).2))),
        closed.next _ _ (reverseStages steps).2
          (Classical.choose_spec (assumption _ (closed.nifs _ (reverseStages steps).2))).1⟩

noncomputable def reverseExtractor (steps : Nat) :
    NifsExtractor FreshWitness ((reverseStages assumption closed adversary admitted steps).1.nifs p) :=
  Classical.choose
    (assumption _ (closed.nifs _ (reverseStages assumption closed adversary admitted steps).2))

theorem reverseExtractor_error (steps : Nat) :
    ((reverseExtractor assumption closed adversary admitted steps).law.toOuterMeasure
        {draw | ExtractionFails p cap folds Holds FreshHolds _
          (reverseExtractor assumption closed adversary admitted steps) draw}).toReal ≤
      error ((reverseStages assumption closed adversary admitted steps).1.nifs p) :=
  (Classical.choose_spec
    (assumption _ (closed.nifs _ (reverseStages assumption closed adversary admitted steps).2))).2

local notation "stages" => reverseStages assumption closed adversary admitted
local notation "extractors" => reverseExtractor assumption closed adversary admitted

theorem reverseStages_succ (steps : Nat) :
    (stages (steps + 1)).1 = (stages steps).1.next p (extractors steps) := rfl

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
        ((stages (steps + extra)).1.input tape, ((stages (steps + extra)).1.results tape).take steps)) =
      (stages steps).1.tape.map (fun tape => ((stages steps).1.input tape, (stages steps).1.results tape)) := by
  induction extra with
  | zero =>
      change (stages steps).1.tape.map (fun tape =>
        ((stages steps).1.input tape, ((stages steps).1.results tape).take steps)) = _
      congr 1
      funext tape
      rw [List.take_of_length_le (results_length assumption closed adversary admitted steps tape).le]
  | succ extra induction =>
      rw [show steps + (extra + 1) = steps + extra + 1 from rfl, reverseStages_succ,
        Stage.next_marginal p _ _ steps (fun tape => by rw [results_length]; omega)]
      exact induction

theorem stage_event (steps extra : Nat)
    (event : Set (Output Key Claim Fresh Proof ClaimWitness × List (ClaimWitness × FreshWitness))) :
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
theorem start_event (total : Nat) (event : Set (Output Key Claim Fresh Proof ClaimWitness)) :
    adversary.tape.toOuterMeasure {tape | adversary.output tape ∈ event} =
      (stages total).1.tape.toOuterMeasure {tape | (stages total).1.input tape ∈ event} := by
  have moved := stage_event assumption closed adversary admitted 0 total {pair | pair.1 ∈ event}
  rw [Nat.zero_add] at moved
  exact moved.symm

/-- Stage `j`'s failure, read on a later stage, is stage `j`'s extraction failure. -/
theorem failure_term (j extra : Nat) :
    (stages (j + 1 + extra)).1.tape.toOuterMeasure
        {tape | StageFails p cap folds Holds FreshHolds ((stages (j + 1 + extra)).1.input tape)
          ((stages (j + 1 + extra)).1.results tape) j} =
      (extractors j).law.toOuterMeasure
        {draw | ExtractionFails p cap folds Holds FreshHolds _ (extractors j) draw} := by
  let event : Set (Output Key Claim Fresh Proof ClaimWitness × List (ClaimWitness × FreshWitness)) :=
    {pair | StageFails p cap folds Holds FreshHolds pair.1 pair.2 j}
  have taken : {tape | StageFails p cap folds Holds FreshHolds ((stages (j + 1 + extra)).1.input tape)
        ((stages (j + 1 + extra)).1.results tape) j} =
      {tape | ((stages (j + 1 + extra)).1.input tape,
        ((stages (j + 1 + extra)).1.results tape).take (j + 1)) ∈ event} := by
    ext tape
    exact fails_take _ _ _ _ j
  rw [taken, stage_event assumption closed adversary admitted (j + 1) extra event]
  rw [reverseStages_succ]
  congr 1
  ext draw
  have length := results_length assumption closed adversary admitted j draw.1
  simp only [Set.mem_ofPred_eq, event, StageFails, Stage.next, ExtractionFails, Stage.nifs, Fails]
  have sameAttempt : ∀ (output : Output Key Claim Fresh Proof ClaimWitness),
      (attempts p output.bound output.witness output.back
          ((stages j).1.results draw.1 ++ [(extractors j).run draw.1 draw.2]))[j]? =
        (attempts p output.bound output.witness output.back
          ((stages j).1.results draw.1))[((stages j).1.results draw.1).length]? := by
    intro output
    rw [attempts_take, List.take_append_of_le_length (by omega), List.take_of_length_le (by omega),
      length]
  have sameAnswer : ((stages j).1.results draw.1 ++ [(extractors j).run draw.1 draw.2])[j]? =
      some ((extractors j).run draw.1 draw.2) := by
    rw [List.getElem?_append_right (by omega), length, Nat.sub_self]
    rfl
  rw [sameAttempt, sameAnswer]
  constructor
  · rintro ⟨attempt, answer, attemptEq, answerEq, wins, fails⟩
    cases answerEq
    exact ⟨attempt, attemptEq, wins, fails⟩
  · rintro ⟨attempt, attemptEq, wins, fails⟩
    exact ⟨attempt, _, attemptEq, rfl, wins, fails⟩

theorem input_bound (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).back.length ≤ depth)
    (steps : Nat) (tape : (stages steps).1.Tape) (supported : tape ∈ (stages steps).1.tape.support) :
    ((stages steps).1.input tape).back.length ≤ depth := by
  have mapped := reverseStages_marginal assumption closed adversary admitted 0 steps
  rw [Nat.zero_add] at mapped
  have member : ((stages steps).1.input tape, ((stages steps).1.results tape).take 0) ∈
      ((stages 0).1.tape.map (fun tape => ((stages 0).1.input tape, (stages 0).1.results tape))).support := by
    rw [← mapped]
    exact (PMF.mem_support_map_iff _ _ _).mpr ⟨tape, supported, rfl⟩
  obtain ⟨origin, originSupported, same⟩ := (PMF.mem_support_map_iff _ _ _).mp member
  have inputs := congrArg Prod.fst same
  change adversary.output origin = (stages steps).1.input tape at inputs
  rw [← inputs]
  exact depthBound origin originSupported

/-- On each supported tape, an accepted history that is not returned has a failed stage. -/
theorem per_tape (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).back.length ≤ depth)
    (tape : (stages depth).1.Tape) (supported : tape ∈ (stages depth).1.tape.support)
    (accepted : Accepted p cap folds Holds ((stages depth).1.input tape))
    (missing : ¬ HistoryReturned p Holds FreshHolds ((stages depth).1.input tape)
      ((stages depth).1.results tape)) :
    ∃ j : Fin depth, StageFails p cap folds Holds FreshHolds ((stages depth).1.input tape)
      ((stages depth).1.results tape) j.val := by
  have historyLength := input_bound assumption closed adversary admitted depth depthBound depth tape
    supported
  have resultsLength := results_length assumption closed adversary admitted depth tape
  obtain ⟨runs, holds, atMostB⟩ := accepted
  refine Classical.byContradiction fun none => missing ?_
  refine returned_of_no_failure _ _ _ _ _ runs holds (by omega) (by omega) fun j small failed => ?_
  exact none ⟨⟨j, by omega⟩, failed⟩

/-- HyperNova Lemma 17 for one slot of option B: the chance that the slot's history is accepted
but the reverse extractor does not return it is at most the stages' Assumption 1 errors. -/
theorem history_failure_le (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).back.length ≤ depth) :
    ((stages depth).1.tape.toOuterMeasure
        {tape | Accepted p cap folds Holds ((stages depth).1.input tape) ∧
          ¬ HistoryReturned p Holds FreshHolds ((stages depth).1.input tape)
            ((stages depth).1.results tape)}).toReal ≤
      ∑ j : Fin depth, error ((stages j.val).1.nifs p) := by
  let law := (stages depth).1.tape
  let failures : Fin depth → Set (stages depth).1.Tape := fun j =>
    {tape | StageFails p cap folds Holds FreshHolds ((stages depth).1.input tape)
      ((stages depth).1.results tape) j.val}
  have covered : {tape | Accepted p cap folds Holds ((stages depth).1.input tape) ∧
        ¬ HistoryReturned p Holds FreshHolds ((stages depth).1.input tape)
          ((stages depth).1.results tape)} ∩ law.support ⊆ ⋃ j, failures j := by
    rintro tape ⟨⟨accepted, missing⟩, supported⟩
    obtain ⟨j, failed⟩ := per_tape assumption closed adversary admitted depth depthBound tape
      supported accepted missing
    exact Set.mem_iUnion.mpr ⟨j, failed⟩
  have termEq : ∀ j : Fin depth, law.toOuterMeasure (failures j) =
      (extractors j.val).law.toOuterMeasure
        {draw | ExtractionFails p cap folds Holds FreshHolds _ (extractors j.val) draw} := by
    intro j
    obtain ⟨extra, total⟩ := Nat.exists_eq_add_of_le (Nat.succ_le_of_lt j.isLt)
    have moved := failure_term assumption closed adversary admitted j.val extra
    rw [← total] at moved
    exact moved
  have bound : law.toOuterMeasure
      {tape | Accepted p cap folds Holds ((stages depth).1.input tape) ∧
        ¬ HistoryReturned p Holds FreshHolds ((stages depth).1.input tape)
          ((stages depth).1.results tape)} ≤
      ∑ j : Fin depth, (extractors j.val).law.toOuterMeasure
        {draw | ExtractionFails p cap folds Holds FreshHolds _ (extractors j.val) draw} := by
    rw [← PMF.toOuterMeasure_apply_inter_support]
    calc _ ≤ law.toOuterMeasure (⋃ j, failures j) := MeasureTheory.measure_mono covered
      _ ≤ ∑ j, law.toOuterMeasure (failures j) := MeasureTheory.measure_iUnion_fintype_le _ _
      _ = _ := Finset.sum_congr rfl fun j _ => termEq j
  have finiteTerm : ∀ j : Fin depth, (extractors j.val).law.toOuterMeasure
      {draw | ExtractionFails p cap folds Holds FreshHolds _ (extractors j.val) draw} ≠ ∞ :=
    fun j => event_ne_top _ _
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono finiteSum bound
  rw [ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  exact realBound.trans (Finset.sum_le_sum fun j _ =>
    reverseExtractor_error assumption closed adversary admitted j.val)

section TwoSlot

open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold (Accepts)
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge
  (Runs Execution MachineStep slotHistory slot_runs)

/-- Each fold of a forward history with the running claim it starts from. -/
def links : Running Claim → List (Fold Key Claim Fresh Proof) → List (Link Key Claim Fresh Proof)
  | _, [] => []
  | start, fold :: rest => (start, fold) :: links fold.output rest

/-- The running claim the oldest fold of a history, last fold first, starts from. -/
def oldest : Running Claim → List (Link Key Claim Fresh Proof) → Running Claim
  | final, [] => final
  | _, link :: older => oldest link.1 older

theorem runsBack_append :
    ∀ (back : List (Link Key Claim Fresh Proof)) (final : Running Claim)
      (link : Link Key Claim Fresh Proof),
      RunsBack cap folds final back → link.2.output = oldest final back →
      link.2.input = link.1.1 →
      Accepts (verifier cap folds link.2.circuit) link.2.key link.1 link.2.fresh link.2.proof
        link.2.output →
      RunsBack cap folds final (back ++ [link]) ∧ oldest final (back ++ [link]) = link.1
  | [], _, _, _, output, input, accepted => ⟨⟨output, input, accepted, trivial⟩, rfl⟩
  | newer :: older, _, link, ⟨outputEq, inputEq, acceptedNewer, olderRuns⟩, output, input,
      accepted => by
      obtain ⟨runs, oldestEq⟩ := runsBack_append older newer.1 link olderRuns output input accepted
      exact ⟨⟨outputEq, inputEq, acceptedNewer, runs⟩, oldestEq⟩

/-- A forward history of accepted folds, read last fold first. -/
theorem runsBack_of_runs :
    ∀ (history : List (Fold Key Claim Fresh Proof)) (start final : Running Claim),
      Runs cap folds start history final →
        RunsBack cap folds final (links start history).reverse ∧
          oldest final (links start history).reverse = start
  | [], start, final, runs => by
      simp only [Runs] at runs
      subst runs
      exact ⟨trivial, rfl⟩
  | fold :: rest, start, final, ⟨inputEq, accepted, later⟩ => by
      obtain ⟨runs, oldestEq⟩ := runsBack_of_runs rest fold.output final later
      simp only [links, List.reverse_cons]
      exact runsBack_append _ final (start, fold) runs oldestEq.symm inputEq accepted

variable {Digest State StepWitness Encoded : Type}

/-- The one-slot output of a two-slot execution: the slot's history, last fold first, and a
witness of its final claim. -/
def slotOutput (verifierKeys : Fin 2 → Key) (start final : Fin 2 → Running Claim) (slot : Fin 2)
    (trace : List (MachineStep Key Claim Fresh Proof Digest State StepWitness)) (bound : Nat)
    (witness : ClaimWitness) : Output Key Claim Fresh Proof ClaimWitness where
  final := final slot
  back := (links (start slot) (slotHistory verifierKeys slot trace)).reverse
  bound := bound
  witness := witness

/-- An accepted two-slot execution and a decider check of one slot's final claim below `B`
make that slot's output accepted, so `history_failure_le` bounds its extraction failure. -/
theorem slot_accepted (verifierKeys : Fin 2 → Key) (defaultClaim : Claim)
    (machine : Paper.Machine Key Digest State StepWitness (Running Claim) Fresh Encoded 2)
    (trace : List (MachineStep Key Claim Fresh Proof Digest State StepWitness))
    (start final : Fin 2 → Running Claim)
    (execution : Execution (setup cap folds verifierKeys defaultClaim) machine start trace final)
    (slot : Fin 2) (bound : Nat) (witness : ClaimWitness)
    (holds : Holds bound (final slot).1 witness) (atMostB : bound ≤ p.bigB) :
    Accepted p cap folds Holds (slotOutput verifierKeys start final slot trace bound witness) :=
  ⟨(runsBack_of_runs _ _ _
      (slot_runs verifierKeys defaultClaim machine slot trace start final execution)).1,
    holds, atMostB⟩

end TwoSlot

end NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthSecurity
