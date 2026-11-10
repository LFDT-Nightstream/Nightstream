import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthReverse
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge

/-! HyperNova Lemma 17 (Appendix H.3) for one slot of the bounded-depth scheme, with probabilities. Stage `0`
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

variable {Prior Key Claim Fresh Proof ClaimWitness FreshWitness : Type}

theorem event_ne_top {Sample : Type} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

/-- What the IVC adversary outputs for one slot: the slot's key, the history with the prior
states, last fold first, and a witness of the final claim at `bound`. -/
structure Output (Prior Key Claim Fresh Proof ClaimWitness : Type) where
  key : Key
  final : Running Claim
  back : List (Link Prior Key Claim Fresh Proof)
  bound : Nat
  witness : ClaimWitness

/-- HyperNova Definition 7 adversary for one fold. -/
structure NifsAdversary (Prior Key Claim Fresh Proof ClaimWitness : Type) where
  Tape : Type
  tape : PMF Tape
  run : Tape → Option (Attempt Prior Key Claim Fresh Proof ClaimWitness)

structure NifsExtractor (FreshWitness : Type)
    (adversary : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness) where
  Coins : Type
  coins : PMF Coins
  run : adversary.Tape → Coins → ClaimWitness × FreshWitness

noncomputable def NifsExtractor.law {adversary : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness}
    (extractor : NifsExtractor FreshWitness adversary) : PMF (adversary.Tape × extractor.Coins) :=
  adversary.tape.bind fun tape => extractor.coins.map (tape, ·)

structure IvcAdversary (Prior Key Claim Fresh Proof ClaimWitness : Type) where
  Tape : Type
  tape : PMF Tape
  output : Tape → Output Prior Key Claim Fresh Proof ClaimWitness

structure Stage (Prior Key Claim Fresh Proof ClaimWitness FreshWitness : Type) where
  Tape : Type
  tape : PMF Tape
  input : Tape → Output Prior Key Claim Fresh Proof ClaimWitness
  results : Tape → List (ClaimWitness × FreshWitness)

section Events

variable (p : GlobalParams) (cap : Nat) (folds : Folds Key Claim Fresh Proof)
  (PriorLink : Prior → Running Claim → Fold Key Claim Fresh Proof → Prop)
  (Holds : Key → Nat → Claim → ClaimWitness → Prop) (FreshHolds : Key → Fresh → FreshWitness → Prop)

/-- The verifier accepts the slot's history, each fold linked to its output prior state, and the
decider accepts the final claim below `B`. -/
def Accepted (output : Output Prior Key Claim Fresh Proof ClaimWitness) : Prop :=
  RunsBack cap folds PriorLink output.key output.final output.back ∧
    Holds output.key output.bound output.final.1 output.witness ∧ output.bound ≤ p.bigB

def ExtractionFails (adversary : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness)
    (extractor : NifsExtractor FreshWitness adversary) (draw : adversary.Tape × extractor.Coins) :
    Prop :=
  ∃ attempt, adversary.run draw.1 = some attempt ∧ Succeeds p cap folds PriorLink Holds attempt ∧
    ¬ Extracted p Holds FreshHolds attempt (extractor.run draw.1 draw.2)

/-- HyperNova errata Assumption 1 in Definition 7 form for the folds of the bounded-depth scheme. -/
def Assumption1 (Admitted : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness → Prop)
    (Efficient : (adversary : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness) →
      NifsExtractor FreshWitness adversary → Prop)
    (error : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness → ℝ) : Prop :=
  ∀ adversary, Admitted adversary → ∃ extractor : NifsExtractor FreshWitness adversary,
    Efficient adversary extractor ∧
      (extractor.law.toOuterMeasure
        {draw | ExtractionFails p cap folds PriorLink Holds FreshHolds adversary extractor draw}).toReal ≤
        error adversary

/-- Stage `j` of the reverse extractor fails on its fold. -/
def StageFails (output : Output Prior Key Claim Fresh Proof ClaimWitness)
    (results : List (ClaimWitness × FreshWitness)) (j : Nat) : Prop :=
  Fails p cap folds PriorLink Holds FreshHolds output.bound output.witness output.back results j

/-- The answers extract every fold of the slot's history. -/
def HistoryReturned (output : Output Prior Key Claim Fresh Proof ClaimWitness)
    (results : List (ClaimWitness × FreshWitness)) : Prop :=
  Returned p Holds FreshHolds output.bound output.witness output.back results

namespace Stage

/-- Stage `j`'s NIFS adversary: the `j`-th fold from the end, with stage `j − 1`'s answer. -/
def nifs (stage : Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness) :
    NifsAdversary Prior Key Claim Fresh Proof ClaimWitness where
  Tape := stage.Tape
  tape := stage.tape
  run tape := (attempts p (stage.input tape).bound (stage.input tape).witness
    (stage.input tape).back (stage.results tape))[(stage.results tape).length]?

def start (adversary : IvcAdversary Prior Key Claim Fresh Proof ClaimWitness) :
    Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness where
  Tape := adversary.Tape
  tape := adversary.tape
  input := adversary.output
  results _ := []

@[implicit_reducible]
noncomputable def next (stage : Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness)
    (extractor : NifsExtractor FreshWitness (stage.nifs p)) :
    Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness where
  Tape := stage.Tape × extractor.Coins
  tape := extractor.law
  input draw := stage.input draw.1
  results draw := stage.results draw.1 ++ [extractor.run draw.1 draw.2]

theorem next_marginal (stage : Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness)
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
structure Closed (Admitted : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness → Prop)
    (StageAdmitted : Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness → Prop)
    (Efficient : (adversary : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness) →
      NifsExtractor FreshWitness adversary → Prop) : Prop where
  nifs : ∀ stage, StageAdmitted stage → Admitted (stage.nifs p)
  next : ∀ (stage : Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness)
      (extractor : NifsExtractor FreshWitness (stage.nifs p)),
    StageAdmitted stage → Efficient (stage.nifs p) extractor →
      StageAdmitted (stage.next p extractor)

end Events

variable {p : GlobalParams} {cap : Nat} {folds : Folds Key Claim Fresh Proof}
  {PriorLink : Prior → Running Claim → Fold Key Claim Fresh Proof → Prop}
  {Holds : Key → Nat → Claim → ClaimWitness → Prop} {FreshHolds : Key → Fresh → FreshWitness → Prop}
  {Admitted : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness → Prop}
  {StageAdmitted : Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness → Prop}
  {Efficient : (adversary : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness) →
    NifsExtractor FreshWitness adversary → Prop}
  {error : NifsAdversary Prior Key Claim Fresh Proof ClaimWitness → ℝ}
  (assumption : Assumption1 p cap folds PriorLink Holds FreshHolds Admitted Efficient error)
  (closed : Closed p Admitted StageAdmitted Efficient)
  (adversary : IvcAdversary Prior Key Claim Fresh Proof ClaimWitness)
  (admitted : StageAdmitted (Stage.start adversary))

/-- HyperNova Lemma 17. -/
noncomputable def reverseStages :
    Nat → {stage : Stage Prior Key Claim Fresh Proof ClaimWitness FreshWitness // StageAdmitted stage}
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
        {draw | ExtractionFails p cap folds PriorLink Holds FreshHolds _
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
    (event : Set (Output Prior Key Claim Fresh Proof ClaimWitness × List (ClaimWitness × FreshWitness))) :
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
theorem start_event (total : Nat) (event : Set (Output Prior Key Claim Fresh Proof ClaimWitness)) :
    adversary.tape.toOuterMeasure {tape | adversary.output tape ∈ event} =
      (stages total).1.tape.toOuterMeasure {tape | (stages total).1.input tape ∈ event} := by
  have moved := stage_event assumption closed adversary admitted 0 total {pair | pair.1 ∈ event}
  rw [Nat.zero_add] at moved
  exact moved.symm

/-- Stage `j`'s failure, read on a later stage, is stage `j`'s extraction failure. -/
theorem failure_term (j extra : Nat) :
    (stages (j + 1 + extra)).1.tape.toOuterMeasure
        {tape | StageFails p cap folds PriorLink Holds FreshHolds ((stages (j + 1 + extra)).1.input tape)
          ((stages (j + 1 + extra)).1.results tape) j} =
      (extractors j).law.toOuterMeasure
        {draw | ExtractionFails p cap folds PriorLink Holds FreshHolds _ (extractors j) draw} := by
  let event : Set (Output Prior Key Claim Fresh Proof ClaimWitness × List (ClaimWitness × FreshWitness)) :=
    {pair | StageFails p cap folds PriorLink Holds FreshHolds pair.1 pair.2 j}
  have taken : {tape | StageFails p cap folds PriorLink Holds FreshHolds ((stages (j + 1 + extra)).1.input tape)
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
  have sameAttempt : ∀ (output : Output Prior Key Claim Fresh Proof ClaimWitness),
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
    (accepted : Accepted p cap folds PriorLink Holds ((stages depth).1.input tape))
    (missing : ¬ HistoryReturned p Holds FreshHolds ((stages depth).1.input tape)
      ((stages depth).1.results tape)) :
    ∃ j : Fin depth, StageFails p cap folds PriorLink Holds FreshHolds ((stages depth).1.input tape)
      ((stages depth).1.results tape) j.val := by
  have historyLength := input_bound assumption closed adversary admitted depth depthBound depth tape
    supported
  have resultsLength := results_length assumption closed adversary admitted depth tape
  obtain ⟨runs, holds, atMostB⟩ := accepted
  refine Classical.byContradiction fun none => missing ?_
  refine returned_of_no_failure _ _ _ _ _ _ runs holds (by omega) (by omega) fun j small failed => ?_
  exact none ⟨⟨j, by omega⟩, failed⟩

/-- HyperNova Lemma 17 for one slot of the bounded-depth scheme: the chance that the slot's history is accepted
but the reverse extractor does not return it is at most the stages' Assumption 1 errors. -/
theorem history_failure_le (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).back.length ≤ depth) :
    ((stages depth).1.tape.toOuterMeasure
        {tape | Accepted p cap folds PriorLink Holds ((stages depth).1.input tape) ∧
          ¬ HistoryReturned p Holds FreshHolds ((stages depth).1.input tape)
            ((stages depth).1.results tape)}).toReal ≤
      ∑ j : Fin depth, error ((stages j.val).1.nifs p) := by
  let law := (stages depth).1.tape
  let failures : Fin depth → Set (stages depth).1.Tape := fun j =>
    {tape | StageFails p cap folds PriorLink Holds FreshHolds ((stages depth).1.input tape)
      ((stages depth).1.results tape) j.val}
  have covered : {tape | Accepted p cap folds PriorLink Holds ((stages depth).1.input tape) ∧
        ¬ HistoryReturned p Holds FreshHolds ((stages depth).1.input tape)
          ((stages depth).1.results tape)} ∩ law.support ⊆ ⋃ j, failures j := by
    rintro tape ⟨⟨accepted, missing⟩, supported⟩
    obtain ⟨j, failed⟩ := per_tape assumption closed adversary admitted depth depthBound tape
      supported accepted missing
    exact Set.mem_iUnion.mpr ⟨j, failed⟩
  have termEq : ∀ j : Fin depth, law.toOuterMeasure (failures j) =
      (extractors j.val).law.toOuterMeasure
        {draw | ExtractionFails p cap folds PriorLink Holds FreshHolds _ (extractors j.val) draw} := by
    intro j
    obtain ⟨extra, total⟩ := Nat.exists_eq_add_of_le (Nat.succ_le_of_lt j.isLt)
    have moved := failure_term assumption closed adversary admitted j.val extra
    rw [← total] at moved
    exact moved
  have bound : law.toOuterMeasure
      {tape | Accepted p cap folds PriorLink Holds ((stages depth).1.input tape) ∧
        ¬ HistoryReturned p Holds FreshHolds ((stages depth).1.input tape)
          ((stages depth).1.results tape)} ≤
      ∑ j : Fin depth, (extractors j.val).law.toOuterMeasure
        {draw | ExtractionFails p cap folds PriorLink Holds FreshHolds _ (extractors j.val) draw} := by
    rw [← PMF.toOuterMeasure_apply_inter_support]
    calc _ ≤ law.toOuterMeasure (⋃ j, failures j) := MeasureTheory.measure_mono covered
      _ ≤ ∑ j, law.toOuterMeasure (failures j) := MeasureTheory.measure_iUnion_fintype_le _ _
      _ = _ := Finset.sum_congr rfl fun j _ => termEq j
  have finiteTerm : ∀ j : Fin depth, (extractors j.val).law.toOuterMeasure
      {draw | ExtractionFails p cap folds PriorLink Holds FreshHolds _ (extractors j.val) draw} ≠ ∞ :=
    fun j => event_ne_top _ _
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono finiteSum bound
  rw [ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  exact realBound.trans (Finset.sum_le_sum fun j _ =>
    reverseExtractor_error assumption closed adversary admitted j.val)

section TwoSlot

open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold (Accepts)
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge
  (Execution MachineStep slotFold)

variable {Digest State StepWitness Encoded : Type}

/-- HyperNova Construction 2 step 4(b) for one fold: the fold's fresh instance encodes the hash
of `prior`, and the program counter of `prior` selects a slot that holds `start`. -/
def StepLink (machine : Paper.Machine Key Digest State StepWitness (Running Claim) Fresh Encoded 2)
    (prior : Paper.HashPreimage Key State (Running Claim) 2) (start : Running Claim)
    (fold : Fold Key Claim Fresh Proof) : Prop :=
  machine.freshPublic fold.fresh = machine.encodeInstance (machine.hash prior) ∧
    ∃ valid : Paper.InRange 2 prior.pc, prior.running (Paper.selectedIndex valid) = start

/-- The hash preimage that a step's fresh instance commits to (Construction 2 step 4(b)). -/
def priorOf (verifierKeys : Fin 2 → Key)
    (step : MachineStep Key Claim Fresh Proof Digest State StepWitness) :
    Paper.HashPreimage Key State (Running Claim) 2 where
  verifierKeys := verifierKeys
  iteration := step.input.iteration
  z0 := step.input.z0
  current := step.input.zi
  running := step.input.running
  pc := step.input.priorPc

/-- The links `slot` receives: one for each step whose prior program counter selects it, with
the step's prior hash preimage and the slot's claim before the fold. -/
def slotLinks (verifierKeys : Fin 2 → Key) (slot : Fin 2) :
    List (MachineStep Key Claim Fresh Proof Digest State StepWitness) →
      List (Link (Paper.HashPreimage Key State (Running Claim) 2) Key Claim Fresh Proof)
  | [] => []
  | step :: rest =>
      if step.input.priorPc = slot.val + 1 then
        ⟨priorOf verifierKeys step, step.input.running slot, slotFold verifierKeys slot step⟩ ::
          slotLinks verifierKeys slot rest
      else slotLinks verifierKeys slot rest

/-- The running claim the oldest fold of a history, last fold first, starts from. -/
def oldest : Running Claim → List (Link Prior Key Claim Fresh Proof) → Running Claim
  | final, [] => final
  | _, link :: older => oldest link.start older

theorem runsBack_append (key : Key) :
    ∀ (back : List (Link Prior Key Claim Fresh Proof)) (final : Running Claim)
      (link : Link Prior Key Claim Fresh Proof),
      RunsBack cap folds PriorLink key final back → link.fold.output = oldest final back →
      link.fold.input = link.start.1 → link.fold.key = key →
      PriorLink link.prior link.start link.fold →
      Accepts (verifier cap folds link.fold.circuit) link.fold.key link.start link.fold.fresh
        link.fold.proof link.fold.output →
      RunsBack cap folds PriorLink key final (back ++ [link]) ∧
        oldest final (back ++ [link]) = link.start
  | [], _, _, _, output, input, keyEq, linked, accepted =>
      ⟨⟨output, input, keyEq, linked, accepted, trivial⟩, rfl⟩
  | newer :: older, _, link, ⟨outputEq, inputEq, keyNewer, linkedNewer, acceptedNewer, olderRuns⟩,
      output, input, keyEq, linked, accepted => by
      obtain ⟨runs, oldestEq⟩ :=
        runsBack_append key older newer.start link olderRuns output input keyEq linked accepted
      exact ⟨⟨outputEq, inputEq, keyNewer, linkedNewer, acceptedNewer, runs⟩, oldestEq⟩

/-- Each slot of an accepted execution receives an accepted history, last fold first, under the
slot's key, with each fold linked to its step's prior hash preimage. -/
theorem slot_runsBack (verifierKeys : Fin 2 → Key) (defaultClaim : Claim)
    (machine : Paper.Machine Key Digest State StepWitness (Running Claim) Fresh Encoded 2)
    (slot : Fin 2) :
    ∀ (trace : List (MachineStep Key Claim Fresh Proof Digest State StepWitness))
      (start final : Fin 2 → Running Claim),
      Execution (setup cap folds verifierKeys defaultClaim) machine start trace final →
        RunsBack cap folds (StepLink machine) (verifierKeys slot) (final slot)
            (slotLinks verifierKeys slot trace).reverse ∧
          oldest (final slot) (slotLinks verifierKeys slot trace).reverse = start slot
  | [], start, final, execution => by
      simp only [Execution] at execution
      subst execution
      exact ⟨trivial, rfl⟩
  | step :: rest, start, final, ⟨runningEq, recursive, holds, later⟩ => by
      obtain ⟨runs, oldestEq⟩ := slot_runsBack verifierKeys defaultClaim machine slot rest
        step.output.runningNext final later
      have holdsCopy := holds
      obtain ⟨_, _, _, _, base | ⟨priorPcValid, _, priorPublic, accepted, unchanged⟩⟩ := holdsCopy
      · omega
      have lower := priorPcValid.1
      by_cases selects : step.input.priorPc = slot.val + 1
      · have same : Paper.selectedIndex priorPcValid = slot := by
          apply Fin.ext
          simp only [Paper.selectedIndex]
          omega
        rw [same] at accepted
        simp only [slotLinks, selects, ↓reduceIte, List.reverse_cons]
        have linked : StepLink machine (priorOf verifierKeys step) (step.input.running slot)
            (slotFold verifierKeys slot step) := ⟨priorPublic, priorPcValid, by rw [← same]; rfl⟩
        obtain ⟨appended, appendedOldest⟩ := runsBack_append (verifierKeys slot) _ (final slot)
          ⟨priorOf verifierKeys step, step.input.running slot, slotFold verifierKeys slot step⟩
          runs oldestEq.symm rfl rfl linked accepted
        exact ⟨appended, appendedOldest.trans (congrFun runningEq slot)⟩
      · have other : Paper.selectedIndex priorPcValid ≠ slot := by
          intro equal
          apply selects
          have value := congrArg Fin.val equal
          simp only [Paper.selectedIndex] at value
          omega
        have kept : step.output.runningNext slot = start slot :=
          (unchanged slot (Ne.symm other)).trans (congrFun runningEq slot)
        simp only [slotLinks, selects, ↓reduceIte]
        rw [kept] at oldestEq
        exact ⟨runs, oldestEq⟩

/-- The one-slot output of a two-slot execution: the slot's key, its history with the prior
hash preimages, last fold first, and a witness of its final claim. -/
def slotOutput (verifierKeys : Fin 2 → Key) (final : Fin 2 → Running Claim) (slot : Fin 2)
    (trace : List (MachineStep Key Claim Fresh Proof Digest State StepWitness)) (bound : Nat)
    (witness : ClaimWitness) :
    Output (Paper.HashPreimage Key State (Running Claim) 2) Key Claim Fresh Proof ClaimWitness where
  key := verifierKeys slot
  final := final slot
  back := (slotLinks verifierKeys slot trace).reverse
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
    (holds : Holds (verifierKeys slot) bound (final slot).1 witness) (atMostB : bound ≤ p.bigB) :
    Accepted p cap folds (StepLink machine) Holds
      (slotOutput verifierKeys final slot trace bound witness) :=
  ⟨(slot_runsBack verifierKeys defaultClaim machine slot trace start final execution).1,
    holds, atMostB⟩

end TwoSlot

end NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthSecurity
