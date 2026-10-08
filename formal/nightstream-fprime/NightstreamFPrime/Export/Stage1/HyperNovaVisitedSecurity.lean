import NightstreamFPrime.Export.Stage1.HyperNovaFirstFailure

/-!
Owns the history security bound of the selected HyperNova IVC under HyperNova
errata Assumption 1, stated as Definition 7 knowledge soundness of the
Poseidon2 NIFS, and the reverse extractor of HyperNova Lemma 17 (Appendix H.3).

Inputs:
- `Assumption1 Admitted Efficient error`: for every admitted NIFS adversary
  there is an efficient extractor that reads the adversary's tape and its own
  coins (so it may rerun the adversary), and fails after a real NIFS success
  with probability at most `error` of that adversary;
- `Closed Admitted Efficient`: one reverse step of an admitted algorithm with
  an efficient extractor is again admitted;
- an admitted IVC adversary and a bound on its advertised iteration.

Outputs:
- `reverseStages`: stage `j + 1` is stage `j` followed by the extractor that
  Assumption 1 gives for stage `j`, as in Lemma 17. Every stage is admitted;
- `history_probability_bound`: the accepted terminal mass is at most the
  reverse extractor's returned-history mass plus, at each stage, the marked
  hash-collision mass of that stage and the Assumption 1 error of that stage.

`Admitted` and `Efficient` are abstract. Their intended meaning is expected
polynomial time; Lean states no running-time model. Assumption 1 is not a
theorem here. `Lifecycle.RandomOracleKnowledge` proves a random-oracle
analogue for one fold, which motivates the value of `error`; the step circuit
recomputes the previous challenges with Poseidon2, so no random-oracle model
covers the history.

Does not own: numerical hardness, collision resistance of the state hash, or
query applicability.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open HyperNovaHistory (Statement Envelope Payload SourceResult)
open HyperNovaHistoryProbability (Sample AdviceReturned)
open HyperNovaVisitedLaw (Visit visitedLaw guardedDraw goodActive pathVisit listSource)
open HyperNovaFirstFailure (MarkedHashCollision MarkedSourceFailure)
open HyperNovaGuardedSourceLaw (inputs realOutput)
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

attribute [local instance] Classical.propDecidable

/-- Every event of a PMF has finite mass. -/
theorem event_ne_top {Sample : Type*} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

/-! ## HyperNova errata Assumption 1 -/

/-- A plain-model adversary against the production NIFS (HyperNova
Definition 7): its random tape, and the NIFS source input and real output
(prior preimage, proof and child witnesses) that it computes from the tape. -/
structure NifsAdversary where
  Tape : Type
  tape : PMF Tape
  run : Tape → PiCCSInputCheck.Input × Option (NifsRealSuccess.RealOutput PiDECInputCheck.relation)

/-- An extractor for one adversary: its own coins, and a source result
computed from the adversary's tape and those coins. It is chosen for the
adversary, so it may rerun it on its tape. -/
structure NifsExtractor (adversary : NifsAdversary) where
  Coins : Type
  coins : PMF Coins
  run : adversary.Tape → Coins → SourceResult

/-- The adversary's tape and the extractor's independent coins. -/
noncomputable def NifsExtractor.law {adversary : NifsAdversary} (extractor : NifsExtractor adversary) :
    PMF (adversary.Tape × extractor.Coins) :=
  adversary.tape.bind fun tape => extractor.coins.map (tape, ·)

/-- The NIFS really succeeds (`NifsRealSuccess.RealSuccess`, with the
prior-state link), and the extractor returns no checked source witness. -/
def ExtractionFails (adversary : NifsAdversary) (extractor : NifsExtractor adversary)
    (draw : adversary.Tape × extractor.Coins) : Prop :=
  NifsRealSuccess.RealSuccess PiDECInputCheck.relation productionAjtaiKey
      (PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup)
      (PiCCSInputCheck.running (adversary.run draw.1).1) (PiCCSInputCheck.fresh (adversary.run draw.1).1)
      (adversary.run draw.1).2 ∧
    ¬ CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
      (PiCCSStoredWitnessCheck.statement (adversary.run draw.1).1) (extractor.run draw.1 draw.2)

/-- HyperNova errata Assumption 1, as Definition 7 knowledge soundness of the
non-interactive NIFS: every admitted adversary has an efficient extractor
whose failure after a real success is at most `error` of that adversary. -/
def Assumption1 (Admitted : NifsAdversary → Prop)
    (Efficient : (adversary : NifsAdversary) → NifsExtractor adversary → Prop)
    (error : NifsAdversary → ℝ) : Prop :=
  ∀ adversary, Admitted adversary → ∃ extractor : NifsExtractor adversary,
    Efficient adversary extractor ∧
      (extractor.law.toOuterMeasure {draw | ExtractionFails adversary extractor draw}).toReal ≤
        error adversary

/-! ## The reverse extractor -/

/-- A plain-model adversary against the selected IVC: its random tape, and
the terminal statement and proof envelope that it outputs. -/
structure IvcAdversary where
  Tape : Type
  tape : PMF Tape
  output : Tape → Statement × Envelope

/-- The reverse extractor after some steps, as one algorithm with an explicit
tape: the IVC adversary's tape, then the coins of each NIFS extractor. It
computes the terminal input and the source results returned so far. -/
structure Stage where
  Tape : Type
  tape : PMF Tape
  input : Tape → Statement × Envelope
  results : Tape → List SourceResult

namespace Stage

/-- The current visited context: the terminal, advanced by each result. -/
noncomputable def visit (stage : Stage) (tape : stage.Tape) : Visit :=
  pathVisit (stage.input tape) (stage.results tape) (stage.results tape).length

/-- The NIFS adversary of a stage: it outputs the decoded NIFS input and the
real output of its current context. -/
noncomputable def nifs (stage : Stage) : NifsAdversary where
  Tape := stage.Tape
  tape := stage.tape
  run tape := (inputs (stage.visit tape), realOutput (stage.visit tape))

/-- Before any extraction: the IVC adversary itself. -/
def start (adversary : IvcAdversary) : Stage where
  Tape := adversary.Tape
  tape := adversary.tape
  input := adversary.output
  results _ := []

/-- One reverse step: run the stage, then the extractor for its NIFS
adversary on its tape and fresh coins. -/
noncomputable def next (stage : Stage) (extractor : NifsExtractor stage.nifs) : Stage where
  Tape := stage.Tape × extractor.Coins
  tape := extractor.law
  input draw := stage.input draw.1
  results draw := stage.results draw.1 ++ [extractor.run draw.1 draw.2]

/-- The reverse program's law when it reads the stage's results. -/
noncomputable def reverseLaw (stage : Stage) : PMF Sample :=
  stage.tape.bind fun tape =>
    HyperNovaHistoryLaw.law (listSource (stage.input tape).1.iteration (stage.results tape))
      (PMF.pure (stage.input tape))

/-- One reverse step keeps the input and every earlier result. -/
theorem next_marginal (stage : Stage) (extractor : NifsExtractor stage.nifs) (count : Nat)
    (within : ∀ tape, count ≤ (stage.results tape).length) :
    (stage.next extractor).tape.map
        (fun draw => ((stage.next extractor).input draw, ((stage.next extractor).results draw).take count)) =
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

/-- One reverse step of an admitted algorithm with an efficient extractor is
again admitted. For a constant number of steps this is the paper's expected
polynomial-time composition. -/
def Closed (Admitted : NifsAdversary → Prop)
    (Efficient : (adversary : NifsAdversary) → NifsExtractor adversary → Prop) : Prop :=
  ∀ (stage : Stage) (extractor : NifsExtractor stage.nifs),
    Admitted stage.nifs → Efficient stage.nifs extractor → Admitted (stage.next extractor).nifs

variable {Admitted : NifsAdversary → Prop}
  {Efficient : (adversary : NifsAdversary) → NifsExtractor adversary → Prop}
  {error : NifsAdversary → ℝ}
  (assumption : Assumption1 Admitted Efficient error) (closed : Closed Admitted Efficient)
  (adversary : IvcAdversary) (admitted : Admitted (Stage.start adversary).nifs)

/-- HyperNova Lemma 17: stage `j + 1` runs stage `j` and then the extractor
that Assumption 1 gives for stage `j`'s NIFS adversary. -/
noncomputable def reverseStages : Nat → {stage : Stage // Admitted stage.nifs}
  | 0 => ⟨Stage.start adversary, admitted⟩
  | steps + 1 =>
      ⟨(reverseStages steps).1.next
          (Classical.choose (assumption _ (reverseStages steps).2)),
        closed _ _ (reverseStages steps).2
          (Classical.choose_spec (assumption _ (reverseStages steps).2)).1⟩

/-- The extractor that Assumption 1 gives for one stage. -/
noncomputable def reverseExtractor (steps : Nat) :
    NifsExtractor (reverseStages assumption closed adversary admitted steps).1.nifs :=
  Classical.choose (assumption _ (reverseStages assumption closed adversary admitted steps).2)

theorem reverseExtractor_error (steps : Nat) :
    ((reverseExtractor assumption closed adversary admitted steps).law.toOuterMeasure
        {draw | ExtractionFails _ (reverseExtractor assumption closed adversary admitted steps) draw}).toReal ≤
      error (reverseStages assumption closed adversary admitted steps).1.nifs :=
  (Classical.choose_spec (assumption _ (reverseStages assumption closed adversary admitted steps).2)).2

local notation "stages" => reverseStages assumption closed adversary admitted
local notation "extractors" => reverseExtractor assumption closed adversary admitted

theorem reverseStages_succ (steps : Nat) :
    (stages (steps + 1)).1 = (stages steps).1.next (extractors steps) := rfl

theorem results_length (steps : Nat) (tape : (stages steps).1.Tape) :
    ((stages steps).1.results tape).length = steps := by
  induction steps with
  | zero => rfl
  | succ steps induction =>
      change ((stages steps).1.results tape.1 ++ [(extractors steps).run tape.1 tape.2]).length =
        steps + 1
      rw [List.length_append, induction, List.length_singleton]

/-- Later stages keep the input and the earlier results of every stage. -/
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
        Stage.next_marginal _ _ steps (fun tape => by rw [results_length]; omega)]
      exact induction

/-- An event of the input and the first `steps` results has the same mass at
every later stage. -/
theorem stage_event (steps extra : Nat) (event : Set ((Statement × Envelope) × List SourceResult)) :
    (stages (steps + extra)).1.tape.toOuterMeasure
        {tape | ((stages (steps + extra)).1.input tape,
          ((stages (steps + extra)).1.results tape).take steps) ∈ event} =
      (stages steps).1.tape.toOuterMeasure
        {tape | ((stages steps).1.input tape, (stages steps).1.results tape) ∈ event} := by
  have mapped := congrArg (fun law => PMF.toOuterMeasure law event)
    (reverseStages_marginal assumption closed adversary admitted steps extra)
  simp only [PMF.toOuterMeasure_map_apply] at mapped
  exact mapped

/-- A weighted sum of point masses is the mass of the preimage. -/
private theorem sum_pure {Tape Value : Type} (law : PMF Tape) (value : Tape → Value) (event : Set Value) :
    ∑' tape, law tape * (PMF.pure (value tape)).toOuterMeasure event =
      law.toOuterMeasure {tape | value tape ∈ event} := by
  rw [← PMF.toOuterMeasure_bind_apply]
  change (law.bind (PMF.pure ∘ value)).toOuterMeasure event = _
  rw [PMF.bind_pure_comp, PMF.toOuterMeasure_map_apply]
  rfl

/-- Every tape that the last stage can draw has the advertised iteration of
some adversary output in the adversary's support. -/
theorem input_bound (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth)
    (steps : Nat) (tape : (stages steps).1.Tape) (supported : tape ∈ (stages steps).1.tape.support) :
    ((stages steps).1.input tape).1.iteration ≤ depth := by
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

/-- The hash term of the per-tape first-failure bound, averaged over the last
stage, is the marked hash-collision mass of stage `steps`. -/
theorem hash_term (steps extra : Nat) :
    ∑' tape, (stages (steps + extra)).1.tape tape *
        (PMF.pure (pathVisit ((stages (steps + extra)).1.input tape)
          ((stages (steps + extra)).1.results tape) steps)).toOuterMeasure
          {visit | MarkedHashCollision visit} =
      (stages steps).1.tape.toOuterMeasure {tape | MarkedHashCollision ((stages steps).1.visit tape)} := by
  rw [sum_pure]
  have event := stage_event assumption closed adversary admitted steps extra
    {pair | MarkedHashCollision (pathVisit pair.1 pair.2 steps)}
  have visitEq (tape : (stages steps).1.Tape) :
      (stages steps).1.visit tape = pathVisit ((stages steps).1.input tape) ((stages steps).1.results tape) steps := by
    simp only [Stage.visit, results_length]
  simp only [Set.mem_setOf_eq, pathVisit, List.take_take, min_self] at event ⊢
  simpa only [visitEq, pathVisit] using event

/-- At a stage's own context, a good active visit is a real NIFS success. -/
theorem realSuccess_of_goodActive (stage : Stage) (tape : stage.Tape)
    (good : goodActive (stage.visit tape)) :
    NifsRealSuccess.RealSuccess PiDECInputCheck.relation productionAjtaiKey
      (PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup)
      (PiCCSInputCheck.running (inputs (stage.visit tape))) (PiCCSInputCheck.fresh (inputs (stage.visit tape)))
      (realOutput (stage.visit tape)) := by
  have supported : stage.visit tape ∈
      (visitedLaw (listSource (stage.input tape).1.iteration (stage.results tape))
        (PMF.pure (stage.input tape)) (stage.results tape).length).support := by
    rw [HyperNovaVisitedLaw.visitedLaw_listSource _ _ _ le_rfl, PMF.support_pure]
    rfl
  exact (HyperNovaVisitedAcceptance.realSuccess_iff_goodActive _ _ _ _ supported).mpr good

/-- The source-failure term of the per-tape first-failure bound, averaged over
the last stage, is at most the Assumption 1 failure mass of stage `steps`. -/
theorem failure_term_le (steps extra : Nat) :
    ∑' tape, (stages (steps + 1 + extra)).1.tape tape *
        (PMF.pure (pathVisit ((stages (steps + 1 + extra)).1.input tape)
            ((stages (steps + 1 + extra)).1.results tape) steps,
          if goodActive (pathVisit ((stages (steps + 1 + extra)).1.input tape)
              ((stages (steps + 1 + extra)).1.results tape) steps)
          then ((stages (steps + 1 + extra)).1.results tape).getD steps none else none)).toOuterMeasure
          {draw | MarkedSourceFailure draw} ≤
      (extractors steps).law.toOuterMeasure
        {draw | ExtractionFails _ (extractors steps) draw} := by
  rw [sum_pure]
  let event : Set ((Statement × Envelope) × List SourceResult) :=
    {pair | goodActive (pathVisit pair.1 pair.2 steps) ∧
      ¬ CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
        (PiCCSStoredWitnessCheck.statement (inputs (pathVisit pair.1 pair.2 steps)))
        (pair.2.getD steps none)}
  have restrict (input : Statement × Envelope) (results : List SourceResult) :
      pathVisit input (results.take (steps + 1)) steps = pathVisit input results steps := by
    simp only [pathVisit, List.take_take, Nat.min_eq_left (Nat.le_succ steps)]
  have entry (results : List SourceResult) :
      (results.take (steps + 1)).getD steps none = results.getD steps none := by
    simp only [List.getD_eq_getElem?_getD, List.getElem?_take, Nat.lt_succ_self, if_true]
  have equal := stage_event assumption closed adversary admitted (steps + 1) extra event
  calc
    _ = (stages (steps + 1 + extra)).1.tape.toOuterMeasure
          {tape | ((stages (steps + 1 + extra)).1.input tape,
            ((stages (steps + 1 + extra)).1.results tape).take (steps + 1)) ∈ event} := by
        congr 1
        ext tape
        simp only [Set.mem_setOf_eq, MarkedSourceFailure, event, restrict, entry]
        constructor
        · rintro ⟨good, failed⟩
          exact ⟨good, by simpa only [if_pos good] using failed⟩
        · rintro ⟨good, failed⟩
          exact ⟨good, by simpa only [if_pos good] using failed⟩
    _ = (stages (steps + 1)).1.tape.toOuterMeasure
          {tape | ((stages (steps + 1)).1.input tape, (stages (steps + 1)).1.results tape) ∈ event} :=
        equal
    _ ≤ _ := by
        apply PMF.toOuterMeasure_mono
        intro draw member
        obtain ⟨⟨good, failed⟩, _⟩ := member
        have current : pathVisit ((stages (steps + 1)).1.input draw)
            ((stages (steps + 1)).1.results draw) steps = (stages steps).1.visit draw.1 := by
          change pathVisit ((stages steps).1.input draw.1)
            ((stages steps).1.results draw.1 ++ [(extractors steps).run draw.1 draw.2]) steps = _
          have length := results_length assumption closed adversary admitted steps draw.1
          simp only [Stage.visit, pathVisit, length,
            List.take_append_of_le_length (le_of_eq length.symm)]
        have returned : ((stages (steps + 1)).1.results draw).getD steps none =
            (extractors steps).run draw.1 draw.2 := by
          change ((stages steps).1.results draw.1 ++ [(extractors steps).run draw.1 draw.2]).getD steps none = _
          have length := results_length assumption closed adversary admitted steps draw.1
          rw [List.getD_append_right _ _ _ _ (le_of_eq length), length, Nat.sub_self]
          rfl
        dsimp only at good failed
        rw [current] at good failed
        rw [returned] at failed
        exact ⟨realSuccess_of_goodActive _ draw.1 good, failed⟩

theorem hash_term_total (steps total : Nat) (within : steps ≤ total) :
    ∑' tape, (stages total).1.tape tape *
        (PMF.pure (pathVisit ((stages total).1.input tape) ((stages total).1.results tape) steps)).toOuterMeasure
          {visit | MarkedHashCollision visit} =
      (stages steps).1.tape.toOuterMeasure {tape | MarkedHashCollision ((stages steps).1.visit tape)} := by
  obtain ⟨extra, rfl⟩ := Nat.exists_eq_add_of_le within
  exact hash_term assumption closed adversary admitted steps extra

theorem failure_term_total (steps total : Nat) (within : steps < total) :
    ∑' tape, (stages total).1.tape tape *
        (PMF.pure (pathVisit ((stages total).1.input tape) ((stages total).1.results tape) steps,
          if goodActive (pathVisit ((stages total).1.input tape) ((stages total).1.results tape) steps)
          then ((stages total).1.results tape).getD steps none else none)).toOuterMeasure
          {draw | MarkedSourceFailure draw} ≤
      (extractors steps).law.toOuterMeasure {draw | ExtractionFails _ (extractors steps) draw} := by
  obtain ⟨extra, rfl⟩ := Nat.exists_eq_add_of_le (Nat.succ_le_of_lt within)
  exact failure_term_le assumption closed adversary admitted steps extra

/-- The terminal law as an event of the last stage's input. -/
theorem start_event (total : Nat) (event : Set (Statement × Envelope)) :
    adversary.tape.toOuterMeasure {tape | adversary.output tape ∈ event} =
      ∑' tape, (stages total).1.tape tape *
        (PMF.pure ((stages total).1.input tape)).toOuterMeasure event := by
  rw [sum_pure]
  have moved := stage_event assumption closed adversary admitted 0 total {pair | pair.1 ∈ event}
  rw [Nat.zero_add] at moved
  exact moved.symm

/-- The per-tape first-failure bound of `HyperNovaFirstFailure`, read along
the deterministic path of one tape of the last stage. -/
theorem per_tape (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth)
    (tape : (stages depth).1.Tape) (supported : tape ∈ (stages depth).1.tape.support) :
    (PMF.pure ((stages depth).1.input tape)).toOuterMeasure {input |
        PerApplicationTerminal.Holds application fits productionSetup input.1 input.2} ≤
      (HyperNovaHistoryLaw.law (listSource ((stages depth).1.input tape).1.iteration
          ((stages depth).1.results tape)) (PMF.pure ((stages depth).1.input tape))).toOuterMeasure
          {sample | AdviceReturned sample} +
        ∑ j : Fin depth,
          ((PMF.pure (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val)).toOuterMeasure
              {visit | MarkedHashCollision visit} +
            (PMF.pure (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val,
              if goodActive (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val)
              then ((stages depth).1.results tape).getD j.val none else none)).toOuterMeasure
              {draw | MarkedSourceFailure draw}) := by
  have length := results_length assumption closed adversary admitted depth tape
  have first := HyperNovaFirstFailure.accepted_probability_le_first_failures
    (listSource ((stages depth).1.input tape).1.iteration ((stages depth).1.results tape))
    (PMF.pure ((stages depth).1.input tape)) depth (by
      intro input member
      rw [PMF.mem_support_pure_iff] at member
      rw [member]
      exact input_bound assumption closed adversary admitted depth depthBound depth tape supported)
  refine first.trans (le_of_eq ?_)
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [HyperNovaVisitedLaw.visitedLaw_listSource _ _ _ (by omega), PMF.pure_bind,
    HyperNovaVisitedLaw.guardedDraw_listSource _ _ _ (by omega)]

/-- History security under Assumption 1 (HyperNova Lemma 17). The accepted
terminal mass is at most the reverse extractor's returned-history mass plus,
at each stage, the marked hash-collision mass and the Assumption 1 error of
that stage's NIFS adversary. Every stage is admitted (`reverseStages`). -/
theorem history_probability_bound (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    (adversary.tape.toOuterMeasure {tape |
      PerApplicationTerminal.Holds application fits productionSetup
        (adversary.output tape).1 (adversary.output tape).2}).toReal ≤
      ((stages depth).1.reverseLaw.toOuterMeasure {sample | AdviceReturned sample}).toReal +
        ∑ j : Fin depth,
          (((stages j.val).1.tape.toOuterMeasure
              {tape | MarkedHashCollision ((stages j.val).1.visit tape)}).toReal +
            error (stages j.val).1.nifs) := by
  let law := (stages depth).1.tape
  let advice : (stages depth).1.Tape → ℝ≥0∞ := fun tape =>
    (HyperNovaHistoryLaw.law (listSource ((stages depth).1.input tape).1.iteration
      ((stages depth).1.results tape)) (PMF.pure ((stages depth).1.input tape))).toOuterMeasure
      {sample | AdviceReturned sample}
  let hash : Fin depth → (stages depth).1.Tape → ℝ≥0∞ := fun j tape =>
    (PMF.pure (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val)).toOuterMeasure
      {visit | MarkedHashCollision visit}
  let failure : Fin depth → (stages depth).1.Tape → ℝ≥0∞ := fun j tape =>
    (PMF.pure (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val,
      if goodActive (pathVisit ((stages depth).1.input tape) ((stages depth).1.results tape) j.val)
      then ((stages depth).1.results tape).getD j.val none else none)).toOuterMeasure
      {draw | MarkedSourceFailure draw}
  have bound : adversary.tape.toOuterMeasure {tape |
      PerApplicationTerminal.Holds application fits productionSetup
        (adversary.output tape).1 (adversary.output tape).2} ≤
      (stages depth).1.reverseLaw.toOuterMeasure {sample | AdviceReturned sample} +
        ∑ j : Fin depth,
          ((stages j.val).1.tape.toOuterMeasure {tape | MarkedHashCollision ((stages j.val).1.visit tape)} +
            (extractors j.val).law.toOuterMeasure {draw | ExtractionFails _ (extractors j.val) draw}) := by
    calc
      _ = ∑' tape, law tape * (PMF.pure ((stages depth).1.input tape)).toOuterMeasure
            {input | PerApplicationTerminal.Holds application fits productionSetup input.1 input.2} :=
          start_event assumption closed adversary admitted depth
            {input | PerApplicationTerminal.Holds application fits productionSetup input.1 input.2}
      _ ≤ ∑' tape, law tape * (advice tape + ∑ j : Fin depth, (hash j tape + failure j tape)) := by
          apply ENNReal.tsum_le_tsum
          intro tape
          by_cases zero : law tape = 0
          · simp only [zero, zero_mul, le_refl]
          · exact mul_le_mul_right (per_tape assumption closed adversary admitted depth depthBound tape
              ((PMF.mem_support_iff _ _).mpr zero)) _
      _ = ∑' tape, law tape * advice tape +
            ∑ j : Fin depth, (∑' tape, law tape * hash j tape + ∑' tape, law tape * failure j tape) := by
          simp only [mul_add, Finset.mul_sum]
          rw [ENNReal.tsum_add, Summable.tsum_finsetSum (fun _ _ => ENNReal.summable)]
          congr 1
          exact Finset.sum_congr rfl fun j _ => ENNReal.tsum_add
      _ ≤ _ := by
          refine add_le_add (le_of_eq ?_) (Finset.sum_le_sum fun j _ => add_le_add (le_of_eq ?_) ?_)
          · exact (PMF.toOuterMeasure_bind_apply _ _ _).symm
          · exact hash_term_total assumption closed adversary admitted j.val depth j.isLt.le
          · exact failure_term_total assumption closed adversary admitted j.val depth j.isLt
  have finiteTerm (j : Fin depth) :
      (stages j.val).1.tape.toOuterMeasure {tape | MarkedHashCollision ((stages j.val).1.visit tape)} +
        (extractors j.val).law.toOuterMeasure {draw | ExtractionFails _ (extractors j.val) draw} ≠ ∞ :=
    ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono (ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, finiteSum⟩) bound
  rw [ENNReal.toReal_add (event_ne_top _ _) finiteSum,
    ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  refine realBound.trans (add_le_add le_rfl (Finset.sum_le_sum fun j _ => ?_))
  rw [ENNReal.toReal_add (event_ne_top _ _) (event_ne_top _ _)]
  exact add_le_add le_rfl (reverseExtractor_error assumption closed adversary admitted j.val)

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
