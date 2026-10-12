import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
import NightstreamFPrime.Lifecycle.Nebula.EndToEnd

/-!
Owns the end-to-end security bound of a memory application (security note §5)
on the joint law of the memory statement that an adversary claims and the
HyperNova reverse extractor's sample.

* `experiment`: the extracted history is the advice that
  `HyperNovaHistory.run` returns, so every event reads data of the sample.
* `stage1Extraction`: Stage 1 extraction for the memory program, proved from
  `HyperNovaHistory.run_correct` with no premise. Its failure is the event
  that `HyperNovaVisitedSecurity.history_failure_le` bounds.
* `MemoryRound`: security note A6 in the form of HyperNova errata
  Assumption 1, for the stages that Stage 1 admits. It is not a theorem here.
* `adversaryTape`: the IVC adversary's tape inside a reverse stage's tape.
* `bad_le`: on the claim that the adversary's tape gives, the bad mass is at
  most the Stage 1 history terms, the state-link and chain collision masses,
  and the A6 error of the last stage.

Does not own: numerical values, collision resistance of Poseidon2, or the
running time of an admitted adversary.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NebulaMemorySecurity

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nebula
open NightstreamFPrime.Lifecycle.Nebula.EndToEnd (MemoryStatement Experiment Stage1Extraction)
open HyperNovaHistory (ValidAdvice)
open HyperNovaHistoryProbability (Accepted AdviceReturned)
open HyperNovaVisitedSecurity (Stage IvcAdversary NifsAdversary NifsExtractor Assumption1 Closed
  reverseStages event_ne_top)
open HyperNovaFirstFailure (MarkedHashCollision)

-- The adversary types do not read the application width; keep it opaque.
attribute [local irreducible] PerApplicationFixedPoint.logicalWidth

variable {p : Plan} (app : PortApp p)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 (MemoryApp.program p app))
  (setup : PerApplicationCanonicalPackage.CommitmentSetup (MemoryApp.program p app))

/-- A sample: the claimed memory statement and the reverse sample (terminal
statement, envelope, and source returns). -/
abbrev Sample := MemoryStatement app.width × HyperNovaHistoryProbability.Sample (MemoryApp.program p app)

/-- The advice that the reverse program returns; empty after a failure. -/
noncomputable def advice (s : HyperNovaHistoryProbability.Sample (MemoryApp.program p app)) :
    List AppWitness :=
  match HyperNovaHistory.run (MemoryApp.program p app) fits s.1 s.2.1 s.2.2 with
  | .ok (advice, _) => advice
  | .error _ => []

/-- The experiment of the memory verifier: the claimed statement, and the
history that the reverse program returns. -/
noncomputable def experiment : Experiment app.width (Sample app) where
  statement s := s.2.1
  memory s := s.1
  states s i := ((advice app fits s.2).take i).foldl (MemoryApp.program p app).step s.2.1.z0
  witness s i := (advice app fits s.2).getD i []

/-- Valid advice is a step history along its prefixes. -/
theorem steps_of_validAdvice (program : Lifecycle.Stage1.Application.Program) :
    ∀ (z0 : AppState) (advice : List AppWitness), ValidAdvice program z0 advice →
      ∀ i < advice.length,
        program.valid ((advice.take i).foldl program.step z0) (advice.getD i []) ∧
          (advice.take (i + 1)).foldl program.step z0 =
            program.step ((advice.take i).foldl program.step z0) (advice.getD i [])
  | _, [], _, i, below => absurd below (Nat.not_lt_zero i)
  | z0, w :: ws, ⟨valid, rest⟩, i, below => by
    cases i with
    | zero => exact ⟨valid, rfl⟩
    | succ i =>
      simpa only [List.take_succ_cons, List.foldl_cons, List.getD_cons_succ] using
        steps_of_validAdvice program (program.step z0 w) ws rest i
          (by simpa only [List.length_cons, Nat.add_lt_add_iff_right] using below)

/-- The Stage 1 failure: the terminal verifier accepts and the reverse program
returns no history of valid steps. -/
def Failure (s : Sample app) : Prop :=
  Accepted (MemoryApp.program p app) fits setup s.2 ∧ ¬ AdviceReturned (MemoryApp.program p app) fits s.2

/-- Stage 1 extraction for the memory program, from the reverse program. -/
theorem stage1Extraction :
    Stage1Extraction (MemoryApp.program p app) (experiment app fits)
      (fun s => Accepted (MemoryApp.program p app) fits setup s.2) (Failure app fits setup) where
  extract s accepted := by
    by_cases returned : AdviceReturned (MemoryApp.program p app) fits s.2
    · obtain ⟨list, _unused, ran, length, reached, valid⟩ := returned
      have adviceEq : advice app fits s.2 = list := by
        unfold advice
        rw [ran]
      refine Or.inl ⟨?_, ?_, fun i below => ?_⟩
      · show ((advice app fits s.2).take 0).foldl (MemoryApp.program p app).step s.2.1.z0 = s.2.1.z0
        rfl
      · show ((advice app fits s.2).take s.2.1.iteration).foldl (MemoryApp.program p app).step
          s.2.1.z0 = s.2.1.zi
        rw [adviceEq, ← length, List.take_length, reached]
      · change i < s.2.1.iteration at below
        show (MemoryApp.program p app).valid
            (((advice app fits s.2).take i).foldl (MemoryApp.program p app).step s.2.1.z0)
            ((advice app fits s.2).getD i []) ∧
          ((advice app fits s.2).take (i + 1)).foldl (MemoryApp.program p app).step s.2.1.z0 =
            (MemoryApp.program p app).step
              (((advice app fits s.2).take i).foldl (MemoryApp.program p app).step s.2.1.z0)
              ((advice app fits s.2).getD i [])
        rw [adviceEq]
        exact steps_of_validAdvice _ _ list valid i (by rw [length]; exact below)
    · exact Or.inr ⟨accepted, returned⟩

/-! ## Security note A6 -/

/-- A reverse stage of the HyperNova extractor, with a memory statement that
its tape gives. `bad_le` uses the claim that reads only the IVC adversary's
part of the tape (`adversaryTape`). -/
structure ClaimedStage where
  stage : Stage (MemoryApp.program p app) fits setup
  claim : stage.Tape → MemoryStatement app.width

/-- The joint law of the claim and the reverse sample. -/
noncomputable def ClaimedStage.law (c : ClaimedStage app fits setup) : PMF (Sample app) :=
  c.stage.tape.bind fun tape =>
    (HyperNovaHistoryLaw.law (MemoryApp.program p app) fits
      (HyperNovaVisitedLaw.listSource (MemoryApp.program p app) (c.stage.input tape).1.iteration
        (c.stage.results tape)) (PMF.pure (c.stage.input tape))).map fun s => (c.claim tape, s)

/-- The claim does not change the reverse law. -/
theorem ClaimedStage.law_snd (c : ClaimedStage app fits setup) :
    c.law.map Prod.snd = c.stage.reverseLaw := by
  rw [ClaimedStage.law, PMF.map_bind, Stage.reverseLaw]
  congr 1
  funext tape
  rw [PMF.map_comp]
  exact PMF.map_id _

/-- Security note A6 for the memory round, in the form of HyperNova errata
Assumption 1: for every claimed stage whose stage is admitted, the mass of
samples whose extracted run the model accepts with a claimed segment that
passes the product test with unbalanced multisets is at most `error c`, one
term for all segment slots. `StageAdmitted` is the class of `Closed`, so A6
and Assumption 1 cover the same stages. The relation computes `η` with
Poseidon2, so no random-oracle model covers it. The per-slot game bound
`Lifecycle.Nebula.badAt_bound` (`2·m/q²` per slot plus the retry term)
motivates the value of `error`; it does not prove this premise. `bad_le`
applies it at the last reverse stage, so its error counts every rerun of the
earlier stages. -/
def MemoryRound (StageAdmitted : Stage (MemoryApp.program p app) fits setup → Prop)
    (error : ClaimedStage app fits setup → ℝ) : Prop :=
  ∀ c : ClaimedStage app fits setup, StageAdmitted c.stage →
    (c.law.toOuterMeasure {s | (experiment app fits).Unbalanced app s}).toReal ≤ error c

/-! ## The end-to-end bound -/

variable {Admitted : NifsAdversary (MemoryApp.program p app) fits setup → Prop}
  {StageAdmitted : Stage (MemoryApp.program p app) fits setup → Prop}
  {Efficient : (adversary : NifsAdversary (MemoryApp.program p app) fits setup) →
    NifsExtractor adversary → Prop}
  {error : NifsAdversary (MemoryApp.program p app) fits setup → ℝ}
  (assumption : Assumption1 Admitted Efficient error) (closed : Closed Admitted StageAdmitted Efficient)
  (adversary : IvcAdversary (MemoryApp.program p app) fits setup)
  (admitted : StageAdmitted (Stage.start adversary))

local notation "stages" => reverseStages assumption closed adversary admitted

/-- The IVC adversary's tape inside the tape of reverse stage `j`: stage
`j + 1` draws a tape of stage `j` and the coins of one extractor. -/
noncomputable def adversaryTape : (j : ℕ) → (stages j).1.Tape → adversary.Tape
  | 0, tape => tape
  | j + 1, draw => adversaryTape j draw.1

/-- Every reverse stage reads its Stage 1 input from the adversary's tape. -/
theorem input_adversaryTape (j : ℕ) (tape : (stages j).1.Tape) :
    (stages j).1.input tape =
      adversary.output (adversaryTape app fits setup assumption closed adversary admitted j tape) := by
  induction j with
  | zero => rfl
  | succ j ih => exact ih tape.1

/-- The adversary's tape inside a reverse stage has the adversary's law. -/
theorem tape_map_adversaryTape (j : ℕ) :
    (stages j).1.tape.map (adversaryTape app fits setup assumption closed adversary admitted j) =
      adversary.tape := by
  induction j with
  | zero => exact PMF.map_id _
  | succ j ih =>
    rw [← ih]
    change ((stages j).1.tape.bind fun tape =>
        (HyperNovaVisitedSecurity.reverseExtractor assumption closed adversary admitted j).coins.map
          (tape, ·)).map
        (fun draw => adversaryTape app fits setup assumption closed adversary admitted j draw.1) = _
    rw [PMF.map_bind, ← PMF.bind_pure_comp]
    congr 1
    funext tape
    rw [PMF.map_comp]
    exact PMF.map_const (HyperNovaVisitedSecurity.reverseExtractor assumption closed adversary admitted j).coins
      (adversaryTape app fits setup assumption closed adversary admitted j tape)

/-- Security note §5 for a memory application: on the joint law of the
memory statement that the adversary claims from its own tape and the reverse
extractor's sample, the mass of samples that the memory verifier accepts
(Stage 1 terminal and spec §13 checks) but whose extracted history does not
attest the claim is at most the Stage 1 history terms of
`history_failure_le`, the state-link and chain collision masses of the
extracted runs, and the A6 error of the last stage. A6 holds for the stages
that `closed` admits. -/
theorem bad_le (valid : p.Valid) (depth : ℕ)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth)
    (claim : adversary.Tape → MemoryStatement app.width)
    {memoryError : ClaimedStage app fits setup → ℝ}
    (round : MemoryRound app fits setup StageAdmitted memoryError) :
    let c : ClaimedStage app fits setup :=
      ⟨(stages depth).1, claim ∘ adversaryTape app fits setup assumption closed adversary admitted depth⟩
    (c.law.toOuterMeasure {s | (experiment app fits).Bad app
        (fun s => Accepted (MemoryApp.program p app) fits setup s.2) s}).toReal ≤
      ∑ j : Fin depth,
          (((stages j.val).1.tape.toOuterMeasure
              {tape | MarkedHashCollision (MemoryApp.program p app) fits setup
                ((stages j.val).1.visit tape)}).toReal +
            error (stages j.val).1.nifs) +
        (c.law.toOuterMeasure {s | (experiment app fits).StateLinkCollision app s}).toReal +
        (c.law.toOuterMeasure {s | (experiment app fits).ChainCollision app s}).toReal +
        memoryError c := by
  intro c
  have cover := EndToEnd.bad_le_sum valid (stage1Extraction app fits setup) c.law
  have failure : c.law.toOuterMeasure {s | Failure app fits setup s} =
      (stages depth).1.reverseLaw.toOuterMeasure
        {sample | Accepted (MemoryApp.program p app) fits setup sample ∧
          ¬ AdviceReturned (MemoryApp.program p app) fits sample} := by
    rw [← ClaimedStage.law_snd app fits setup c, PMF.toOuterMeasure_map_apply]
    rfl
  have history := HyperNovaVisitedSecurity.history_failure_le assumption closed adversary admitted
    depth depthBound
  rw [← failure] at history
  have finite (event : Set (Sample app)) : c.law.toOuterMeasure event ≠ ∞ := event_ne_top c.law event
  have real := ENNReal.toReal_mono
    (ENNReal.add_ne_top.mpr ⟨ENNReal.add_ne_top.mpr ⟨ENNReal.add_ne_top.mpr
      ⟨finite _, finite _⟩, finite _⟩, finite _⟩) cover
  rw [ENNReal.toReal_add (ENNReal.add_ne_top.mpr ⟨ENNReal.add_ne_top.mpr ⟨finite _, finite _⟩,
      finite _⟩) (finite _),
    ENNReal.toReal_add (ENNReal.add_ne_top.mpr ⟨finite _, finite _⟩) (finite _),
    ENNReal.toReal_add (finite _) (finite _)] at real
  refine real.trans ?_
  gcongr
  exact round c (stages depth).2

end NightstreamFPrime.Export.Stage1.NebulaMemorySecurity
