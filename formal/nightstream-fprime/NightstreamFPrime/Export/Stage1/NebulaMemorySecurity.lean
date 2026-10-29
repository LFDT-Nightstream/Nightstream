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
* `badBound`: the right side of security note §5: the Stage 1 history terms,
  the state-link and chain collision masses, and the A6 error.
* `bad_le`: on the claim that the adversary's tape gives, the bad mass is at
  most `badBound` at the adversary's step bound.
* `bad_le_clamp`: the same for an adversary whose outputs can be longer than
  `S_max · N` steps, on its own outputs, through the reverse extractor of its
  clamped form (`clamp`), if that clamped form is admitted.

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

variable {p : Plan} (app : PortApp p.toGeometry)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 (MemoryApp.program p.toGeometry app))
  (setup : PerApplicationCanonicalPackage.CommitmentSetup (MemoryApp.program p.toGeometry app))

/-- A sample: the claimed memory statement and the reverse sample (terminal
statement, envelope, and source returns). -/
abbrev Sample := MemoryStatement app.width × HyperNovaHistoryProbability.Sample (MemoryApp.program p.toGeometry app)

/-- The advice that the reverse program returns; empty after a failure. -/
noncomputable def advice (s : HyperNovaHistoryProbability.Sample (MemoryApp.program p.toGeometry app)) :
    List AppWitness :=
  match HyperNovaHistory.run (MemoryApp.program p.toGeometry app) fits s.1 s.2.1 s.2.2 with
  | .ok (advice, _) => advice
  | .error _ => []

/-- The experiment of the memory verifier: the claimed statement, and the
history that the reverse program returns. -/
noncomputable def experiment : Experiment app.width (Sample app) where
  statement s := s.2.1
  memory s := s.1
  states s i := ((advice app fits s.2).take i).foldl (MemoryApp.program p.toGeometry app).step s.2.1.z0
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
  Accepted (MemoryApp.program p.toGeometry app) fits setup s.2 ∧ ¬ AdviceReturned (MemoryApp.program p.toGeometry app) fits s.2

/-- Stage 1 extraction for the memory program, from the reverse program. -/
theorem stage1Extraction :
    Stage1Extraction (MemoryApp.program p.toGeometry app) (experiment app fits)
      (fun s => Accepted (MemoryApp.program p.toGeometry app) fits setup s.2) (Failure app fits setup) where
  extract s accepted := by
    by_cases returned : AdviceReturned (MemoryApp.program p.toGeometry app) fits s.2
    · obtain ⟨list, _unused, ran, length, reached, valid⟩ := returned
      have adviceEq : advice app fits s.2 = list := by
        unfold advice
        rw [ran]
      refine Or.inl ⟨?_, ?_, fun i below => ?_⟩
      · show ((advice app fits s.2).take 0).foldl (MemoryApp.program p.toGeometry app).step s.2.1.z0 = s.2.1.z0
        rfl
      · show ((advice app fits s.2).take s.2.1.iteration).foldl (MemoryApp.program p.toGeometry app).step
          s.2.1.z0 = s.2.1.zi
        rw [adviceEq, ← length, List.take_length, reached]
      · change i < s.2.1.iteration at below
        show (MemoryApp.program p.toGeometry app).valid
            (((advice app fits s.2).take i).foldl (MemoryApp.program p.toGeometry app).step s.2.1.z0)
            ((advice app fits s.2).getD i []) ∧
          ((advice app fits s.2).take (i + 1)).foldl (MemoryApp.program p.toGeometry app).step s.2.1.z0 =
            (MemoryApp.program p.toGeometry app).step
              (((advice app fits s.2).take i).foldl (MemoryApp.program p.toGeometry app).step s.2.1.z0)
              ((advice app fits s.2).getD i [])
        rw [adviceEq]
        exact steps_of_validAdvice _ _ list valid i (by rw [length]; exact below)
    · exact Or.inr ⟨accepted, returned⟩

/-! ## Security note A6 -/

/-- A reverse stage of the HyperNova extractor, with a memory statement that
its tape gives. `bad_le` uses the claim that reads only the IVC adversary's
part of the tape (`adversaryTape`). -/
structure ClaimedStage where
  stage : Stage (MemoryApp.program p.toGeometry app) fits setup
  claim : stage.Tape → MemoryStatement app.width

/-- On stage tape `tape`, the claim and one reverse sample for terminal input
`x`: the reverse program runs on `x` and reads the stage's source returns from
`x`'s iteration. -/
noncomputable def ClaimedStage.sampleAt (c : ClaimedStage app fits setup) (tape : c.stage.Tape)
    (x : HyperNovaHistory.Statement × HyperNovaHistory.Envelope (MemoryApp.program p.toGeometry app)) :
    PMF (Sample app) :=
  (HyperNovaHistoryLaw.law (MemoryApp.program p.toGeometry app) fits
    (HyperNovaVisitedLaw.listSource (MemoryApp.program p.toGeometry app) x.1.iteration (c.stage.results tape))
    (PMF.pure x)).map fun s => (c.claim tape, s)

/-- The joint law of the claim and a reverse sample whose terminal input is
`input tape`. -/
noncomputable def ClaimedStage.lawAt (c : ClaimedStage app fits setup)
    (input : c.stage.Tape →
      HyperNovaHistory.Statement × HyperNovaHistory.Envelope (MemoryApp.program p.toGeometry app)) :
    PMF (Sample app) :=
  c.stage.tape.bind fun tape => ClaimedStage.sampleAt app fits setup c tape (input tape)

/-- The joint law of the claim and the reverse sample. -/
noncomputable def ClaimedStage.law (c : ClaimedStage app fits setup) : PMF (Sample app) :=
  ClaimedStage.lawAt app fits setup c c.stage.input

/-- The claim does not change the reverse law. -/
theorem ClaimedStage.law_snd (c : ClaimedStage app fits setup) :
    c.law.map Prod.snd = c.stage.reverseLaw := by
  rw [ClaimedStage.law, ClaimedStage.lawAt, PMF.map_bind, Stage.reverseLaw]
  congr 1
  funext tape
  rw [ClaimedStage.sampleAt, PMF.map_comp]
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
def MemoryRound (StageAdmitted : Stage (MemoryApp.program p.toGeometry app) fits setup → Prop)
    (error : ClaimedStage app fits setup → ℝ) : Prop :=
  ∀ c : ClaimedStage app fits setup, StageAdmitted c.stage →
    (c.law.toOuterMeasure {s | (experiment app fits).Unbalanced app s}).toReal ≤ error c

/-! ## The end-to-end bound -/

/-- The samples of security note §5: the memory verifier accepts (Stage 1
terminal and spec §13 checks), but the extracted history does not attest the
claim. -/
def badSamples : Set (Sample app) :=
  {s | (experiment app fits).Bad app (fun s => Accepted (MemoryApp.program p.toGeometry app) fits setup s.2) s}

/-- The right side of security note §5 for the reverse stages `stage` run to
`depth`, the claimed stage `c`, and its A6 error `memoryError`: the Stage 1
history terms, the state-link and chain collision masses, and the A6 error. It
is the §5 bound only when `c.stage` is `stage depth`, as in `bad_le` and
`bad_le_clamp`. -/
noncomputable def badBound (error : NifsAdversary (MemoryApp.program p.toGeometry app) fits setup → ℝ)
    (stage : ℕ → Stage (MemoryApp.program p.toGeometry app) fits setup) (depth : ℕ)
    (c : ClaimedStage app fits setup) (memoryError : ℝ) : ℝ :=
  ∑ j : Fin depth,
      (((stage j.val).tape.toOuterMeasure
          {tape | MarkedHashCollision (MemoryApp.program p.toGeometry app) fits setup ((stage j.val).visit tape)}).toReal +
        error (stage j.val).nifs) +
    (c.law.toOuterMeasure {s | (experiment app fits).StateLinkCollision app s}).toReal +
    (c.law.toOuterMeasure {s | (experiment app fits).ChainCollision app s}).toReal +
    memoryError

variable {Admitted : NifsAdversary (MemoryApp.program p.toGeometry app) fits setup → Prop}
  {StageAdmitted : Stage (MemoryApp.program p.toGeometry app) fits setup → Prop}
  {Efficient : (adversary : NifsAdversary (MemoryApp.program p.toGeometry app) fits setup) →
    NifsExtractor adversary → Prop}
  {error : NifsAdversary (MemoryApp.program p.toGeometry app) fits setup → ℝ}
  (assumption : Assumption1 Admitted Efficient error) (closed : Closed Admitted StageAdmitted Efficient)
  (adversary : IvcAdversary (MemoryApp.program p.toGeometry app) fits setup)
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
    (c.law.toOuterMeasure (badSamples app fits setup)).toReal ≤
      badBound app fits setup error (fun j => (stages j).1) depth c (memoryError c) := by
  intro c
  rw [badBound]
  have cover := EndToEnd.bad_le_sum valid (stage1Extraction app fits setup) c.law
  have failure : c.law.toOuterMeasure {s | Failure app fits setup s} =
      (stages depth).1.reverseLaw.toOuterMeasure
        {sample | Accepted (MemoryApp.program p.toGeometry app) fits setup sample ∧
          ¬ AdviceReturned (MemoryApp.program p.toGeometry app) fits sample} := by
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

/-! ## Adversaries with longer outputs -/

/-- An output with more than `bound` steps, replaced by the same output at
iteration `0`; other outputs unchanged. -/
def clampOutput (bound : ℕ)
    (output : HyperNovaHistory.Statement × HyperNovaHistory.Envelope (MemoryApp.program p.toGeometry app)) :
    HyperNovaHistory.Statement × HyperNovaHistory.Envelope (MemoryApp.program p.toGeometry app) :=
  if output.1.iteration ≤ bound then output else ({ output.1 with iteration := 0 }, output.2)

/-- A clamped output has at most `bound` steps. -/
theorem clampOutput_iteration_le (bound : ℕ)
    (x : HyperNovaHistory.Statement × HyperNovaHistory.Envelope (MemoryApp.program p.toGeometry app)) :
    (clampOutput app bound x).1.iteration ≤ bound := by
  unfold clampOutput
  split
  · assumption
  · exact Nat.zero_le bound

/-- A clamped output that passes the spec §13 checks is unchanged: the memory
verifier rejects iteration `0`. -/
theorem clampOutput_eq_of_checks (valid : p.Valid) (bound : ℕ)
    (x : HyperNovaHistory.Statement × HyperNovaHistory.Envelope (MemoryApp.program p.toGeometry app))
    {m : MemoryStatement app.width} (checks : EndToEnd.VerifierChecks p (clampOutput app bound x).1 m) :
    clampOutput app bound x = x := by
  unfold clampOutput at checks ⊢
  split at checks
  · rw [if_pos ‹_›]
  · have steps : 0 = m.segments * p.n := checks.terminal.stepCount
    have : 0 < m.segments * p.n := Nat.mul_pos checks.terminal.segmentsRange.1 valid.positive.2.2.1
    omega

/-- With `S_max · N ≤ bound`, an output that passes the spec §13 checks is
unchanged by the clamp. -/
theorem clampOutput_eq_of_input_checks (bound : ℕ) (below : p.sMax * p.n ≤ bound)
    (x : HyperNovaHistory.Statement × HyperNovaHistory.Envelope (MemoryApp.program p.toGeometry app))
    {m : MemoryStatement app.width} (checks : EndToEnd.VerifierChecks p x.1 m) :
    clampOutput app bound x = x := by
  have short : x.1.iteration ≤ bound := by
    rw [checks.terminal.stepCount]
    exact (Nat.mul_le_mul_right p.n checks.terminal.segmentsRange.2).trans below
  unfold clampOutput
  rw [if_pos short]

/-- The adversary on its own tape with clamped outputs. With
`S_max · N ≤ bound`, a clamped output passes the spec §13 checks exactly when
the adversary's output does, and the two are then equal
(`clampOutput_eq_of_checks`, `clampOutput_eq_of_input_checks`). -/
def clamp (bound : ℕ) : IvcAdversary (MemoryApp.program p.toGeometry app) fits setup :=
  { adversary with output := clampOutput app bound ∘ adversary.output }

/-- The clamped adversary meets the step bound of `bad_le`. -/
theorem clamp_depthBound (bound : ℕ) :
    ∀ tape ∈ (clamp app fits setup adversary bound).tape.support,
      ((clamp app fits setup adversary bound).output tape).1.iteration ≤ bound :=
  fun tape _ => clampOutput_iteration_le app bound (adversary.output tape)

/-- On the clamped adversary's reverse stages at depth `S_max · N`, the bad
mass with the adversary's own outputs as terminal inputs equals the bad mass
of the clamped run: on each tape the two outputs are equal when either passes
the spec §13 checks, and neither is bad otherwise. -/
private theorem clamp_bad_eq (valid : p.Valid)
    (clampedAdmitted : StageAdmitted (Stage.start (clamp app fits setup adversary (p.sMax * p.n))))
    (claim : adversary.Tape → MemoryStatement app.width) :
    let clamped := clamp app fits setup adversary (p.sMax * p.n)
    let own := adversaryTape app fits setup assumption closed clamped clampedAdmitted (p.sMax * p.n)
    let c : ClaimedStage app fits setup :=
      ⟨(reverseStages assumption closed clamped clampedAdmitted (p.sMax * p.n)).1, claim ∘ own⟩
    (ClaimedStage.lawAt app fits setup c (adversary.output ∘ own)).toOuterMeasure
        (badSamples app fits setup) =
      c.law.toOuterMeasure (badSamples app fits setup) := by
  intro clamped own c
  have zero : ∀ (tape : c.stage.Tape)
      (x : HyperNovaHistory.Statement × HyperNovaHistory.Envelope (MemoryApp.program p.toGeometry app)),
      ¬ EndToEnd.VerifierChecks p x.1 (claim (own tape)) →
      (ClaimedStage.sampleAt app fits setup c tape x).toOuterMeasure (badSamples app fits setup) = 0 := by
    intro tape x no
    rw [ClaimedStage.sampleAt, PMF.toOuterMeasure_map_apply, HyperNovaHistoryLaw.law, PMF.pure_bind,
      PMF.toOuterMeasure_map_apply]
    convert MeasureTheory.measure_empty (μ := (HyperNovaHistoryLaw.results (MemoryApp.program p.toGeometry app) fits
      (HyperNovaVisitedLaw.listSource (MemoryApp.program p.toGeometry app) x.1.iteration (c.stage.results tape))
      x.1 x.2).toOuterMeasure)
    ext r
    simp only [badSamples, Set.mem_preimage, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    exact fun bad => no bad.2.1
  rw [ClaimedStage.law, ClaimedStage.lawAt, ClaimedStage.lawAt, PMF.toOuterMeasure_bind_apply,
    PMF.toOuterMeasure_bind_apply]
  congr 1
  funext tape
  congr 1
  have input : c.stage.input tape = clampOutput app (p.sMax * p.n) (adversary.output (own tape)) :=
    input_adversaryTape app fits setup assumption closed clamped clampedAdmitted (p.sMax * p.n) tape
  rw [show (adversary.output ∘ own) tape = adversary.output (own tape) from rfl, input]
  by_cases same : clampOutput app (p.sMax * p.n) (adversary.output (own tape)) = adversary.output (own tape)
  · rw [same]
  · rw [zero tape _ (fun checks => same
        (clampOutput_eq_of_input_checks app _ le_rfl _ checks)),
      zero tape _ (fun checks => same (clampOutput_eq_of_checks app valid _ _ checks))]

/-- Security note §5 for an adversary whose outputs can be longer than
`S_max · N` steps (note A4): with the reverse extractor of its clamped form at
depth `S_max · N`, the mass of samples where the memory verifier accepts the
adversary's own output but the extracted history does not attest the claim is
at most `badBound` of the clamped form, the bound of `bad_le` for it. The
premise is that the clamped adversary's start stage is admitted. -/
theorem bad_le_clamp (valid : p.Valid)
    (clampedAdmitted : StageAdmitted (Stage.start (clamp app fits setup adversary (p.sMax * p.n))))
    (claim : adversary.Tape → MemoryStatement app.width)
    {memoryError : ClaimedStage app fits setup → ℝ}
    (round : MemoryRound app fits setup StageAdmitted memoryError) :
    let clamped := clamp app fits setup adversary (p.sMax * p.n)
    let stage := reverseStages assumption closed clamped clampedAdmitted
    let own := adversaryTape app fits setup assumption closed clamped clampedAdmitted (p.sMax * p.n)
    let c : ClaimedStage app fits setup := ⟨(stage (p.sMax * p.n)).1, claim ∘ own⟩
    ((ClaimedStage.lawAt app fits setup c (adversary.output ∘ own)).toOuterMeasure
        (badSamples app fits setup)).toReal ≤
      badBound app fits setup error (fun j => (stage j).1) (p.sMax * p.n) c (memoryError c) := by
  intro clamped stage own c
  rw [clamp_bad_eq app fits setup assumption closed adversary valid clampedAdmitted claim]
  exact bad_le app fits setup assumption closed clamped clampedAdmitted valid (p.sMax * p.n)
    (clamp_depthBound app fits setup adversary _) claim round

end NightstreamFPrime.Export.Stage1.NebulaMemorySecurity
