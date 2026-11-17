import NightstreamFPrime.Export.BoundedDepth.FinalHistory
import NightstreamFPrime.Export.BoundedDepth.NebulaSecurity

/-! The memory verifier of a final proof, Rust `nebula::Verifier::verify_final` (SPEC.md,
Algorithms 8 and 10; `SECURITY_MODEL.md`, adversaries 5 and 6). It accepts when the checks of
Algorithm 8 hold for the memory statement and the Stage 1 statement that they give, and the final
verifier accepts that Stage 1 statement.

A prover of the final verifier with `q` queries names a memory statement with each result of the
state-restoration experiment (`claim`, a law for each result). With that statement on its tape it
is an IVC adversary (`adversary`), whose output is the output of `FinalHistory.adversary`. A false
memory statement is a tape of the walk's last stage on which the final verifier accepts
`(i, z0, zi)`, the checks of Algorithm 8 hold, and the history that the walk decodes does not
attest the memory statement (`FalseMemory`). On such a tape `Verifies` rejects the proof from `W`,
or the sample is bad for `NebulaSecurity.bad_le`. So its probability is at most `q · finalError`
plus the bound of `NebulaSecurity.bad_le` for this adversary, which starts with
`IvcSecurity.walkBound` (`soundness`). The walk has `S_max · N` stages: the checks of Algorithm 8
accept at most `S_max · N` steps (`EndToEnd.VerifierChecks.iteration_le`), so no premise bounds
the iteration of the prover's statements. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.NebulaFinal

open scoped ENNReal
open scoped NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Spec.Compression.Game (Strategy prob restore law prob_eq_law restore_sound)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.Nebula
open NightstreamFPrime.Lifecycle.Nebula.EndToEnd (MemoryStatement VerifierChecks History)
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.IvcWalk
open NightstreamFPrime.Export.BoundedDepth.IvcSecurity (Output walkAdversary walkBound)
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier
open NightstreamFPrime.Export.BoundedDepth.FinalHistory (outputOf)
open NightstreamFPrime.Export.BoundedDepth.NebulaSecurity

/-- The first part of a draw from `base` and then from `kernel` has the law `base`. -/
theorem map_fst_bind {α β : Type} (base : PMF α) (kernel : α → PMF β) :
    (base.bind fun a => (kernel a).map (a, ·)).map Prod.fst = base := by
  rw [PMF.map_bind]
  conv_rhs => rw [← PMF.bind_pure base]
  congr 1
  funext a
  rw [PMF.map_comp]
  exact PMF.map_const (kernel a) a

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  (setup : Setup logicalWidth publicFits)
  {p : Plan} (app : PortApp p.toGeometry)

/-- A tape of the prover with a memory statement: the result of the experiment and the memory
statement that the prover names with it. -/
abbrev Tape (width : ℕ) (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) :=
  Option (Option (Outcome logicalWidth publicFits)) × MemoryStatement width

/-- The prover `s` of the final verifier in the state-restoration experiment with `q` queries,
with the memory statement of law `claim r` for the result `r`, as an IVC adversary. -/
noncomputable def adversary (s : Strategy (Option (Option (Outcome logicalWidth publicFits))))
    (q : ℕ)
    (claim : Option (Option (Outcome logicalWidth publicFits)) → PMF (MemoryStatement app.width)) :
    IvcAdversary (Output logicalWidth publicFits) :=
  ⟨Tape app.width logicalWidth publicFits,
    (law s (restore setup.game q [])).bind fun r => (claim r).map (r, ·),
    fun tape => outputOf setup tape.1⟩

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {step : AppState → AppWitness → AppState}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : Exports relation setup.ajtai setup.circuitKeys step cubeFits)

/-- The final verifier accepts `(i, z0, zi)`, the checks of SPEC.md Algorithm 8 hold for it and
the sample's memory statement, and the history that the walk decodes from the sample is not a
history of `(i, z0, zi)` that attests the memory statement. On the law of `soundness` the
sample's statement is `(i, z0, zi)`. -/
def FalseMemory (r : Option (Option (Outcome logicalWidth publicFits)))
    (sample : Sample app.width logicalWidth publicFits) : Prop :=
  ∃ o, r = some (some o) ∧ VerifierChecks p o.1 sample.1 ∧
    ¬ (History (MemoryApp.program p.toGeometry app) o.1
        ((experiment app setup.ajtai setup.circuitKeys exports).states sample)
        ((experiment app setup.ajtai setup.circuitKeys exports).witness sample) ∧
      (experiment app setup.ajtai setup.circuitKeys exports).Attested app sample)

variable {Admitted : NifsAdversary (WalkAttempt logicalWidth publicFits) → Prop}
  {StageAdmitted : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits) → Prop}
  {Efficient : (adversary : NifsAdversary (WalkAttempt logicalWidth publicFits)) →
    NifsExtractor (Answer logicalWidth publicFits) adversary → Prop}
  {error : NifsAdversary (WalkAttempt logicalWidth publicFits) → ℝ}
  (assumption : Assumption1 (WalkSucceeds setup.ajtai) (WalkExtracted setup.ajtai) Admitted
    Efficient error)
  (closed : Closed (attemptOf setup.ajtai setup.circuitKeys exports)
    (back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
  (s : Strategy (Option (Option (Outcome logicalWidth publicFits)))) (q : ℕ)
  (claim : Option (Option (Outcome logicalWidth publicFits)) → PMF (MemoryStatement app.width))
  (admitted : StageAdmitted (Stage.start (walkAdversary (adversary setup app s q claim))))

local notation "stages" =>
  reverseStages assumption closed (walkAdversary (adversary setup app s q claim)) admitted

/-- The prover's tape inside a tape of the walk's stage `depth`. -/
noncomputable def tapeOf (depth : ℕ) (tape : (stages depth).1.Tape) :
    Tape app.width logicalWidth publicFits :=
  origin assumption closed (walkAdversary (adversary setup app s q claim)) admitted depth tape

/-- The walk's last stage with the memory statement of the prover's tape. -/
noncomputable def claimed (depth : ℕ) :
    ClaimedStage app (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  ⟨(stages depth).1, fun tape => (tapeOf setup app exports assumption closed s q claim admitted
    depth tape).2⟩

/-- The sample of a tape of the walk's stage `depth`: the memory statement, the walk's start and
the extractor's answers. `(claimed …).law` is its law. -/
noncomputable def sampleOf (depth : ℕ) (tape : (stages depth).1.Tape) :
    Sample app.width logicalWidth publicFits :=
  ((tapeOf setup app exports assumption closed s q claim admitted depth tape).2,
    (stages depth).1.input tape, (stages depth).1.results tape)

/-- The bound for Rust `nebula::Verifier::verify_final`. In the state-restoration experiment with
`q` queries and the walk's `S_max · N` stages (`p.sMax * p.n`), the probability that the final
verifier accepts, the checks of SPEC.md Algorithm 8 hold, and the decoded history does not attest
the memory statement is at most `q · (1/2^119 + 38/fieldModulus²)` plus the terms of
`NebulaSecurity.bad_le` for the prover's IVC adversary: `IvcSecurity.walkBound`, the state-link
and chain collision masses, and the memory-round error. The checks of Algorithm 8 bound the
iteration by `S_max · N` (`EndToEnd.VerifierChecks.iteration_le`), so the answers of the last
stage cover the iteration of each statement in the event. -/
theorem soundness (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (valid : p.Valid) (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (programEq : exports.program = MemoryApp.program p.toGeometry app)
    {memoryError : ClaimedStage app (logicalWidth := logicalWidth) (publicFits := publicFits) → ℝ}
    (round : MemoryRound app setup.ajtai setup.circuitKeys exports StageAdmitted memoryError) :
    ((stages (p.sMax * p.n)).1.tape.toOuterMeasure
        {tape | FalseMemory setup app exports
          (tapeOf setup app exports assumption closed s q claim admitted (p.sMax * p.n) tape).1
          (sampleOf setup app exports assumption closed s q claim admitted (p.sMax * p.n)
            tape)}).toReal ≤
      q * finalError +
        (walkBound setup.ajtai setup.circuitKeys exports assumption closed
            (adversary setup app s q claim) admitted (p.sMax * p.n) +
          ((claimed setup app exports assumption closed s q claim admitted
              (p.sMax * p.n)).law.toOuterMeasure
            {sample | (experiment app setup.ajtai setup.circuitKeys exports).StateLinkCollision app
              sample}).toReal +
          ((claimed setup app exports assumption closed s q claim admitted
              (p.sMax * p.n)).law.toOuterMeasure
            {sample | (experiment app setup.ajtai setup.circuitKeys exports).ChainCollision app
              sample}).toReal +
          memoryError
            (claimed setup app exports assumption closed s q claim admitted (p.sMax * p.n))) := by
  classical
  let prover := adversary setup app s q claim
  let stage := (stages (p.sMax * p.n)).1
  let tapeAt : stage.Tape → Tape app.width logicalWidth publicFits :=
    tapeOf setup app exports assumption closed s q claim admitted (p.sMax * p.n)
  let sampleAt : stage.Tape → Sample app.width logicalWidth publicFits :=
    sampleOf setup app exports assumption closed s q claim admitted (p.sMax * p.n)
  let Rejected : Option (Option (Outcome logicalWidth publicFits)) → Prop := fun r =>
    ¬ ∀ a, r = some a → ∀ o, a = some o →
      Verifies setup.ajtai setup.circuitKeys step complete.packageWidth 0 (fun _ _ => 0) o.1
        (setup.proofOf o)
  let Bad : Sample app.width logicalWidth publicFits → Prop :=
    (experiment app setup.ajtai setup.circuitKeys exports).Bad app
      (Accepted app setup.ajtai setup.circuitKeys (step := step) (fun _ _ => 0) 0)
  have origins : stage.tape.map tapeAt = prover.tape :=
    tape_map_origin assumption closed (walkAdversary prover) admitted (p.sMax * p.n)
  have results : prover.tape.map Prod.fst = law s (restore setup.game q []) := map_fst_bind _ _
  have cover : {tape | FalseMemory setup app exports (tapeAt tape).1 (sampleAt tape)} ∩
      stage.tape.support ⊆ {tape | Rejected (tapeAt tape).1} ∪ {tape | Bad (sampleAt tape)} := by
    rintro tape ⟨⟨o, accepted, checks, notGood⟩, -⟩
    have input : stage.input tape = initial (o.1, setup.proofOf o) := by
      rw [input_origin assumption closed (walkAdversary prover) admitted (p.sMax * p.n) tape]
      change initial (outputOf setup (tapeAt tape).1) = _
      rw [accepted]
      rfl
    have statement : (sampleAt tape).2.1.statement = o.1 := congrArg Visit.statement input
    by_cases verifies : Verifies setup.ajtai setup.circuitKeys step complete.packageWidth 0 (fun _ _ => 0)
      o.1 (setup.proofOf o)
    · refine Or.inr ⟨⟨?_, congrArg Visit.bounds input, ?_⟩, ?_, ?_⟩
      · change Terminal.HoldsFor 0 setup.ajtai setup.circuitKeys step (fun _ _ => 0)
          (stage.input tape).statement (stage.input tape).proof
        rw [input]
        exact holds_of_verifies verifies
      · change (stage.input tape).statement.iteration ≤ (stage.results tape).length
        rw [input,
          results_length assumption closed (walkAdversary prover) admitted (p.sMax * p.n) tape]
        exact checks.iteration_le
      · change VerifierChecks p (sampleAt tape).2.1.statement (sampleAt tape).1
        rw [statement]
        exact checks
      · change ¬ (History (MemoryApp.program p.toGeometry app) (sampleAt tape).2.1.statement _ _ ∧ _)
        rw [statement]
        exact notGood
    · exact Or.inl fun all => verifies (all _ accepted o rfl)
  have rejected : (stage.tape.toOuterMeasure {tape | Rejected (tapeAt tape).1}).toReal ≤
      q * finalError := by
    have moved : stage.tape.toOuterMeasure {tape | Rejected (tapeAt tape).1} =
        (law s (restore setup.game q [])).toOuterMeasure {r | Rejected r} := by
      rw [← results, ← origins, PMF.toOuterMeasure_map_apply, PMF.toOuterMeasure_map_apply]
      rfl
    rw [moved, ← prob_eq_law]
    exact restore_sound _ s _ _ (game_avoids setup step 0 (fun _ _ => 0)
      (FinalRows.premises_of_rows setup exports complete rows))
      (game_densityLe setup) (by unfold finalError; positivity) q
  have bad := bad_le app setup.ajtai setup.circuitKeys exports assumption closed prover admitted
    valid keysShaped programEq (fun _ _ => 0) 0 (p.sMax * p.n) Prod.snd round
  have sampled : stage.tape.toOuterMeasure {tape | Bad (sampleAt tape)} =
      (claimed setup app exports assumption closed s q claim admitted
        (p.sMax * p.n)).law.toOuterMeasure {sample | Bad sample} := by
    rw [ClaimedStage.law, PMF.toOuterMeasure_map_apply]
    rfl
  calc (stage.tape.toOuterMeasure
          {tape | FalseMemory setup app exports (tapeAt tape).1 (sampleAt tape)}).toReal
      ≤ (stage.tape.toOuterMeasure {tape | Rejected (tapeAt tape).1} +
          stage.tape.toOuterMeasure {tape | Bad (sampleAt tape)}).toReal := by
        rw [← PMF.toOuterMeasure_apply_inter_support]
        exact ENNReal.toReal_mono
          (ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩)
          ((MeasureTheory.measure_mono cover).trans (MeasureTheory.measure_union_le _ _))
    _ = (stage.tape.toOuterMeasure {tape | Rejected (tapeAt tape).1}).toReal +
          (stage.tape.toOuterMeasure {tape | Bad (sampleAt tape)}).toReal :=
        ENNReal.toReal_add (event_ne_top _ _) (event_ne_top _ _)
    _ ≤ _ := add_le_add rejected (by rw [sampled]; exact bad)

/-- With Akita openings, the probability `memory` of a false memory statement is at most
`q · (1/2^119 + 38/fieldModulus²) + εAkita` plus the terms of `NebulaSecurity.bad_le`, over the
walk's `S_max · N` stages, as in `soundness`. `akita` is Akita's knowledge soundness
(`FinalVerifier.AkitaKnowledge`) for the event `FalseMemory`: `s` is the prover of the game with
ideal openings that it gives for the prover with Akita openings. -/
theorem real_soundness (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (valid : p.Valid) (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (programEq : exports.program = MemoryApp.program p.toGeometry app)
    {memoryError : ClaimedStage app (logicalWidth := logicalWidth) (publicFits := publicFits) → ℝ}
    (round : MemoryRound app setup.ajtai setup.circuitKeys exports StageAdmitted memoryError)
    (memory εAkita : ℝ)
    (akita : memory ≤ ((stages (p.sMax * p.n)).1.tape.toOuterMeasure
        {tape | FalseMemory setup app exports
          (tapeOf setup app exports assumption closed s q claim admitted (p.sMax * p.n) tape).1
          (sampleOf setup app exports assumption closed s q claim admitted (p.sMax * p.n)
            tape)}).toReal +
      εAkita) :
    memory ≤ q * finalError + εAkita +
        (walkBound setup.ajtai setup.circuitKeys exports assumption closed
            (adversary setup app s q claim) admitted (p.sMax * p.n) +
          ((claimed setup app exports assumption closed s q claim admitted
              (p.sMax * p.n)).law.toOuterMeasure
            {sample | (experiment app setup.ajtai setup.circuitKeys exports).StateLinkCollision app
              sample}).toReal +
          ((claimed setup app exports assumption closed s q claim admitted
              (p.sMax * p.n)).law.toOuterMeasure
            {sample | (experiment app setup.ajtai setup.circuitKeys exports).ChainCollision app
              sample}).toReal +
          memoryError
            (claimed setup app exports assumption closed s q claim admitted (p.sMax * p.n))) := by
  linarith [soundness setup app exports assumption closed s q claim admitted complete rows valid
    keysShaped programEq round]

end NightstreamFPrime.Export.BoundedDepth.NebulaFinal
