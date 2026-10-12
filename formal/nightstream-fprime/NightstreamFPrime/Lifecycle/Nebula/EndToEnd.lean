import NightstreamFPrime.Lifecycle.Nebula.ProgramSoundness
import NightstreamFPrime.Spec.HyperNova.Construction2.Paper
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-! Owns the end-to-end soundness of a memory application on the history
that the Stage 1 extractor returns (security note §5, Lemma 6). No event or
conclusion quantifies over other histories: for a compressing state digest,
some other history of a statement always contains a collision.

* `Stage1Extraction`: on an accepted sample, the extracted history is a step
  history of the sample's statement for the program, or a named Stage 1
  failure occurs. `Export/Stage1/NebulaMemorySecurity` proves it for the
  HyperNova reverse extractor.
* `bad_subset`: a bad sample has a Stage 1 failure, a state-link collision, a
  chain collision, or an accepted extracted run with an unbalanced claimed
  segment that passes the product test (`Unbalanced`).
* `bad_le_sum`: the same cover as a bound on the mass of any law.

This module gives no term a number and states no premise about the memory
round; `Export/Stage1/NebulaMemorySecurity` owns security note A6. -/

namespace NightstreamFPrime.Lifecycle.Nebula.EndToEnd

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper (TerminalStatement)
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

/-- A step history of `statement` for `program`: states from `z0` to `zi` in
`iteration` steps, each step valid with its witness. -/
def History (program : Application.Program) (statement : TerminalStatement AppState)
    (z witness : ℕ → List F) : Prop :=
  z 0 = statement.z0 ∧ z statement.iteration = statement.zi ∧
    ∀ i < statement.iteration,
      program.valid (z i) (witness i) ∧ z (i + 1) = program.step (z i) (witness i)

/-- The memory fields of the spec §13 statement, with `width` application
words, and the prover's final carry words. -/
structure MemoryStatement (width : ℕ) where
  initialApp : Fin width → F
  finalApp : Fin width → F
  finalCarry : Fin 39 → F
  segments : ℕ
  finalTs : ℕ
  finalRoot : Digest

variable {p : Plan} {app : PortApp p}

/-- The checks of `Context::verify` (spec §13) on a Stage 1 statement: it
opens to the memory statement's states, and the terminal checks hold. -/
structure VerifierChecks {width : ℕ} (p : Plan) (statement : TerminalStatement AppState)
    (m : MemoryStatement width) : Prop where
  start : statement.z0 = stateWords (List.ofFn m.initialApp) (carryWords (startCarry (context p)))
  terminal : TerminalChecks p statement.zi m.finalApp m.finalCarry statement.iteration m.segments
    m.finalTs m.finalRoot

/-- The typed statement that a memory statement claims after `T` steps. -/
def MemoryStatement.typed (app : PortApp p) (m : MemoryStatement app.width) (T : ℕ) :
    Nebula.Statement app.State Digest :=
  ⟨T, m.initialApp, m.finalApp, m.segments, m.finalTs, m.finalRoot⟩

/-- What the end-to-end layer reads from one sample: the Stage 1 statement,
the memory statement with the prover's final carry, and the history that
the Stage 1 extractor returns. A randomized extractor keeps its coins in the
sample. -/
structure Experiment (width : ℕ) (Sample : Type) where
  statement : Sample → TerminalStatement AppState
  memory : Sample → MemoryStatement width
  states : Sample → ℕ → List F
  witness : Sample → ℕ → List F

/-- The Stage 1 extraction premise for `program`: on an accepted sample, the
extracted history is a history of the sample's statement, or the named
Stage 1 failure occurs. -/
structure Stage1Extraction (program : Application.Program) {width : ℕ} {Sample : Type}
    (x : Experiment width Sample) (Accepted Failure : Sample → Prop) : Prop where
  extract : ∀ s, Accepted s →
    History program (x.statement s) (x.states s) (x.witness s) ∨ Failure s

namespace Experiment

variable (app) {Sample : Type} (x : Experiment app.width Sample)

/-- The step witnesses that the extractor returns. -/
def steps (s : Sample) (i : ℕ) : StepWitness p app.toAppShape :=
  MemoryApp.decode p app (x.witness s i)

/-- The extracted run of the model. -/
def run (s : Sample) : List (Invocation app.State Digest) :=
  runOf app (x.steps app s) (x.statement s).iteration

/-- The typed statement that the memory statement claims. -/
def claim (s : Sample) : Nebula.Statement app.State Digest :=
  (x.memory s).typed app (x.statement s).iteration

/-- The extracted run attests an execution for the claim. -/
def Attested (s : Sample) : Prop :=
  Attests (context p) app.model (x.claim app s) (x.run app s)

/-- A state-digest collision between two inputs of the extracted run
(security note Lemma 6 steps 2 and 6). Both sides are data of the sample: the
extracted words, and the claimed final words. -/
def StateLinkCollision (s : Sample) : Prop :=
  RunStateCollision (x.steps app s) (x.memory s).initialApp (x.memory s).finalApp
    (x.memory s).finalCarry (x.statement s).iteration

/-- A Poseidon2 transcript collision among the chain inputs of the extracted
run and the `D_init` chain (Lemma 2). -/
def ChainCollision (s : Sample) : Prop :=
  ∃ a ∈ runInputs (context p) (x.run app s) (x.memory s).segments,
    ∃ b ∈ runInputs (context p) (x.run app s) (x.memory s).segments,
      TranscriptCollision (blocks a) (blocks b)

/-- The model accepts the extracted run, and a claimed segment passes the
product test with unbalanced multisets: the one event that A6 bounds, for all
segment slots at once. -/
def Unbalanced (s : Sample) : Prop :=
  Accepts (context p) app.model (x.claim app s) (x.run app s) ∧
    ∃ k < (x.memory s).segments, BadChallenge (context p) (segmentView (context p) (x.run app s) k)

/-- An accepted sample that passes the spec §13 checks, but whose extracted
history is not a history that attests the claim (security note §5). -/
def Bad (Accepted : Sample → Prop) (s : Sample) : Prop :=
  Accepted s ∧ VerifierChecks p (x.statement s) (x.memory s) ∧
    ¬ (History (MemoryApp.program p app) (x.statement s) (x.states s) (x.witness s) ∧
      x.Attested app s)

end Experiment

variable {Sample : Type} {x : Experiment app.width Sample} {Accepted Failure : Sample → Prop}

/-- The deterministic cover (security note Lemma 6 on the extracted run): a
bad sample has a Stage 1 failure, a state-link collision, a chain collision,
or an accepted extracted run with a claimed segment that passes the product
test with unbalanced multisets. -/
theorem bad_subset (valid : p.Valid)
    (extraction : Stage1Extraction (MemoryApp.program p app) x Accepted Failure) :
    {s | x.Bad app Accepted s} ⊆
      {s | Failure s} ∪ {s | x.StateLinkCollision app s} ∪ {s | x.ChainCollision app s} ∪
        {s | x.Unbalanced app s} := by
  intro s ⟨accepted, checks, notGood⟩
  rcases extraction.extract s accepted with ⟨z0, zT, steps⟩ | failure
  · rcases accepts_or_collision valid (z0.trans checks.start)
        (fun i below => MemoryApp.holds_of_valid (steps i below).1 (steps i below).2)
        (zT ▸ checks.terminal) with ran | link
    · rcases poseidon2_soundness rfl valid ran with attests | chain | bad
      · exact absurd ⟨⟨z0, zT, steps⟩, attests⟩ notGood
      · exact Or.inl (Or.inr chain)
      · exact Or.inr ⟨ran, bad⟩
    · exact Or.inl (Or.inl (Or.inr link))
  · exact Or.inl (Or.inl (Or.inl failure))

/-! ### Probability -/

open MeasureTheory
open scoped ENNReal

/-- The deterministic cover as a bound on the mass of any law of samples. -/
theorem bad_le_sum (valid : p.Valid)
    (extraction : Stage1Extraction (MemoryApp.program p app) x Accepted Failure)
    (law : PMF Sample) :
    law.toOuterMeasure {s | x.Bad app Accepted s} ≤
      law.toOuterMeasure {s | Failure s} + law.toOuterMeasure {s | x.StateLinkCollision app s} +
        law.toOuterMeasure {s | x.ChainCollision app s} +
        law.toOuterMeasure {s | x.Unbalanced app s} := by
  calc law.toOuterMeasure {s | x.Bad app Accepted s}
      ≤ law.toOuterMeasure ({s | Failure s} ∪ {s | x.StateLinkCollision app s} ∪
          {s | x.ChainCollision app s} ∪ {s | x.Unbalanced app s}) :=
        measure_mono (bad_subset valid extraction)
    _ ≤ _ := by
        refine (measure_union_le _ _).trans ?_
        gcongr
        refine (measure_union_le _ _).trans ?_
        gcongr
        exact measure_union_le _ _

end NightstreamFPrime.Lifecycle.Nebula.EndToEnd
