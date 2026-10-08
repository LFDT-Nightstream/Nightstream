import NightstreamFPrime.Lifecycle.Nebula.ProgramSoundness
import NightstreamFPrime.Lifecycle.Nebula.MemoryBound
import NightstreamFPrime.Spec.HyperNova.Construction2.Paper
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-! Owns the end-to-end soundness of the first memory application over two
named premises (owner decision 2026-10-08). It does not prove either premise.

* `Stage1Extraction`: what Stage 1 must deliver for the program. An accepted
  sample's terminal statement has a step history that satisfies the program's
  step function and validity predicate, or a named Stage 1 failure occurs.
  Stage 1 extraction is today written for the hash-chain application only.
* `MemoryRoundTransfer`: security note A6 for the memory round. The real
  mass of accepted runs that are not executions is at most the failure
  frequency of one translated game, plus `delta`.

The deterministic theorem joins the first premise to Lemma 6 for the memory
relation. The probability theorem adds the second premise and the game bound
of `memory_bound`. Every remaining term is a collision or an extraction
event of the experiment; this module gives none of them a number. -/

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

/-- The Stage 1 extraction premise for `program` in an experiment with
samples `Sample`. -/
structure Stage1Extraction (program : Application.Program) {Sample : Type}
    (statementOf : Sample → TerminalStatement AppState) (Accepted Failure : Sample → Prop) :
    Prop where
  extract : ∀ s, Accepted s → (∃ z witness, History program (statementOf s) z witness) ∨ Failure s

/-- The memory fields of the spec §13 statement and the envelope's final
carry words. -/
structure MemoryStatement where
  initialApp : Fin 2 → F
  finalApp : Fin 2 → F
  finalCarry : Fin 39 → F
  segments : ℕ
  finalTs : ℕ
  finalRoot : Digest

variable {p : Plan} {two : p.bOps = 2}

/-- The checks of `Context::verify` (spec §13) on a Stage 1 statement: it
opens to the memory statement's states, and the terminal checks hold. -/
structure VerifierChecks (p : Plan) (statement : TerminalStatement AppState)
    (m : MemoryStatement) : Prop where
  start : statement.z0 = stateWords (List.ofFn m.initialApp) (carryWords (startCarry (context p)))
  terminal : TerminalChecks p statement.zi m.finalApp m.finalCarry statement.iteration m.segments
    m.finalTs m.finalRoot

/-- The typed statement that a memory statement claims after `T` steps. -/
def MemoryStatement.typed (m : MemoryStatement) (T : ℕ) : Nebula.Statement Machine.State Digest :=
  ⟨T, Machine.State.ofWords m.initialApp, Machine.State.ofWords m.finalApp, m.segments, m.finalTs,
    m.finalRoot⟩

/-- The run that a witness history gives the model. -/
def runOfWitness (p : Plan) (witness : ℕ → List F) (T : ℕ) :
    List (Invocation Machine.State Digest) :=
  runOf (fun i => MemoryApp.decode p (witness i)) T

/-- The memory result of one history: it attests a machine execution, or it
gives a state-link collision, a collision among its chain inputs, or a bad
challenge. -/
def Outcome (p : Plan) (two : p.bOps = 2) (m : MemoryStatement) (T : ℕ)
    (witness : ℕ → List F) : Prop :=
  let run := runOfWitness p witness T
  Attests (context p) (Machine.application p two) (m.typed T) run ∨
    RunStateCollision (fun i => MemoryApp.decode p (witness i)) m.initialApp m.finalApp
      m.finalCarry T ∨
    (∃ a ∈ runInputs (context p) run m.segments, ∃ b ∈ runInputs (context p) run m.segments,
      TranscriptCollision (blocks a) (blocks b)) ∨
    ∃ k < m.segments, BadChallenge (context p) (segmentView (context p) run k)

/-- End-to-end soundness, deterministic: an accepted sample that passes the
spec §13 checks has a history with the Lemma 6 outcome, or a named Stage 1
failure occurs. -/
theorem soundness (valid : p.Valid) {Sample : Type}
    {statementOf : Sample → TerminalStatement AppState} {Accepted Failure : Sample → Prop}
    (extraction : Stage1Extraction (MemoryApp.program p two) statementOf Accepted Failure)
    {s : Sample} (accepted : Accepted s) {m : MemoryStatement}
    (checks : VerifierChecks p (statementOf s) m) :
    (∃ witness, Outcome p two m (statementOf s).iteration witness) ∨ Failure s := by
  rcases extraction.extract s accepted with ⟨z, witness, z0, zT, steps⟩ | failure
  · refine Or.inl ⟨witness, ?_⟩
    exact MemoryApp.chain_soundness valid (z0.trans checks.start) steps (zT ▸ checks.terminal)
  · exact Or.inr failure

/-! ### Probability -/

open MeasureTheory
open scoped ENNReal NNReal

section Probability

variable {Sample : Type} (p) (two) (statementOf : Sample → TerminalStatement AppState)
  (memoryOf : Sample → MemoryStatement) (Accepted Failure : Sample → Prop)

/-- An accepted sample that passes the spec §13 checks, but whose memory
statement no history attests. -/
def Bad (s : Sample) : Prop :=
  Accepted s ∧ VerifierChecks p (statementOf s) (memoryOf s) ∧
    ¬ ∃ z witness, History (MemoryApp.program p two) (statementOf s) z witness ∧
      Attests (context p) (Machine.application p two)
        ((memoryOf s).typed (statementOf s).iteration)
        (runOfWitness p witness (statementOf s).iteration)

/-- A history of the sample with a state-link collision. -/
def StateLinkFailure (s : Sample) : Prop :=
  ∃ z witness, History (MemoryApp.program p two) (statementOf s) z witness ∧
    RunStateCollision (fun i => MemoryApp.decode p (witness i)) (memoryOf s).initialApp
      (memoryOf s).finalApp (memoryOf s).finalCarry (statementOf s).iteration

/-- A history of the sample whose run the model accepts but which does not
attest an execution: the event that A6 transfers to the game. -/
def RunFailure (s : Sample) : Prop :=
  ∃ z witness, History (MemoryApp.program p two) (statementOf s) z witness ∧
    Accepts (context p) (Machine.application p two) ((memoryOf s).typed (statementOf s).iteration)
      (runOfWitness p witness (statementOf s).iteration) ∧
    ¬ Attests (context p) (Machine.application p two)
      ((memoryOf s).typed (statementOf s).iteration)
      (runOfWitness p witness (statementOf s).iteration)

variable {p two statementOf memoryOf Accepted Failure}

/-- The deterministic cover: a bad sample has a Stage 1 failure, a
state-link collision, or a run that the model accepts without an
execution. -/
theorem bad_subset (valid : p.Valid)
    (extraction : Stage1Extraction (MemoryApp.program p two) statementOf Accepted Failure) :
    {s | Bad p two statementOf memoryOf Accepted s} ⊆
      ({s | Failure s} ∪ {s | StateLinkFailure p two statementOf memoryOf s}) ∪
        {s | RunFailure p two statementOf memoryOf s} := by
  intro s ⟨accepted, checks, notAttested⟩
  rcases extraction.extract s accepted with ⟨z, witness, history⟩ | failure
  · obtain ⟨z0, zT, steps⟩ := history
    rcases accepts_or_collision valid (z0.trans checks.start)
        (fun i below => MemoryApp.holds_of_valid (steps i below).1 (steps i below).2)
        (zT ▸ checks.terminal) with accepted' | collision
    · refine Or.inr ⟨z, witness, ⟨z0, zT, steps⟩, accepted', fun attests => ?_⟩
      exact notAttested ⟨z, witness, ⟨z0, zT, steps⟩, attests⟩
    · exact Or.inl (Or.inr ⟨z, witness, ⟨z0, zT, steps⟩, collision⟩)
  · exact Or.inl (Or.inl failure)

/-- Security note A6 for the memory round (owner decision 2026-10-08: its own
premise, not tied to a Stage 1 model). The real mass of `event` is at most
the failure frequency of one translated game `g`, plus `delta`. A game's runs
are consistent by definition; an extracted run whose segment transcript
inputs differ from the committed ones is an extraction failure, which `delta`
or the Stage 1 failure must cover. No game, `delta`, or proof is given here. -/
structure MemoryRoundTransfer (law : PMF Sample) (event : Sample → Prop) {Coins : Type}
    [Fintype Coins] (g : Game K Digest Machine.State Coins p.sMax) (delta : ℝ≥0∞) : Prop where
  transfer : law.toOuterMeasure {s | event s} ≤
    ((freq (g.Fails (context p) (Machine.application p two)) : ℝ≥0) : ℝ≥0∞) + delta

/-- End-to-end soundness with probability: the mass of bad samples is at most
the Stage 1 failure mass, the state-link collision mass, the game's
collision frequency, `S_max · 2·m_mem/q²`, the retry terms, and `delta`. -/
theorem bad_probability (valid : p.Valid)
    (extraction : Stage1Extraction (MemoryApp.program p two) statementOf Accepted Failure)
    (law : PMF Sample) {Coins : Type} [Fintype Coins]
    {g : Game K Digest Machine.State Coins p.sMax} {delta : ℝ≥0∞}
    (transfer : MemoryRoundTransfer (two := two) law (RunFailure p two statementOf memoryOf) g delta) :
    law.toOuterMeasure {s | Bad p two statementOf memoryOf Accepted s} ≤
      law.toOuterMeasure {s | Failure s} +
        law.toOuterMeasure {s | StateLinkFailure p two statementOf memoryOf s} +
        ((freq (g.Collides (context p) (Machine.application p two)) +
            p.sMax * (2 * (p.maxTuples : ℚ≥0) / (goldilocksModulus ^ 2 : ℕ)) +
            ∑ k, g.retryTerm (context p) (Machine.application p two) k : ℚ≥0) : ℝ≥0∞) +
        delta := by
  have game : ((freq (g.Fails (context p) (Machine.application p two)) : ℝ≥0) : ℝ≥0∞) ≤
      ((freq (g.Collides (context p) (Machine.application p two)) +
          p.sMax * (2 * (p.maxTuples : ℚ≥0) / (goldilocksModulus ^ 2 : ℕ)) +
          ∑ k, g.retryTerm (context p) (Machine.application p two) k : ℚ≥0) : ℝ≥0∞) :=
    ENNReal.coe_le_coe.mpr (NNRat.cast_le.mpr (memory_bound (Machine.application p two) g valid))
  calc law.toOuterMeasure {s | Bad p two statementOf memoryOf Accepted s}
      ≤ law.toOuterMeasure (({s | Failure s} ∪ {s | StateLinkFailure p two statementOf memoryOf s}) ∪
          {s | RunFailure p two statementOf memoryOf s}) :=
        measure_mono (bad_subset valid extraction)
    _ ≤ law.toOuterMeasure {s | Failure s} +
          law.toOuterMeasure {s | StateLinkFailure p two statementOf memoryOf s} +
          law.toOuterMeasure {s | RunFailure p two statementOf memoryOf s} := by
        refine (measure_union_le _ _).trans ?_
        gcongr
        exact measure_union_le _ _
    _ ≤ _ := by
        rw [add_assoc _ _ delta]
        gcongr
        calc law.toOuterMeasure {s | RunFailure p two statementOf memoryOf s}
            ≤ ((freq (g.Fails (context p) (Machine.application p two)) : ℝ≥0) : ℝ≥0∞) + delta :=
              transfer.transfer
          _ ≤ _ := by gcongr

end Probability

end NightstreamFPrime.Lifecycle.Nebula.EndToEnd
