import NightstreamFPrime.Lifecycle.Nebula.ProgramSoundness
import NightstreamFPrime.Lifecycle.Nebula.MemoryBound
import NightstreamFPrime.Spec.HyperNova.Construction2.Paper
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-! Owns the end-to-end soundness of the first memory application over two
named premises (owner decision 2026-10-08). It does not prove either premise.
Every event reads the history that the Stage 1 extractor returns for the
sample. No event or conclusion quantifies over other histories: for a
compressing state digest, some other history of a statement always contains a
collision.

* `Stage1Extraction`: on an accepted sample, the extracted history is a step
  history of the sample's statement for the program, or a named Stage 1
  failure occurs. Stage 1 extraction is today written for the hash-chain
  application only.
* `MemoryRoundTransfer`: security note A6 for the memory round, one segment
  slot at a time, at the first unbalanced segment of the extracted run.

The deterministic theorem joins the first premise to Lemma 6 for the memory
relation. The probability theorem adds the second premise and the per-slot
game bound `badAt_bound`. This module gives no term a number. -/

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

/-- What the end-to-end layer reads from one sample: the Stage 1 statement,
the memory statement with the envelope's final carry, and the history that
the Stage 1 extractor returns. A randomized extractor keeps its coins in the
sample. -/
structure Experiment (Sample : Type) where
  statement : Sample → TerminalStatement AppState
  memory : Sample → MemoryStatement
  states : Sample → ℕ → List F
  witness : Sample → ℕ → List F

/-- The Stage 1 extraction premise for `program`: on an accepted sample, the
extracted history is a history of the sample's statement, or the named
Stage 1 failure occurs. -/
structure Stage1Extraction (program : Application.Program) {Sample : Type}
    (x : Experiment Sample) (Accepted Failure : Sample → Prop) : Prop where
  extract : ∀ s, Accepted s →
    History program (x.statement s) (x.states s) (x.witness s) ∨ Failure s

namespace Experiment

variable {Sample : Type} (x : Experiment Sample) (p) (two)

/-- The step witnesses that the extractor returns. -/
def steps (s : Sample) (i : ℕ) : StepWitness p := MemoryApp.decode p (x.witness s i)

/-- The extracted run of the model. -/
def run (s : Sample) : List (Invocation Machine.State Digest) :=
  runOf (x.steps p s) (x.statement s).iteration

/-- The typed statement that the memory statement claims. -/
def claim (s : Sample) : Nebula.Statement Machine.State Digest :=
  (x.memory s).typed (x.statement s).iteration

/-- The extracted run attests a machine execution for the claim. -/
def Attested (s : Sample) : Prop :=
  Attests (context p) (Machine.application p two) (x.claim s) (x.run p s)

/-- A state-digest collision between two inputs of the extracted run
(security note Lemma 6 step 2). The extractor outputs both sides. -/
def StateLinkCollision (s : Sample) : Prop :=
  RunStateCollision (x.steps p s) (x.memory s).initialApp (x.memory s).finalApp
    (x.memory s).finalCarry (x.statement s).iteration

/-- A Poseidon2 transcript collision among the chain inputs of the extracted
run and the `D_init` chain (Lemma 2). -/
def ChainCollision (s : Sample) : Prop :=
  ∃ a ∈ runInputs (context p) (x.run p s) (x.memory s).segments,
    ∃ b ∈ runInputs (context p) (x.run p s) (x.memory s).segments,
      TranscriptCollision (blocks a) (blocks b)

/-- The model accepts the extracted run, and `k` is its first unbalanced
segment: the only event that A6 transfers. The earlier segments are balanced,
so they close with any challenge. -/
def FirstUnbalanced (k : Fin p.sMax) (s : Sample) : Prop :=
  Accepts (context p) (Machine.application p two) (x.claim s) (x.run p s) ∧
    k.val < (x.memory s).segments ∧
    ¬ ((segmentView (context p) (x.run p s) k).multisets p).Balanced ∧
    ∀ j < k.val, ((segmentView (context p) (x.run p s) j).multisets p).Balanced

/-- Security note Lemma 6 on the extracted run: an execution, a state-link
collision, a chain collision, or a segment whose challenges pass the product
test with unbalanced multisets. -/
def Outcome (s : Sample) : Prop :=
  x.Attested p two s ∨ x.StateLinkCollision p s ∨ x.ChainCollision p s ∨
    ∃ k < (x.memory s).segments, BadChallenge (context p) (segmentView (context p) (x.run p s) k)

/-- An accepted sample that passes the spec §13 checks, but whose extracted
history is not a history that attests the claim (security note §5). -/
def Bad (Accepted : Sample → Prop) (s : Sample) : Prop :=
  Accepted s ∧ VerifierChecks p (x.statement s) (x.memory s) ∧
    ¬ (History (MemoryApp.program p two) (x.statement s) (x.states s) (x.witness s) ∧
      x.Attested p two s)

end Experiment

variable {Sample : Type} {x : Experiment Sample} {Accepted Failure : Sample → Prop}

/-- End-to-end soundness, deterministic: on an accepted sample that passes the
spec §13 checks, the extracted history is a history with the Lemma 6 outcome,
or the named Stage 1 failure occurs. -/
theorem soundness (valid : p.Valid)
    (extraction : Stage1Extraction (MemoryApp.program p two) x Accepted Failure)
    {s : Sample} (accepted : Accepted s)
    (checks : VerifierChecks p (x.statement s) (x.memory s)) :
    (History (MemoryApp.program p two) (x.statement s) (x.states s) (x.witness s) ∧
      x.Outcome p two s) ∨ Failure s := by
  rcases extraction.extract s accepted with ⟨z0, zT, steps⟩ | failure
  · exact Or.inl ⟨⟨z0, zT, steps⟩,
      MemoryApp.chain_soundness valid (z0.trans checks.start) steps (zT ▸ checks.terminal)⟩
  · exact Or.inr failure

/-- The deterministic cover: a bad sample has a Stage 1 failure, a state-link
collision, a chain collision, or a first unbalanced segment in an accepted
extracted run. -/
theorem bad_subset (valid : p.Valid)
    (extraction : Stage1Extraction (MemoryApp.program p two) x Accepted Failure) :
    {s | x.Bad p two Accepted s} ⊆
      {s | Failure s} ∪ {s | x.StateLinkCollision p s} ∪ {s | x.ChainCollision p s} ∪
        ⋃ k : Fin p.sMax, {s | x.FirstUnbalanced p two k s} := by
  intro s ⟨accepted, checks, notGood⟩
  rcases extraction.extract s accepted with ⟨z0, zT, steps⟩ | failure
  · rcases accepts_or_collision valid (z0.trans checks.start)
        (fun i below => MemoryApp.holds_of_valid (steps i below).1 (steps i below).2)
        (zT ▸ checks.terminal) with ran | link
    · rcases poseidon2_soundness rfl valid ran with attests | chain | ⟨k, hk, bad⟩
      · exact absurd ⟨⟨z0, zT, steps⟩, attests⟩ notGood
      · exact Or.inl (Or.inr chain)
      · classical
        have unbalanced : ∃ j, j < (x.memory s).segments ∧
            ¬ ((segmentView (context p) (x.run p s) j).multisets p).Balanced := ⟨k, hk, bad.1⟩
        have first := Nat.find_spec unbalanced
        refine Or.inr (Set.mem_iUnion.mpr
          ⟨⟨Nat.find unbalanced, lt_of_lt_of_le first.1 ran.segmentsRange.2⟩,
            ran, first.1, first.2, fun j below => ?_⟩)
        by_contra notBalanced
        exact Nat.find_min unbalanced below ⟨lt_trans below first.1, notBalanced⟩
    · exact Or.inl (Or.inl (Or.inr link))
  · exact Or.inl (Or.inl (Or.inl failure))

/-! ### Probability -/

open MeasureTheory
open scoped ENNReal NNReal

/-- Security note A6 for the memory round, one segment slot at a time (owner
decision 2026-10-08: its own premise). The real mass of samples whose
extracted run has first unbalanced segment `k` is at most `loss` times the
frequency of `BadAt k` in `games k`, plus `delta`.

The intended `games k` runs the adversary and the extractor. It answers the
oracle query for segment `k`'s transcript input with the slot-`k` challenge,
commits that input before the challenge, and cuts the run after segment `k`.
The earlier segments are balanced, so they close with any challenge. A §9.3
transcript input has no segment index, so an earlier segment with the same
input gets the same challenge; this does not matter for the same reason.
`loss` is the query factor: a translation that guesses which of the `Q`
evaluations of the §9.3 transcript answers segment `k` (the relation evaluates
it in every invocation, and off-circuit evaluations count too), or that none
does, has `loss = Q + 1`. `delta` holds the rest of A6, including the
extractor's rewinds. No game, `loss`, `delta`, or proof is given here. -/
structure MemoryRoundTransfer (law : PMF Sample) (x : Experiment Sample) {Coins : Type}
    [Fintype Coins] (games : Fin p.sMax → Game K Digest Machine.State Coins p.sMax)
    (loss delta : ℝ≥0∞) : Prop where
  transfer : ∀ k, law.toOuterMeasure {s | x.FirstUnbalanced p two k s} ≤
    loss * ((freq ((games k).BadAt (context p) (Machine.application p two) k) : ℝ≥0) : ℝ≥0∞) +
      delta

/-- End-to-end soundness with probability: the mass of bad samples is at most
the Stage 1 failure mass, the state-link and chain collision masses of the
extracted runs, and, per segment slot, `loss` times the Lemma 3 game bound
plus `delta`. -/
theorem bad_probability (valid : p.Valid)
    (extraction : Stage1Extraction (MemoryApp.program p two) x Accepted Failure)
    (law : PMF Sample) {Coins : Type} [Fintype Coins]
    {games : Fin p.sMax → Game K Digest Machine.State Coins p.sMax} {loss delta : ℝ≥0∞}
    (transfer : MemoryRoundTransfer (two := two) law x games loss delta) :
    law.toOuterMeasure {s | x.Bad p two Accepted s} ≤
      law.toOuterMeasure {s | Failure s} + law.toOuterMeasure {s | x.StateLinkCollision p s} +
        law.toOuterMeasure {s | x.ChainCollision p s} +
        ∑ k : Fin p.sMax, (loss * (((2 * (p.maxTuples : ℚ≥0) / (goldilocksModulus ^ 2 : ℕ) +
          (games k).retryTerm (context p) (Machine.application p two) k : ℚ≥0) : ℝ≥0) : ℝ≥0∞) +
          delta) := by
  calc law.toOuterMeasure {s | x.Bad p two Accepted s}
      ≤ law.toOuterMeasure ({s | Failure s} ∪ {s | x.StateLinkCollision p s} ∪
          {s | x.ChainCollision p s} ∪ ⋃ k : Fin p.sMax, {s | x.FirstUnbalanced p two k s}) :=
        measure_mono (bad_subset valid extraction)
    _ ≤ law.toOuterMeasure {s | Failure s} + law.toOuterMeasure {s | x.StateLinkCollision p s} +
          law.toOuterMeasure {s | x.ChainCollision p s} +
          ∑ k : Fin p.sMax, law.toOuterMeasure {s | x.FirstUnbalanced p two k s} := by
        refine (measure_union_le _ _).trans ?_
        gcongr
        · refine (measure_union_le _ _).trans ?_
          gcongr
          exact measure_union_le _ _
        · exact measure_iUnion_fintype_le _ _
    _ ≤ _ := by
        gcongr with k
        refine (transfer.transfer k).trans ?_
        gcongr
        exact badAt_bound (Machine.application p two) (games k) valid k

end NightstreamFPrime.Lifecycle.Nebula.EndToEnd
