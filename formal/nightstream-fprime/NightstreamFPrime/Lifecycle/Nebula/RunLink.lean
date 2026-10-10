import NightstreamFPrime.Lifecycle.Nebula.StepInvoke
import NightstreamFPrime.Lifecycle.Nebula.MachineRefinement

/-! Owns the run-level link of the first memory application (spec §11.1, §12,
§13). The input is what the Stage 1 relation chain and its terminal check
authenticate: the state words `z_0 … z_T`, one step witness per invocation
whose rows hold from `z_i` to `z_{i+1}`, and the §13 terminal checks with the
envelope's final carry. The output is the model's `Accepts` on the extracted
run, or a Poseidon2 collision between two state-digest inputs of the run. It
does not own the Stage 1 fold, the circuit that checks the rows, or the memory
security argument. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

variable {p : Plan} (two : p.bOps = 2)

namespace StepWitness

/-- One Stage 1 step of the memory application: the rows hold on the input
state words, and the output state words are the state digest of the step's
output words (spec §11.1). -/
structure Holds (w : StepWitness p) (zIn zOut : List F) : Prop where
  rows : w.RowsHold zIn
  machine : w.MachineRows two
  stateOut : zOut = stateWords (List.ofFn w.appOut) (List.ofFn w.carryOut)

/-- The invocation that a step witness gives the model. -/
def invocation (w : StepWitness p) : Invocation Machine.State Digest :=
  ⟨w.proposals, w.records, Machine.State.ofWords w.appOut⟩

end StepWitness

/-- The spec §13 terminal checks on the final state words `zT`, with the
statement fields and the envelope's final carry words. -/
structure TerminalChecks (p : Plan) (zT : List F) (finalApp : Fin 2 → F) (carry : Fin 39 → F)
    (steps segments finalTs : ℕ) (finalRoot : Digest) : Prop where
  state : zT = stateWords (List.ofFn finalApp) (List.ofFn carry)
  closed : (decodeCarry carry).idx = p.n
  segmentsRange : 1 ≤ segments ∧ segments ≤ p.sMax
  stepCount : steps = segments * p.n
  segIdx : (decodeCarry carry).segIdx = segments
  ts : (decodeCarry carry).ts = finalTs
  root : (decodeCarry carry).memRoot = finalRoot

theorem decodeCarry_startCarry (valid : p.Valid) :
    decodeCarry (carryVector (startCarry (context p))) = startCarry (context p) :=
  decodeCarry_carryVector ⟨by simp [startCarry, goldilocksModulus], Plan.n_lt valid,
    by simp [startCarry, goldilocksModulus]⟩

/-- Equal state words over two application words and 39 carry words give
equal words, or a collision between these two inputs. -/
theorem state_eq_or_collision {app app' : Fin 2 → F} {carry carry' : Fin 39 → F}
    (same : stateWords (List.ofFn app) (List.ofFn carry) =
      stateWords (List.ofFn app') (List.ofFn carry')) :
    (app = app' ∧ carry = carry') ∨
      StateCollision (List.ofFn app) (List.ofFn carry) (List.ofFn app') (List.ofFn carry') := by
  have small : ∀ n : ℕ, n ≤ 39 → n < goldilocksModulus := fun n h => by
    unfold goldilocksModulus
    omega
  rcases stateWords_eq_or_collision (by simpa using small 2 (by norm_num))
      (by simpa using small 2 (by norm_num)) (by simpa using small 39 le_rfl)
      (by simpa using small 39 le_rfl) same with ⟨appEq, carryEq⟩ | collision
  · exact Or.inl ⟨List.ofFn_injective appEq, List.ofFn_injective carryEq⟩
  · exact Or.inr collision

private theorem continueRun_snoc {E Digest σ : Type} [CommRing E] {ctx : Context E Digest}
    {c d d' : Carry E Digest} {run : List (Invocation σ Digest)} {inv : Invocation σ Digest}
    (before : continueRun ctx c run = some d) (step : invoke ctx d inv = some d') :
    continueRun ctx c (run ++ [inv]) = some d' := by
  unfold continueRun at before ⊢
  rw [List.foldlM_append, before]
  simp [step]

private theorem appThread_snoc {Digest σ : Type} {q : Plan} {app : Application q σ} :
    ∀ {s s' : σ} {run : List (Invocation σ Digest)} {inv : Invocation σ Digest},
      AppThread app s run s' → app.Step s' inv.ports inv.next →
        AppThread app s (run ++ [inv]) inv.next
  | _, _, [], _, thread, step => by
    cases thread
    exact ⟨step, rfl⟩
  | _, _, _ :: _, _, thread, step => ⟨thread.1, appThread_snoc thread.2 step⟩

/-- The application words and the carry words after `k` invocations. -/
def wordsAfter (w : ℕ → StepWitness p) (initialApp : Fin 2 → F) : ℕ → (Fin 2 → F) × (Fin 39 → F)
  | 0 => (initialApp, carryVector (startCarry (context p)))
  | k + 1 => ((w k).appOut, (w k).carryOut)

/-- The extracted run of the first `k` invocations. -/
def runOf (w : ℕ → StepWitness p) (k : ℕ) : List (Invocation Machine.State Digest) :=
  (List.range k).map fun i => (w i).invocation

/-- A state-digest collision at link `k` of the chain: the input words of
invocation `k` and the words after `k` invocations differ and have the same
state digest. -/
def LinkCollision (w : ℕ → StepWitness p) (initialApp : Fin 2 → F) (k : ℕ) : Prop :=
  StateCollision (List.ofFn (w k).appIn) (List.ofFn (w k).carryIn)
    (List.ofFn (wordsAfter w initialApp k).1) (List.ofFn (wordsAfter w initialApp k).2)

/-- A state-digest collision between two inputs of the run: at a link of the
chain, or between the words after the last invocation and the statement's
final words. Both sides of each collision are data of the run, so a
compressing hash does not make this event certain. -/
def RunStateCollision (w : ℕ → StepWitness p) (initialApp finalApp : Fin 2 → F)
    (finalCarry : Fin 39 → F) (T : ℕ) : Prop :=
  (∃ k < T, LinkCollision w initialApp k) ∨
    StateCollision (List.ofFn (wordsAfter w initialApp T).1)
      (List.ofFn (wordsAfter w initialApp T).2) (List.ofFn finalApp) (List.ofFn finalCarry)

variable {two}

/-- The state words after `k` invocations open to `wordsAfter k`. -/
theorem state_wordsAfter {T : ℕ} {z : ℕ → List F} {w : ℕ → StepWitness p}
    {initialApp : Fin 2 → F}
    (start : z 0 = stateWords (List.ofFn initialApp) (carryWords (startCarry (context p))))
    (steps : ∀ i < T, (w i).Holds two (z i) (z (i + 1))) :
    ∀ k ≤ T, z k = stateWords (List.ofFn (wordsAfter w initialApp k).1)
      (List.ofFn (wordsAfter w initialApp k).2)
  | 0, _ => start
  | k + 1, hk => (steps k (by omega)).stateOut

/-- The model runs the first `k` invocations to the decoded carry of
`wordsAfter k`, along the application thread, or a link before `k` collides. -/
theorem prefix_or_collision (valid : p.Valid) {T : ℕ} {z : ℕ → List F} {w : ℕ → StepWitness p}
    {initialApp : Fin 2 → F}
    (start : z 0 = stateWords (List.ofFn initialApp) (carryWords (startCarry (context p))))
    (steps : ∀ i < T, (w i).Holds two (z i) (z (i + 1))) :
    ∀ k ≤ T, (continueRun (context p) (startCarry (context p)) (runOf w k) =
        some (decodeCarry (wordsAfter w initialApp k).2) ∧
      Reach p (decodeCarry (wordsAfter w initialApp k).2) ∧
      AppThread (Machine.application p two) (Machine.State.ofWords initialApp) (runOf w k)
        (Machine.State.ofWords (wordsAfter w initialApp k).1)) ∨
      ∃ j < k, LinkCollision w initialApp j
  | 0, _ => by
    refine Or.inl ⟨?_, ?_, rfl⟩
    · rw [wordsAfter, decodeCarry_startCarry valid]
      rfl
    · rw [wordsAfter, decodeCarry_startCarry valid]
      exact reach_startCarry
  | k + 1, hk => by
    rcases prefix_or_collision valid start steps k (by omega) with
      ⟨before, reach, thread⟩ | collision
    · have holds := steps k (by omega)
      have opens := holds.rows.stateIn.trans (state_wordsAfter start steps k (by omega))
      rcases state_eq_or_collision opens with ⟨appIn, carryIn⟩ | collision
      · have inCarry : (w k).inCarry = decodeCarry (wordsAfter w initialApp k).2 := by
          rw [StepWitness.inCarry, carryIn]
        rw [← inCarry] at before reach
        obtain ⟨invoked, reachOut⟩ :=
          holds.rows.invoke valid reach (Machine.State.ofWords (w k).appOut)
        have machine := holds.rows.machineStep two valid holds.machine
        rw [appIn] at machine
        refine Or.inl ⟨?_, reachOut, ?_⟩
        · rw [runOf, List.range_succ, List.map_append, List.map_singleton]
          exact continueRun_snoc before invoked
        · rw [runOf, List.range_succ, List.map_append, List.map_singleton]
          exact appThread_snoc thread machine
      · exact Or.inr ⟨k, by omega, collision⟩
    · obtain ⟨j, below, collision⟩ := collision
      exact Or.inr ⟨j, by omega, collision⟩

/-- The run-level link: a chain of Stage 1 states whose steps satisfy the
memory rows and the machine rows, with the spec §13 terminal checks, is
accepted by the model on the extracted run, or two state-digest inputs of the
run collide. -/
theorem accepts_or_collision (valid : p.Valid) {T : ℕ} {z : ℕ → List F}
    {w : ℕ → StepWitness p} {initialApp finalApp : Fin 2 → F} {finalCarry : Fin 39 → F}
    {segments finalTs : ℕ} {finalRoot : Digest}
    (start : z 0 = stateWords (List.ofFn initialApp) (carryWords (startCarry (context p))))
    (steps : ∀ i < T, (w i).Holds two (z i) (z (i + 1)))
    (terminal : TerminalChecks p (z T) finalApp finalCarry T segments finalTs finalRoot) :
    Accepts (context p) (Machine.application p two)
        ⟨T, Machine.State.ofWords initialApp, Machine.State.ofWords finalApp, segments, finalTs,
          finalRoot⟩ (runOf w T) ∨ RunStateCollision w initialApp finalApp finalCarry T := by
  rcases prefix_or_collision valid start steps T le_rfl with ⟨ran, -, thread⟩ | collision
  · rcases state_eq_or_collision ((state_wordsAfter start steps T le_rfl).symm.trans
        terminal.state) with ⟨appEq, carryEq⟩ | collision
    · rw [appEq] at thread
      rw [carryEq] at ran
      exact Or.inl
        { steps := by simp [runOf]
          segmentsRange := terminal.segmentsRange
          stepCount := terminal.stepCount
          terminal := ⟨_, ran, terminal.closed, terminal.segIdx, terminal.ts, terminal.root⟩
          application := thread }
    · exact Or.inr (Or.inr collision)
  · exact Or.inr (Or.inl collision)

/-- Security note Lemma 6 for the memory application, at the relation level:
a chain of Stage 1 states whose steps satisfy the memory rows and the machine
rows, with the spec §13 terminal checks, attests a machine execution with the
statement's final timestamp and memory root, or gives a Poseidon2 transcript
collision (between two state-digest inputs of the run, or among the run's
chain inputs), or a segment
whose challenges pass the product test with unbalanced multisets. -/
theorem chain_soundness (valid : p.Valid) {T : ℕ} {z : ℕ → List F}
    {w : ℕ → StepWitness p} {initialApp finalApp : Fin 2 → F} {finalCarry : Fin 39 → F}
    {segments finalTs : ℕ} {finalRoot : Digest}
    (start : z 0 = stateWords (List.ofFn initialApp) (carryWords (startCarry (context p))))
    (steps : ∀ i < T, (w i).Holds two (z i) (z (i + 1)))
    (terminal : TerminalChecks p (z T) finalApp finalCarry T segments finalTs finalRoot) :
    Attests (context p) (Machine.application p two)
        ⟨T, Machine.State.ofWords initialApp, Machine.State.ofWords finalApp, segments, finalTs,
          finalRoot⟩ (runOf w T) ∨ RunStateCollision w initialApp finalApp finalCarry T ∨
      (∃ a ∈ runInputs (context p) (runOf w T) segments,
        ∃ b ∈ runInputs (context p) (runOf w T) segments, TranscriptCollision (blocks a) (blocks b)) ∨
      ∃ k < segments, BadChallenge (context p) (segmentView (context p) (runOf w T) k) := by
  rcases accepts_or_collision valid start steps terminal with accepted | collision
  · rcases poseidon2_soundness rfl valid accepted with attests | rest
    · exact Or.inl attests
    · exact Or.inr (Or.inr rest)
  · exact Or.inr (Or.inl collision)

end NightstreamFPrime.Lifecycle.Nebula
