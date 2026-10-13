import NightstreamFPrime.Lifecycle.Nebula.HonestStep
import NightstreamFPrime.Lifecycle.Nebula.EndToEnd

/-! Owns Ob5 for runs of a memory application: the model's `invoke` keeps a
carry reachable, a run that the model accepts is a Stage 1 step history of the
memory program, and every execution of the application has such a history
whose statement passes the spec §13 checks. It does not own the per-step
witness (`HonestStep`) or the Stage 1 fold. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper (TerminalStatement)
open NightstreamFPrime.Lifecycle.Stage1

variable {p : Plan} {app : PortApp p}

/-- The model's `invoke` keeps the carry reachable. -/
theorem MemoryApp.reach_invoke (valid : p.Valid) {c c' : MemoryCarry} {σ : Type}
    {inv : Invocation σ Digest} (reach : Reach p c)
    (accepts : invoke (context p) c inv = some c') : Reach p c' := by
  obtain ⟨accepted, rfl⟩ := Honest.accepted p c inv accepts
  have below : c.segIdx < p.sMax := by
    by_cases opens : c.idx = p.n
    · exact accepted.segBelow opens
    · exact reach.openSeg opens
  have idx : (Honest.finished p c inv).idx = (Honest.start p c inv).idx + 1 := by
    rw [Honest.finished_idx]
    rfl
  have idxLt := Honest.start_idx_lt (inv := inv) valid reach
  have boundary := accepted.rows.boundary
  have seg := Honest.finished_segIdx (p := p) (c := c) (inv := inv)
  refine ⟨?_, ?_, fun open_ => ?_, ?_⟩
  · rw [seg]
    split <;> omega
  · omega
  · rw [seg, if_neg (by rw [← Honest.finished_idx, idx]; omega)]
    rw [Honest.finished_idx] at open_
    omega
  · rw [Honest.finished_ts, Honest.stepped_ts]
    exact boundary

/-- A run that the model accepts from a reachable carry ends at a reachable
carry. -/
theorem MemoryApp.reach_continueRun (valid : p.Valid) {σ : Type} :
    ∀ {run : List (Invocation σ Digest)} {c c' : MemoryCarry},
      Reach p c → continueRun (context p) c run = some c' → Reach p c'
  | [], _, _, reach, ran => by
    simp only [continueRun, List.foldlM_nil, Option.pure_def, Option.some.injEq] at ran
    exact ran ▸ reach
  | _ :: _, _, _, reach, ran => by
    simp only [continueRun, List.foldlM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff] at ran
    obtain ⟨_, step, tail⟩ := ran
    exact MemoryApp.reach_continueRun valid (MemoryApp.reach_invoke valid reach step) tail

/-- A run that the model accepts from a reachable carry, along the
application thread, is a step history of the memory program between the two
state digests. -/
private theorem MemoryApp.history_of_run (valid : p.Valid) :
    ∀ {run : List (Invocation app.State Digest)} {c c' : MemoryCarry} {s s' : app.State},
      Reach p c → continueRun (context p) c run = some c' →
      AppThread app.model s run s' →
      ∃ z witness : ℕ → List F,
        z 0 = stateWords (List.ofFn s) (carryWords c) ∧
        z run.length = stateWords (List.ofFn (s')) (carryWords c') ∧
        (∀ i < run.length, (MemoryApp.program p app).valid (z i) (witness i) ∧
          z (i + 1) = (MemoryApp.program p app).step (z i) (witness i)) ∧
        ∀ i < run.length, (witness i).length = Words.count p app.toAppShape
  | [], c, _, s, _, _, ran, thread => by
    simp only [continueRun, List.foldlM_nil, Option.pure_def, Option.some.injEq] at ran
    subst ran
    have same : s = _ := thread
    subst same
    exact ⟨fun _ => stateWords (List.ofFn s) (carryWords c), fun _ => [], rfl, rfl,
      fun i below => absurd below (Nat.not_lt_zero i), fun i below => absurd below (Nat.not_lt_zero i)⟩
  | inv :: rest, c, _, s, _, reach, ran, thread => by
    simp only [continueRun, List.foldlM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff] at ran
    obtain ⟨d, step, tail⟩ := ran
    obtain ⟨machine, thread⟩ := thread
    obtain ⟨w, lengthW, validW, stepW⟩ := MemoryApp.complete valid reach step machine
    have stepW : (MemoryApp.program p app).step (stateWords (List.ofFn s) (carryWords c)) w =
        stateWords (List.ofFn (inv.next)) (carryWords d) := stepW
    obtain ⟨z, witness, z0, zT, steps, lengths⟩ :=
      MemoryApp.history_of_run valid (MemoryApp.reach_invoke valid reach step) tail thread
    refine ⟨fun i => if i = 0 then stateWords (List.ofFn s) (carryWords c) else z (i - 1),
      fun i => if i = 0 then w else witness (i - 1), rfl, ?_, ?_, ?_⟩
    · simpa using zT
    · intro i below
      cases i with
      | zero => exact ⟨validW, by simpa [z0] using stepW.symm⟩
      | succ j => simpa using steps j (by simpa using below)
    · intro i below
      cases i with
      | zero => simpa using lengthW
      | succ j => simpa using lengths j (by simpa using below)

/-- Every step of an execution uses `B_ops` ports. -/
private theorem executes_width {s s' : app.State} {m m' : Machine}
    {steps : List (List (Option PortAccess))} (exec : Executes p app.model s m steps s' m') :
    ∀ ports ∈ steps, ports.length = p.bOps := by
  induction exec with
  | nil => intro ports member; simp at member
  | step machine _ _ ih =>
    intro ports member
    rcases List.mem_cons.mp member with rfl | member
    · exact app.portsLength machine
    · exact ih ports member

/-- Ob5 for a memory application at the run level: every execution of
`S · N` steps with `1 ≤ S ≤ S_max` has a Stage 1 step history of the memory
program for the statement that the verifier computes, and that statement
passes the spec §13 checks. This includes `S = S_max`, idle steps, and every
reachable counter value. -/
theorem MemoryApp.history_of_execution (valid : p.Valid)
    {s₀ s : app.State} {final : Machine} {steps : List (List (Option PortAccess))}
    {segments : ℕ} (count : steps.length = segments * p.n)
    (range : 1 ≤ segments ∧ segments ≤ p.sMax)
    (exec : Executes p app.model s₀ (initialMachine p) steps s final) :
    ∃ (carry : Fin 39 → F) (z witness : ℕ → List F),
      let m : EndToEnd.MemoryStatement app.width :=
        ⟨s₀, s, carry, segments, final.ts,
          memoryRoot hash p (planDigest p) final.memory⟩
      let statement : TerminalStatement AppState :=
        ⟨segments * p.n, stateWords (List.ofFn m.initialApp) (carryWords (startCarry (context p))),
          stateWords (List.ofFn m.finalApp) (List.ofFn m.finalCarry)⟩
      EndToEnd.History (MemoryApp.program p app) statement z witness ∧
        (∀ i < segments * p.n, (witness i).length = Words.count p app.toAppShape) ∧
        EndToEnd.VerifierChecks p statement m := by
  obtain ⟨run, accepted, -⟩ := Spec.Nebula.completeness (ctx := context p)
    (app := app.model) valid (executes_width exec) count range exec
  obtain ⟨c, ran, idx, seg, ts, root⟩ := accepted.terminal
  have decoded := decodeCarry_carryVector
    (Honest.canonical valid (MemoryApp.reach_continueRun valid reach_startCarry ran))
  obtain ⟨z, witness, z0, zT, steps, lengths⟩ :=
    MemoryApp.history_of_run valid reach_startCarry ran accepted.application
  have length : run.length = segments * p.n := accepted.steps
  rw [length] at zT steps lengths
  refine ⟨carryVector c, z, witness, ⟨z0, zT, steps⟩, lengths, rfl, ?_⟩
  exact { state := rfl
          closed := by rw [decoded]; exact idx
          segmentsRange := range
          stepCount := rfl
          segIdx := by rw [decoded]; exact seg
          ts := by rw [decoded]; exact ts
          root := by rw [decoded]; exact root }

end NightstreamFPrime.Lifecycle.Nebula
