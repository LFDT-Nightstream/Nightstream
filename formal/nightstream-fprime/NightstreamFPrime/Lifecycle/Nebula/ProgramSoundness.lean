import NightstreamFPrime.Lifecycle.Nebula.MemoryProgram
import NightstreamFPrime.Lifecycle.Nebula.RunLink

/-! Owns the join of the memory program's contract to the run link: each step
that the Stage 1 relation accepts for `MemoryApp.program` is a
`StepWitness.Holds` step, so a chain of accepted steps with the spec §13
terminal checks gives security note Lemma 6. It does not own the Stage 1 fold
that authenticates the chain. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

variable {p : Plan} {app : PortApp p}

/-- A valid program step with the program's output state is a run-link step. -/
theorem MemoryApp.holds_of_valid {input witness output : List F}
    (valid : MemoryApp.valid p app input witness) (out : output = MemoryApp.step p app input witness) :
    (MemoryApp.decode p app witness).Holds app input output :=
  ⟨valid.1, valid.2, out⟩

/-- Security note Lemma 6 for the memory program: a chain of states whose steps
satisfy the program's step and validity predicate, with the spec §13 terminal
checks, attests an execution, or gives a Poseidon2 transcript collision
between two inputs of the run, or a segment whose challenges pass the product
test with unbalanced multisets. -/
theorem MemoryApp.chain_soundness (valid : p.Valid) {T : ℕ} {z : ℕ → List F}
    {witness : ℕ → List F} {initialApp finalApp : Fin app.width → F} {finalCarry : Fin 39 → F}
    {segments finalTs : ℕ} {finalRoot : Digest}
    (start : z 0 = stateWords (List.ofFn initialApp) (carryWords (startCarry (context p))))
    (steps : ∀ i < T, MemoryApp.valid p app (z i) (witness i) ∧
      z (i + 1) = MemoryApp.step p app (z i) (witness i))
    (terminal : TerminalChecks p (z T) finalApp finalCarry T segments finalTs finalRoot) :
    let run := runOf app (fun i => MemoryApp.decode p app (witness i)) T
    Attests (context p) app.model
        ⟨T, app.ofWords initialApp, app.ofWords finalApp, segments, finalTs, finalRoot⟩ run ∨
        RunStateCollision (fun i => MemoryApp.decode p app (witness i)) initialApp finalApp
          finalCarry T ∨
      (∃ a ∈ runInputs (context p) run segments, ∃ b ∈ runInputs (context p) run segments,
        TranscriptCollision (blocks a) (blocks b)) ∨
      ∃ k < segments, BadChallenge (context p) (segmentView (context p) run k) :=
  Nebula.chain_soundness valid start
    (fun i below => MemoryApp.holds_of_valid (steps i below).1 (steps i below).2) terminal

end NightstreamFPrime.Lifecycle.Nebula
