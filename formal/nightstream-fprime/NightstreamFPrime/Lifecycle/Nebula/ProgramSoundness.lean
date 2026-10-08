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

variable {p : Plan} {two : p.bOps = 2}

/-- A valid program step with the program's output state is a run-link step. -/
theorem MemoryApp.holds_of_valid {input witness output : List F}
    (valid : MemoryApp.valid p two input witness) (out : output = MemoryApp.step input witness) :
    (MemoryApp.decode p witness).Holds two input output :=
  ⟨valid.1, valid.2, out⟩

/-- Security note Lemma 6 for the memory program: a chain of states whose steps
satisfy the program's step and validity predicate, with the spec §13 terminal
checks, attests a machine execution, or gives a Poseidon2 transcript collision
between two inputs of the run, or a segment whose challenges pass the product
test with unbalanced multisets. -/
theorem MemoryApp.chain_soundness (valid : p.Valid) {T : ℕ} {z : ℕ → List F}
    {witness : ℕ → List F} {initialApp finalApp : Fin 2 → F} {finalCarry : Fin 39 → F}
    {segments finalTs : ℕ} {finalRoot : Digest}
    (start : z 0 = stateWords (List.ofFn initialApp) (carryWords (startCarry (context p))))
    (steps : ∀ i < T, MemoryApp.valid p two (z i) (witness i) ∧
      z (i + 1) = MemoryApp.step (z i) (witness i))
    (terminal : TerminalChecks p (z T) finalApp finalCarry T segments finalTs finalRoot) :
    let run := runOf (fun i => MemoryApp.decode p (witness i)) T
    Attests (context p) (Machine.application p two)
        ⟨T, Machine.State.ofWords initialApp, Machine.State.ofWords finalApp, segments, finalTs,
          finalRoot⟩ run ∨
        RunStateCollision (fun i => MemoryApp.decode p (witness i)) initialApp finalApp finalCarry T ∨
      (∃ a ∈ runInputs (context p) run segments, ∃ b ∈ runInputs (context p) run segments,
        TranscriptCollision (blocks a) (blocks b)) ∨
      ∃ k < segments, BadChallenge (context p) (segmentView (context p) run k) :=
  Nebula.chain_soundness valid start
    (fun i below => MemoryApp.holds_of_valid (steps i below).1 (steps i below).2) terminal

end NightstreamFPrime.Lifecycle.Nebula
