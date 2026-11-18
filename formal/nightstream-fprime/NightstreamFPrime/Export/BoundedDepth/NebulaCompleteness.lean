import NightstreamFPrime.Export.BoundedDepth.RunCompleteness
import NightstreamFPrime.Lifecycle.Nebula.HonestRun
import NightstreamFPrime.Lifecycle.Nebula.MemoryProgram

/-! Completeness of a memory application on the bounded-depth scheme (CW-10.7, SPEC.md Algorithm 8). Every
execution of `S · N` steps with `1 ≤ S ≤ S_max` has a statement that passes the memory verifier's
checks (steps 3 and 4, `MemoryApp.history_of_execution`), and the bounded-depth verifier accepts the
honest prover's recursive proof of it (step 5, `memory_complete`). The fresh witness of that proof
is zero on every cell from its package width (`RunCompleteness.FreshZeroTail`). The proof is
recursive because `T = S · N ≥ 1`. The join is generic: a step history is a valid run
(`validFrom_of_history`), so `RunCompleteness.run_complete` accepts it (`accepts_of_history`), and
`S · N < q` follows from the geometry rules (`steps_lt`). When `S_max · N ≤ cap + 1`, as for each
WASM size (`Spec.Nebula.WasmSize.steps_le_light`), the memory verifier's checks bound the run by
`cap + 1` steps, so the honest proof has the light circuit last and slot 1 at the default slot of
depth 0, the light mode of SPEC.md Algorithm 9 (`memory_light_complete`, from
`RunCompleteness.light_run_complete`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.NebulaCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nebula
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.StepExtend
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open scoped NightstreamFPrime.Spec.FieldExtensionRing

/-- The first `k` witnesses of a step history are a valid run from its first state, and they reach
state `k`. -/
theorem validFrom_of_history (program : Stage1.Application.Program)
    (statement : TerminalStatement AppState) {z witness : ℕ → List F}
    (history : EndToEnd.History program statement z witness) :
    ∀ k ≤ statement.iteration,
      IvcWalk.ValidFrom program (z 0) ((List.range k).map witness) ∧
        ((List.range k).map witness).foldl program.step (z 0) = z k
  | 0, _ => ⟨trivial, rfl⟩
  | k + 1, below => by
      obtain ⟨valid, reaches⟩ := validFrom_of_history program statement history k (by omega)
      obtain ⟨stepValid, stepEq⟩ := history.2.2 k (by omega)
      rw [List.range_succ, List.map_append, List.map_singleton, IvcWalk.validFrom_append,
        List.foldl_append, reaches]
      exact ⟨⟨valid, stepValid, trivial⟩, stepEq.symm⟩

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
  {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
  {step : AppState → AppWitness → AppState}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}

/-- A step history of a statement for the exported program, with witnesses of its witness length,
a canonical iteration and a state of four words, has an honest proof: the verifier accepts it,
and its fresh witness is zero on every cell from its package width. -/
theorem accepts_of_history (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) {program : Stage1.Application.Program}
    (programEq : exports.program = program) {statement : TerminalStatement AppState}
    {z witness : ℕ → List F} (history : EndToEnd.History program statement z witness)
    (lengths : ∀ i < statement.iteration, (witness i).length = program.witnessWordCount)
    (z0Length : statement.z0.length = Stage1.Application.stateWordCount)
    (below : statement.iteration < fieldModulus) :
    ∃ proof, Verifies ajtai circuitKeys step complete.packageWidth cap control statement proof ∧
      FreshZeroTail complete.packageWidth proof := by
  subst programEq
  obtain ⟨valid, reaches⟩ :=
    validFrom_of_history exports.program statement history statement.iteration le_rfl
  have start : z 0 = statement.z0 := history.1
  rw [start] at valid reaches
  have witnessLengths : ∀ other ∈ (List.range statement.iteration).map witness,
      other.length = exports.program.witnessWordCount := by
    intro other inside
    obtain ⟨i, member, rfl⟩ := List.mem_map.mp inside
    exact lengths i (List.mem_range.mp member)
  have count : ((List.range statement.iteration).map witness).length = statement.iteration := by
    rw [List.length_map, List.length_range]
  have accepted := run_complete exports complete keysShaped cap control statement.z0 z0Length
    ((List.range statement.iteration).map witness) valid witnessLengths
    (by rw [count]; exact below)
  have last : ((List.range statement.iteration).map witness).foldl step statement.z0 =
      statement.zi := by
    rw [← exports.programStep, reaches]
    exact history.2.1
  rw [count, last] at accepted
  exact accepted

/-- A step history of `1` to `cap + 1` steps for the exported program, with witnesses of its
witness length and a state of four words, has a recursive proof that the verifier accepts, with
the light circuit last and slot 1 at the default slot of depth 0, and whose fresh witness is zero
on every cell from its package width (`RunCompleteness.FreshZeroTail`). -/
theorem light_of_history (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) {program : Stage1.Application.Program}
    (programEq : exports.program = program) {statement : TerminalStatement AppState}
    {z witness : ℕ → List F} (history : EndToEnd.History program statement z witness)
    (lengths : ∀ i < statement.iteration, (witness i).length = program.witnessWordCount)
    (z0Length : statement.z0.length = Stage1.Application.stateWordCount)
    (positive : 0 < statement.iteration)
    (short : statement.iteration ≤ Schedule.productionLightFoldCap + 1) :
    ∃ payload, Verifies ajtai circuitKeys step complete.packageWidth cap control statement
        (.recursive payload) ∧ payload.pc = 1 ∧ payload.running 1 = (defaultSlot, 0) ∧
      FreshZeroTail complete.packageWidth (.recursive payload) := by
  subst programEq
  obtain ⟨valid, reaches⟩ :=
    validFrom_of_history exports.program statement history statement.iteration le_rfl
  have start : z 0 = statement.z0 := history.1
  rw [start] at valid reaches
  have witnessLengths : ∀ other ∈ (List.range statement.iteration).map witness,
      other.length = exports.program.witnessWordCount := by
    intro other inside
    obtain ⟨i, member, rfl⟩ := List.mem_map.mp inside
    exact lengths i (List.mem_range.mp member)
  have count : ((List.range statement.iteration).map witness).length = statement.iteration := by
    rw [List.length_map, List.length_range]
  obtain ⟨payload, verifies, pcOne, slotOne, -, tail⟩ := light_run_complete exports complete keysShaped
    cap control statement.z0 z0Length ((List.range statement.iteration).map witness) valid
    witnessLengths (by rw [count]; exact positive) (by rw [count]; exact short)
  have last : ((List.range statement.iteration).map witness).foldl step statement.z0 =
      statement.zi := by
    rw [← exports.programStep, reaches]
    exact history.2.1
  rw [count, last] at verifies
  exact ⟨payload, verifies, pcOne, slotOne, tail⟩

/-- `S · N < q` for at most `S_max` segments of a valid geometry. -/
theorem steps_lt {p : Plan} (valid : p.Valid) {segments : ℕ} (most : segments ≤ p.sMax) :
    segments * p.n < fieldModulus := by
  have encoding : 2 ^ p.wTs < fieldModulus :=
    lt_of_le_of_lt (Nat.pow_le_pow_right (by norm_num) valid.fieldEncoding)
      (by norm_num [fieldModulus])
  calc segments * p.n ≤ p.sMax * p.n := Nat.mul_le_mul_right _ most
    _ ≤ p.sMax * p.n * p.bOps := Nat.le_mul_of_pos_right _ valid.positive.1
    _ < 2 ^ p.wTs := valid.timestampRange
    _ < fieldModulus := encoding

/-- Completeness of a memory application (SPEC.md Algorithm 8): for every execution of `S · N`
steps with `1 ≤ S ≤ S_max`, the statement that the verifier computes passes its checks, and the
bounded-depth verifier accepts the honest prover's recursive proof of it. The fresh witness of that
proof is zero on every cell from its package width. -/
theorem memory_complete {p : Plan} {app : PortApp p.toGeometry} (valid : p.Valid)
    (exports : Exports relation ajtai circuitKeys step cubeFits)
    (programEq : exports.program = MemoryApp.program p.toGeometry app)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2)
    {s₀ s : app.State} {final : Machine} {steps : List (List (Option PortAccess))}
    {segments : ℕ} (count : steps.length = segments * p.n)
    (range : 1 ≤ segments ∧ segments ≤ p.sMax)
    (exec : Executes p.toGeometry app.model s₀ (initialMachine p) steps s final) :
    ∃ carry : Fin 39 → F,
      let m : EndToEnd.MemoryStatement app.width :=
        ⟨s₀, s, carry, segments, final.ts,
          memoryRoot hash p.toGeometry (planDigest p.toGeometry) final.memory⟩
      let statement : TerminalStatement AppState :=
        ⟨segments * p.n, stateWords (List.ofFn m.initialApp) (carryWords (startCarry (context p))),
          stateWords (List.ofFn m.finalApp) (List.ofFn m.finalCarry)⟩
      EndToEnd.VerifierChecks p statement m ∧
        ∃ payload, Verifies ajtai circuitKeys step complete.packageWidth cap control statement
            (.recursive payload) ∧ FreshZeroTail complete.packageWidth (.recursive payload) := by
  obtain ⟨carry, z, witness, history, lengths, checks⟩ :=
    MemoryApp.history_of_execution valid count range exec
  refine ⟨carry, checks, ?_⟩
  obtain ⟨proof, verifies, tail⟩ := accepts_of_history exports complete keysShaped cap control
    programEq history lengths (stateWords_length _ _) (steps_lt valid range.2)
  cases proof with
  | bottom =>
    obtain ⟨-, iterationZero, -⟩ :=
      (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step control _).mp
        (holds_of_verifies verifies)
    have zero : segments * p.n = 0 := iterationZero
    exact absurd zero (Nat.mul_pos range.1 valid.positive.2.2.1).ne'
  | recursive payload => exact ⟨payload, verifies, tail⟩

/-- Completeness of the light mode of a memory application (SPEC.md Algorithms 8 and 9): at a
geometry with `S_max · N ≤ cap + 1`, for every execution of `S · N` steps with `1 ≤ S ≤ S_max`, the
statement that the verifier computes passes its checks, and the verifier accepts the honest
prover's recursive proof of it, whose last circuit is the light circuit, whose slot 1 is the
default slot at depth 0, and whose fresh witness is zero on every cell from its package width
(`RunCompleteness.FreshZeroTail`). So the memory verifier's bound on the segments keeps every accepted run
of such a geometry in the light mode. -/
theorem memory_light_complete {p : Plan} {app : PortApp p.toGeometry} (valid : p.Valid)
    (light : p.sMax * p.n ≤ Schedule.productionLightFoldCap + 1)
    (exports : Exports relation ajtai circuitKeys step cubeFits)
    (programEq : exports.program = MemoryApp.program p.toGeometry app)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2)
    {s₀ s : app.State} {final : Machine} {steps : List (List (Option PortAccess))}
    {segments : ℕ} (count : steps.length = segments * p.n)
    (range : 1 ≤ segments ∧ segments ≤ p.sMax)
    (exec : Executes p.toGeometry app.model s₀ (initialMachine p) steps s final) :
    ∃ carry : Fin 39 → F,
      let m : EndToEnd.MemoryStatement app.width :=
        ⟨s₀, s, carry, segments, final.ts,
          memoryRoot hash p.toGeometry (planDigest p.toGeometry) final.memory⟩
      let statement : TerminalStatement AppState :=
        ⟨segments * p.n, stateWords (List.ofFn m.initialApp) (carryWords (startCarry (context p))),
          stateWords (List.ofFn m.finalApp) (List.ofFn m.finalCarry)⟩
      EndToEnd.VerifierChecks p statement m ∧
        ∃ payload, Verifies ajtai circuitKeys step complete.packageWidth cap control statement
            (.recursive payload) ∧ payload.pc = 1 ∧ payload.running 1 = (defaultSlot, 0) ∧
          FreshZeroTail complete.packageWidth (.recursive payload) := by
  obtain ⟨carry, z, witness, history, lengths, checks⟩ :=
    MemoryApp.history_of_execution valid count range exec
  exact ⟨carry, checks, light_of_history exports complete keysShaped cap control programEq history
    lengths (stateWords_length _ _) (Nat.mul_pos range.1 valid.positive.2.2.1)
    ((Nat.mul_le_mul_right _ range.2).trans light)⟩

end NightstreamFPrime.Export.BoundedDepth.NebulaCompleteness
