import NightstreamFPrime.Export.BoundedDepth.EtaSponge
import NightstreamFPrime.Export.BoundedDepth.NebulaCompleteness
import NightstreamFPrime.Export.BoundedDepth.NebulaSecurity
import NightstreamFPrime.Lifecycle.Nebula.RunLink
import NightstreamFPrime.Spec.Nebula.Completeness
import NightstreamFPrime.Spec.Nebula.EventBinding
import NightstreamFPrime.Spec.Nebula.Soundness

/-! # Claims: Nebula memory

Soundness and completeness of the memory checks (`SPEC.md` section 7 and Algorithm 8,
`SECURITY_MODEL.md` adversary 5).

- Paper: Nebula offline memory checking.
- `model_sound`, `model_complete` and `accepts_or_collision` are deterministic.
- `memory_light_complete` is `memory_complete` in the light mode of `SPEC.md` Algorithm 9, at a
  geometry with `S_max · N ≤ cap + 1`.
- `catalog_agrees` is deterministic. Its premises are the conditions of a catalog application
  (`SPEC.md` section 7.6). No Lean result shows that the Rust rows enforce them.
- Assumptions: `eta_balanced` takes `[SPONGE]` under `[P2-PERM]`. `memory_sound` takes `[HN-A1]`.
- Open: `memory_sound` takes the memory round as a premise (`SECURITY_MODEL.md` trust item 6).
  A memory application that Rust builds, the WASM relation included, has no Lean instance. -/

namespace NightstreamFPrime.Claims.Nebula

open Spec Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped Spec.FieldExtensionRing

/-- Deterministic memory soundness (`SECURITY_MODEL.md`, adversary 5). -/
theorem model_sound
    {E : Type}
    {Digest : Type}
    {σ : Type}
    [ring : CommRing E]
    {ctx : Nebula.Context E Digest}
    {app : Nebula.Application ctx.plan.toGeometry σ}
    {stmt : Nebula.Statement σ Digest}
    {run : List (Nebula.Invocation σ Digest)}
    (valid : ctx.plan.Valid)
    (accepted : Nebula.Accepts ctx app stmt run) :
    Nebula.Attests ctx app stmt run ∨
        Nebula.RunCollision ctx run stmt.segments ∨
          ∃ k < stmt.segments, Nebula.BadChallenge ctx (Nebula.segmentView ctx run k) :=
  @Spec.Nebula.soundness E Digest σ ring ctx app stmt run valid accepted

/-- Completeness of the model. -/
theorem model_complete
    {E : Type}
    {Digest : Type}
    {σ : Type}
    [ring : CommRing E]
    {ctx : Nebula.Context E Digest}
    {app : Nebula.Application ctx.plan.toGeometry σ}
    {s₀ : σ}
    {s : σ}
    {final : Nebula.Machine}
    {steps : List (List (Option Nebula.PortAccess))}
    {segments : ℕ}
    (valid : ctx.plan.Valid)
    (width : ∀ ports ∈ steps, ports.length = ctx.plan.bOps)
    (count : steps.length = segments * ctx.plan.n)
    (range : 1 ≤ segments ∧ segments ≤ ctx.plan.sMax)
    (exec : Nebula.Executes ctx.plan.toGeometry app s₀ (Nebula.initialMachine ctx.plan) steps s final) :
    ∃ (run : List (Nebula.Invocation σ Digest)),
        Nebula.Accepts ctx app (Nebula.honestStatement ctx s₀ s final segments) run ∧
          List.map Nebula.Invocation.ports run = steps :=
  @Spec.Nebula.completeness E Digest σ ring ctx app s₀ s final steps segments valid width count
    range exec

/-- The run-level link: a chain of Stage 1 states whose steps satisfy the
memory rows and the application's rows, with the SPEC.md Algorithm 8 terminal checks, is
accepted by the model on the extracted run, or two state-digest inputs of the
run collide. -/
theorem accepts_or_collision
    {p : Nebula.Plan}
    {app : Nebula.PortApp p.toGeometry}
    (valid : p.Valid)
    {T : ℕ}
    {z : ℕ → List F}
    {w : ℕ → Nebula.StepWitness p.toGeometry app.toAppShape}
    {initialApp : Fin app.width → F}
    {finalApp : Fin app.width → F}
    {finalCarry : Fin 39 → F}
    {segments : ℕ}
    {finalTs : ℕ}
    {finalRoot : Nebula.Digest}
    (start : z 0 =
        Nebula.stateWords (List.ofFn initialApp)
          (Nebula.carryWords (Nebula.startCarry (Nebula.context p))))
    (steps : ∀ i < T, Nebula.StepWitness.Holds app (w i) (z i) (z (i + 1)))
    (terminal : Nebula.TerminalChecks p.toGeometry (z T) finalApp finalCarry T segments finalTs finalRoot) :
    Nebula.Accepts (Nebula.context p) app.model
          { steps := T, initial := initialApp, final := finalApp, segments := segments,
            finalTs := finalTs, finalRoot := finalRoot }
          (Nebula.runOf app w T) ∨
        Nebula.RunStateCollision w initialApp finalApp finalCarry T :=
  @Lifecycle.Nebula.accepts_or_collision p app valid T z w initialApp finalApp finalCarry segments
    finalTs finalRoot start steps terminal

/-- Adversary 5's memory round in the ideal-permutation model: under `Spec.Sponge.Transfer` of the
unbalanced event with the `η` conditions, the wins of each `world` are at most
`(queries + 1) · 2 · m_mem/q²`, plus the retry term of the transfer's sampler, plus
`spongeError calls`. Each pair that the retry term counts is a Poseidon2 transcript collision. -/
theorem eta_balanced
    {p : Nebula.Plan}
    (valid : p.Valid)
    (app : Nebula.PortApp p.toGeometry)
    {Sample : Type}
    (x : Nebula.EndToEnd.Experiment app.width Sample)
    {queries : ℕ}
    {World : Type}
    (winsOf : World → ℝ)
    (calls : ℕ)
    (premise : Sponge.Transfer EtaSponge.Conditions winsOf
        (fun (sampler : Nebula.EtaRound.EtaSampler Sample queries) =>
          Nebula.EtaRound.EtaSampler.unbalancedChance app x sampler)
        calls)
    (world : World) :
    ∃ sampler : Nebula.EtaRound.EtaSampler Sample queries,
      winsOf world ≤
        (↑(queries + 1) : ℝ) * (2 * (↑p.maxTuples : ℝ) / (↑(fieldModulus ^ 2) : ℝ)) +
          Nebula.EtaRound.EtaSampler.retryCollision app x sampler + Sponge.spongeError calls :=
  @Export.BoundedDepth.EtaSponge.unbalanced_le p valid app Sample x queries World winsOf calls
    premise world

/-- The memory round in the random-oracle model of `η` (CW-10.8): for an `η` sampler of samples
with `queries` queries and a secure geometry, the chance over the sampler's coins and a uniform
oracle that the extracted run is unbalanced with the oracle's challenges is at most
`(queries + 1) · 2^−109.91` plus the sampler's retry collision term. At the Poseidon2 oracle the
event is `Unbalanced` (`EtaRound.unbalancedIn_deployed`), but no result bounds it there, so
`MemoryRound` stays a premise. -/
theorem memory_round_in_random_oracle_model
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {p : Nebula.Plan}
    (app : Nebula.PortApp p.toGeometry)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    (valid : p.Valid)
    (secure : p.Secure)
    {queries : ℕ}
    (sampler : Nebula.EtaRound.EtaSampler (NebulaSecurity.Sample app.width logicalWidth publicFits) queries) :
    Nebula.EtaRound.EtaSampler.unbalancedChance app
          (NebulaSecurity.experiment app ajtai circuitKeys exports) sampler ≤
        (↑(queries + 1) : ℝ) * (2 : ℝ) ^ (-(10991 / 100 : ℝ)) +
          Nebula.EtaRound.EtaSampler.retryCollision app
            (NebulaSecurity.experiment app ajtai circuitKeys exports) sampler :=
  @Export.BoundedDepth.NebulaSecurity.romMemoryRound logicalWidth publicFits p app relation ajtai
    circuitKeys step cubeFits exports valid secure queries sampler

/-- The bound of adversary 5 for a memory application on the bounded-depth scheme: on the joint law of the
memory statement that the adversary claims from its own tape and the sample of the walk's
`S_max · N` stages, the mass of samples that the memory verifier accepts but whose extracted
history does not attest the claim is at most the walk's finder and stage terms (`IvcSecurity.walkBound`), the state-link and
chain collision masses of the extracted runs, and the memory-round error of the last stage. -/
theorem memory_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {p : Nebula.Plan}
    (app : Nebula.PortApp p.toGeometry)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
          (IvcWalk.Answer logicalWidth publicFits) →
        Prop}
    {Efficient : (adversary :
          HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
        HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
          Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds ajtai)
        (IvcWalk.WalkExtracted ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf ajtai circuitKeys exports)
        (IvcWalk.back ajtai circuitKeys exports) Admitted StageAdmitted Efficient)
    (adversary : HyperNova.ReverseStages.IvcAdversary (IvcSecurity.Output logicalWidth publicFits))
    (admitted : StageAdmitted (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary adversary)))
    (valid : p.Valid)
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (programEq : exports.program = Nebula.MemoryApp.program p.toGeometry app)
    (control : AppState → AppWitness → Fin 2)
    (cap : ℕ)
    (claim : adversary.Tape → Nebula.EndToEnd.MemoryStatement app.width)
    {memoryError : NebulaSecurity.ClaimedStage (logicalWidth := logicalWidth) (publicFits := publicFits) app →
        ℝ}
    (round : NebulaSecurity.MemoryRound app ajtai circuitKeys exports StageAdmitted memoryError) :
    have c :=
        {
          stage :=
            (↑(HyperNova.ReverseStages.reverseStages (Succeeds := IvcWalk.WalkSucceeds ajtai)
                  (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                  (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption closed
                  (IvcSecurity.walkAdversary adversary) admitted (p.sMax * p.n)) :
              HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                (IvcWalk.Answer logicalWidth publicFits)),
          claim :=
            claim ∘
              HyperNova.ReverseStages.origin (Succeeds := IvcWalk.WalkSucceeds ajtai)
                (Extracted := IvcWalk.WalkExtracted ajtai) (Admitted := Admitted)
                (StageAdmitted := StageAdmitted) (Efficient := Efficient) assumption closed
                (IvcSecurity.walkAdversary adversary) admitted (p.sMax * p.n) :
          NebulaSecurity.ClaimedStage (logicalWidth := logicalWidth) (publicFits := publicFits)
            app };
      ENNReal.toReal
          (((NebulaSecurity.ClaimedStage.law app c).toOuterMeasure :
              Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
            {s : NebulaSecurity.Sample app.width logicalWidth publicFits |
              Nebula.EndToEnd.Experiment.Bad app
                (NebulaSecurity.experiment app ajtai circuitKeys exports)
                (NebulaSecurity.Checked app ajtai circuitKeys (step := step) control cap) s}) ≤
        IvcSecurity.walkBound ajtai circuitKeys exports assumption closed adversary admitted
                (p.sMax * p.n) +
              ENNReal.toReal
                (((NebulaSecurity.ClaimedStage.law app c).toOuterMeasure :
                    Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                  {s : NebulaSecurity.Sample app.width logicalWidth publicFits |
                    Nebula.EndToEnd.Experiment.StateLinkCollision app
                      (NebulaSecurity.experiment app ajtai circuitKeys exports) s}) +
            ENNReal.toReal
              (((NebulaSecurity.ClaimedStage.law app c).toOuterMeasure :
                  Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                {s : NebulaSecurity.Sample app.width logicalWidth publicFits |
                  Nebula.EndToEnd.Experiment.ChainCollision app
                    (NebulaSecurity.experiment app ajtai circuitKeys exports) s}) +
          memoryError c :=
  @Export.BoundedDepth.NebulaSecurity.checked_le logicalWidth publicFits p app relation ajtai
    circuitKeys step cubeFits exports Admitted StageAdmitted Efficient error assumption closed
    adversary admitted valid keysShaped programEq control cap claim memoryError round

/-- Completeness of a memory application (SPEC.md Algorithm 8): for every execution of `S · N`
steps with `1 ≤ S ≤ S_max`, the statement that the verifier computes passes its checks, and the
verifier of SPEC.md Algorithm 6 accepts the honest prover's recursive proof of it. The fresh
witness of that proof is zero on every cell from the width of its circuit's package, as Rust's
direct verifier requires (`RunCompleteness.FreshZeroTail`). -/
theorem memory_complete
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    {p : Nebula.Plan}
    {app : Nebula.PortApp p.toGeometry}
    (valid : p.Valid)
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    (programEq : exports.program = Nebula.MemoryApp.program p.toGeometry app)
    (complete : StepExtend.Complete exports)
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {s₀ : app.State}
    {s : app.State}
    {final : Nebula.Machine}
    {steps : List (List (Option Nebula.PortAccess))}
    {segments : ℕ}
    (count : steps.length = segments * p.n)
    (range : 1 ≤ segments ∧ segments ≤ p.sMax)
    (exec : Nebula.Executes p.toGeometry app.model s₀ (Nebula.initialMachine p) steps s final) :
    ∃ (carry : Fin 39 → F),
        have m :=
          { initialApp := s₀, finalApp := s, finalCarry := carry, segments := segments,
            finalTs := final.ts,
            finalRoot :=
              Nebula.memoryRoot Nebula.hash p.toGeometry (Nebula.planDigest p.toGeometry)
                final.memory :
            Nebula.EndToEnd.MemoryStatement app.width };
        have statement :=
          { iteration := segments * p.n,
            z0 :=
              Nebula.stateWords (List.ofFn m.initialApp)
                (Nebula.carryWords (Nebula.startCarry (Nebula.context p))),
            zi := Nebula.stateWords (List.ofFn m.finalApp) (List.ofFn m.finalCarry) :
            HyperNova.Construction2.Paper.TerminalStatement AppState };
        Nebula.EndToEnd.VerifierChecks p statement m ∧
          ∃ (payload :
            HyperNova.Construction2.Paper.TerminalProof (SlotData logicalWidth publicFits)
              (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
              (StepFresh logicalWidth publicFits)
              (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2),
            RunCompleteness.Verifies ajtai circuitKeys step complete.packageWidth cap control
                statement (HyperNova.Construction2.Paper.OuterTerminalProof.recursive payload) ∧
              RunCompleteness.FreshZeroTail complete.packageWidth
                (HyperNova.Construction2.Paper.OuterTerminalProof.recursive payload) :=
  @Export.BoundedDepth.NebulaCompleteness.memory_complete logicalWidth publicFits relation ajtai
    circuitKeys step cubeFits p app valid exports programEq complete keysShaped cap control s₀ s
    final steps segments count range exec

/-- Completeness of the light mode of a memory application (SPEC.md Algorithms 8 and 9): at a
geometry with `S_max · N ≤ cap + 1`, as for each WASM size, for every execution of `S · N` steps
with `1 ≤ S ≤ S_max`, the statement that the verifier computes passes its checks, and the verifier
of SPEC.md Algorithm 6 accepts the honest prover's recursive proof of it. That proof has the light
circuit last and slot 1 at the default slot of depth 0, and its fresh witness is zero on every cell
from the width of its circuit's package (`RunCompleteness.FreshZeroTail`). -/
theorem memory_light_complete
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    {p : Nebula.Plan}
    {app : Nebula.PortApp p.toGeometry}
    (valid : p.Valid)
    (light : p.sMax * p.n ≤ Folding.Schedule.productionLightFoldCap + 1)
    (exports : TerminalBridge.Exports relation ajtai circuitKeys step cubeFits)
    (programEq : exports.program = Nebula.MemoryApp.program p.toGeometry app)
    (complete : StepExtend.Complete exports)
    (keysShaped : ∀ (key : Fin 2), List.length (circuitKeys key).digest = 4)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {s₀ : app.State}
    {s : app.State}
    {final : Nebula.Machine}
    {steps : List (List (Option Nebula.PortAccess))}
    {segments : ℕ}
    (count : steps.length = segments * p.n)
    (range : 1 ≤ segments ∧ segments ≤ p.sMax)
    (exec : Nebula.Executes p.toGeometry app.model s₀ (Nebula.initialMachine p) steps s final) :
    ∃ (carry : Fin 39 → F),
        have m :=
          { initialApp := s₀, finalApp := s, finalCarry := carry, segments := segments,
            finalTs := final.ts,
            finalRoot :=
              Nebula.memoryRoot Nebula.hash p.toGeometry (Nebula.planDigest p.toGeometry)
                final.memory :
            Nebula.EndToEnd.MemoryStatement app.width };
        have statement :=
          { iteration := segments * p.n,
            z0 :=
              Nebula.stateWords (List.ofFn m.initialApp)
                (Nebula.carryWords (Nebula.startCarry (Nebula.context p))),
            zi := Nebula.stateWords (List.ofFn m.finalApp) (List.ofFn m.finalCarry) :
            HyperNova.Construction2.Paper.TerminalStatement AppState };
        Nebula.EndToEnd.VerifierChecks p statement m ∧
          ∃ (payload :
            HyperNova.Construction2.Paper.TerminalProof (SlotData logicalWidth publicFits)
              (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
              (StepFresh logicalWidth publicFits)
              (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2),
            RunCompleteness.Verifies ajtai circuitKeys step complete.packageWidth cap control
                statement (HyperNova.Construction2.Paper.OuterTerminalProof.recursive payload) ∧
              payload.pc = 1 ∧ payload.running 1 = (defaultSlot, 0) ∧
              RunCompleteness.FreshZeroTail complete.packageWidth
                (HyperNova.Construction2.Paper.OuterTerminalProof.recursive payload) :=
  @Export.BoundedDepth.NebulaCompleteness.memory_light_complete logicalWidth publicFits relation
    ajtai circuitKeys step cubeFits p app valid light exports programEq complete keysShaped cap
    control s₀ s final steps segments count range exec

/-- The memory guarantee for a catalog application (`SPEC.md` section 7.6): for an execution from
the plan images whose active ports bind a reorder of the events of the trace, the events replay
from the initial catalog state to a state that agrees with the memory of the final machine on
every live mapped cell. -/
theorem catalog_agrees
    {catalog : List Nebula.MemoryTrace.MemorySpec}
    {b : Nebula.MemoryTrace.Binding}
    {p : Nebula.Plan}
    {σ : Type}
    {app : Nebula.Application p.toGeometry σ}
    {s : σ}
    {s' : σ}
    {final : Nebula.Machine}
    {steps : List (List (Option Nebula.PortAccess))}
    {initial : Nebula.MemoryTrace.MemoryState}
    {rows : List Nebula.MemoryTrace.Witness}
    {order : List Nebula.MemoryTrace.Event}
    (exec : Nebula.Executes p.toGeometry app s (Nebula.initialMachine p) steps s' final)
    (disjoint : b.Disjoint catalog)
    (inRange : ∀ e ∈ Nebula.MemoryTrace.eventsForTrace catalog rows, b.InRange e.memory e.address)
    (images : b.Initial catalog p initial)
    (reorders : Nebula.MemoryTrace.Reorders (Nebula.MemoryTrace.eventsForTrace catalog rows) order)
    (bound : List.Forall₂ (b.Binds catalog) order (steps.flatMap fun ports => ports.filterMap id)) :
    ∃ state,
      Nebula.MemoryTrace.Replay initial (Nebula.MemoryTrace.eventsForTrace catalog rows) state ∧
        Nebula.MemoryTrace.Agrees p.toGeometry catalog b state final.memory :=
  @Spec.Nebula.MemoryTrace.agrees_of_executes catalog b p σ app s s' final steps initial
    rows order exec disjoint inRange images reorders bound

end NightstreamFPrime.Claims.Nebula
