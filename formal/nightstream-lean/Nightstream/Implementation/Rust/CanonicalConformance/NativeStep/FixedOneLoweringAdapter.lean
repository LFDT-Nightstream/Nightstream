import Nightstream.Implementation.Lowering.FPrimeFixedOne.Step
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal
import Nightstream.Implementation.Lowering.Goldilocks.Codec
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter

/-!
Contract: semantic alignment between the universal fixed-one native adapter
and the artifact-independent fixed-one typed lowering vocabulary.

Assurance tier: model-level.

Owns:
- the exact lowering parameterization induced by the native adapter;
- exact semantics for the six calls whose physical recipes remain open;
- equivalence of the typed Step program with the direct paper step relation;
- equivalence of the typed Terminal program with the supplied paper terminal
  relation.

Does not own: codecs, physical recipes, row or column numbers, generated
artifacts, compiled-Rust semantics, or concrete terminal relations.  Widths
and footprints are carried only because the typed vocabulary separates
semantics from a later encoding profile.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter

open Nightstream.HyperNova.Construction2
open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Protocol.FPrime

namespace Native

open FixedOneCanonicalAdapter

abbrev AdapterParameters
    (Params StructureDigest Header Digest Running Fresh NifsProof Nebula
      NebulaDigest NebulaOpen : Type) :=
  FixedOneCanonicalAdapter.Parameters
    Params StructureDigest Header Digest Running Fresh NifsProof Nebula
      NebulaDigest NebulaOpen

abbrev DirectState
    (Digest Running Fresh Nebula : Type) :=
  State Digest Running Fresh Nebula

abbrev AdapterWitness
    (Digest Fresh NifsProof Nebula NebulaOpen : Type) :=
  FixedOneCanonicalAdapter.Witness
    Digest Fresh NifsProof Nebula NebulaOpen

abbrev AdapterFresh
    (Digest Fresh Nebula : Type) :=
  FixedOneCanonicalAdapter.FreshInput Digest Fresh Nebula

abbrev AdapterEncoded (Digest : Type) :=
  FixedOneCanonicalAdapter.Encoded Digest

end Native

namespace Lowering

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

abbrev Parameters :=
  Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Parameters

end Lowering

section

variable
  {Params StructureDigest Header Digest Running Fresh NifsProof Nebula
    NebulaDigest NebulaOpen : Type}

local notation "AdapterParameters" =>
  Native.AdapterParameters Params StructureDigest Header Digest Running Fresh
    NifsProof Nebula NebulaDigest NebulaOpen

local notation "DirectState" =>
  Native.DirectState Digest Running Fresh Nebula

local notation "AdapterWitness" =>
  Native.AdapterWitness Digest Fresh NifsProof Nebula NebulaOpen

local notation "AdapterFresh" =>
  Native.AdapterFresh Digest Fresh Nebula

local notation "AdapterEncoded" =>
  Native.AdapterEncoded Digest

local notation "PaperKey" =>
  XOut.Context Params StructureDigest Header Digest

/-- Terminal semantics and encoding-shape placeholders needed to place one
native adapter in the closed lowering vocabulary.

The shape fields are not certified here.  A later concrete codec/profile must
derive the widths and every physical recipe must prove its exact footprint. -/
structure Configuration
    (adapter : AdapterParameters) where
  RunningWitness : Type
  FreshWitness : Type
  terminalRelations :
    TerminalRelations PaperKey Running RunningWitness AdapterFresh
      FreshWitness 1
  terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations
  widths :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Widths
  footprints :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Footprints

/-- The exact typed-lowering semantics induced by the fixed-one native
adapter.  No physical encoding fact is asserted by this construction. -/
def parameters
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : Configuration adapter) :
    Lowering.Parameters where
  Field := Nightstream.Implementation.Lowering.Goldilocks.Field
  fieldZero := 0
  fieldAdd := fun left right => left + right
  fieldMul := fun left right => left * right
  Key := PaperKey
  Digest := Option Digest
  State := Option DirectState
  Witness := AdapterWitness
  Running := Running
  Fresh := AdapterFresh
  NifsProof := Step.FoldProof NifsProof
  Encoded := AdapterEncoded
  RunningWitness := configuration.RunningWitness
  FreshWitness := configuration.FreshWitness
  stateDecidableEq := inferInstance
  encodedDecidableEq := inferInstance
  setup := FixedOneCanonicalAdapter.setup adapter
  machine := FixedOneCanonicalAdapter.machine adapter
  terminalRelations := configuration.terminalRelations
  terminalChecks := configuration.terminalChecks
  widths := configuration.widths
  footprints := configuration.footprints

namespace CallAlignment

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

variable
  [DecidableEq Digest]
  [DecidableEq Running]
  [DecidableEq Fresh]
  [DecidableEq NifsProof]
  [DecidableEq Nebula]
  [DecidableEq NebulaOpen]

/-- The lowering application call is definitionally the totalized fixed-one
native application, including every enumerated rejection branch. -/
theorem step
    (adapter : AdapterParameters)
    (configuration : Configuration adapter)
    (state : Option DirectState)
    (witness : AdapterWitness) :
    callEval (parameters adapter configuration) Call.step
        (.cons state (.cons witness .nil)) =
      some
        (.cons
          (FixedOneCanonicalAdapter.application adapter state witness)
          .nil) :=
  rfl

/-- The prior-hash call uses the authoritative iteration without adjustment
and the unique fixed-one program counter. -/
theorem hashPrior
    (adapter : AdapterParameters)
    (configuration : Configuration adapter)
    (iteration : Nat)
    (z0 current : Option DirectState)
    (running : Running) :
    callEval (parameters adapter configuration) Call.hashPrior
        (.cons iteration
          (.cons z0 (.cons current (.cons running .nil)))) =
      some
        (.cons
          (FixedOneCanonicalAdapter.paperHash adapter {
            verifierKeys :=
              (FixedOneCanonicalAdapter.setup adapter).verifierKeys
            iteration := iteration
            z0 := z0
            current := current
            running := fun _ => running
            pc := oneBased
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          })
          .nil) :=
  rfl

/-- The next-hash call differs from the prior call only by the paper-mandated
single increment of the iteration coordinate. -/
theorem hashNext
    (adapter : AdapterParameters)
    (configuration : Configuration adapter)
    (iteration : Nat)
    (z0 current : Option DirectState)
    (running : Running) :
    callEval (parameters adapter configuration) Call.hashNext
        (.cons iteration
          (.cons z0 (.cons current (.cons running .nil)))) =
      some
        (.cons
          (FixedOneCanonicalAdapter.paperHash adapter {
            verifierKeys :=
              (FixedOneCanonicalAdapter.setup adapter).verifierKeys
            iteration := iteration + 1
            z0 := z0
            current := current
            running := fun _ => running
            pc := oneBased
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          })
          .nil) :=
  rfl

/-- The lowering NIFS call is exactly the adapter setup verifier, with key,
running value, fresh value, and proof in paper order. -/
theorem nifsVerify
    (adapter : AdapterParameters)
    (configuration : Configuration adapter)
    (running : Running)
    (fresh : AdapterFresh)
    (proof : Step.FoldProof NifsProof) :
    callEval (parameters adapter configuration) Call.nifsVerify
        (.cons running (.cons fresh (.cons proof .nil))) =
      match proof with
      | .noFold => none
      | .recursive nifsProof =>
          match
              adapter.step.nifsVerify fresh.nifsContext running fresh.ordered
                nifsProof with
          | none => none
          | some folded => some (.cons folded .nil) :=
  by
    cases proof with
    | noFold =>
      simp [
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.callEval,
        parameters,
        FixedOneCanonicalAdapter.setup,
        FixedOneCanonicalAdapter.nifsVerifier
      ]
    | recursive nifsProof =>
      simp only [
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.callEval,
        parameters,
        FixedOneCanonicalAdapter.setup,
        FixedOneCanonicalAdapter.nifsVerifier
      ]
      cases adapter.step.nifsVerify fresh.nifsContext running fresh.ordered
          nifsProof <;>
        rfl

/-- The running terminal call uses the supplied exact relation checker at the
unique fixed-one slot and verifier key. -/
theorem runningCheck
    (adapter : AdapterParameters)
    (configuration : Configuration adapter)
    (running : Running)
    (witness : configuration.RunningWitness) :
    callEval (parameters adapter configuration) Call.runningCheck
        (.cons running (.cons witness .nil)) =
      some
        (.cons
          (configuration.terminalChecks.runningCheck
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            ((FixedOneCanonicalAdapter.setup adapter).verifierKeys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running witness)
          .nil) :=
  rfl

/-- The fresh terminal call uses the supplied exact relation checker at the
same unique slot and verifier key. -/
theorem freshCheck
    (adapter : AdapterParameters)
    (configuration : Configuration adapter)
    (fresh : AdapterFresh)
    (witness : configuration.FreshWitness) :
    callEval (parameters adapter configuration) Call.freshCheck
        (.cons fresh (.cons witness .nil)) =
      some
        (.cons
          (configuration.terminalChecks.freshCheck
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            ((FixedOneCanonicalAdapter.setup adapter).verifierKeys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            fresh witness)
          .nil) :=
  rfl

end CallAlignment

/-- The intrinsic typed Step program instantiated through this adapter accepts
exactly the direct paper step relation. -/
theorem stepAccepts_iff_directHolds
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : Configuration adapter)
    (prior next : DirectState)
    (nextInput : Step.Input Fresh Nebula NebulaOpen)
    (proof : Step.Proof Digest NifsProof NebulaOpen) :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        (parameters adapter configuration)
        (FixedOneCanonicalAdapter.input adapter prior nextInput proof)
        (FixedOneCanonicalAdapter.output adapter next proof) ↔
      Step.Holds adapter.hash adapter.step adapter.mode adapter.context
        prior next nextInput proof := by
  rw [
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_transition
  ]
  exact
    FixedOneCanonicalAdapter.transition_iff_holds
      adapter prior next nextInput proof

/-- The intrinsic typed Terminal program instantiated through this adapter
accepts exactly the supplied independent paper terminal relation. -/
theorem terminalAccepts_iff_transition
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration : Configuration adapter)
    (statement : TerminalStatement (Option DirectState))
    (proof :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Proof
        Running configuration.RunningWitness AdapterFresh
          configuration.FreshWitness) :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
        (parameters adapter configuration) statement proof ↔
      TerminalTransition
        (FixedOneCanonicalAdapter.setup adapter)
        (FixedOneCanonicalAdapter.machine adapter)
        configuration.terminalRelations statement proof.toGeneric := by
  exact
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.accepts_iff_transition
      (parameters adapter configuration) statement proof

end

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter
