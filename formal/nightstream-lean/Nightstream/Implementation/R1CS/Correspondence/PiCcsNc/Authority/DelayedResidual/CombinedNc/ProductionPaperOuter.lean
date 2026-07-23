import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTrace
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.PaperConstruction2

/-!
Opening-derived production paper transitions at the active outer boundary.

Protocol: HyperNova Construction 2 over the SuperNeo fixed-active NIFS.
Phase: selected outer invocation after production NIFS closure.
Assurance tier: model-level composition.

Owns: a small outer adapter whose fresh statement, selected parent, fourteen
selected children, source alignment, and Split-NC public input are all
computed from one `SourceInput.Carrier`; exact equality of the resulting
no-pending active context with the installed opening-derived context; and
projection of an already-closed production paper transition to Construction
2.

Does not own: recursive delayed-pending installation, production Rust/R1CS
refinement, transcript or commitment primitive internals, costs, or row
removal.

Authority boundary: no public source equality is a theorem premise.  The
outer source and verifier context are constructed from the same carrier.
The explicit `context.pending = none` premise is the base/singleton lifecycle
boundary.  It cannot be dropped for recursive delayed steps: the current
active `Context.Template.build` hard-codes `pending := none`, while a delayed
step binds `some outgoingPending` into `piCcsStatement` before deriving its
children.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperOuter

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

universe uOuterKey uAppState uWitness uDigest uState

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {State : Type uState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

private theorem sourceProduct_eq_of_fields
    (left right : SourceProduct shape publicRingColumns publicFits
      (CommitmentValue verifierRows) productionGlobalParams FixedActive.arity)
    (fresh : left.fresh = right.fresh)
    (running : left.running = right.running) :
    left = right := by
  cases left with
  | mk leftFresh leftRunning =>
    cases right with
    | mk rightFresh rightRunning =>
      cases fresh
      cases running
      rfl

private theorem context_eq_of_fields
    (left right : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (covers : left.covers = right.covers)
    (key : left.key = right.key)
    (alignment : left.alignment = right.alignment)
    (input : left.input = right.input)
    (runningParent : left.runningParent = right.runningParent)
    (pending : left.pending = right.pending)
    (piCcsInput : left.piCcsInput = right.piCcsInput)
    (priorState : left.priorState = right.priorState)
    (piCcsSchedule : left.piCcsSchedule = right.piCcsSchedule)
    (piRlcMachine : left.piRlcMachine = right.piRlcMachine)
    (profile : left.profile = right.profile)
    (challengeSetSize : left.challengeSetSize = right.challengeSetSize) :
    left = right := by
  cases left with
  | mk leftCovers leftKey leftAlignment leftInput leftRunningParent leftPending
      leftPiCcsInput leftPriorState leftPiCcsSchedule leftPiRlcMachine
      leftProfile leftChallengeSetSize =>
    cases right with
    | mk rightCovers rightKey rightAlignment rightInput rightRunningParent
      rightPending rightPiCcsInput rightPriorState rightPiCcsSchedule
      rightPiRlcMachine rightProfile rightChallengeSetSize =>
      cases covers
      cases key
      cases alignment
      cases input
      cases runningParent
      cases pending
      cases piCcsInput
      cases priorState
      cases piCcsSchedule
      cases piRlcMachine
      cases profile
      cases challengeSetSize
      rfl

/-- The exact selected rich slot reconstructed from one complete opening. -/
def selectedSlot
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) :
    Slot shape publicRingColumns publicFits verifierRows where
  parent := carrier.opening.parent key carrier.system
  children := carrier.opening.children key carrier.system

/-- Replace only the selected outer slot with the opening-derived family. -/
def runningOf
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount) :
    Running shape publicRingColumns publicFits verifierRows slotCount :=
  fun slot => if slot = selected then selectedSlot carrier key else inactive slot

@[simp] theorem runningOf_selected
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount) :
    runningOf carrier key selected inactive selected = selectedSlot carrier key := by
  simp [runningOf]

/-- Canonical active setup for one closed opening-derived production step.

The outer key remains a HyperNova-level identifier.  The concrete verifier
key and all transcript machines come from the checked opening context. -/
def setupOf
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows) :
    Setup OuterKey AppState Witness State shape publicRingColumns publicFits
      verifierRows slotCount where
  template := fun _ _ => {
    covers := context.covers
    key := context.key
    alignment := carrier.alignment
    piCcsSchedule := context.piCcsSchedule
    piRlcMachine := context.piRlcMachine
    profile := context.profile
    challengeSetSize := context.challengeSetSize
  }
  expectedStructure := fun _ _ => carrier.system
  piCcsInput := fun _ _ => carrier.piCcsInput
  priorTranscriptState := fun _ _ => context.priorState

/-- Canonical outer input for the selected opening-derived production step.

The one fresh public statement and the complete selected running family are
read from the same carrier.  `inactive` is used only away from `selected`. -/
def inputOf
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness) :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount where
  verifierKey := outerKey
  iteration := iteration
  z0 := z0
  zi := zi
  running := runningOf carrier context.key selected inactive
  fresh := (carrier.sourceProduct context.key).fresh
    ⟨0, FixedActive.arity.freshPositive⟩
  priorPc := priorPc
  witness := witness

/-- The source product installed by the selected outer invocation is exactly
the direct source product computed from the authoritative carrier. -/
theorem invocation_sourceProduct_eq
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness) :
    (invocationAt (setupOf carrier context)
      (inputOf carrier context selected outerKey iteration z0 zi inactive
        priorPc witness) selected).sourceProduct =
      carrier.sourceProduct context.key := by
  apply sourceProduct_eq_of_fields
  · funext fresh
    have freshEq : fresh = ⟨0, FixedActive.arity.freshPositive⟩ := by
      apply Fin.ext
      have freshLt : fresh.val < 1 := by
        simpa only [FixedActive.arity_freshCount] using fresh.isLt
      change fresh.val = 0
      omega
    subst fresh
    rfl
  · funext child
    change
      (runningOf carrier context.key selected inactive selected).children
          child =
        (carrier.sourceProduct context.key).running child
    rw [runningOf_selected]
    change carrier.opening.children context.key carrier.system child =
      (carrier.sourceProduct context.key).running child
    exact carrier.opening_children_eq_sourceProduct_running context.key child

/-- At the no-pending boundary, the active outer constructor reaches exactly
the opening-derived production context.  No source/context equality is
supplied by the caller. -/
theorem contextAt_eq_installed_of_pending_none
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (noPending : context.pending = none) :
    contextAt (setupOf carrier context)
        (inputOf carrier context selected outerKey iteration z0 zi inactive
          priorPc witness) selected =
      (carrier.install context).full := by
  apply context_eq_of_fields
  · rfl
  · rfl
  · rfl
  · exact
      (invocation_sourceProduct_eq carrier context selected outerKey iteration
        z0 zi inactive priorPc witness).trans
        (carrier.input_materialize_eq_sourceProduct context.key).symm
  · change
      some (runningOf carrier context.key selected inactive selected).parent =
        some (carrier.opening.parent context.key carrier.system)
    rw [runningOf_selected]
  · exact noPending.symm
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl

/-- Active certificate whose selected NIFS message is the message already
checked by the opening-derived production step.  Its dependent index is the
computed Split-NC public input on both sides. -/
def certificateOf
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.Certificate
      (setupOf carrier context)
      (inputOf carrier context selected outerKey iteration z0 zi inactive
        priorPc witness) where
  selected := selected
  nifs := certificate

/-- A closed opening-derived production transition composes with the active
outer checks into HyperNova Construction 2 at the explicit no-pending
boundary.

Every public source field is computed above.  The only semantic premise is
the exact paper transition already established by the production trace; no
raw-old-block authority, source match, child sidecar, output-unbound branch,
or implementation-refinement failure is assumed. -/
theorem run_refinesConstruction2_of_openingPaperTransition
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (nifsCertificate : FixedActive.Certificate (carrier.install context).full)
    (paperTransition :
      FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf (carrier.install context).full)
        (carrier.install context).full.input
        (outputChildren (carrier.install context).full nifsCertificate))
    (noPending : context.pending = none)
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (checkers :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.Checkers
        (setupOf carrier context) machine functionIndex)
    (output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount)
    (executed :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run checkers
      (inputOf carrier context selected outerKey iteration z0 zi inactive
        priorPc witness)
      (certificateOf carrier context selected outerKey iteration z0 zi
        inactive priorPc witness nifsCertificate) = some output) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
      (Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.family
        (Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsSetup
          (setupOf carrier context)))
      machine functionIndex
      (inputOf carrier context selected outerKey iteration z0 zi inactive
        priorPc witness).toPaper
      output.toPaper := by
  apply
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_refinesConstruction2_of_paperTransition
      checkers
    (inputOf carrier context selected outerKey iteration z0 zi inactive
      priorPc witness)
    (certificateOf carrier context selected outerKey iteration z0 zi inactive
      priorPc witness nifsCertificate) output
  · rw [contextAt_eq_installed_of_pending_none carrier context selected
      outerKey iteration z0 zi inactive priorPc witness noPending]
    exact paperTransition
  · exact executed

/-- Direct trace-facing form of the outer bridge.  It consumes the exact
paper transition in `OutputClosed`; the trace's base proof supplies
`noPending` for the first step. -/
theorem run_refinesConstruction2_of_traceOutput
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (step : ProductionPaperTrace.CheckedStep (State := State) key)
    (closed : ProductionPaperTrace.OutputClosed step)
    (noPending : step.context.pending = none)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (checkers :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.Checkers
        (setupOf step.carrier step.context) machine functionIndex)
    (output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount)
    (executed :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run checkers
      (inputOf step.carrier step.context selected outerKey iteration z0 zi
        inactive priorPc witness)
      (certificateOf step.carrier step.context selected outerKey iteration z0
        zi inactive priorPc witness step.certificate) = some output) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
      (Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.family
        (Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsSetup
          (setupOf step.carrier step.context)))
      machine functionIndex
      (inputOf step.carrier step.context selected outerKey iteration z0 zi
        inactive priorPc witness).toPaper
      output.toPaper := by
  exact run_refinesConstruction2_of_openingPaperTransition step.carrier
    step.context selected outerKey iteration z0 zi inactive priorPc witness
    step.certificate closed.paper noPending machine functionIndex checkers
    output executed

/-! ## Pending-compatible canonical outer verifier

The legacy active evaluator reinterprets its certificate under a context whose
pending field is fixed to `none`.  The canonical verifier below does not do
that.  It executes the outer checks on the paper-visible input and executes
the production paper checker on the opening context that owns the actual
lifecycle pending value.  The two results meet only at the independently
stated paper transition, whose profile consists of the verifier key and
source alignment and whose input is the carrier-computed source product.
-/

/-- Canonical output for one opening-derived active step.  The rich parent is
computed from the same certificate, while the paper projection exposes only
the children required by Construction 2. -/
def canonicalOuterOutput
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (certificate : FixedActive.Certificate (carrier.install context).full)
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount) :
    Output Digest AppState shape publicRingColumns publicFits verifierRows
      slotCount :=
  outputOf machine
    (inputOf carrier context selected outerKey iteration z0 zi inactive
      priorPc witness)
    selected (FixedActive.resultOf (carrier.install context).full certificate)

/-- Executable canonical active check.  The first conjunct owns only the
HyperNova outer equalities.  The second executes the opening-derived
`Pi_CCS`/sampler/paper-`Pi_DEC`/parent-opening checker under the context's
actual pending value.  The legacy NIFS callback inside `checkers` is not
executed here. -/
def canonicalCheck
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (certificate : FixedActive.Certificate (carrier.install context).full)
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (checkers :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.Checkers
        (setupOf carrier context) machine functionIndex) : Bool :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.outerCheck checkers
      (inputOf carrier context selected outerKey iteration z0 zi inactive
        priorPc witness) selected &&
    ProductionPaperChecker.check carrier context certificate

/-- An exact paper transition under the lifecycle-owned pending context is
also the public selected-NIFS transition used by Construction 2.  No
certificate is reinterpreted under the legacy pending-none context. -/
private theorem paperTransition_implies_selectedTransition
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (target : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (paperTransition :
      FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf (carrier.install context).full)
        (carrier.install context).full.input target) :
    Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.Transition
      (Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsSetup
        (setupOf carrier context))
      outerKey selected
      (Nightstream.Protocol.FPrime.Paper.Construction2.selectedInput
        (inputOf carrier context selected outerKey iteration z0 zi inactive
          priorPc witness).toPaper selected)
      target := by
  apply
    Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.transition_of_paper
      (incomingParent := carrier.opening.parent context.key carrier.system)
      (polynomial := carrier.piCcsInput)
      (priorState := context.priorState)
  simpa [
    Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.contextOf,
    Nightstream.Protocol.FPrime.ConcretePhi81.SelectedNifsSemantics.contextOf,
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsSetup,
    setupOf, inputOf, invocationAt,
    Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template.build,
    Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct,
    FixedActive.CanonicalOpening.SourceInput.Carrier.install,
    FixedActive.paperProfileOf] using paperTransition

/-- Pending-compatible active outer composition.  The executable outer check
and one exact opening-derived paper transition imply HyperNova Construction
2 for the canonical output.  There is no `pending = none` premise and no
generic implementation or output-unbound alternative. -/
theorem outerCheck_and_paperTransition_imply_construction2
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (certificate : FixedActive.Certificate (carrier.install context).full)
    (paperTransition :
      FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf (carrier.install context).full)
        (carrier.install context).full.input
        (outputChildren (carrier.install context).full certificate))
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (checkers :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.Checkers
        (setupOf carrier context) machine functionIndex)
    (outerChecked :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.outerCheck
        checkers
        (inputOf carrier context selected outerKey iteration z0 zi inactive
          priorPc witness) selected = true) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
      (Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.family
        (Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsSetup
          (setupOf carrier context)))
      machine functionIndex
      (inputOf carrier context selected outerKey iteration z0 zi inactive
        priorPc witness).toPaper
      (canonicalOuterOutput carrier context selected outerKey iteration z0 zi
        inactive priorPc witness certificate machine).toPaper := by
  let input := inputOf carrier context selected outerKey iteration z0 zi
    inactive priorPc witness
  let selectedNext :=
    FixedActive.resultOf (carrier.install context).full certificate
  have outer :=
    (Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.outerCheck_eq_true_iff
      checkers input selected).1 outerChecked
  have priorPcValid : Nightstream.Protocol.FPrime.Paper.InRange slotCount
      input.priorPc := by
    rw [outer.priorSlot]
    exact
      (Nightstream.Protocol.FPrime.Paper.ProgramCounter.ofIndex selected).valid
  have selectedEq :
      Nightstream.Protocol.FPrime.Paper.selectedIndex priorPcValid = selected := by
    apply Fin.ext
    simp [Nightstream.Protocol.FPrime.Paper.selectedIndex,
      Nightstream.Protocol.FPrime.Paper.ProgramCounter.index,
      outer.priorSlot]
  have selectedTransition :=
    paperTransition_implies_selectedTransition carrier context selected
      outerKey iteration z0 zi inactive priorPc witness
      (outputChildren (carrier.install context).full certificate)
      paperTransition
  change Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds _ _ _
    input.toPaper (outputOf machine input selected selectedNext).toPaper
  refine {
    iterationPositive := outer.iterationPositive
    priorPcValid := priorPcValid
    priorPublicInput := outer.priorPublicInput
    application := ?_
    selectedStructures := ?_
    selectedNifs := ?_
    unchanged := ?_
    outputHash := ?_
  }
  · have derived := Nightstream.Protocol.FPrime.Paper.derivedOutput_application
      machine input.toPaper (updatedRunning input selected selectedNext).toPaper
    have indexEq :
        (machine.control input.toPaper.zi input.toPaper.witness).index =
          functionIndex := by
      rw [outer.dispatch]
      exact Nightstream.Protocol.FPrime.Paper.ProgramCounter.index_ofIndex
        functionIndex
    rw [indexEq] at derived
    simpa using derived
  · constructor
    · change input.fresh.constraintSystem =
        (setupOf carrier context).expectedStructure input.verifierKey
          (Nightstream.Protocol.FPrime.Paper.selectedIndex priorPcValid)
      rw [selectedEq]
      exact outer.expectedStructure
    · intro child
      change ((input.running
          (Nightstream.Protocol.FPrime.Paper.selectedIndex priorPcValid)).children
          child).constraintSystem =
        (setupOf carrier context).expectedStructure input.verifierKey
          (Nightstream.Protocol.FPrime.Paper.selectedIndex priorPcValid)
      rw [selectedEq]
      change
        ((runningOf carrier context.key selected inactive selected).children
          child).constraintSystem = carrier.system
      rw [runningOf_selected]
      rfl
  · change
      Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.Transition
        (Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsSetup
          (setupOf carrier context))
        input.verifierKey
        (Nightstream.Protocol.FPrime.Paper.selectedIndex priorPcValid)
        (Nightstream.Protocol.FPrime.Paper.Construction2.selectedInput
          input.toPaper
          (Nightstream.Protocol.FPrime.Paper.selectedIndex priorPcValid))
        ((outputOf machine input selected selectedNext).toPaper.runningNext
          (Nightstream.Protocol.FPrime.Paper.selectedIndex priorPcValid))
    rw [selectedEq]
    have selectedOutput :
        (outputOf machine input selected selectedNext).toPaper.runningNext
            selected =
          outputChildren (carrier.install context).full certificate := by
      funext child
      exact congrArg (fun result => result.children child)
        (outputOf_runningNext_selected machine input selected selectedNext)
    rw [selectedOutput]
    exact selectedTransition
  · intro slot notSelected
    have notConcrete : slot ≠ selected := by
      intro slotEq
      apply notSelected
      exact slotEq.trans selectedEq.symm
    have richEq := outputOf_runningNext_other machine input selected
      selectedNext slot notConcrete
    simpa [Output.toPaper, Running.toPaper, Input.toPaper] using
      congrArg (fun result => result.children) richEq
  · exact Nightstream.Protocol.FPrime.Paper.derivedOutput_outputHolds machine
      input.toPaper (updatedRunning input selected selectedNext).toPaper

/-- The complete pending-compatible canonical check, plus the packed equation
closed by the successor or terminal step, yields Construction 2 or only the
separate `y_ring`/raw-`Pi_CCS` events.  In particular there is no
`outputUnbound`, source-binding premise, or implementation-refinement branch. -/
theorem canonicalCheck_and_packed_imply_construction2_or_yRingUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (selected : Fin slotCount)
    (outerKey : OuterKey)
    (iteration : Nat)
    (z0 zi : AppState)
    (inactive : Running shape publicRingColumns publicFits verifierRows
      slotCount)
    (priorPc : Nat)
    (witness : Witness)
    (certificate : FixedActive.Certificate (carrier.install context).full)
    (machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (checkers :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.Checkers
        (setupOf carrier context) machine functionIndex)
    (checked : canonicalCheck carrier context selected outerKey iteration z0
      zi inactive priorPc witness certificate machine functionIndex checkers =
        true)
    (packed : Terminal.PackedYZcolBoundAtBlock
      (carrier.install context).full.covers carrier.data
      (ProductionPiCcs.ncPoint (carrier.install context).full certificate).block
      certificate.piCcs.output) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics.family
          (Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsSetup
            (setupOf carrier context)))
        machine functionIndex
        (inputOf carrier context selected outerKey iteration z0 zi inactive
          priorPc witness).toPaper
        (canonicalOuterOutput carrier context selected outerKey iteration z0
          zi inactive priorPc witness certificate machine).toPaper \/
      ProductionPiCcs.YRingUnbound (carrier.install context).full carrier.data
        certificate \/
      ProductionPiCcs.BadEvent (carrier.install context).full carrier.data
        certificate := by
  have checkedParts :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.outerCheck
          checkers
          (inputOf carrier context selected outerKey iteration z0 zi inactive
            priorPc witness) selected = true /\
        ProductionPaperChecker.check carrier context certificate = true := by
    simpa [canonicalCheck] using checked
  have stepAccepted :=
    (ProductionPaperChecker.check_eq_true_iff_accepted carrier context
      certificate).1 checkedParts.2
  rcases
      ProductionPaperNifs.paperStepAccepted_and_packed_implies_refinement_or_yRingUnbound_or_badEvent
        noZeroDivisors carrier context certificate stepAccepted packed with
    refinement | yRingUnbound | bad
  · exact Or.inl
      (outerCheck_and_paperTransition_imply_construction2 carrier context
        selected outerKey iteration z0 zi inactive priorPc witness certificate
        (refinement.toPaperTransition packed) machine functionIndex checkers
        checkedParts.1)
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr bad)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperOuter
