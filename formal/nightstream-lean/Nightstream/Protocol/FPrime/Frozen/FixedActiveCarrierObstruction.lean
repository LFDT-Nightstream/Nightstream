import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Types
import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
import Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs

/-!
Conditional 257-erasure obstruction and full-carrier necessity result for the
fixed-active public-input boundary.

Protocol: SuperNeo Sections 7.3--7.5 and HyperNova Construction 2.
Phase: comparison of a complete five-ring paper NIFS carrier with a hypothetical
257-coordinate running projection.
Constraint family: typed semantic carrier ownership only; this file emits no
rows.

Owns: the exact fixed-active paper shape (`K = 1`, `k = 14`); one explicit
257-coordinate erasure of the complete 270-coordinate running carrier; two exact
paper NIFS running states with the same erased view; impossibility of a lossless
decoder from that view; and the same conditional obstruction lifted through the
carrier-polymorphic fixed-one and generic Construction-2 input types.

Does not own: a claim that production or the frozen facade selects the erased
carrier; a production decoder; Poseidon2; commitments; extraction;
PiCCS/PiRLC/PiDEC soundness; Rust; R1CS; generated rows; or cost claims.

Authority boundary: fresh construction may set the thirteen inserted
coordinates to zero, but exact paper running inputs retain all 270 coordinates.
The existing `PiCcsSources` adapter already accepts full-carrier running
assignments and passes them through unchanged.  The frozen NIFS/F' interfaces
remain polymorphic until a separate fixed-active carrier pin is supplied.  No
digest or 257-coordinate projection is promoted into authority.

The countermodel is finite and extensional.  The two carriers differ at exact
paper coordinate 257, have identical first 257 coordinates, inhabit a concrete
270-wide instantiation of `PaperNonInteractive.Running`, and remain
indistinguishable after lifting through matching 257-wide instantiations of the
carrier-polymorphic fixed-one and Construction-2 input types.  It proves that a
257-wide pin would be lossy; it does not prove that such a pin currently exists.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint


universe uExtension uCommitment uState uWitness uFresh uProof uKey uScalar

private theorem running_ext
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uFresh}
    {shape : Shape}
    {left right :
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Running
        Extension Commitment PublicInput shape}
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem fixedOneInput_ext
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uExtension}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {left right :
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input
        State Witness Running Fresh Proof}
    (iteration : left.iteration = right.iteration)
    (z0 : left.z0 = right.z0)
    (zi : left.zi = right.zi)
    (running : left.running = right.running)
    (fresh : left.fresh = right.fresh)
    (witness : left.witness = right.witness)
    (nifsProof : left.nifsProof = right.nifsProof) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem construction2Input_ext
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uExtension}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {slotCount : Nat}
    {left right : Nightstream.HyperNova.Construction2.Paper.Input
      Key State Witness Running Fresh Proof slotCount}
    (iteration : left.iteration = right.iteration)
    (z0 : left.z0 = right.z0)
    (zi : left.zi = right.zi)
    (running : left.running = right.running)
    (fresh : left.fresh = right.fresh)
    (priorPc : left.priorPc = right.priorPc)
    (witness : left.witness = right.witness)
    (nifsProof : left.nifsProof = right.nifsProof) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Exact paper residual shape for the selected fixed-active profile. -/
def fixedShape (dimensions : Dimensions) : Shape :=
  (semanticShape dimensions 1 14).paperShape

@[simp] theorem fixedShape_freshCount (dimensions : Dimensions) :
    (fixedShape dimensions).freshCount = 1 := by
  rfl

@[simp] theorem fixedShape_runningCount (dimensions : Dimensions) :
    (fixedShape dimensions).runningCount = 14 := by
  rfl

@[simp] theorem fixedShape_sourceCount (dimensions : Dimensions) :
    (fixedShape dimensions).sourceCount = 15 := by
  rfl

/-- Complete 270-coordinate instantiation of the paper non-interactive NIFS
running carrier. -/
abbrev ExactRunning
    (dimensions : Dimensions)
    (Extension : Type uExtension)
    (Commitment : Type uCommitment) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Running Extension Commitment (LIn dimensions) (fixedShape dimensions)

/-- Hypothetical 257-coordinate erasure used only to state the conditional
lossiness result. -/
abbrev ErasedRunning
    (dimensions : Dimensions)
    (Extension : Type uExtension)
    (Commitment : Type uCommitment) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Running Extension Commitment ExternalInput (fixedShape dimensions)

/-- Exact paper fresh carrier at `K = 1`. -/
abbrev ExactFresh
    (dimensions : Dimensions)
    (Commitment : Type uCommitment) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Fresh
    Commitment (LIn dimensions) (fixedShape dimensions)

/-- Exact paper proof message at the fixed-active shape. -/
abbrev ExactProof
    (dimensions : Dimensions)
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (degreeBound : Nat) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Proof
    Extension Commitment (fixedShape dimensions) degreeBound

/-- Exact paper verifier key type specialized only to the complete fixed-active
carrier. Its fields remain the audited paper contracts; this is not a
production Poseidon2 or commitment instantiation. -/
abbrev ExactKey
    (dimensions : Dimensions)
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (TranscriptState : Type uState)
    (columns blockCount degreeBound : Nat) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key
    Extension Commitment (LIn dimensions) Scalar TranscriptState
    (fixedShape dimensions) columns blockCount degreeBound

/-- The already-audited paper verifier, instantiated at the complete 270-field
fixed-active carrier. This deliberately does not claim production refinement. -/
def exactPaperVerifier
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {TranscriptState : Type uState}
    [DecidableEq Extension]
    {columns blockCount degreeBound : Nat} :
    Nightstream.HyperNova.NonInteractiveMultiFold.Verifier
      (ExactKey dimensions Extension Commitment Scalar TranscriptState
        columns blockCount degreeBound)
      (ExactRunning dimensions Extension Commitment)
      (ExactFresh dimensions Commitment)
      (ExactProof dimensions Extension Commitment degreeBound) :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.nifsVerifier

/-- Exact frozen `NifsSoundAndCompleteModulo` theorem for the chosen 270-wide
paper instantiation.  This selects a concrete carrier for this theorem only; it
does not prove that an existing production or frozen F' setup uses the same
instantiation.  It is the paper verifier theorem, not the missing
production-shaped PiCCS/PiRLC/PiDEC refinement. -/
theorem exactPaperVerifier_soundAndCompleteModulo
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {TranscriptState : Type uState}
    [DecidableEq Extension]
    {columns blockCount degreeBound : Nat} :
    Nightstream.Protocol.FPrime.Frozen.Obligations.NifsSoundAndCompleteModulo
      (exactPaperVerifier (dimensions := dimensions)
        (Extension := Extension) (Commitment := Commitment)
        (Scalar := Scalar) (TranscriptState := TranscriptState)
        (columns := columns) (blockCount := blockCount)
        (degreeBound := degreeBound))
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.nifsSoundAndCompleteModulo

/-- Forget precisely the thirteen authoritative running coordinates in every
public input, preserving all other exact NIFS fields. -/
def eraseRunning
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    (running : ExactRunning dimensions Extension Commitment) :
    ErasedRunning dimensions Extension Commitment where
  point := running.point
  commitments := running.commitments
  publicInputs := fun index =>
    projectExternal dimensions (running.publicInputs index)
  evaluations := running.evaluations

/-- Canonical first running source in the fixed `k = 14` product. -/
def firstRunning (dimensions : Dimensions) :
    Fin (fixedShape dimensions).runningCount :=
  ⟨0, by simpa using (show 0 < 14 by decide)⟩

/-- Replace every exact running public input by the complete all-zero carrier. -/
def zeroPublicRunning
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    (seed : ExactRunning dimensions Extension Commitment) :
    ExactRunning dimensions Extension Commitment :=
  { seed with publicInputs := fun _ => zeroLIn dimensions }

/-- Mutate only paper coordinate 257 of the first running public input. -/
def tailMutatedRunning
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    (seed : ExactRunning dimensions Extension Commitment) :
    ExactRunning dimensions Extension Commitment :=
  { seed with
    publicInputs := fun index =>
      if index = firstRunning dimensions then
        firstPaddingOne dimensions
      else
        zeroLIn dimensions }

/-- The two exact NIFS running states have identical 257-coordinate views. -/
theorem eraseRunning_zero_eq_tail
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    (seed : ExactRunning dimensions Extension Commitment) :
    eraseRunning (zeroPublicRunning seed) =
      eraseRunning (tailMutatedRunning seed) := by
  apply running_ext
  · rfl
  · rfl
  · funext index
    by_cases selected : index = firstRunning dimensions
    · simp [eraseRunning, zeroPublicRunning, tailMutatedRunning, selected,
        projectExternal_firstPaddingOne_eq_zero]
    · simp [eraseRunning, zeroPublicRunning, tailMutatedRunning, selected]
  · rfl

/-- The exact NIFS running states are different at the authoritative first
tail coordinate. -/
theorem zeroPublicRunning_ne_tailMutatedRunning
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    (seed : ExactRunning dimensions Extension Commitment) :
    zeroPublicRunning seed ≠ tailMutatedRunning seed := by
  intro equal
  have inputsEqual := congrArg Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Running.publicInputs equal
  have firstEqual := congrFun inputsEqual (firstRunning dimensions)
  have carrierEqual := congrFun firstEqual
    (carrierColumn dimensions (.inr firstPadding))
  have zeroEqOne : (0 : Nightstream.SuperNeo.Concrete.F) = 1 := by
    simpa [zeroPublicRunning, tailMutatedRunning, zeroLIn, firstPaddingOne]
      using carrierEqual
  exact (by decide : (0 : Nightstream.SuperNeo.Concrete.F) ≠ 1) zeroEqOne

/-- Exact erasure at the paper NIFS input is non-injective. -/
theorem eraseRunning_not_injective
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    (seed : ExactRunning dimensions Extension Commitment) :
    ¬ Function.Injective
      (eraseRunning (dimensions := dimensions)
        (Extension := Extension) (Commitment := Commitment)) := by
  intro injective
  exact zeroPublicRunning_ne_tailMutatedRunning seed
    (injective (eraseRunning_zero_eq_tail seed))

/-- Conditional pinning obstruction: if the fixed-active paper NIFS running
carrier is exposed only through this 257-coordinate erasure, no decoder can be
a left inverse on the complete 270-coordinate domain.  The theorem does not
assert that the frozen facade currently chooses the erased instantiation. -/
theorem no_exact_paperNifs_running_decoder
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    (seed : ExactRunning dimensions Extension Commitment) :
    ¬ ∃ decode : ErasedRunning dimensions Extension Commitment ->
        ExactRunning dimensions Extension Commitment,
      ∀ running, decode (eraseRunning running) = running := by
  rintro ⟨decode, leftInverse⟩
  apply eraseRunning_not_injective seed
  intro left right erased
  calc
    left = decode (eraseRunning left) := (leftInverse left).symm
    _ = decode (eraseRunning right) := congrArg decode erased
    _ = right := leftInverse right

/-- Exact one-slot input consumed by the frozen fixed-one Construction-2
checker. -/
abbrev ExactFixedOneInput
    (dimensions : Dimensions)
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (State : Type uState)
    (Witness : Type uWitness)
    (Fresh : Type uFresh)
    (Proof : Type uProof) :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input State Witness
    (ExactRunning dimensions Extension Commitment) Fresh Proof

/-- A hypothetical lossy one-slot instantiation with only 257 public
coordinates per running claim. -/
abbrev ErasedFixedOneInput
    (dimensions : Dimensions)
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (State : Type uState)
    (Witness : Type uWitness)
    (Fresh : Type uFresh)
    (Proof : Type uProof) :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input State Witness
    (ErasedRunning dimensions Extension Commitment) Fresh Proof

/-- Apply the same 257-coordinate erasure inside the exact fixed-one F' input. -/
def eraseFixedOneInput
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input : ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof) :
    ErasedFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof where
  iteration := input.iteration
  z0 := input.z0
  zi := input.zi
  running := fun slot => eraseRunning (input.running slot)
  fresh := input.fresh
  witness := input.witness
  nifsProof := input.nifsProof

/-- Fixed-one input with the zero-tail exact running state installed. -/
def zeroFixedOneInput
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof) :
    ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof :=
  { seed with
    running := fun _ => zeroPublicRunning (seed.running Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected) }

/-- Fixed-one input with the invisible exact tail mutation installed. -/
def tailMutatedFixedOneInput
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof) :
    ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof :=
  { seed with
    running := fun _ => tailMutatedRunning (seed.running Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected) }

/-- The frozen fixed-one checker receives the same erased input for two
different exact Construction-2 inputs. -/
theorem eraseFixedOneInput_zero_eq_tail
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof) :
    eraseFixedOneInput (zeroFixedOneInput seed) =
      eraseFixedOneInput (tailMutatedFixedOneInput seed) := by
  apply fixedOneInput_ext
  · rfl
  · rfl
  · rfl
  · funext slot
    exact eraseRunning_zero_eq_tail (seed.running Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
  · rfl
  · rfl
  · rfl

/-- The two exact fixed-one F' inputs remain different. -/
theorem zeroFixedOneInput_ne_tailMutatedFixedOneInput
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof) :
    zeroFixedOneInput seed ≠ tailMutatedFixedOneInput seed := by
  intro equal
  have runningEqual := congrArg Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.running equal
  have selectedEqual := congrFun runningEqual Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
  exact zeroPublicRunning_ne_tailMutatedRunning
    (seed.running Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected) selectedEqual

/-- Conditional fixed-one obstruction: a 257-wide instantiation cannot
reconstruct the matching complete 270-wide advice domain. -/
theorem no_exact_fixedOne_fprime_decoder
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof) :
    ¬ ∃ decode : ErasedFixedOneInput dimensions Extension Commitment
          State Witness Fresh Proof ->
        ExactFixedOneInput dimensions Extension Commitment
          State Witness Fresh Proof,
      ∀ input, decode (eraseFixedOneInput input) = input := by
  rintro ⟨decode, leftInverse⟩
  have erased := eraseFixedOneInput_zero_eq_tail seed
  have exactEqual : zeroFixedOneInput seed = tailMutatedFixedOneInput seed := by
    calc
      zeroFixedOneInput seed =
          decode (eraseFixedOneInput (zeroFixedOneInput seed)) :=
        (leftInverse (zeroFixedOneInput seed)).symm
      _ = decode (eraseFixedOneInput (tailMutatedFixedOneInput seed)) :=
        congrArg decode erased
      _ = tailMutatedFixedOneInput seed :=
        leftInverse (tailMutatedFixedOneInput seed)
  exact zeroFixedOneInput_ne_tailMutatedFixedOneInput seed exactEqual

/-- Exact generic Construction-2 input, including its one-based prior counter. -/
abbrev ExactConstruction2Input
    (dimensions : Dimensions)
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (Key : Type uKey)
    (State : Type uState)
    (Witness : Type uWitness)
    (Fresh : Type uFresh)
    (Proof : Type uProof) :=
  Nightstream.HyperNova.Construction2.Paper.Input Key State Witness
    (ExactRunning dimensions Extension Commitment) Fresh Proof 1

/-- Lossy generic Construction-2 input. -/
abbrev ErasedConstruction2Input
    (dimensions : Dimensions)
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (Key : Type uKey)
    (State : Type uState)
    (Witness : Type uWitness)
    (Fresh : Type uFresh)
    (Proof : Type uProof) :=
  Nightstream.HyperNova.Construction2.Paper.Input Key State Witness
    (ErasedRunning dimensions Extension Commitment) Fresh Proof 1

/-- Erase the thirteen exact running coordinates without changing any other
Construction-2 advice, including the prior counter. -/
def eraseConstruction2Input
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input : ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof) :
    ErasedConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof where
  iteration := input.iteration
  z0 := input.z0
  zi := input.zi
  running := fun slot => eraseRunning (input.running slot)
  fresh := input.fresh
  priorPc := input.priorPc
  witness := input.witness
  nifsProof := input.nifsProof

/-- Generic Construction-2 input with the zero-tail running state. -/
def zeroConstruction2Input
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof) :
    ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof :=
  { seed with
    running := fun _ => zeroPublicRunning (seed.running Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected) }

/-- Generic Construction-2 input with the exact tail mutation. -/
def tailMutatedConstruction2Input
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof) :
    ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof :=
  { seed with
    running := fun _ => tailMutatedRunning (seed.running Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected) }

/-- The exact generic Construction-2 interface has the same ambiguity. -/
theorem eraseConstruction2Input_zero_eq_tail
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof) :
    eraseConstruction2Input (zeroConstruction2Input seed) =
      eraseConstruction2Input (tailMutatedConstruction2Input seed) := by
  apply construction2Input_ext
  · rfl
  · rfl
  · rfl
  · funext slot
    exact eraseRunning_zero_eq_tail (seed.running Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
  · rfl
  · rfl
  · rfl
  · rfl

/-- The exact generic Construction-2 inputs are different. -/
theorem zeroConstruction2Input_ne_tailMutatedConstruction2Input
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof) :
    zeroConstruction2Input seed ≠ tailMutatedConstruction2Input seed := by
  intro equal
  have runningEqual := congrArg Nightstream.HyperNova.Construction2.Paper.Input.running equal
  have selectedEqual := congrFun runningEqual Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
  exact zeroPublicRunning_ne_tailMutatedRunning
    (seed.running Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected) selectedEqual

/-- Conditional Construction-2 obstruction: a generic F' setup pinned to the
257-wide erasure cannot decode all inputs of the matching 270-wide setup.  The
carrier-polymorphic frozen relation itself does not choose either pin. -/
theorem no_exact_construction2_fprime_decoder
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof) :
    ¬ ∃ decode : ErasedConstruction2Input dimensions Extension Commitment
          Key State Witness Fresh Proof ->
        ExactConstruction2Input dimensions Extension Commitment
          Key State Witness Fresh Proof,
      ∀ input, decode (eraseConstruction2Input input) = input := by
  rintro ⟨decode, leftInverse⟩
  have erased := eraseConstruction2Input_zero_eq_tail seed
  have exactEqual :
      zeroConstruction2Input seed = tailMutatedConstruction2Input seed := by
    calc
      zeroConstruction2Input seed =
          decode (eraseConstruction2Input (zeroConstruction2Input seed)) :=
        (leftInverse (zeroConstruction2Input seed)).symm
      _ = decode
          (eraseConstruction2Input (tailMutatedConstruction2Input seed)) :=
        congrArg decode erased
      _ = tailMutatedConstruction2Input seed :=
        leftInverse (tailMutatedConstruction2Input seed)
  exact zeroConstruction2Input_ne_tailMutatedConstruction2Input seed exactEqual

end Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction
