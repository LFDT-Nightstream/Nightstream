import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.PaperStep
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker

/-!
Executable checker for one protocol-owned delayed packed-`yZcol` step.

Assurance tier: model-level executable semantics.

Owns: exact finite checks for accumulator-state recomputation, public
combined-NC message acceptance, the five operational `Pi_DEC` output
equations, the canonical outgoing-parent opening, and the no-pending base
boundary.

Does not own: extraction of raw NC assignments, successor continuity,
terminal closure, commitment security, Rust/R1CS refinement, costs, or row
removal.

Emits constraints: no.

Authority boundary: the step checker receives one authoritative
opening-derived carrier and executes only public transcript and typed-carrier
checks. It never accepts an incoming child `yZcol` sidecar, a digest, or a
caller-provided semantic proposition as authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.check.state` | compare the carried state with the digest recomputed from the complete typed payload | checked/security boundary | `Binding.stateBindingCheck`, `Binding.stateBindingCheck_eq_true_iff` |
| `fprime.delayed.check.pi_ccs` | check the public combined-NC message and its exact acceptance predicate | checked | `piCcsMessageCheck`, `piCcsMessageCheck_eq_true_iff_accepted` |
| `fprime.delayed.check.pi_dec` | check the five paper output equations and canonical public/evaluation dimensions | checked | `paperOutputCheck`, `paperOutputCheck_eq_true_iff` |
| `fprime.delayed.check.parent` | check the outgoing parent against its canonical opening-derived carrier | checked/security boundary | `canonicalParentOpeningCheck`, `canonicalParentOpeningCheck_eq_true_iff` |
| `fprime.delayed.check.step` | compose the active step checks, with a separate no-pending base boundary | checked | `check`, `check_eq_true_iff_accepted`, `baseCheck`, `baseCheck_eq_true_iff_accepted` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Checker

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

namespace Binding

open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

universe uEncoding uDigest

/-- Fail-closed equality between the carried state coordinate and the digest
recomputed from the complete parent, ordered child family, and optional
pending projection. The digest is compression, never authority. -/
def stateBindingCheck
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (parent : CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (children : Fin productionGlobalParams.k -> CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (pending : Option ProductionDelayedBlockLane) : Bool :=
  decide (stateDigest = pendingFamilyDigest scheme parent children pending)

/-- The executable state-coordinate check is exactly recomputation from the
complete typed payload. -/
theorem stateBindingCheck_eq_true_iff
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (parent : CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (children : Fin productionGlobalParams.k -> CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (pending : Option ProductionDelayedBlockLane) :
    stateBindingCheck scheme stateDigest parent children pending = true <->
      StateBinds scheme stateDigest parent children pending := by
  simp [stateBindingCheck, StateBinds]

end Binding

universe uState

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

private def natEqual (left right : Nat) : Bool :=
  decide (left = right)

private theorem natEqual_eq_true_iff (left right : Nat) :
    natEqual left right = true <-> left = right := by
  exact decide_eq_true_iff

/-- Execute the production FE phase and the public combined-NC transcript
against the terminal computed from the public output message. -/
def piCcsMessageCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Bool :=
  Fe.check context.feMachine context.initialState context.profile
      context.piCcsInput context.feCoins certificate.piCcs.output
      certificate.piCcs.fe &&
    Transcript.Nc.BlockLane.check context.ncMachine
      (BlockLaneCombinedNc.ncTranscriptState context certificate)
      (BlockLaneCombinedNc.rawInitial context)
      (BlockLaneCombinedNc.messageTerminal context certificate)
      certificate.piCcs.nc

/-- The public combined-NC Boolean check is exact to the protocol-owned
claims-level predicate. -/
theorem piCcsMessageCheck_eq_true_iff_accepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    piCcsMessageCheck context certificate = true <->
      BlockLaneCombinedNc.MessageAccepted context certificate := by
  simp only [piCcsMessageCheck, Bool.and_eq_true,
    Fe.check_eq_true_iff_accepted,
    Transcript.Nc.BlockLane.check_eq_true_iff_accepted]
  constructor
  · rintro ⟨fe, nc⟩
    exact { fe := fe, nc := nc }
  · intro accepted
    exact ⟨accepted.fe, accepted.nc⟩

/-- Compare every computed public child with the verifier-owned radix split. -/
def canonicalPublicInputCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Bool :=
  functionEqual publicInputEqual
    (fun child => (outputChildren context certificate child).publicInput)
    ((FixedActive.PaperProfile.decPublicInputSplit
      (FixedActive.paperProfileOf context)).split
      (derive context certificate).piRlcOutput.publicInput)

theorem canonicalPublicInputCheck_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    canonicalPublicInputCheck context certificate = true <->
      forall child,
        (outputChildren context certificate child).publicInput =
          (FixedActive.PaperProfile.decPublicInputSplit
            (FixedActive.paperProfileOf context)).split
            (derive context certificate).piRlcOutput.publicInput child := by
  rw [canonicalPublicInputCheck,
    functionEqual_eq_true_iff publicInputEqual publicInputEqual_eq_true_iff]
  constructor
  · intro equal child
    exact congrFun equal child
  · intro equal
    funext child
    exact equal child

/-- Check the verifier-owned evaluation-vector arity for the parent. -/
def parentEvaluationSizeCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Bool :=
  natEqual (derive context certificate).piRlcOutput.evaluations.size
    ((FixedActive.PaperProfile.decEvaluationArity
      (FixedActive.paperProfileOf context)).count
      (derive context certificate).piRlcOutput.constraintSystem)

theorem parentEvaluationSizeCheck_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    parentEvaluationSizeCheck context certificate = true <->
      (derive context certificate).piRlcOutput.evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem := by
  exact natEqual_eq_true_iff _ _

/-- Check the same verifier-owned arity for every ordered child. -/
def childEvaluationSizeCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Bool :=
  functionEqual natEqual
    (fun child => (outputChildren context certificate child).evaluations.size)
    (fun _ => (FixedActive.PaperProfile.decEvaluationArity
      (FixedActive.paperProfileOf context)).count
      (derive context certificate).piRlcOutput.constraintSystem)

theorem childEvaluationSizeCheck_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    childEvaluationSizeCheck context certificate = true <->
      forall child,
        (outputChildren context certificate child).evaluations.size =
          (FixedActive.PaperProfile.decEvaluationArity
            (FixedActive.paperProfileOf context)).count
            (derive context certificate).piRlcOutput.constraintSystem := by
  rw [childEvaluationSizeCheck,
    functionEqual_eq_true_iff natEqual natEqual_eq_true_iff]
  constructor
  · intro equal child
    exact congrFun equal child
  · intro equal
    funext child
    exact equal child

/-- Execute exactly the five non-computed output equations of the paper
`Pi_DEC` verifier. -/
def paperOutputCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Bool :=
  canonicalPublicInputCheck context certificate &&
    (parentEvaluationSizeCheck context certificate &&
      (childEvaluationSizeCheck context certificate &&
        (commitmentEqual
          (derive context certificate).piRlcOutput.commitment
          ((decAlgebra context.key).recomposeCommitment
            (fun child =>
              (outputChildren context certificate child).commitment)) &&
          evaluationsEqual
            (derive context certificate).piRlcOutput.evaluations
            ((decAlgebra context.key).recomposeEvaluations
              (fun child =>
                (outputChildren context certificate child).evaluations)))))

theorem paperOutputCheck_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    paperOutputCheck context certificate = true <->
      FixedActive.PaperProfile.PhysicalOutput.PaperOutputEquations context
        certificate := by
  simp only [paperOutputCheck, Bool.and_eq_true,
    canonicalPublicInputCheck_eq_true_iff,
    parentEvaluationSizeCheck_eq_true_iff,
    childEvaluationSizeCheck_eq_true_iff,
    commitmentEqual_eq_true_iff, evaluationsEqual_eq_true_iff]
  constructor
  · rintro ⟨publicInput, parentSize, childSize, commitment, evaluations⟩
    exact {
      canonicalPublicInput := publicInput
      parentEvaluationSize := parentSize
      childEvaluationSize := childSize
      commitment := commitment
      evaluations := evaluations
    }
  · intro equations
    exact ⟨equations.canonicalPublicInput, equations.parentEvaluationSize,
      equations.childEvaluationSize, equations.commitment,
      equations.evaluations⟩

private def canonicalParentCarrier
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    CanonicalParentVerifier.Generic.Carrier
      (RelationShape shape publicRingColumns publicFits) verifierRows where
  point := (derive context certificate).piRlcOutput.point
  commitment := (derive context certificate).piRlcOutput.commitment

/-- Check the canonical source/challenge parent assignment directly against
the carried parent commitment and the combined norm bound. -/
def canonicalParentOpeningCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Bool :=
  CanonicalParentVerifier.verify context.key
    (canonicalParentCarrier context certificate)
    (PackedYZcol.canonicalParentAssignment context data certificate)

theorem canonicalParentOpeningCheck_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) :
    canonicalParentOpeningCheck context data certificate = true <->
      DelayedRawChildren.CanonicalParentBinding context data certificate := by
  rw [canonicalParentOpeningCheck,
    CanonicalParentVerifier.verify_eq_true_iff]
  rfl

/-- Execute every retained public obligation for one opening-derived paper
step. Incoming running authority is reconstructed later from raw NC truth and
is not a retained check. -/
def check
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    Bool :=
  piCcsMessageCheck (carrier.install context).full certificate &&
    (Sampler.Checker.certificateCheck (carrier.install context).full
        certificate &&
      (paperOutputCheck (carrier.install context).full certificate &&
        canonicalParentOpeningCheck (carrier.install context).full carrier.data
          certificate))

/-- The complete Boolean step checker is exact to `PaperStepAccepted`; no
semantic acceptance proposition is supplied to the checker. -/
theorem check_eq_true_iff_accepted
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    check carrier context certificate = true <->
      PaperStep.PaperStepAccepted carrier context certificate := by
  simp only [check, Bool.and_eq_true,
    piCcsMessageCheck_eq_true_iff_accepted,
    Sampler.Checker.certificateCheck_eq_true_iff_accepted,
    paperOutputCheck_eq_true_iff,
    canonicalParentOpeningCheck_eq_true_iff]
  constructor
  · rintro ⟨piCcs, sampler, paperOutput, canonicalParent⟩
    exact {
      piCcs := piCcs
      sampler := sampler
      paperOutput := paperOutput
      canonicalParent := canonicalParent
    }
  · intro accepted
    exact ⟨accepted.piCcs, accepted.sampler, accepted.paperOutput,
      accepted.canonicalParent⟩

/-- Exact base-step predicate. The absence of a delayed predecessor is an
executed shape check, not an informal trace convention. -/
structure BaseAccepted
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    Prop where
  step : PaperStep.PaperStepAccepted carrier context certificate
  noPending : (carrier.install context).full.pending = none

/-- A base starts with no pending delayed projection. Its own output remains
to be closed by the next recursive step or by the terminal checker. -/
def baseCheck
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    Bool :=
  check carrier context certificate && context.pending.isNone

theorem baseCheck_eq_true_iff_accepted
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    baseCheck carrier context certificate = true <->
      BaseAccepted carrier context certificate := by
  rw [baseCheck, Bool.and_eq_true, check_eq_true_iff_accepted]
  constructor
  · rintro ⟨step, noPending⟩
    exact ⟨step, by simpa using noPending⟩
  · intro accepted
    exact ⟨accepted.step, by simpa using accepted.noPending⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Checker
