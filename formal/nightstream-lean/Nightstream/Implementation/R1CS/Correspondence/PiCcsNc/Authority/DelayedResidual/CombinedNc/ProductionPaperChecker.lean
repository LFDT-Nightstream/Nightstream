import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperNifs
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality

/-!
Executable paper-step checker over the compact opening-derived carrier.

Protocol: SuperNeo Sections 7.3--7.5.
Phase: one claims-level production NIFS step before its delayed packed output
is closed by a successor or the terminal verifier.

Assurance tier: model-level executable semantics.

Owns: exact finite checks for the five operational `Pi_DEC` output equations,
the outgoing canonical-parent opening, and their composition with public
`Pi_CCS` messages and sampler replay.

Does not own: raw NC extraction, one-fold state continuity, terminal closure,
Rust differential execution, physical rows, primitive security, costs, or
row-removal authority.

Authority boundary: the checker receives one authoritative opening-derived
source carrier.  It never accepts incoming running authority, child
`y_zcol` sidecars, a source-projection premise, or a digest as authority.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperChecker

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

/-- Claims-level paper checker. Incoming running authority is reconstructed
later from raw NC truth and therefore is not a retained check. -/
def check
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    Bool :=
  ProductionChecker.piCcsMessageCheck (carrier.install context).full
      certificate &&
    (Sampler.Checker.certificateCheck (carrier.install context).full
        certificate &&
      (paperOutputCheck (carrier.install context).full certificate &&
        canonicalParentOpeningCheck (carrier.install context).full carrier.data
          certificate))

theorem check_eq_true_iff_accepted
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full) :
    check carrier context certificate = true <->
      ProductionPaperNifs.PaperStepAccepted carrier context certificate := by
  simp only [check, Bool.and_eq_true,
    ProductionChecker.piCcsMessageCheck_eq_true_iff_accepted,
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
  step : ProductionPaperNifs.PaperStepAccepted carrier context certificate
  noPending : (carrier.install context).full.pending = none

/-- A base starts with no pending delayed projection. Its own output is still
closed by the next recursive step or by the terminal checker. -/
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

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperChecker
