import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.PaperBridge
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalOutput

/-!
Minimal strict-`PiDEC` shape bridge to the physical fixed-active certificate.

Protocol: SuperNeo Section 7.5.
Phase: strict source `PiDEC` to the public physical output family.
Constraint family: public-X identity and evaluation-vector arity only.

Assurance tier: model-level representation refinement.

Owns: the structural parent evaluation arity; the exact three-field mapping
needed to identify uniform-X decodes and ordered child evaluation lengths
with one physical certificate; and derivation of the three non-tail premises
of `PhysicalOutput.paperOutputAccepted_of_tail`.

Does not own: generated rows, selected-row satisfaction, Rust dataflow,
commitment or evaluation values, child private openings, points, delayed
projection authority, transcript authority, costs, or row removal.

Authority boundary: `CertificatePaperShapeBound` deliberately maps only the
strict parent/child public-X projections and child array lengths. It is not a
full decoded-result equality and carries no commitment, evaluation value,
private assignment, sidecar, or digest authority.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.PhysicalOutputShape

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.Implementation.R1CS

universe uState

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {layout : PiDecStrictProductionCompiler.Layout}

/-- The exact compact execution-audit contract still required from the active
Rust exporter.

The first two fields map only the 270-coordinate uniform-X decodes. The
third maps only each ordered child's evaluation-array length. In particular,
this structure does not equate decoded commitments, evaluation values, points,
private witnesses, or complete child payloads. -/
structure CertificatePaperShapeBound
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows
      layout.base)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context) : Prop where
  parentPublicX :
    (PiDecTypedCarrier.decodedParent profile context.system
      assignment).publicInput =
      (derive context certificate).piRlcOutput.publicInput
  childPublicX : forall child,
    (PiDecTypedCarrier.decodedOutput profile context.system
      assignment child).publicInput =
      (outputChildren context certificate child).publicInput
  childEvaluationLength : forall child,
    (PiDecTypedCarrier.decodedOutput profile context.system
      assignment child).evaluations.size =
      (outputChildren context certificate child).evaluations.size

/-- The physical `PiRLC` parent has one evaluation per verifier-owned matrix
by construction. This fact needs neither a row artifact nor a certificate
binding premise. -/
theorem parentEvaluationSize_structural
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    (derive context certificate).piRlcOutput.evaluations.size =
      (FixedActive.PaperProfile.decEvaluationArity
        (FixedActive.paperProfileOf context)).count
        (derive context certificate).piRlcOutput.constraintSystem := by
  simp [derive, rlcAlgebra, PiRLC.combinedOutput,
    PiRLCAlgebra.Algebra.concrete, PiRLCFinite.combineEvaluations,
    FixedActive.PaperProfile.decEvaluationArity,
    FixedActive.paperProfileOf,
    PiDECAlgebra.PaperVerifier.evaluationArity]

/-- Uniform-X acceptance plus the two public-X mapping fields gives
the exact canonical public child equation on the physical certificate. -/
theorem canonicalPublicInput_of_uniformX
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows
      layout.base)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context)
    (accepted : PiDecStrictProductionCompiler.UniformXAccepted layout
      assignment)
    (bound : CertificatePaperShapeBound profile context assignment
      certificate) :
    forall child,
      (outputChildren context certificate child).publicInput =
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf context)).split
          (derive context certificate).piRlcOutput.publicInput child := by
  intro child
  rw [← bound.childPublicX child, ← bound.parentPublicX]
  simpa [PiDecTypedCarrier.decodedParent,
    PiDecTypedCarrier.decodedOutput,
    FixedActive.PaperProfile.decPublicInputSplit,
    FixedActive.paperProfileOf,
    PiDECAlgebra.PaperVerifier.publicInputSplit] using
      (PiDecStrictProductionCompiler.PaperBridge.canonicalPublicInput_of_uniformX
        valid profile accepted child)

/-- The ordered child-length mapping and the strict layout profile give the
exact physical child evaluation arity. No evaluation value is decoded or
compared. -/
theorem childEvaluationSize_of_bound
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows
      layout.base)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context)
    (bound : CertificatePaperShapeBound profile context assignment
      certificate) :
    forall child,
      (outputChildren context certificate child).evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem := by
  intro child
  rw [← bound.childEvaluationLength child]
  simpa [FixedActive.PaperProfile.decEvaluationArity,
    FixedActive.paperProfileOf] using
      (PiDecTypedCarrier.decodedOutput_evaluations_size profile
        context.system assignment child)

/-- The uniform-X endpoint and compact execution mapping discharge exactly the
three paper-only premises left by the physical tail. -/
theorem paperShapeFacts_of_uniformX
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows
      layout.base)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context)
    (accepted : PiDecStrictProductionCompiler.UniformXAccepted layout
      assignment)
    (bound : CertificatePaperShapeBound profile context assignment
      certificate) :
    (forall child,
      (outputChildren context certificate child).publicInput =
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf context)).split
          (derive context certificate).piRlcOutput.publicInput child) /\
    (derive context certificate).piRlcOutput.evaluations.size =
      (FixedActive.PaperProfile.decEvaluationArity
        (FixedActive.paperProfileOf context)).count
        (derive context certificate).piRlcOutput.constraintSystem /\
    (forall child,
      (outputChildren context certificate child).evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem) := by
  exact ⟨
    canonicalPublicInput_of_uniformX valid profile context assignment
      certificate accepted bound,
    parentEvaluationSize_structural context certificate,
    childEvaluationSize_of_bound profile context assignment certificate
      bound⟩

/-- With the already retained physical tail, the uniform-X endpoint and compact
shape mapping give exact operational paper output acceptance for the actual
public certificate children. -/
theorem paperOutputAccepted_of_tail_and_uniformX
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile
      (RelationShape shape publicRingColumns publicFits) verifierRows
      layout.base)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : Nat -> Nat)
    (certificate : FixedActive.Certificate context)
    (tail : TailAccepted context certificate)
    (accepted : PiDecStrictProductionCompiler.UniformXAccepted layout
      assignment)
    (bound : CertificatePaperShapeBound profile context assignment
      certificate) :
    PiDEC.PaperVerifier.OutputAccepted
      (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.decAlgebra context.key)
      (FixedActive.PaperProfile.decPublicInputSplit
        (FixedActive.paperProfileOf context))
      (FixedActive.PaperProfile.decEvaluationArity
        (FixedActive.paperProfileOf context))
      (derive context certificate).piRlcOutput
      (outputChildren context certificate) := by
  rcases paperShapeFacts_of_uniformX valid profile context assignment
      certificate accepted bound with
    ⟨canonicalPublicInput, parentEvaluationSize, childEvaluationSize⟩
  exact FixedActive.PaperProfile.PhysicalOutput.paperOutputAccepted_of_tail
    context certificate tail canonicalPublicInput parentEvaluationSize
    childEvaluationSize

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.PhysicalOutputShape
