import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalOutput
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Transition

/-!
Honest/refined physical certificates satisfy the exact paper output equations.

Assurance tier: model-level.

Owns: derivation of verifier-computed child public inputs and exact
parent/child evaluation arities from the existing semantic opening
refinement, followed by reuse of the retained physical recomposition tail.

Does not own: construction of source openings, physical acceptance,
commitment binding, extraction, Rust, R1CS, rows, or costs.

Authority boundary: actual public certificate children are retained. The
proof uses their semantic openings only to derive public-input and arity
facts; it never requires equality with a deterministic private child tuple.

Emits constraints: no.

| Boundary | Exact equation or ownership rule | Lean owner |
|---|---|---|
| child public input | each public child equals the PiDEC split of the parent | `paperOutputEquations` |
| parent arity | parent evaluation size equals the paper PiDEC arity | `paperOutputEquations` |
| child arity | every child evaluation size equals the paper PiDEC arity | `paperOutputEquations` |
| recomposition | retained physical equations bind parent and child outputs | `PhysicalOutput.paperOutputEquations_of_recomposition` |
-/

set_option autoImplicit false
set_option maxRecDepth 2048

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.CertificateRefinement

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
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

/-- Every accepted physical certificate carrying the existing semantic
refinement satisfies all five operational paper output equations. -/
theorem paperOutputEquations
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (accepted : ConcretePhi81.Accepted context certificate)
    (refinement : ConcretePhi81.CertificateRefinement context data certificate) :
    FixedActive.PaperProfile.PhysicalOutput.PaperOutputEquations context
      certificate := by
  have publicInput : forall child,
      (outputChildren context certificate child).publicInput =
    (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf context)).split
          (derive context certificate).piRlcOutput.publicInput child := by
    intro child
    simpa only [FixedActive.paperProfileOf_decPublicInputSplit_split] using
      refinement.childPublicInput_eq_splitParent child
  have parentSize :
      (derive context certificate).piRlcOutput.evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem := by
    simpa only [FixedActive.paperProfileOf_decEvaluationArity_count] using
      refinement.parentEvaluations_size
  have childSize : forall child,
      (outputChildren context certificate child).evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem := by
    intro child
    simpa only [FixedActive.paperProfileOf_decEvaluationArity_count] using
      refinement.childEvaluations_size child
  exact
    FixedActive.PaperProfile.PhysicalOutput.equations_of_tail context
      certificate accepted.tail publicInput parentSize childSize

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.CertificateRefinement
