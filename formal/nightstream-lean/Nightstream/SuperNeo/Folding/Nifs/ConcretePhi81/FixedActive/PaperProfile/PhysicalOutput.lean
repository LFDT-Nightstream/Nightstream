import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

/-!
Exact operational `Pi_DEC` output boundary for the fixed active carrier.

Protocol: SuperNeo Section 7.5.
Phase: physical `Pi_DEC` output to the independent fixed-active paper profile.
Constraint family: typed semantic obligations only; this file emits no rows.

Assurance tier: model-level.

Owns: an exact characterization of operational paper output acceptance for the
parent and public child family computed by one physical certificate; and the
reduction of its arithmetic equations to the retained physical tail.

Does not own: `Pi_CCS` soundness, public source binding, transcript replay,
generated rows, Rust refinement, evaluation-column decoding, costs, or row
removal.

Authority boundary: generic `PiDEC.Accepted` is intentionally insufficient.
The paper verifier additionally computes every child public input and enforces
the fixed evaluation-vector arity. Those obligations remain explicit here;
neither child CE openings nor equality with an honest private split is used.
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalOutput

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

/-- Exactly the five non-computed equations of the Section-7.5 operational
verifier on the parent and actual public children carried by a physical
certificate. Structure, point, and fresh child stage are definitionally
inherited by `outputChildren`. -/
structure PaperOutputEquations
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Prop where
  canonicalPublicInput : forall child,
    (outputChildren context certificate child).publicInput =
      (FixedActive.PaperProfile.decPublicInputSplit
        (FixedActive.paperProfileOf context)).split
        (derive context certificate).piRlcOutput.publicInput child
  parentEvaluationSize :
    (derive context certificate).piRlcOutput.evaluations.size =
      (FixedActive.PaperProfile.decEvaluationArity
        (FixedActive.paperProfileOf context)).count
        (derive context certificate).piRlcOutput.constraintSystem
  childEvaluationSize : forall child,
    (outputChildren context certificate child).evaluations.size =
      (FixedActive.PaperProfile.decEvaluationArity
        (FixedActive.paperProfileOf context)).count
        (derive context certificate).piRlcOutput.constraintSystem
  commitment :
    (derive context certificate).piRlcOutput.commitment =
      (ConcretePhi81.decAlgebra context.key).recomposeCommitment
        (fun child => (outputChildren context certificate child).commitment)
  evaluations :
    (derive context certificate).piRlcOutput.evaluations =
      (ConcretePhi81.decAlgebra context.key).recomposeEvaluations
        (fun child => (outputChildren context certificate child).evaluations)

/-- The five explicit equations are exactly operational paper acceptance for
the physical parent and actual public child family. -/
theorem paperOutputAccepted_iff_equations
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    PiDEC.PaperVerifier.OutputAccepted (ConcretePhi81.decAlgebra context.key)
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf context))
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context))
        (derive context certificate).piRlcOutput
        (outputChildren context certificate) ↔
      PaperOutputEquations context certificate := by
  constructor
  · intro accepted
    exact {
      canonicalPublicInput := accepted.childPublicInput_eq
      parentEvaluationSize := accepted.parentEvaluations_size
      childEvaluationSize := accepted.childEvaluations_size
      commitment := accepted.checks.commitmentEquation
      evaluations := accepted.checks.evaluationEquation
    }
  · intro equations
    refine {
      outputComputed := ?_
      checks := {
        parentCombined := rfl
        parentEvaluationSize := equations.parentEvaluationSize
        messageEvaluationSize := equations.childEvaluationSize
        commitmentEquation := equations.commitment
        evaluationEquation := equations.evaluations
      }
    }
    funext child
    have publicInput := equations.canonicalPublicInput child
    simp only [outputChildren, Execution.piDecChildren,
      PiDecChildPayload.materialize] at publicInput
    simp only [PiDEC.PaperVerifier.children,
      PiDEC.PaperVerifier.attemptForOutput,
      PiDEC.PaperVerifier.messagesOf, outputChildren,
      Execution.piDecChildren, PiDecChildPayload.materialize]
    rw [← publicInput]

/-- Physical tail acceptance already supplies both arithmetic equations in the
operational paper contract. The only additional obligations are the
verifier-computed child public inputs and the two fixed-arity facts. -/
theorem equations_of_tail
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (tail : TailAccepted context certificate)
    (canonicalPublicInput : forall child,
      (outputChildren context certificate child).publicInput =
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf context)).split
          (derive context certificate).piRlcOutput.publicInput child)
    (parentEvaluationSize :
      (derive context certificate).piRlcOutput.evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem)
    (childEvaluationSize : forall child,
      (outputChildren context certificate child).evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem) :
    PaperOutputEquations context certificate := by
  refine {
    canonicalPublicInput := canonicalPublicInput
    parentEvaluationSize := parentEvaluationSize
    childEvaluationSize := childEvaluationSize
    commitment := ?_
    evaluations := ?_
  }
  · simpa [DerivedPiDec.RecompositionEquations, outputChildren,
      Execution.piDecChildren, PiDecChildPayload.materialize] using
      tail.piDecRecomposition.commitment
  · simpa [DerivedPiDec.RecompositionEquations, outputChildren,
      Execution.piDecChildren, PiDecChildPayload.materialize] using
      tail.piDecRecomposition.evaluations

/-- The retained physical tail plus the three paper-only shape obligations
construct exact operational paper output acceptance. -/
theorem paperOutputAccepted_of_tail
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (tail : TailAccepted context certificate)
    (canonicalPublicInput : forall child,
      (outputChildren context certificate child).publicInput =
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf context)).split
          (derive context certificate).piRlcOutput.publicInput child)
    (parentEvaluationSize :
      (derive context certificate).piRlcOutput.evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem)
    (childEvaluationSize : forall child,
      (outputChildren context certificate child).evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf context)).count
          (derive context certificate).piRlcOutput.constraintSystem) :
    PiDEC.PaperVerifier.OutputAccepted (ConcretePhi81.decAlgebra context.key)
      (FixedActive.PaperProfile.decPublicInputSplit
        (FixedActive.paperProfileOf context))
      (FixedActive.PaperProfile.decEvaluationArity
        (FixedActive.paperProfileOf context))
      (derive context certificate).piRlcOutput
      (outputChildren context certificate) := by
  exact (paperOutputAccepted_iff_equations context certificate).2
    (equations_of_tail context certificate tail canonicalPublicInput
      parentEvaluationSize childEvaluationSize)

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalOutput
