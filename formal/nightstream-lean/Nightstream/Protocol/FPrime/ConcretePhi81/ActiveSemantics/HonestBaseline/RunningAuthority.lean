import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

/-!
Model-level checked running-authority construction for fixed-active F-prime.

Owns: one reusable lift from a valid combined Phi81 CE opening and its
canonical `PiDEC.childrenOf` family to
`ConcretePhi81.RunningAuthority.Accepted`.

Does not own: source data, a Split-NC input, an F-prime `Setup`, construction
of a production context, derivation of the incoming accumulator from a prior
step, transcript execution, commitment binding security, Rust, R1CS, costs,
or row removal.

Emits constraints: no.

Authority boundary: the generic theorem retains five explicit inputs that it
must not manufacture: an opening assignment, the parent's combined-stage
tag, proof that the parent is valid at that opening, exact installation of the
parent in `context.runningParent`, and equality of the public running children
with the canonical `PiDEC.childrenOf` family. This module deliberately does
not invent a zero specialization: choosing a relation structure, evaluation
point, and public context is a separate source-authority obligation. This
theorem does not claim that production supplies any of its premises.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.honest_baseline.running.parent_stage` | the installed parent is at the combined norm stage | explicit semantic input | `accepted_of_combinedOpening` |
| `fprime.active.honest_baseline.running.parent_opening` | one installed combined CE parent has a valid semantic opening | explicit semantic input | `accepted_of_combinedOpening` |
| `fprime.active.honest_baseline.running.children` | the installed running family is exactly `PiDEC.childrenOf` for that opening | explicit equality input | `accepted_of_combinedOpening` |
| `fprime.active.honest_baseline.running.pi_dec` | honest split/recomposition satisfies every public PiDEC equation | derived | `accepted_of_combinedOpening` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

section

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- A valid combined parent opening whose exact canonical children are
installed in a fixed-active context satisfies the checked incoming
running-authority predicate.

The opening is used only to invoke PiDEC perfect completeness. It is not added
to the public context or exposed as verifier authority. -/
theorem accepted_of_combinedOpening
    (context :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows)
    (assignment : SourceAssignment shape)
    (parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (parentBound : context.runningParent = some parent)
    (parentCombined : parent.stage = .combined)
    (parentValid :
      CE.Holds (ConcretePhi81.semantics context.key) productionGlobalParams
        parent assignment)
    (childrenCanonical :
      ConcretePhi81.RunningAuthority.children context FixedActive.arity_mode =
        PiDEC.childrenOf (ConcretePhi81.decAlgebra context.key) parent
          assignment) :
    ConcretePhi81.RunningAuthority.Accepted context := by
  have complete :=
    PiDEC.complete (ConcretePhi81.semantics context.key)
      productionGlobalParams (ConcretePhi81.decAlgebra context.key) parent
      assignment parentCombined parentValid
  refine .active {
    active := FixedActive.arity_mode
    parent := parent
    parentBound := parentBound
    piDec := ?_
  }
  simpa [ConcretePhi81.RunningAuthority.attempt, childrenCanonical] using
    complete.1

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority
