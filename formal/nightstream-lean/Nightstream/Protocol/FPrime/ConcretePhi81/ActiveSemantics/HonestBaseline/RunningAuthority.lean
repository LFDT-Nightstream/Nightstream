import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
import Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren

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

Authority boundary: the generic theorem retains an explicit opening, exact
installation of the parent in `context.runningParent`, and one
`CanonicalChildren.ForOpening` proof. That proof packages the combined-stage,
valid-opening, and exact-child-split obligations under their generic PiDEC
owner. This module deliberately does not invent a zero specialization or
claim that production supplies any premise.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.honest_baseline.running.parent` | install the same parent used by canonical-child authority | direct dataflow | `accepted_of_combinedOpening` |
| `fprime.active.honest_baseline.running.canonical_children` | bind the complete running family to one valid combined parent opening | semantic input | `PiDEC.CanonicalChildren.ForOpening` |
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
    (canonical :
      PiDEC.CanonicalChildren.ForOpening
        (ConcretePhi81.decAlgebra context.key) parent assignment
        (ConcretePhi81.RunningAuthority.children context
          FixedActive.arity_mode)) :
    ConcretePhi81.RunningAuthority.Accepted context := by
  have complete := canonical.complete
  refine .active {
    active := FixedActive.arity_mode
    parent := parent
    parentBound := parentBound
    piDec := ?_
  }
  simpa [ConcretePhi81.RunningAuthority.attempt] using
    complete.1

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority
