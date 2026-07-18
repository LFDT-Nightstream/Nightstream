import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiRlcSidecar
import Nightstream.SuperNeo.ProjectionCheck

/-!
Typed delayed projection authority for the packed `yZcol` sidecar.

Protocol: SuperNeo `Pi_CCS -> Pi_RLC`.
Phases: packed source aggregation followed by one generic delayed pair check.
Constraint family: one 54-coefficient projection identity; this file emits no
rows.

Assurance tier: model-level.

Owns: the fixed-width `RingK` coefficient codec; a generic degree-53 packed
pair identity; its exact-or-bad-root soundness boundary; the canonical
`Pi_RLC` source aggregate; and a scalar interface that separates left-side
acceptance from right-side semantic equality.

Does not own: the optional `Pi_DEC` child-recomposition specialization,
commitment openings, construction of the physical aggregate equation,
Fiat--Shamir timing, bad-root or mixing probability, Poseidon2, Rust/R1CS
refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: the generic pair check is only semantic equality at one
point. A physical consumer must separately prove that each side comes from its
authoritative source. The producer challenge supports the soundness argument
only up to `ProjectionCheck.BadRoot` and must later be proved transcript-
derived after both vectors are fixed. A digest, scalar evaluation, or self-
consistent parent is never treated as an opening.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.shared.delayed_packed.coefficients` | preserve all 54 `RingK` lanes in canonical order | computed | `coefficients`, `coefficients_injective` |
| `nifs.shared.delayed_packed.pair` | compare any two packed values at one producer challenge | checked / security boundary | `pairIdentity`, `PairAccepted`, `pairAccepted_implies_exact_or_badRoot` |
| `nifs.shared.delayed_packed.scalar.left` | one scalar equals the left packed projection at `beta` | semantic match | `PairLeftScalarMatches` |
| `nifs.shared.delayed_packed.scalar.right` | that scalar equals the expected right packed projection | semantic match | `PairRightScalarMatches` |
| `nifs.shared.delayed_packed.scalar.compose` | the two scalar links instantiate the fixed-width pair identity | derived | `pairAccepted_of_scalar_matches` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism

/-- Canonical fixed-width coefficient representation of one packed sidecar. -/
def coefficients (value : RingK) : List K :=
  List.ofFn value

@[simp] theorem coefficients_length (value : RingK) :
    (coefficients value).length = ringDegree := by
  simp [coefficients]

/-- The fixed-width coefficient codec loses no packed lane. -/
theorem coefficients_injective : Function.Injective coefficients := by
  intro left right equal
  funext lane
  have atLane := congrArg (fun values => values.getD lane.val K.zero) equal
  simpa [coefficients] using atLane

/-- One fixed-width identity between arbitrary packed values. -/
def pairIdentity
    (left right : RingK)
    (producerBeta : K) : ProjectionCheck.Identity K where
  lhs := coefficients left
  rhs := coefficients right
  beta := producerBeta
  maxDegree := ringDegree - 1

/-- Concrete extension-field operations used by the one-point check. -/
def projectionOps : ProjectionCheck.Ops K where
  zero := K.zero
  add := K.add
  mul := K.mul

/-- Model-level acceptance predicate for a direct packed pair. -/
def PairAccepted
    (left right : RingK)
    (producerBeta : K) : Prop :=
  ProjectionCheck.Accepted projectionOps
    (pairIdentity left right producerBeta)

/-- Canonical source aggregate. This value is computed from public source
claims and transcript challenges; it is never a prover-owned parent sidecar. -/
def sourceAggregate
    {count : Nat}
    (challenges : Fin count -> RingF)
    (claims : Fin count -> RingK) : RingK :=
  PiRLCFinite.combineEvaluation challenges claims

/-- Evaluate one fixed-width packed value at the producer challenge. This is
one scalar, not authority for the underlying coefficient vector. -/
def projectedValue (value : RingK) (producerBeta : K) : K :=
  ProjectionCheck.eval projectionOps (coefficients value) producerBeta

/-- The claimed scalar is the projection of the left packed value. -/
def PairLeftScalarMatches
    (left : RingK) (claimedProjection producerBeta : K) : Prop :=
  projectedValue left producerBeta = claimedProjection

/-- The same scalar equals the expected projection of the right packed value.
This predicate is a semantic target, not evidence of a commitment opening. -/
def PairRightScalarMatches
    (right : RingK) (claimedProjection producerBeta : K) : Prop :=
  claimedProjection = projectedValue right producerBeta

/-- The checked scalar is the projection of the computed source aggregate. -/
def SourceProjectionMatches
    {count : Nat}
    (challenges : Fin count -> RingF)
    (sourceClaims : Fin count -> RingK)
    (claimedProjection producerBeta : K) : Prop :=
  PairLeftScalarMatches (sourceAggregate challenges sourceClaims)
    claimedProjection producerBeta

/-- The two scalar links instantiate one direct packed-pair check. -/
theorem pairAccepted_of_scalar_matches
    (left right : RingK)
    (claimedProjection producerBeta : K)
    (leftMatches : PairLeftScalarMatches left claimedProjection producerBeta)
    (rightMatches : PairRightScalarMatches right claimedProjection producerBeta) :
    PairAccepted left right producerBeta := by
  refine ⟨?_, ?_⟩
  · simp [pairIdentity, ProjectionCheck.Identity.WellFormed]
    decide
  change projectedValue left producerBeta = projectedValue right producerBeta
  exact leftMatches.trans rightMatches

/-- Every direct packed pair has exactly 54 coefficients and degree at most
53. -/
theorem pairIdentity_wellFormed
    (left right : RingK)
    (producerBeta : K) :
    (pairIdentity left right producerBeta).WellFormed := by
  simp [pairIdentity, ProjectionCheck.Identity.WellFormed]
  decide

/-- Exactness of the direct identity is packed-value equality. -/
theorem pairIdentity_exact_iff
    (left right : RingK)
    (producerBeta : K) :
    (pairIdentity left right producerBeta).Exact <-> left = right := by
  constructor
  · intro exact
    apply coefficients_injective
    exact exact
  · intro exact
    simpa [ProjectionCheck.Identity.Exact, pairIdentity] using
      congrArg coefficients exact

/-- Deterministic direct-pair soundness. -/
theorem pairAccepted_implies_exact_or_badRoot
    (left right : RingK)
    (producerBeta : K)
    (accepted : PairAccepted left right producerBeta) :
    left = right ∨
      ProjectionCheck.BadRoot projectionOps
        (pairIdentity left right producerBeta) := by
  rcases ProjectionCheck.accepted_implies_exact_or_badRoot projectionOps
      (pairIdentity left right producerBeta) accepted with exact | badRoot
  · exact Or.inl <| (pairIdentity_exact_iff left right producerBeta).1 exact
  · exact Or.inr badRoot

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
