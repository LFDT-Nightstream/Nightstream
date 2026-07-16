import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc
import Nightstream.SuperNeo.Concrete.Phi81StrongSet

/-!
Production Phi81 challenge algebra for the paper-facing `Pi_RLC` verifier.

Assurance tier: executable mathematical semantics. This module instantiates
the abstract paper-facing `RingAlgebra` from independent Phi81 semantics. It
keeps unary challenge membership and the pairwise strong-set security law as
different propositions, so a range check cannot masquerade as invertibility.

Owns: the canonical 54-field list carrier; exact list-to-Phi81 decoding;
production challenge membership as the image of the five-symbol semantic
embedding; the concrete `RingAlgebra`; and the pairwise Definition-17 law
conditional only on the isolated low-norm invertibility theorem.

Does not own: Fiat-Shamir sampling, transcript authority, R1CS columns or
rows, Rust refinement, the external Lyubashevsky-Seiler theorem, constraint
removal, or cost totals.

Emits constraints: no.

Authority boundary: a valid list is exactly a canonical serialization of a
typed 54-coordinate production scalar. Arbitrary lists that merely decode to
the same ring element are not members. The global strong-set theorem cannot
be constructed from unary membership alone; it additionally consumes the
explicit `LowNormInvertibility` mathematical boundary.

| Protocol | Phase | Mathematical object | Exact obligation | Status |
|---|---|---|---|---|
| `Pi_RLC` | carrier | `canonicalRing` | serialize exactly 54 Phi81 coefficients | proved |
| `Pi_RLC` | decoding | `ringOfList_canonicalRing` | serialization round-trips to the same `RingF` | proved |
| `Pi_RLC` | membership | `ChallengeMember` | list is the image of one typed five-symbol scalar | definition |
| `Pi_RLC` | algebra | `productionRingAlgebra` | public combination is exactly `phi81Combine` | proved by construction |
| Definition 17 | pairwise security | `StrongChallengeSet` | distinct valid lists have invertible ring difference | conditional only on `LowNormInvertibility` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ProductionRingAlgebra

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81StrongSet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

/-- Canonical coefficient-order list for a concrete Phi81 ring element. -/
def canonicalRing (value : RingF) : Ring :=
  List.ofFn value

@[simp] theorem canonicalRing_length (value : RingF) :
    (canonicalRing value).length = ringDegree := by
  simp [canonicalRing]

/-- Decoding a canonical 54-coordinate list recovers the original ring
element at every coefficient. -/
theorem ringOfList_canonicalRing (value : RingF) :
    ringOfList (canonicalRing value) = value := by
  funext position
  simp [ringOfList, canonicalRing]

/-- Exact unary membership in the production challenge carrier. -/
def ChallengeMember (value : Ring) : Prop :=
  exists scalar : ProductionStrongSet.Scalar,
    value = canonicalRing (embedScalar scalar)

theorem embeddedScalar_member (scalar : ProductionStrongSet.Scalar) :
    ChallengeMember (canonicalRing (embedScalar scalar)) :=
  ⟨scalar, rfl⟩

theorem challengeMember_length {value : Ring}
    (member : ChallengeMember value) : value.length = ringDegree := by
  obtain ⟨scalar, rfl⟩ := member
  exact canonicalRing_length (embedScalar scalar)

/-- Unary list membership refines to semantic quotient-ring membership. -/
theorem challengeMember_ringOfList {value : Ring}
    (member : ChallengeMember value) :
    ProductionMember (ringOfList value) := by
  obtain ⟨scalar, rfl⟩ := member
  rw [ringOfList_canonicalRing]
  exact embedScalar_member scalar

/-- Concrete paper-facing algebra. Its validity predicate is exact canonical
membership, while its public operation is the already independent Phi81
combination. -/
def productionRingAlgebra : RingAlgebra where
  challengeValid := ChallengeMember
  combine := phi81Combine
  phi81 := by
    intros
    rfl

theorem productionRingAlgebra_membership_iff (value : Ring) :
    productionRingAlgebra.challengeValid value ↔ ChallengeMember value := by
  rfl

/-- Definition-17 law on the paper list carrier. This is intentionally
separate from `RingAlgebra.challengeValid`, which is unary. -/
def StrongChallengeSet (ring : RingAlgebra) : Prop :=
  forall {left right : Ring},
    ring.challengeValid left ->
    ring.challengeValid right ->
    left ≠ right ->
    RingFInvertible (ringFSub (ringOfList left) (ringOfList right))

/-- Every implementation-independent premise of the production strong-set
law is proved. Only the explicitly isolated external low-norm theorem remains
as a parameter. -/
theorem productionRingAlgebra_strong
    (theorem8 : LowNormInvertibility) :
    StrongChallengeSet productionRingAlgebra := by
  intro left right leftMember rightMember different
  change ChallengeMember left at leftMember
  change ChallengeMember right at rightMember
  obtain ⟨leftScalar, rfl⟩ := leftMember
  obtain ⟨rightScalar, rfl⟩ := rightMember
  rw [ringOfList_canonicalRing, ringOfList_canonicalRing]
  exact productionSet_strong theorem8
    (embedScalar_member leftScalar)
    (embedScalar_member rightScalar)
    (fun equal => different (congrArg canonicalRing equal))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ProductionRingAlgebra
