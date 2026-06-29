import SuperNeo.Primitives.PolynomialBridge
import SuperNeo.ProofSystem.Lattice

namespace Nightstream.PCSOpeningSemantics

open SuperNeo
open SuperNeo.ProofSystem

abbrev F := SuperNeo.Fq

structure RawScalarClaim (Family Point : Type*) where
  family : Family
  point : Point
  value : F
deriving DecidableEq, Repr

structure OpeningWitness (params : AjtaiParams) where
  commitment : Commitment
  opening : Opening
  opens : opensTo params commitment opening

def extractsRawScalar
  (extract : Commitment → Opening → Family → Point → F)
  {params : AjtaiParams}
  (w : OpeningWitness params)
  (claim : RawScalarClaim Family Point) : Prop :=
  claim.value = extract w.commitment w.opening claim.family claim.point

structure OpeningRefinement
  (params : AjtaiParams)
  (extract : Commitment → Opening → Family → Point → F)
  (claim : RawScalarClaim Family Point) where
  witness : OpeningWitness params
  extracts : extractsRawScalar extract witness claim

structure AcceptedScalarOpening
  (params : AjtaiParams)
  (extract : Commitment → Opening → Family → Point → F)
  (claim : RawScalarClaim Family Point) where
  refinement : OpeningRefinement params extract claim
  valueEq :
    claim.value =
      extract refinement.witness.commitment refinement.witness.opening
        claim.family claim.point
  normSound : Opening.NormSound refinement.witness.opening

def RawOpeningSeparation
  (params : AjtaiParams)
  (extract : Commitment → Opening → Family → Point → F)
  (claim : RawScalarClaim Family Point) : Prop :=
  ∃ refinement : OpeningRefinement params extract claim,
    claim.value =
        extract refinement.witness.commitment refinement.witness.opening
          claim.family claim.point ∧
      Opening.NormSound refinement.witness.opening

theorem opensTo_of_refinement
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : OpeningRefinement params extract claim) :
  opensTo params h.witness.commitment h.witness.opening := by
  exact h.witness.opens

theorem commitmentWellFormed_of_refinement
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : OpeningRefinement params extract claim) :
  Commitment.WellFormed params h.witness.commitment := by
  exact (opensTo_of_refinement h).1

theorem openingWellFormed_of_refinement
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : OpeningRefinement params extract claim) :
  Opening.WellFormed params h.witness.opening := by
  exact (opensTo_of_refinement h).2.1

theorem openingNormSound_of_refinement
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : OpeningRefinement params extract claim) :
  Opening.NormSound h.witness.opening := by
  exact (opensTo_of_refinement h).2.2.1

theorem rawScalarMatches_of_refinement
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : OpeningRefinement params extract claim) :
  claim.value =
    extract h.witness.commitment h.witness.opening claim.family claim.point := by
  exact h.extracts

def acceptedOpening_of_refinement
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : OpeningRefinement params extract claim) :
  AcceptedScalarOpening params extract claim := by
  exact
    { refinement := h
      valueEq := rawScalarMatches_of_refinement h
      normSound := openingNormSound_of_refinement h }

def refinement_of_acceptedOpening
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : AcceptedScalarOpening params extract claim) :
  OpeningRefinement params extract claim :=
  h.refinement

theorem rawOpeningSeparation_of_acceptedOpening
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : AcceptedScalarOpening params extract claim) :
  RawOpeningSeparation params extract claim := by
  exact ⟨h.refinement, h.valueEq, h.normSound⟩

theorem rawOpeningSeparation_of_refinement
  {params : AjtaiParams}
  {extract : Commitment → Opening → Family → Point → F}
  {claim : RawScalarClaim Family Point}
  (h : OpeningRefinement params extract claim) :
  RawOpeningSeparation params extract claim := by
  refine ⟨h, rawScalarMatches_of_refinement h, openingNormSound_of_refinement h⟩

end Nightstream.PCSOpeningSemantics
