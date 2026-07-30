import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews

/-!
Contract: bundle the exact selected-proof coordinate views required by one
operational NIFS profile.

Owns: only assembly of codec-derived views.  Every index is constructed in
`ConcreteNifsCanonicalViews` from the Lean codec product.

Does not own: verifier acceptance, transcript challenges, matrices, rows,
application data, Rust, or artifacts.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProfileViews

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

noncomputable def messageViews
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    MessageViews
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits)
      constraintPolynomial where
  feRow round slot :=
    proofFeRowView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits round slot
  feLane round slot :=
    proofFeLaneView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits round slot
  nc round slot :=
    proofNcView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits round slot

noncomputable def samplerViews
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    SamplerViews
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits) where
  challenge coordinate lane :=
    proofChallengeView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits coordinate lane

noncomputable def endpointViews
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    ProofViews
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits) where
  priorPoint coordinate :=
    proofPriorPointView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits coordinate
  claimedYRing running matrix lane :=
    proofClaimedYRingView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits running matrix lane
  outputYRing source matrix lane :=
    proofOutputYRingView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits source matrix lane
  outputYZcol source lane :=
    proofOutputYZcolView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits source lane

noncomputable def payloadViews
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (priorAbsorbed : Nat)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    PayloadViews
      (proofCodec shape constraintPolynomial priorAbsorbed
        publicRingColumns verifierRows publicFits) where
  commitment child row lane :=
    proofPayloadCommitmentView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits child row lane
  publicInput child column :=
    proofPayloadPublicView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits child column
  evaluation child matrix lane :=
    proofPayloadEvaluationView shape constraintPolynomial priorAbsorbed
      publicRingColumns verifierRows publicFits child matrix lane
  evaluationsSize proof admissible child := by
    change
      ProofAdmissible constraintPolynomial priorAbsorbed proof
      at admissible
    exact admissible.piDecEvaluations_size child

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProfileViews
