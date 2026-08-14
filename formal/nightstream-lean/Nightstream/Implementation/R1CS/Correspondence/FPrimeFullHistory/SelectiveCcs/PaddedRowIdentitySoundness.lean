import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentity
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction

/-!
Contract: deterministic soundness reduction for the selected
`PaddedRowIdentity` relation.

Owns: construction of the exact identity-first public statement; refinement
of the generic joint-paper source relation to the direct logical selective CCS
relation; and specialization of the accepted-probe extraction theorem to the
24-round, degree-nine profile.

Does not own: probability bounds for the two named bad events, Fiat--Shamir,
commitment binding, Rust or R1CS conformance, or release security.

Emits constraints: no.

Assurance tier: model-level. The main theorem does not assume source
satisfaction. It derives direct logical source truth from verifier acceptance
and an ambient output witness, unless an explicit joint-mixing root or
SumCheck bad challenge occurs. Thus it is a real reduction statement, not an
honest-prover-only check.

| Code owner | Protocol object | Mathematical obligation | Proven result |
|---|---|---|---|
| `statement` | selected identity-first joint statement | use 24 row variables and the exact padded relation | typed construction |
| `statement_sumcheckDegree_exact` | joint SumCheck | the equality-gated polynomial degree must be nine | exact equality |
| `sourceHolds_iff_logicalSourceHolds` | source relation | generic padded truth must equal direct logical truth | exact iff |
| `acceptedProbe_extracts_logicalSource_or_badEvent` | accepted public-coin probe | acceptance must imply logical truth or a named algebraic bad event | deterministic disjunction |
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySoundness

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.SumCheck
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open PaddedRowIdentity

universe uCommitment uPublicInput

/-- Public statement for the exact selected matrix and source dimensions. -/
def statement
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) :
    Statement K Commitment PublicInput shape assignmentColumns
      (Phi81ColumnLayout.blockCount assignmentColumns) baseOps where
  cubeLayout := assignmentLayout
  matrixSource := matrixSource matrices
  commitments := commitments
  publicInputs := publicInputs
  priorPoint := priorPoint
  claimedCoefficient := claimedCoefficient
  matrixCountPositive := by decide
  identityFirstEntry := by
    intro vertex column
    rfl

/-- The verifier degree is exactly nine after both identity-variable
prepending and base-to-extension coefficient lifting. -/
theorem statement_sumcheckDegree_exact
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) :
    ((statement matrices commitments publicInputs priorPoint
      claimedCoefficient).verifierInput K.embed).sumcheckDegreeBound = 9 := by
  unfold Statement.verifierInput ProtocolPolynomial.VerifierInput.sumcheckDegreeBound
  rw [ConstraintPolynomialLift.liftConstraintPolynomial_canonicalEqualityGatedDegreeBound]
  change Nat.max
      ((ConstraintPolynomialPrepend.prependIgnoredVariable
        Semantics.polynomial).canonicalEqualityGatedDegreeBound) 4 = 9
  rw [ConstraintPolynomialPrepend.prependIgnoredVariable_canonicalEqualityGatedDegreeBound]
  rw [Semantics.canonicalEqualityGatedDegreeBound_exact]
  decide

/-- Source truth with CCS stated on the direct logical row set. -/
def LogicalSourceHolds
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (params : GlobalParams)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (witness : OutputWitness shape assignmentColumns) : Prop :=
  (forall source,
    Opening.Holds
      (paperRelationSemantics (shape := shape)
        (blockCount := Phi81ColumnLayout.blockCount assignmentColumns)
        baseOps extensionOps K.embed openingMaps)
      params.b (commitments source) (publicInputs source)
      (witness.assignments source)) /\
  LogicalSemanticTruth matrices witness.assignments priorPoint
    claimedCoefficient

/-- The generic connected source relation and the direct logical source
relation are exactly equivalent for this profile. -/
theorem sourceHolds_iff_logicalSourceHolds
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (params : GlobalParams)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (witness : OutputWitness shape assignmentColumns) :
    SourceHolds extensionOps K.embed openingMaps params
        (statement matrices commitments publicInputs priorPoint
          claimedCoefficient) witness <->
      LogicalSourceHolds openingMaps params matrices commitments publicInputs
        priorPoint claimedCoefficient witness := by
  unfold SourceHolds LogicalSourceHolds
  change
    ((forall source,
        Opening.Holds
          (paperRelationSemantics (shape := shape)
            (blockCount := Phi81ColumnLayout.blockCount assignmentColumns)
            baseOps extensionOps K.embed openingMaps)
          params.b (commitments source) (publicInputs source)
          (witness.assignments source)) /\
      (connectedInputs matrices witness.assignments priorPoint
        claimedCoefficient).SemanticTruth baseOps extensionOps K.embed) <-> _
  rw [connectedSemanticTruth_iff_logicalSemanticTruth]

/-- The two explicit algebraic failure branches left by deterministic
extraction. -/
def BadEvent
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (challengeSetSize : Nat)
    (probe : Probe K shape)
    (witness : OutputWitness shape assignmentColumns) : Prop :=
  let selected :=
    statement matrices commitments publicInputs priorPoint claimedCoefficient
  let data := selected.sourceProtocolData K.embed witness
  SignedCoefficientObject.MixingRoot extensionOps
      (data.toJointData extensionOps)
      probe.coins.alpha probe.coins.gamma \/
    (exists round,
      SumCheck.BadChallenge
        (SumCheckInitial.symbolicInstance extensionOps
          (data.toJointData extensionOps)
          probe.coins.alpha probe.coins.gamma
          data.toVerifierInput.sumcheckDegreeBound
          challengeSetSize probe.coins.roundPoint.coordinates
          (ProtocolPolynomial.terminalFromMessage extensionOps
            data.toVerifierInput probe.coins.alpha probe.coins.gamma
            probe.coins.roundPoint
            (selected.projectOutput probe.response.fullOutput))
          probe.response.rounds
          (ProtocolPolynomial.canonicalExpected extensionOps data
            probe.coins.alpha probe.coins.gamma
            probe.coins.roundPoint.coordinates))
        round)

/-- Deterministic soundness gate for the selected padded protocol. An accepted
probe with a valid corrected-ambient output witness yields the direct logical
source relation, or one of the two named polynomial-collision events. -/
theorem acceptedProbe_extracts_logicalSource_or_badEvent
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (params : GlobalParams)
    (freshBound : params.b = 2)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (challengeSetSize : Nat)
    (probe : Probe K shape)
    (witness : OutputWitness shape assignmentColumns)
    (ambient : AmbientOutputHolds extensionOps K.embed openingMaps params
      (statement matrices commitments publicInputs priorPoint
        claimedCoefficient) probe witness)
    (accepted : probe.Accepted extensionOps K.embed
      (statement matrices commitments publicInputs priorPoint
        claimedCoefficient)) :
    LogicalSourceHolds openingMaps params matrices commitments publicInputs
        priorPoint claimedCoefficient witness \/
      BadEvent matrices commitments publicInputs priorPoint claimedCoefficient
        challengeSetSize probe witness := by
  let selected :=
    statement matrices commitments publicInputs priorPoint claimedCoefficient
  rcases acceptedProbe_extracts_source_or_badEvent
      baseLaws baseZeroAgreement
      (NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
        Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime)
      extensionOps extensionLaws extensionZeroLaws K.embed
      ConcreteCarrier.protocolLift
      openingMaps params freshBound selected
      Phi81CoefficientKernel.phi81ConstantTermLaw challengeSetSize
      probe witness ambient accepted with source | bad
  · left
    exact (sourceHolds_iff_logicalSourceHolds openingMaps params matrices
      commitments publicInputs priorPoint claimedCoefficient witness).mp source
  · right
    exact bad

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySoundness
