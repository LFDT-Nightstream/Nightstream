import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Lifecycle.Transcript
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Folding.Nifs.PaperProfile
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixLayout
import NightstreamFPrime.Spec.Folding.Nifs

/-!
Owns the one concrete Stage 1 SuperNeo NIFS verifier key. Every algebraic law
field is discharged by a concrete theorem (Goldilocks primality, the Φ₈₁
carrier laws, the Ajtai commitment algebra); the transcript fields are the
Poseidon2 sponge of `Lifecycle.Transcript`. The only inputs are the F′ logical
relation itself (matrices, constraint polynomial, its cube capacity and its
identity-first matrix proof) and the verifier-owned Ajtai key. Nothing is a
caller-supplied `Prop`.
-/

namespace NightstreamFPrime.Lifecycle.ProductionKey

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- The F′ logical relation as the key consumes it: the Φ₈₁ structure at the
Stage 1 shape, the proof that its carrier fits the `2^24` cube, and the
paper's `M₁ = [I; 0]` requirement. These are produced by the circuit builder
(Lifecycle composition) once the fixed point closes. -/
structure LogicalRelation (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  system : Phi81Relation.Structure (FullShape logicalWidth publicFits)
  cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth <= 2 ^ cubeVariables
  identityFirstEntry : ∀ (vertex : BooleanVertex cubeVariables)
      (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)),
    (matrixSource system).matrices ⟨0, by decide⟩ vertex column =
      (PrefixLayout.layout cubeVariables
        (Phi81CarrierLayout.carrierWidth logicalWidth) cubeFits).paddedIdentityEntry
        baseOps.zero baseOps.one vertex column

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}


/-- Per-round sum-check degree bound computed from the lifted constraint
polynomial: `max(D_f + 1, 4)` (paper D.4 with `b = 2`). -/
def degreeBound (relation : LogicalRelation logicalWidth publicFits) : Nat :=
  Nat.max
    (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
      (matrixSource relation.system).constraintPolynomial).canonicalEqualityGatedDegreeBound 4

abbrev KeyType (relation : LogicalRelation logicalWidth publicFits) :=
  Key K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    RingF Transcript.State productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
    (degreeBound relation)

/-- Absorb the complete public NIFS statement: the running claims, then the
fresh claims, each as self-delimiting blocks. -/
def absorbPublicInput (s : Transcript.State)
    (running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape) :
    Transcript.State :=
  let s := Transcript.absorbBlock s (serializePoint running.point)
  let s := (List.finRange productionShape.runningCount).foldl (fun s i =>
    let s := Transcript.absorbBlock s (running.commitments i |> fun c =>
      (List.finRange productionProfile.commitmentWidth).flatMap fun r => serializeRingF (c r))
    let s := Transcript.absorbBlock s ((List.finRange (FullShape logicalWidth publicFits).publicWidth).map
      fun j => running.publicInputs i j)
    Transcript.absorbBlock s (serializeEvaluations (running.evaluations i))) s
  (List.finRange productionShape.freshCount).foldl (fun s i =>
    let s := Transcript.absorbBlock s
      ((List.finRange productionProfile.commitmentWidth).flatMap fun r =>
        serializeRingF (fresh.commitments i r))
    Transcript.absorbBlock s ((List.finRange (FullShape logicalWidth publicFits).publicWidth).map
      fun j => fresh.publicInputs i j)) s

/-- Absorb the complete paper `y′` family after the sum-check. -/
def absorbFullOutput (s : Transcript.State)
    (out : FullOutputCoordinates.FullOutput K productionShape) : Transcript.State :=
  Transcript.absorbBlock s ((List.finRange productionShape.sourceCount).flatMap fun i =>
    (List.finRange productionShape.matrixCount).flatMap fun j =>
      (List.finRange productionShape.coefficientCount).flatMap fun l =>
        serializeK (out.coordinate i j l))

/-- `ρ_i` for source `i`, squeezed from the post-output state. -/
def piRlcResponse (s : Transcript.State) (index : Fin (Nifs.PaperProfile.arity).total) : RingF :=
  (Transcript.piRlcChallenges s (Nifs.PaperProfile.arity).total).getD index.val ringFZero

theorem piRlcResponse_valid (s : Transcript.State) (index : Fin (Nifs.PaperProfile.arity).total) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (piRlcResponse s index) := by
  unfold piRlcResponse
  have hlen : index.val < (Transcript.piRlcChallenges s (Nifs.PaperProfile.arity).total).length := by
    rw [Transcript.piRlcChallenges_length]; exact index.isLt
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlen, Option.getD_some]
  exact Transcript.piRlcChallenges_member s _ _ (List.getElem_mem hlen)

/-- The Stage 1 production NIFS key for one logical relation and one
verifier-owned Ajtai key. -/
noncomputable def key (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    KeyType relation where
  baseOps := baseOps
  baseLaws := baseLaws
  baseZero := baseZeroAgreement
  noZeroDivisors := GoldilocksPrime.baseFieldNoZeroDivisors
  extensionOps := extensionOps
  extensionLaws := extensionLaws
  extensionZeroLaws := extensionZeroLaws
  lift := K.embed
  liftLaws := protocolLift
  openingMaps := openingMaps ajtai
  params := productionGlobalParams
  freshBound := rfl
  arity := Nifs.PaperProfile.arity
  freshCount_eq := rfl
  runningCount_eq := rfl
  outputCount_eq := rfl
  kPositive := by decide
  cubeLayout := PrefixLayout.layout cubeVariables
    (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits
  matrixSource := matrixSource relation.system
  degreeBoundExact := rfl
  matrixCountPositive := by decide
  identityFirstEntry := relation.identityFirstEntry
  constantLaw := Phi81CoefficientKernel.phi81ConstantTermLaw
  challengeSetSize := 5 ^ ringDegree
  piRlcSemantics := semantics ajtai
  openingAgreement := openingAgreement ajtai
  ambientAgreement := ambientAgreement ajtai relation.system
  evaluationAgreement := by
    intro assignment point
    exact ⟨True.intro, evaluations_eq_paper ajtai relation.system assignment point⟩
  piRlcEvaluationsSize := semantics_evaluations_size ajtai
  piRlcAlgebra := piRlcAlgebra ajtai
  piDecAlgebra := piDecAlgebra ajtai
  piDecPublicInputSplit := publicInputSplit ajtai
  piDecEvaluationArity := evaluationArity ajtai
  piDecEvaluationCount := rfl
  piDecDecision := fun _ => Classical.propDecidable _
  oracle := Transcript.piCcsOracle
  initialTranscriptState := Transcript.absorb Transcript.initialState Transcript.domainTag
  absorbPublicInput := absorbPublicInput
  absorbPiCcsOutput := absorbFullOutput
  piRlcResponse := piRlcResponse
  piRlcResponseValid := piRlcResponse_valid

end

end NightstreamFPrime.Lifecycle.ProductionKey
