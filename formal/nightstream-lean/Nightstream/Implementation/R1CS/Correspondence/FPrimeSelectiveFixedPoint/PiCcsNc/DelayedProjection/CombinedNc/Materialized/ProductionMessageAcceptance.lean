import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectedRowsSoundness
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs

/-!
Exact dataflow contract from the materialized combined-NC assignment to the
production claims-level NC verifier.

Owns: the generic algebraic handoff from the claimed chain derived by
`SourceRowsSoundness.Consequences` to `ProductionPiCcs.NcMessageAccepted`.
The explicit carrier homomorphism transports the projection interpreter's
independently named Goldilocks extension into the production semantic carrier.
The four equalities in `ExactDataflow` then identify, on one reconstructed
assignment, the transported first claimed value, all 25 five-coefficient
round messages, verifier-derived challenge coordinates, and final claim.

Does not own: reconstruction of the assignment from production columns,
proof of any `ExactDataflow` field, source-row satisfaction, transcript
scheduling, parent or raw-child authority, projection truth, commitment
binding, Rust conformance, costs, or row removal. Final production refinement
must derive every field from the encoder, generated rows, and transcript
replay; none may be supplied as an acceptance or semantic-authority premise.

Emits constraints: none.

Assurance tier: model-level intermediate Rust/R1CS refinement contract.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.production_message_acceptance` | Derive typed combined-NC message acceptance from the bound production columns. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

universe uSource uTarget uState

/-- The exact operations that a carrier map must preserve for finite
claimed-chain replay. No field laws or inverse map are needed. -/
structure OpsHomomorphism
    {Source : Type uSource}
    {Target : Type uTarget}
    (sourceOps : Ops Source)
    (targetOps : Ops Target)
    (map : Source → Target) : Prop where
  zero : map sourceOps.zero = targetOps.zero
  one : map sourceOps.one = targetOps.one
  add : ∀ left right,
    map (sourceOps.add left right) = targetOps.add (map left) (map right)
  mul : ∀ left right,
    map (sourceOps.mul left right) = targetOps.mul (map left) (map right)

/-- Coefficient-wise transport preserves the statically checked width. -/
def mapFixedPolynomial
    {Source : Type uSource}
    {Target : Type uTarget}
    {degree : Nat}
    (map : Source → Target)
    (polynomial : FixedPolynomial Source degree) :
    FixedPolynomial Target degree where
  coefficients := polynomial.coefficients.map map
  coefficients_length := by
    simpa using polynomial.coefficients_length

private theorem evaluateCoefficients_map
    {Source : Type uSource}
    {Target : Type uTarget}
    (sourceOps : Ops Source)
    (targetOps : Ops Target)
    (map : Source → Target)
    (homomorphism : OpsHomomorphism sourceOps targetOps map)
    (point : Source)
    (coefficients : List Source) :
    map (Message.evaluateCoefficients sourceOps point coefficients) =
      Message.evaluateCoefficients targetOps (map point)
        (coefficients.map map) := by
  induction coefficients with
  | nil =>
      simpa only [Message.evaluateCoefficients, List.map_nil] using
        homomorphism.zero
  | cons coefficient coefficients inductionHypothesis =>
      simp only [Message.evaluateCoefficients, List.map_cons]
      rw [homomorphism.add, homomorphism.mul, inductionHypothesis]

/-- Horner evaluation commutes with every operations homomorphism. -/
theorem mapFixedPolynomial_evaluate
    {Source : Type uSource}
    {Target : Type uTarget}
    {degree : Nat}
    (sourceOps : Ops Source)
    (targetOps : Ops Target)
    (map : Source → Target)
    (homomorphism : OpsHomomorphism sourceOps targetOps map)
    (polynomial : FixedPolynomial Source degree)
    (point : Source) :
    map (polynomial.evaluate sourceOps point) =
      (mapFixedPolynomial map polynomial).evaluate targetOps (map point) := by
  change map
      (Message.evaluateCoefficients sourceOps point polynomial.coefficients) =
    Message.evaluateCoefficients targetOps (map point)
      (polynomial.coefficients.map map)
  exact evaluateCoefficients_map sourceOps targetOps map homomorphism point
    polynomial.coefficients

/-- A complete finite claimed chain transports coefficient-wise across any
map preserving zero, one, addition, and multiplication. This theorem changes
only representation; it cannot establish message or transcript dataflow. -/
theorem chain_transport
    {Source : Type uSource}
    {Target : Type uTarget}
    {degree : Nat}
    (sourceOps : Ops Source)
    (targetOps : Ops Target)
    (map : Source → Target)
    (homomorphism : OpsHomomorphism sourceOps targetOps map)
    {current terminal : Source}
    {rounds : List (FixedPolynomial Source degree)}
    {challenges : List Source}
    (chain : FixedPhase.Chain sourceOps current rounds challenges terminal) :
    FixedPhase.Chain targetOps (map current)
      (rounds.map (mapFixedPolynomial map))
      (challenges.map map) (map terminal) := by
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges with
      | nil =>
          simp only [FixedPhase.Chain, List.map_nil] at chain ⊢
          exact congrArg map chain
      | cons challenge challenges =>
          simp only [FixedPhase.Chain] at chain
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil =>
          simp only [FixedPhase.Chain] at chain
      | cons challenge challenges =>
          simp only [FixedPhase.Chain, List.map_cons] at chain ⊢
          constructor
          · calc
              map current =
                  map (sourceOps.add
                    (polynomial.evaluate sourceOps sourceOps.zero)
                    (polynomial.evaluate sourceOps sourceOps.one)) :=
                congrArg map chain.1
              _ = targetOps.add
                    (map (polynomial.evaluate sourceOps sourceOps.zero))
                    (map (polynomial.evaluate sourceOps sourceOps.one)) :=
                homomorphism.add _ _
              _ = targetOps.add
                    ((mapFixedPolynomial map polynomial).evaluate targetOps
                      (map sourceOps.zero))
                    ((mapFixedPolynomial map polynomial).evaluate targetOps
                      (map sourceOps.one)) := by
                rw [mapFixedPolynomial_evaluate sourceOps targetOps map
                    homomorphism,
                  mapFixedPolynomial_evaluate sourceOps targetOps map
                    homomorphism]
              _ = targetOps.add
                    ((mapFixedPolynomial map polynomial).evaluate targetOps
                      targetOps.zero)
                    ((mapFixedPolynomial map polynomial).evaluate targetOps
                      targetOps.one) := by
                rw [homomorphism.zero, homomorphism.one]
          · have tail := inductionHypothesis
                (current := polynomial.evaluate sourceOps challenge)
                (challenges := challenges) chain.2
            simpa only [mapFixedPolynomial_evaluate sourceOps targetOps map
              homomorphism polynomial challenge] using tail

/-- The two independently named base carriers use the same production
Goldilocks modulus. The conversion is explicit so no cross-layer theorem
depends on a definitional coincidence between their aliases. -/
def toConcreteField (value : ProjectionProgram.F) : F :=
  ⟨value.val, by
    simpa [goldilocksP, goldilocksModulus] using value.isLt⟩

@[simp] theorem toConcreteField_add
    (left right : ProjectionProgram.F) :
    toConcreteField (left + right) =
      toConcreteField left + toConcreteField right := by
  apply Fin.ext
  rfl

@[simp] theorem toConcreteField_mul
    (left right : ProjectionProgram.F) :
    toConcreteField (left * right) =
      toConcreteField left * toConcreteField right := by
  apply Fin.ext
  rfl

@[simp] theorem toConcreteField_seven :
    toConcreteField (7 : ProjectionProgram.F) = (7 : F) := by
  apply Fin.ext
  rfl

/-- Structure-preserving conversion from the projection interpreter carrier
to the independent production semantic carrier. -/
def toConcreteK (value : ProjectionProgram.K) : K :=
  ⟨toConcreteField value.c0, toConcreteField value.c1⟩

@[simp] theorem toConcreteK_zero :
    toConcreteK ProjectionProgram.K.zero = K.zero := by
  rfl

@[simp] theorem toConcreteK_one :
    toConcreteK ProjectionProgram.K.one = K.one := by
  rfl

@[simp] theorem toConcreteK_add (left right : ProjectionProgram.K) :
    toConcreteK (ProjectionProgram.K.add left right) =
      K.add (toConcreteK left) (toConcreteK right) := by
  rfl

@[simp] theorem toConcreteK_mul (left right : ProjectionProgram.K) :
    toConcreteK (ProjectionProgram.K.mul left right) =
      K.mul (toConcreteK left) (toConcreteK right) := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [toConcreteK, ProjectionProgram.K.mul, K.mul, K.mk.injEq,
    toConcreteField_add, toConcreteField_mul, toConcreteField_seven]
  constructor
  · apply Fin.ext
    simp [Fin.mul_assoc]
  · trivial

/-- The concrete carrier conversion preserves exactly the operations consumed
by the materialized and production claimed-chain machines. -/
def projectionCarrierHomomorphism :
    OpsHomomorphism ClaimedChain.ops
      ConcreteCarrier.extensionOps.toOps toConcreteK where
  zero := toConcreteK_zero
  one := toConcreteK_one
  add := toConcreteK_add
  mul := toConcreteK_mul

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Exact field-level identification of one materialized assignment with the
production NC message verifier. `RoundMaps.values` contains the fixed 25-round
production profile; this contract compares its proof-free assignment reads,
not stage labels, row counts, or digests. -/
structure ExactDataflow
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (assignment : Nat → Nat) : Prop where
  claimedInitial :
    toConcreteK (ClaimedChain.initial RoundMaps.values assignment) =
      Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.rawInitial
        context
  roundMessages :
    (ClaimedChain.certificate RoundMaps.values assignment).rounds.map
        (mapFixedPolynomial toConcreteK) =
      certificate.piCcs.nc.toSumCheck.rounds
  challenges :
    (ClaimedChain.challenges RoundMaps.values assignment).map toConcreteK =
      (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.ncPoint
        context certificate).coordinates
  finalClaim :
    toConcreteK (ClaimedChain.terminal RoundMaps.values assignment) =
      Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.messageTerminal
        context certificate

/-- The source-row consequences plus exact assignment-to-message dataflow
give the production claims-level NC acceptance relation. This theorem performs
only substitution into the already proved claimed chain: it assumes neither
acceptance, source-row satisfaction, nor any projection statement. -/
theorem consequences_imply_ncMessageAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (assignment : Nat → Nat)
    (consequences : SourceRowsSoundness.Consequences assignment)
    (dataflow : ExactDataflow context certificate assignment) :
    Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.NcMessageAccepted
      context certificate := by
  have chain := chain_transport ClaimedChain.ops
    ConcreteCarrier.extensionOps.toOps toConcreteK
    projectionCarrierHomomorphism consequences.roundChain
  rw [dataflow.claimedInitial, dataflow.roundMessages, dataflow.challenges,
    dataflow.finalClaim] at chain
  simpa [Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.NcMessageAccepted,
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane.Accepted,
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Accepted,
    Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.ncPoint_eq_transcriptPoint]
    using chain

/-- Literal satisfaction of the selected production rows, together with the
two still-explicit pin equations and exact assignment/message dataflow,
implies the production claims-level NC check. The source-row obligations are
derived by the artifact-checked selective refinement rather than assumed. -/
theorem generatedEmittedRowsSatisfy_implies_ncMessageAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (assignment : Nat → Nat)
    (selectedRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1)
    (dataflow : ExactDataflow context certificate
      (PhysicalAgreement.reconstructedAssignment assignment)) :
    Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs.NcMessageAccepted
      context certificate := by
  exact consequences_imply_ncMessageAccepted context certificate
    (PhysicalAgreement.reconstructedAssignment assignment)
    (SelectedRowsSoundness.generatedEmittedRowsSatisfy_implies_consequences
      selectedRows selectorOne constantOne)
    dataflow

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance
