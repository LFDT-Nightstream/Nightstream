import Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine
import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteAlgebra
import Nightstream.SuperNeo.Folding.Nifs.PaperProfile
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Types

/-!
Contract: exact Poseidon2 transcript and bounded `Pi_RLC` sampler for the
selected `PaddedRowIdentity` protocol.

Owns:
- the versioned public-NIFS-input field order;
- the existing one-joint `Pi_CCS` tags and field order;
- the exact post-SumCheck output absorption;
- the selected width-8 Poseidon2 constants;
- the four-digest, 54-of-64 `Pi_RLC` sampler; and
- a total strong-set response with an explicit bounded-shortfall event.

Does not own: collision or random-oracle probability bounds, the external
Phi81 low-norm invertibility theorem, Ajtai/Module-SIS security, Rust, R1CS
rows, or byte encoding.

Assurance tier: model-level. The one-joint round and challenge schedule reuses
the existing canonical Poseidon2 semantics. A sampler shortfall is not hidden:
the total verifier response uses the centered-zero scalar on that branch, and
`SamplerShortfall` names the exact event that a security proof must bound.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra

abbrev State := Poseidon2Duplex.State
abbrev PaperShape := Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape
abbrev SelectedCommitment :=
  PaddedRowIdentityConcreteAlgebra.Commitment
abbrev SelectedPublicInput :=
  PaddedRowIdentityConcreteAlgebra.PublicInput
abbrev SelectedEvaluation :=
  PaddedRowIdentityConcreteAlgebra.Evaluation

/-- Selected, Lean-owned width-8 Poseidon2 constants. -/
def constants : Poseidon2Schedule.Constants :=
  Poseidon2CanonicalConstants.selected

/-- Empty state before the statement identifier starts one NIFS transcript. -/
def initialState : State := Poseidon2Duplex.empty

/-! ## Canonical field serialization -/

/-- Numeric words are reduced exactly as the existing one-joint schedule. -/
def word (value : Nat) : Nat := value % goldilocksModulus

/-- Construction 3 domain tag for the full-statement identifier. -/
def statementIdentifierTag : Nat := 39

/-- The fixed-length identifier is absorbed before all public NIFS inputs. -/
def statementIdentifierFields (statementId : F) : List Nat :=
  [word statementIdentifierTag, statementId.val]

/-- Transcript state after the verifier binds the full public statement. -/
def initialStateForStatement (statementId : F) : State :=
  Poseidon2Duplex.absorbList constants
    (statementIdentifierFields statementId) initialState

/-- A base-field element has one canonical Goldilocks coordinate. -/
def fFields (value : F) : List Nat := [value.val]

/-- A quadratic-extension element is low limb followed by high limb. -/
def kFields (value : K) : List Nat := [value.c0.val, value.c1.val]

/-- Encode a finite function in increasing `Fin` order. -/
def finFields
    {count : Nat} {Value : Type}
    (encode : Value -> List Nat) (values : Fin count -> Value) : List Nat :=
  (canonicalFinIndices count).flatMap fun index => encode (values index)

/-- Ring coefficients use increasing polynomial degree. -/
def ringFFields (value : RingF) : List Nat :=
  finFields fFields value

/-- The selected paper shape, in the same order as `KPiCcsTranscript`. -/
def shapeFields (value : PaperShape) : List Nat :=
  [word value.cubeVariables, word value.freshCount,
    word value.runningCount, word value.matrixCount,
    word value.coefficientCount]

def monomialFields
    {valueShape : PaperShape}
    (monomial : CCSResidualTable.Monomial K valueShape.matrixCount) : List Nat :=
  kFields monomial.coefficient ++
    (canonicalFinIndices valueShape.matrixCount).map fun index =>
      word (monomial.exponents index)

def polynomialFields
    {valueShape : PaperShape}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial K valueShape.matrixCount) : List Nat :=
  word polynomial.degreeBound :: word polynomial.terms.length ::
    polynomial.terms.flatMap monomialFields

def pointFields
    {variables : Nat} (point : CubePoint K variables) : List Nat :=
  point.coordinates.flatMap kFields

def commitmentFields (commitment : SelectedCommitment) : List Nat :=
  finFields ringFFields commitment

def publicInputFields (input : SelectedPublicInput) : List Nat :=
  finFields fFields input

def evaluationFields (evaluation : SelectedEvaluation) : List Nat :=
  finFields (fun coefficients => finFields kFields coefficients) evaluation

/-- Selected protocol identifier. It is distinct from every one-joint phase
tag and fixes the padded-row profile before public claims are absorbed. -/
def publicInputTag : Nat := 40

/-- Wire/profile version for the selected padded-row NIFS. -/
def protocolVersion : Nat := 1

/-- Static selected profile fields. They make the relation dimensions part of
the public transcript prefix instead of relying on an unnamed build profile. -/
def profileFields : List Nat :=
  shapeFields shape ++
    [word assignmentColumns,
      word (Phi81ColumnLayout.blockCount assignmentColumns),
      word verifierRows,
      word relationShape.publicWidth,
      word 9]

def runningFields
    (running : Running K SelectedCommitment SelectedPublicInput shape) : List Nat :=
  pointFields running.point ++
    finFields commitmentFields running.commitments ++
    finFields publicInputFields running.publicInputs ++
    finFields evaluationFields running.evaluations

def freshFields
    (fresh : Fresh SelectedCommitment SelectedPublicInput shape) : List Nat :=
  finFields commitmentFields fresh.commitments ++
    finFields publicInputFields fresh.publicInputs

/-- Complete public NIFS input, before any `Pi_CCS` challenge. -/
def publicNifsFields
    (running : Running K SelectedCommitment SelectedPublicInput shape)
    (fresh : Fresh SelectedCommitment SelectedPublicInput shape) : List Nat :=
  [word publicInputTag, word protocolVersion] ++ profileFields ++
    runningFields running ++ freshFields fresh

/-- Verifier-owned public-input absorption. -/
def absorbPublicInput
    (state : State)
    (running : Running K SelectedCommitment SelectedPublicInput shape)
    (fresh : Fresh SelectedCommitment SelectedPublicInput shape) : State :=
  Poseidon2Duplex.absorbList constants
    (publicNifsFields running fresh) state

/-! ## Exact one-joint PiCCS schedule -/

/-- Complete verifier statement. Tag `41` and all following fields are the
value-level form of the existing `KPiCcsTranscript.statementFields`. -/
def statementFields
    (statement : ProtocolVerifier.Statement K State shape) : List Nat :=
  [word 41] ++ shapeFields shape ++
    polynomialFields statement.input.constraintPolynomial ++
    [word shape.cubeVariables] ++ pointFields statement.input.priorPoint ++
    [word shape.carriedEvaluationCount] ++
    (canonicalCarriedCoordinates shape).flatMap fun coordinate =>
      kFields (statement.input.claimedCoefficient coordinate)

/-- One SumCheck round. Tag `45`, round index, coefficient count, then
constant-first coefficients. -/
def roundFields
    (round : Fin shape.cubeVariables) (message : Message K) : List Nat :=
  [word 45, word round.val, word message.coefficients.length] ++
    message.coefficients.flatMap kFields

/-- Scalar projection used only by the algebraic terminal checker. This is
not the authority-bearing NIFS handoff because it omits fresh nonconstant
ring coefficients. -/
def projectedOutputFields
    (message : ProtocolPolynomial.OutputMessage K shape) : List Nat :=
  [word 47] ++
    finFields (fun matrices => finFields kFields matrices)
      message.freshMatrixImage ++
    finFields kFields message.sourceAssignment ++
    (canonicalCarriedCoordinates shape).flatMap fun coordinate =>
      kFields (message.carriedImage coordinate)

/-- Complete paper output in source-major, matrix-major, coefficient-major
order. Tag `47` is followed by both extension-field limbs of every `y'`
coordinate sent in Step 3 of Section 7.3. -/
def outputFields
    (message : FullOutputCoordinates.FullOutput K shape) : List Nat :=
  [word 47] ++
    finFields
      (fun matrices => finFields
        (fun coefficients => finFields kFields coefficients) matrices)
      message.coordinate

/-- Exact post-SumCheck handoff used before `Pi_RLC` challenge sampling. -/
def absorbFullOutput
    (state : State)
    (message : FullOutputCoordinates.FullOutput K shape) : State :=
  Poseidon2Duplex.absorbList constants (outputFields message) state

private theorem coefficientOutputFields_length
    (values : Fin shape.coefficientCount -> K) :
    (finFields kFields values).length = shape.coefficientCount * 2 := by
  unfold finFields
  calc
    _ = (canonicalFinIndices shape.coefficientCount).length * 2 := by
      apply Poseidon2Program.length_flatMap_uniform
      intro coefficient
      rfl
    _ = shape.coefficientCount * 2 := by
      rw [canonicalFinIndices_length]

private theorem matrixOutputFields_length
    (values : Fin shape.matrixCount -> Fin shape.coefficientCount -> K) :
    (finFields (fun coefficients => finFields kFields coefficients)
      values).length =
      shape.matrixCount * (shape.coefficientCount * 2) := by
  unfold finFields
  calc
    _ = (canonicalFinIndices shape.matrixCount).length *
        (shape.coefficientCount * 2) := by
      apply Poseidon2Program.length_flatMap_uniform
      intro matrix
      exact coefficientOutputFields_length (values matrix)
    _ = shape.matrixCount * (shape.coefficientCount * 2) := by
      rw [canonicalFinIndices_length]

private theorem sourceOutputFields_length
    (values : Fin shape.sourceCount -> Fin shape.matrixCount ->
      Fin shape.coefficientCount -> K) :
    (finFields
      (fun matrices => finFields
        (fun coefficients => finFields kFields coefficients) matrices)
      values).length =
      shape.sourceCount *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
  unfold finFields
  calc
    _ = (canonicalFinIndices shape.sourceCount).length *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
      apply Poseidon2Program.length_flatMap_uniform
      intro source
      exact matrixOutputFields_length (values source)
    _ = shape.sourceCount *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
      rw [canonicalFinIndices_length]

/-- The selected complete output contains one tag and all 15x14x54
quadratic-extension coordinates. -/
@[simp] theorem outputFields_length
    (message : FullOutputCoordinates.FullOutput K shape) :
    (outputFields message).length = 22681 := by
  unfold outputFields
  rw [List.length_append]
  change 1 + _ = 22681
  rw [sourceOutputFields_length message.coordinate]
  rfl

/-- Interpret the first two freshly permuted lanes as the selected concrete
quadratic extension. -/
def challengeValue (state : State) : K where
  c0 := ⟨state.lanes ⟨0, by decide⟩ % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩
  c1 := ⟨state.lanes ⟨1, by decide⟩ % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

def squeezeK (state : State) : K × State :=
  let next := Poseidon2Duplex.gate constants state
  (challengeValue next, next)

def squeezeLabel (label : Nat) (state : State) : K × State :=
  squeezeK (Poseidon2Duplex.absorbElem constants (word label) state)

def squeezeAt (label index : Nat) (state : State) : K × State :=
  squeezeK (Poseidon2Duplex.absorbList constants
    [word label, word index] state)

/-- Exact typed paper transcript, with no caller-provided challenges. -/
def transcript :
    FiatShamir.Oracle (ProtocolVerifier.Statement K State shape) K State shape where
  initialState statement :=
    Poseidon2Duplex.absorbList constants (statementFields statement)
      statement.priorState
  absorbRound state round message :=
    Poseidon2Duplex.absorbList constants (roundFields round message) state
  squeeze state label :=
    match label with
    | .alpha coordinate => squeezeAt 42 coordinate.val state
    | .gamma => squeezeLabel 43 state
    | .sumcheck round => squeezeAt 46 round.val state

/-- Algebraic `Pi_CCS` oracle. Its output operation is retained for the
generic projected terminal checker. The selected NIFS uses
`absorbFullOutput` for its authority-bearing handoff. -/
def oracle : ProtocolVerifier.Oracle K State shape where
  transcript := transcript
  absorbOutput state message :=
    Poseidon2Duplex.absorbList constants (projectedOutputFields message) state

@[simp] theorem transcript_absorbRound_eq_canonical
    (state : State) (round : Fin shape.cubeVariables) (message : Message K) :
    transcript.absorbRound state round message =
      Poseidon2Duplex.absorbList constants
        ([word 45, word round.val, word message.coefficients.length] ++
          message.coefficients.flatMap kFields) state := by
  rfl

@[simp] theorem transcript_alpha_eq_canonical
    (state : State) (coordinate : Fin shape.cubeVariables) :
    transcript.squeeze state (.alpha coordinate) =
      squeezeAt 42 coordinate.val state := by
  rfl

@[simp] theorem transcript_sumcheck_eq_canonical
    (state : State) (round : Fin shape.cubeVariables) :
    transcript.squeeze state (.sumcheck round) =
      squeezeAt 46 round.val state := by
  rfl

/-! ## Total bounded PiRLC sampling -/

abbrev SamplerSpecification :=
  Specification State Chunk Coefficient Scalar

def samplerSpecification : SamplerSpecification :=
  PiRlcCanonicalMachine.specification constants

def SamplerAvailable (state : State) : Prop :=
  Available samplerSpecification PaperProfile.arity.total candidateBound state

noncomputable def selectedBatch
    (state : State) (available : SamplerAvailable state) :
    BatchExecution samplerSpecification PaperProfile.arity.total
      candidateBound state :=
  Classical.choose available

/-- Centered zero is symbol `2`, since the semantic value is `symbol - 2`. -/
def zeroCoefficient : Coefficient := ⟨2, by decide⟩

def zeroScalar : Scalar := fun _ => zeroCoefficient

/-- Total coefficient-vector response. A successful bounded batch is used
exactly. The centered-zero fallback is valid but its use is a security event. -/
noncomputable def scalarResponse
    (state : State) (coordinate : Fin PaperProfile.arity.total) : Scalar := by
  classical
  exact if available : SamplerAvailable state then
      challenge (selectedBatch state available) coordinate
    else
      zeroScalar

/-- Ring-valued response used by the selected `Pi_RLC` verifier. -/
noncomputable def piRlcResponse
    (state : State) (coordinate : Fin PaperProfile.arity.total) : RingF :=
  Phi81StrongSet.embedScalar (scalarResponse state coordinate)

theorem piRlcResponse_valid (state : State)
    (coordinate : Fin PaperProfile.arity.total) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.challengeValid
      (piRlcResponse state coordinate) := by
  exact
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.embedScalar_valid _

/-- Exact failure event for the fixed 64-candidate budget. -/
def SamplerShortfall (state : State) : Prop :=
  Exists fun coordinate : Fin PaperProfile.arity.total =>
    ShortfallAt samplerSpecification candidateBound state coordinate.val

theorem available_or_shortfall (state : State) :
    SamplerAvailable state \/ SamplerShortfall state :=
  available_or_exists_shortfall samplerSpecification
    PaperProfile.arity.total candidateBound state

theorem available_excludes_shortfall
    {state : State} (available : SamplerAvailable state) :
    ¬ SamplerShortfall state := by
  rintro ⟨coordinate, shortfall⟩
  exact
    (Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.available_excludes_shortfall
      available coordinate)
    shortfall

theorem not_available_iff_shortfall (state : State) :
    ¬ SamplerAvailable state ↔ SamplerShortfall state := by
  constructor
  · intro unavailable
    rcases available_or_shortfall state with available | shortfall
    · exact False.elim (unavailable available)
    · exact shortfall
  · intro shortfall available
    exact available_excludes_shortfall available shortfall

/-- Outside the named shortfall event, the total response is exactly the
canonical transcript-chained bounded sampler. -/
theorem piRlcResponse_refines_of_available
    {state : State} (available : SamplerAvailable state) :
    ResponseRefinesAt scalarResponse samplerSpecification candidateBound state := by
  classical
  refine ⟨selectedBatch state available, ?_⟩
  intro coordinate
  unfold scalarResponse
  rw [dif_pos available]

theorem piRlcResponse_refines_of_no_shortfall
    {state : State} (noShortfall : ¬ SamplerShortfall state) :
    ResponseRefinesAt scalarResponse samplerSpecification candidateBound state := by
  rcases available_or_shortfall state with available | shortfall
  · exact piRlcResponse_refines_of_available available
  · exact False.elim (noShortfall shortfall)

/-- Concrete transcript-security event added by the bounded sampler. The four
paper transcript collision classes remain those in
`PaperNonInteractive.TranscriptSecurityEvent`. -/
inductive Poseidon2SecurityEvent (state : State) where
  | boundedSamplerShortfall (failure : SamplerShortfall state)

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2
