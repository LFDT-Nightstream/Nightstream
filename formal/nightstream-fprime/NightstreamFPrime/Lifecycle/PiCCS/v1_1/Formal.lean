import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementBinding
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding

/-!
Paper authority: SuperNeo v1.1, Section 7.3, complete `Pi_CCS` reduction.
Obligation: Assemble every proved PiCCS leaf into the only production PiCCS
logical circuit, with definitionally shared values and exact transcript order.

Child order:
1. statement binding and absorption;
2. verifier-owned `alpha` and `gamma` derivation;
3. indexed round transcript and fixed SumCheck chain;
4. separate `Eval_K`, `Eval_A`, CCS, and norm terminal values;
5. exact v1.1 final identity;
6. reduced-output and outgoing-state binding.

The parent owns only wiring and child order. It does not unfold a child's
operations. In particular, `Eval_K` is never represented as matrix zero.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The one shared symbolic carrier for all PiCCS children. -/
structure Interface (logicalWidth degreeBound : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  /-- Proof-only phase base. `atOffset` overwrites every caller value. -/
  baseOffset : Nat := 0
  running : Nat → StatementAbsorption.RunningExpr logicalWidth publicFits
  fresh : Nat → StatementAbsorption.FreshExpr logicalWidth publicFits
  round : Nat → Fin productionShape.cubeVariables →
    RoundTranscript.Message degreeBound
  output : Nat → OutputBinding.OutputExpr

/-- Freeze every shared value at the parent entry offset. Child allocation
offsets affect only private witness ranges, never protocol wiring. -/
def atOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) : Interface logicalWidth degreeBound publicFits where
  baseOffset := parentOffset
  running := fun _ => interface.running parentOffset
  fresh := fun _ => interface.fresh parentOffset
  round := fun _ => interface.round parentOffset
  output := fun _ => interface.output parentOffset

/-- Exactly the caller-owned PiCCS expressions. These values must precede
the phase allocation. Derived transcript and arithmetic outputs are excluded. -/
structure ExternalInputsBelow
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Prop where
  runningPoint : ∀ coordinate,
    ((interface.running offset).point coordinate).VarsBelow offset
  runningCommitment : ∀ source row coefficient,
    ((interface.running offset).commitment source row coefficient).VarsBelow
      offset
  runningPublicInput : ∀ source column,
    ((interface.running offset).publicInput source column).VarsBelow offset
  runningEval_K : ∀ source coefficient,
    ((interface.running offset).evaluation source).eval_K coefficient
      |>.VarsBelow offset
  runningEval_A : ∀ source matrix coefficient,
    ((interface.running offset).evaluation source).eval_A matrix coefficient
      |>.VarsBelow offset
  freshCommitment : ∀ source row coefficient,
    ((interface.fresh offset).commitment source row coefficient).VarsBelow
      offset
  freshPublicInput : ∀ source column,
    ((interface.fresh offset).publicInput source column).VarsBelow offset
  roundCoefficient : ∀ roundIndex coefficient,
    ((interface.round offset roundIndex).coefficient coefficient).VarsBelow
      offset
  outputEval_K : ∀ source coefficient,
    ((interface.output offset).padCoordinate source coefficient).VarsBelow
      offset
  outputEval_A : ∀ source matrix coefficient,
    ((interface.output offset).matrixCoordinate source matrix coefficient
      ).VarsBelow offset

def evalRunning {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env) :=
  StatementAbsorption.evalRunning (interface.running offset) env

def evalFresh {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env) :=
  StatementAbsorption.evalFresh (interface.fresh offset) env

def evalOutput {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env) :
    FullOutputCoordinates.FullOutput K productionShape where
  padCoordinate := fun source coefficient =>
    ((interface.output offset).padCoordinate source coefficient).eval env
  matrixCoordinate := fun source matrix coefficient =>
    ((interface.output offset).matrixCoordinate source matrix coefficient).eval
      env

/-- Replace only the PiCCS-owned part of the one NIFS proof. Later PiDEC
message fields remain opaque to this phase. -/
def evalProof
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation)) :
    Proof (ProductionKey.degreeBound relation) where
  piCcsRounds := fun roundIndex =>
    (interface.round offset roundIndex).semanticPolynomial env
  piCcsOutput := evalOutput interface offset env
  piDecCommitments := template.piDecCommitments
  piDecEvaluations := template.piDecEvaluations

def statementBindingInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
    StatementBinding.Interface where
  priorPoint := fun offset => (interface.running offset).point
  eval_K := fun offset coordinate =>
    ((interface.running offset).evaluation coordinate.running).eval_K
      coordinate.coefficient
  eval_A := fun offset coordinate =>
    ((interface.running offset).evaluation coordinate.running).eval_A
      coordinate.matrix coordinate.coefficient

def statementAbsorptionInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
    StatementAbsorption.Interface logicalWidth publicFits where
  running := interface.running
  fresh := interface.fresh

/-- The first transcript child owns this state. Statement binding has zero
local length, so statement absorption starts at the parent offset. -/
def statementFinalState {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) : Layer.EState :=
  StatementAbsorption.finalState
    (statementAbsorptionInterface (atOffset interface parentOffset))
    parentOffset

def challengeInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) :
    ChallengeDerivation.Interface where
  initialState := fun _ => statementFinalState interface parentOffset

/-- Fixed start of the owned challenge child in a frozen phase view. -/
def challengeStart {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Nat :=
  interface.baseOffset + 10298432

def challengeAlpha {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : Fin productionShape.cubeVariables → KExpr :=
  ChallengeDerivation.alpha
    (challengeInterface interface interface.baseOffset)
      (challengeStart interface)

def challengeGamma {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  ChallengeDerivation.gamma
    (challengeInterface interface interface.baseOffset)
      (challengeStart interface)

def challengeFinalState {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : Layer.EState :=
  ChallengeDerivation.finalState
    (challengeInterface interface interface.baseOffset)
      (challengeStart interface)

def roundTranscriptInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
  RoundTranscript.Interface degreeBound where
  initialState := challengeFinalState interface
  round := interface.round

def roundTranscriptStart {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Nat :=
  interface.baseOffset + 10298432 + 44400

def roundTranscriptRound {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) (roundIndex : Fin productionShape.cubeVariables) :
    FixedChain.Round degreeBound :=
  RoundTranscript.round (roundTranscriptInterface interface)
    (roundTranscriptStart interface) roundIndex

def roundTranscriptFinalState {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : Layer.EState :=
  RoundTranscript.finalState (roundTranscriptInterface interface)
    (roundTranscriptStart interface)

/-- `r′` is derived by the round transcript and is never a separate witness
field. -/
def roundPoint {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) (coordinate : Fin productionShape.cubeVariables) : KExpr :=
  RoundTranscript.challenge (roundTranscriptInterface interface)
    (roundTranscriptStart interface) coordinate

def initialClaimInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
    InitialClaim.Interface where
  gamma := challengeGamma interface
  eval_K := fun offset coordinate =>
    ((interface.running offset).evaluation coordinate.running).eval_K
      coordinate.coefficient
  eval_A := fun offset coordinate =>
    ((interface.running offset).evaluation coordinate.running).eval_A
      coordinate.matrix coordinate.coefficient

/-- First private variable of the child-owned initial-claim Horner program. -/
def initialClaimStart {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Nat :=
  roundTranscriptStart interface +
    (RoundTranscript.program (roundTranscriptInterface interface)
      (roundTranscriptStart interface)).recipes.length

/-- The initial SumCheck claim is an output of the preceding Horner child. -/
def initialClaimOutput {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  InitialClaim.output (initialClaimInterface interface)
    (initialClaimStart interface)

def sumcheckInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
  SumcheckChain.Interface degreeBound where
  initial := initialClaimOutput interface
  round := roundTranscriptRound interface

/-- First row position of the zero-private-variable SumCheck child. -/
def sumcheckStart {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Nat :=
  initialClaimStart interface + InitialClaim.privateCount

/-- The chain-owned final `p_i(r_i)` expression. -/
def sumcheckOutput {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  SumcheckChain.output (sumcheckInterface interface)
    (sumcheckStart interface)

def evalKInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
    EvalKTerminal.Interface where
  roundPoint := roundPoint interface
  priorPoint := fun offset => (interface.running offset).point
  gamma := challengeGamma interface
  outputEval_K := fun offset coordinate =>
    (interface.output offset).padCoordinate
      (runningSourceIndex coordinate.running) coordinate.coefficient

/-- Eval_K starts where the zero-private-variable SumCheck child ends. -/
def evalKStart {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Nat :=
  sumcheckStart interface

/-- Child-owned unshifted `E_K`. -/
def evalKOutput {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  EvalKTerminal.output (evalKInterface interface) (evalKStart interface)

def evalAInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
    EvalATerminal.Interface where
  roundPoint := roundPoint interface
  priorPoint := fun offset => (interface.running offset).point
  gamma := challengeGamma interface
  outputEval_A := fun offset coordinate =>
    (interface.output offset).matrixCoordinate
      (runningSourceIndex coordinate.running) coordinate.matrix
        coordinate.coefficient

def evalAStart {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Nat :=
  evalKStart interface + EvalKTerminal.privateCount

def evalAOutput {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  EvalATerminal.output (evalAInterface interface) (evalAStart interface)

def constantCoefficient
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Fin productionShape.coefficientCount :=
  (PaperAlgebra.matrixSource relation.system).kernel.constant

def freshIndex : Fin productionShape.freshCount :=
  ⟨0, by
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape]⟩

def ccsInterface
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    CcsTerminal.Interface where
  freshMatrix := fun offset matrix =>
    (interface.output offset).matrixCoordinate
      (freshSourceIndex freshIndex) matrix
        (constantCoefficient relation)

def ccsStart
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Nat :=
  evalAStart interface + EvalATerminal.privateCount

def ccsOutput
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  CcsTerminal.output relation (ccsInterface relation interface)
    (ccsStart interface)

def normInterface
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    NormTerminal.Interface where
  gamma := challengeGamma interface
  sourceAssignment := fun offset source =>
    (interface.output offset).padCoordinate source
      (constantCoefficient relation)

/-- The norm leaf starts after the zero-private-variable CCS leaf. -/
def normStart
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Nat :=
  ccsStart interface

/-- Child-owned strict-base-2 norm term. -/
def normOutput
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  NormTerminal.output (normInterface relation interface)
    (normStart interface)

def finalIdentityInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    FinalIdentity.Interface where
  roundPoint := roundPoint interface
  alpha := challengeAlpha interface
  gamma := challengeGamma interface
  eval_K := evalKOutput interface
  eval_A := evalAOutput interface
  ccs := ccsOutput relation interface
  norm := normOutput relation interface
  terminal := sumcheckOutput interface

/-- The norm and final-identity children share the same verifier-derived
gamma expression at every child offset. -/
theorem normGamma_eq_finalIdentityGamma
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (normAt finalAt : Nat) :
    (normInterface relation interface).gamma normAt =
      (finalIdentityInterface relation interface).gamma finalAt := by
  rfl

/-- The final-identity norm input is exactly the norm child's owned output. -/
theorem finalIdentityNorm_eq_normOutput
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (normAt finalAt : Nat) (startEq : normStart interface = normAt) :
    (finalIdentityInterface relation interface).norm finalAt =
      NormTerminal.output (normInterface relation interface) normAt := by
  unfold finalIdentityInterface normOutput
  rw [startEq]

def outputBindingInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
  OutputBinding.Interface where
  roundPoint := roundPoint interface
  initialState := roundTranscriptFinalState interface
  output := interface.output

def statementBindingCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (StatementBinding.circuit (statementBindingInterface interface)) 0 0
    (StatementBinding.localLength_eq (statementBindingInterface interface))
    (StatementBinding.flatConstraints_length (statementBindingInterface interface))

def statementAbsorptionCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (StatementAbsorption.circuit (statementAbsorptionInterface interface))
    10298432 10298432
    (StatementAbsorption.localLength_eq (statementAbsorptionInterface interface))
    (StatementAbsorption.flatConstraints_length
      (statementAbsorptionInterface interface))

def challengeCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) : FormalCircuit :=
  let childInterface :=
    challengeInterface (atOffset interface parentOffset) parentOffset
  FormalCircuit.withConstantFootprint
    (ChallengeDerivation.circuit childInterface) 44400 44400
    (ChallengeDerivation.localLength_eq childInterface)
    (ChallengeDerivation.flatConstraints_length childInterface)

def roundTranscriptCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (RoundTranscript.circuit (roundTranscriptInterface interface))
    (24 * RoundTranscript.perRoundRecipeCount degreeBound)
    (24 * RoundTranscript.perRoundRecipeCount degreeBound)
    (RoundTranscript.localLength_eq (roundTranscriptInterface interface))
    (RoundTranscript.flatConstraints_length (roundTranscriptInterface interface))

def initialClaimCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (InitialClaim.circuit (initialClaimInterface interface)) 25918 25918
    (InitialClaim.localLength_eq (initialClaimInterface interface))
    (InitialClaim.flatConstraints_length (initialClaimInterface interface))

def sumcheckCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (SumcheckChain.circuit (sumcheckInterface interface)) 0 48
    (SumcheckChain.localLength_eq (sumcheckInterface interface))
    (SumcheckChain.flatConstraints_length (sumcheckInterface interface))

def evalKCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (EvalKTerminal.circuit (evalKInterface interface)) 1820 1820
    (EvalKTerminal.localLength_eq (evalKInterface interface))
    (EvalKTerminal.flatConstraints_length (evalKInterface interface))

def evalACircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (EvalATerminal.circuit (evalAInterface interface)) 24284 24284
    (EvalATerminal.localLength_eq (evalAInterface interface))
    (EvalATerminal.flatConstraints_length (evalAInterface interface))

def ccsCircuit
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (CcsTerminal.circuit relation (ccsInterface relation interface)) 0 0
    (CcsTerminal.localLength_eq relation (ccsInterface relation interface))
    (CcsTerminal.flatConstraints_length relation (ccsInterface relation interface))

def normCircuit
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (NormTerminal.circuit (normInterface relation interface)) 32 32
    (NormTerminal.localLength_eq (normInterface relation interface))
    (NormTerminal.flatConstraints_length (normInterface relation interface))

def finalIdentityCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (FinalIdentity.circuit (finalIdentityInterface relation interface)) 27742 27744
    (FinalIdentity.localLength_eq (finalIdentityInterface relation interface))
    (FinalIdentity.flatConstraints_length (finalIdentityInterface relation interface))

def outputBindingCircuit {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : FormalCircuit :=
  FormalCircuit.withConstantFootprint
    (OutputBinding.circuit (outputBindingInterface interface)) 4076512 4076512
    (OutputBinding.localLength_eq (outputBindingInterface interface))
    (OutputBinding.flatConstraints_length (outputBindingInterface interface))

def childLength (child : FormalCircuit) (offset : Nat) : Nat :=
  localLength (Circuit.ops child.main offset)

def nextOffset (child : FormalCircuit) (offset : Nat) : Nat :=
  offset + childLength child offset

theorem nextStart_eq (base previousLength completedLength start : Nat)
    (child : FormalCircuit)
    (startEq : base + previousLength = start)
    (lengthEq : completedLength = previousLength +
      localLength (Circuit.ops child.main (base + previousLength))) :
    base + completedLength = nextOffset child start := by
  unfold nextOffset childLength
  rw [lengthEq, ← Nat.add_assoc, startEq]

def statementBindingOffset (offset : Nat) : Nat := offset

def statementAbsorptionOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
  (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (statementBindingCircuit (atOffset interface offset)) offset

@[simp] theorem statementAbsorptionOffset_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : statementAbsorptionOffset interface offset = offset := by
  unfold statementAbsorptionOffset nextOffset childLength
    statementBindingCircuit
  rw [FormalCircuit.withConstantFootprint_main,
    StatementBinding.localLength_eq]
  omega

def challengeOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (statementAbsorptionCircuit (atOffset interface offset))
    (statementAbsorptionOffset interface offset)

@[simp] theorem challengeOffset_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : challengeOffset interface offset = offset + 10298432 := by
  unfold challengeOffset nextOffset childLength statementAbsorptionCircuit
  rw [statementAbsorptionOffset_eq, FormalCircuit.withConstantFootprint_main,
    StatementAbsorption.localLength_eq]

@[simp] theorem challengeStart_atOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    challengeStart (atOffset interface offset) =
      challengeOffset interface offset := by
  simp [challengeStart, atOffset]

def roundTranscriptOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (challengeCircuit interface offset)
    (challengeOffset interface offset)

@[simp] theorem roundTranscriptOffset_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    roundTranscriptOffset interface offset =
      challengeOffset interface offset + 44400 := by
  unfold roundTranscriptOffset nextOffset childLength challengeCircuit
  rw [FormalCircuit.withConstantFootprint_main,
    ChallengeDerivation.localLength_eq]

@[simp] theorem roundTranscriptStart_atOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    roundTranscriptStart (atOffset interface offset) =
      roundTranscriptOffset interface offset := by
  simp [roundTranscriptStart, atOffset]

def initialClaimOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (roundTranscriptCircuit (atOffset interface offset))
    (roundTranscriptOffset interface offset)

@[simp] theorem initialClaimStart_atOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    initialClaimStart (atOffset interface offset) =
      initialClaimOffset interface offset := by
  unfold initialClaimStart initialClaimOffset nextOffset childLength
    roundTranscriptCircuit
  rw [roundTranscriptStart_atOffset,
    RoundTranscript.program_recipes_length,
    FormalCircuit.withConstantFootprint_main,
    RoundTranscript.localLength_eq]

theorem roundTranscriptOffset_le_initialClaimOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    roundTranscriptOffset interface offset ≤
      initialClaimOffset interface offset := by
  unfold initialClaimOffset nextOffset
  omega

theorem challengeOffset_le_initialClaimOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    challengeOffset interface offset ≤ initialClaimOffset interface offset := by
  unfold initialClaimOffset roundTranscriptOffset nextOffset childLength
  omega

def sumcheckOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (initialClaimCircuit (atOffset interface offset))
    (initialClaimOffset interface offset)

@[simp] theorem sumcheckStart_atOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    sumcheckStart (atOffset interface offset) =
      sumcheckOffset interface offset := by
  unfold sumcheckStart sumcheckOffset nextOffset childLength
    initialClaimCircuit InitialClaim.privateCount
  rw [initialClaimStart_atOffset, FormalCircuit.withConstantFootprint_main,
    InitialClaim.localLength_eq]

def evalKOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (sumcheckCircuit (atOffset interface offset))
    (sumcheckOffset interface offset)

@[simp] theorem evalKStart_atOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    evalKStart (atOffset interface offset) = evalKOffset interface offset := by
  unfold evalKStart evalKOffset nextOffset childLength sumcheckCircuit
  rw [sumcheckStart_atOffset, FormalCircuit.withConstantFootprint_main,
    SumcheckChain.localLength_eq]
  omega

def evalAOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (evalKCircuit (atOffset interface offset))
    (evalKOffset interface offset)

@[simp] theorem evalAStart_atOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    evalAStart (atOffset interface offset) = evalAOffset interface offset := by
  unfold evalAStart evalAOffset nextOffset childLength evalKCircuit
    EvalKTerminal.privateCount
  rw [evalKStart_atOffset, FormalCircuit.withConstantFootprint_main,
    EvalKTerminal.localLength_eq]

def ccsOffset {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (evalACircuit (atOffset interface offset))
    (evalAOffset interface offset)

@[simp] theorem ccsStart_atOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    ccsStart (atOffset interface offset) = ccsOffset interface offset := by
  unfold ccsStart ccsOffset nextOffset childLength evalACircuit
    EvalATerminal.privateCount
  rw [evalAStart_atOffset, FormalCircuit.withConstantFootprint_main,
    EvalATerminal.localLength_eq]

def normOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (ccsCircuit relation (atOffset interface offset))
    (ccsOffset interface offset)

@[simp] theorem normStart_atOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    normStart (atOffset interface offset) =
      normOffset relation interface offset := by
  unfold normStart normOffset nextOffset childLength ccsCircuit
  rw [ccsStart_atOffset, FormalCircuit.withConstantFootprint_main,
    CcsTerminal.localLength_eq]
  omega

def finalIdentityOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (normCircuit relation (atOffset interface offset))
    (normOffset relation interface offset)

def outputBindingOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (finalIdentityCircuit relation (atOffset interface offset))
    (finalIdentityOffset relation interface offset)

def finalOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  nextOffset (outputBindingCircuit (atOffset interface offset))
    (outputBindingOffset relation interface offset)

/-- The output-binding child owns the post-PiCCS transcript state. -/
def outputBindingFinalState
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Layer.EState :=
  OutputBinding.finalState
    (outputBindingInterface (atOffset interface offset))
    (outputBindingOffset relation interface offset)

/-- Complete deterministic meaning of the PiCCS phase. The outgoing state
is computed by the final child and is not part of the caller interface. -/
structure PhaseHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation)) : Prop where
  accepted : NightstreamFPrime.Spec.Folding.PiCCS.v1_1.Accepted
    (ProductionKey.key relation ajtai)
    (evalRunning interface offset env)
    (evalFresh interface offset env)
    (evalProof relation interface offset env template)
  outgoingState : StatementAbsorption.evalState env
      (outputBindingFinalState relation interface offset) =
    ((ProductionKey.key relation ajtai).piCcsExecution
      (evalRunning interface offset env)
      (evalFresh interface offset env)
      (evalProof relation interface offset env template)).outgoingState

def childOp (name : String) (child : FormalCircuit) (offset : Nat) : Op :=
  .subcircuit (child.asSubcircuit name offset)

@[simp] theorem childOp_privateCount (name : String) (child : FormalCircuit)
    (offset : Nat) :
    (childOp name child offset).localLength = child.privateCount offset := by
  rfl

@[simp] theorem childOp_rowCount (name : String) (child : FormalCircuit)
    (offset : Nat) :
    (childOp name child offset).rowCount = child.rowCount offset := by
  rfl

def opsAt
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Op :=
  [childOp "piccs.v1_1.statement_binding"
      (statementBindingCircuit (atOffset interface offset)) offset,
    childOp "piccs.v1_1.statement_absorption"
      (statementAbsorptionCircuit (atOffset interface offset))
      (statementAbsorptionOffset interface offset),
    childOp "piccs.v1_1.challenge_derivation"
      (challengeCircuit interface offset)
      (challengeOffset interface offset),
    childOp "piccs.v1_1.round_transcript"
      (roundTranscriptCircuit (atOffset interface offset))
      (roundTranscriptOffset interface offset),
    childOp "piccs.v1_1.initial_claim"
      (initialClaimCircuit (atOffset interface offset))
      (initialClaimOffset interface offset),
    childOp "piccs.v1_1.sumcheck_chain"
      (sumcheckCircuit (atOffset interface offset))
      (sumcheckOffset interface offset),
    childOp "piccs.v1_1.eval_K_terminal"
      (evalKCircuit (atOffset interface offset))
      (evalKOffset interface offset),
    childOp "piccs.v1_1.eval_A_terminal"
      (evalACircuit (atOffset interface offset))
      (evalAOffset interface offset),
    childOp "piccs.v1_1.ccs_terminal"
      (ccsCircuit relation (atOffset interface offset))
      (ccsOffset interface offset),
    childOp "piccs.v1_1.norm_terminal"
      (normCircuit relation (atOffset interface offset))
      (normOffset relation interface offset),
    childOp "piccs.v1_1.final_identity"
      (finalIdentityCircuit relation (atOffset interface offset))
      (finalIdentityOffset relation interface offset),
    childOp "piccs.v1_1.output_binding"
      (outputBindingCircuit (atOffset interface offset))
      (outputBindingOffset relation interface offset)]

def main
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) : Circuit Unit :=
  fun offset => ((), finalOffset relation interface offset,
    opsAt relation interface offset)

@[simp] theorem main_ops
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    Circuit.ops (main relation interface) offset =
      opsAt relation interface offset := by
  rfl

structure Assumptions
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env) : Prop where
  external : ExternalInputsBelow interface offset
  statementBinding : (statementBindingCircuit
    (atOffset interface offset)).assumptions offset env
  statementAbsorption : (statementAbsorptionCircuit
    (atOffset interface offset)).assumptions
    (statementAbsorptionOffset interface offset) env
  challenge : (challengeCircuit interface offset).assumptions
    (challengeOffset interface offset) env
  roundTranscript : (roundTranscriptCircuit
    (atOffset interface offset)).assumptions
    (roundTranscriptOffset interface offset) env
  initialClaim : (initialClaimCircuit (atOffset interface offset)).assumptions
    (initialClaimOffset interface offset) env
  sumcheck : (sumcheckCircuit (atOffset interface offset)).assumptions
    (sumcheckOffset interface offset) env
  eval_K : (evalKCircuit (atOffset interface offset)).assumptions
    (evalKOffset interface offset) env
  eval_A : (evalACircuit (atOffset interface offset)).assumptions
    (evalAOffset interface offset) env
  ccs : (ccsCircuit relation (atOffset interface offset)).assumptions
    (ccsOffset interface offset) env
  norm : (normCircuit relation (atOffset interface offset)).assumptions
    (normOffset relation interface offset) env
  finalIdentity : (finalIdentityCircuit relation
    (atOffset interface offset)).assumptions
    (finalIdentityOffset relation interface offset) env
  outputBinding : (outputBindingCircuit
    (atOffset interface offset)).assumptions
    (outputBindingOffset relation interface offset) env

structure SpecHolds
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env) : Prop where
  statementBinding : (statementBindingCircuit
    (atOffset interface offset)).spec offset env
  statementAbsorption : (statementAbsorptionCircuit
    (atOffset interface offset)).spec
    (statementAbsorptionOffset interface offset) env
  challenge : (challengeCircuit interface offset).spec
    (challengeOffset interface offset) env
  roundTranscript : (roundTranscriptCircuit
    (atOffset interface offset)).spec
    (roundTranscriptOffset interface offset) env
  initialClaim : (initialClaimCircuit (atOffset interface offset)).spec
    (initialClaimOffset interface offset) env
  sumcheck : (sumcheckCircuit (atOffset interface offset)).spec
    (sumcheckOffset interface offset) env
  eval_K : (evalKCircuit (atOffset interface offset)).spec
    (evalKOffset interface offset) env
  eval_A : (evalACircuit (atOffset interface offset)).spec
    (evalAOffset interface offset) env
  ccs : (ccsCircuit relation (atOffset interface offset)).spec
    (ccsOffset interface offset) env
  norm : (normCircuit relation (atOffset interface offset)).spec
    (normOffset relation interface offset) env
  finalIdentity : (finalIdentityCircuit relation
    (atOffset interface offset)).spec
    (finalIdentityOffset relation interface offset) env
  outputBinding : (outputBindingCircuit
    (atOffset interface offset)).spec
    (outputBindingOffset relation interface offset) env

private theorem childSpec_of_rows (name : String) (child : FormalCircuit)
    (childOffset : Nat) (env : Env) (ops : List Op)
    (rows : holds env ops)
    (member : childOp name child childOffset ∈ ops)
    (assumptions : child.assumptions childOffset env) :
    child.spec childOffset env := by
  have callHolds := rows (childOp name child childOffset) member
  change child.assumptions childOffset env → child.spec childOffset env at callHolds
  exact callHolds assumptions

theorem soundness
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (rows : holds env (Circuit.ops (main relation interface) offset)) :
    SpecHolds relation interface offset env := by
  rw [main_ops] at rows
  refine {
    statementBinding := childSpec_of_rows "piccs.v1_1.statement_binding" _ _
      env _ rows (by simp [opsAt]) assumptions.statementBinding
    statementAbsorption := childSpec_of_rows
      "piccs.v1_1.statement_absorption" _ _ env _ rows
      (by simp [opsAt]) assumptions.statementAbsorption
    challenge := childSpec_of_rows "piccs.v1_1.challenge_derivation" _ _
      env _ rows (by simp [opsAt]) assumptions.challenge
    roundTranscript := childSpec_of_rows "piccs.v1_1.round_transcript" _ _
      env _ rows (by simp [opsAt]) assumptions.roundTranscript
    initialClaim := childSpec_of_rows "piccs.v1_1.initial_claim" _ _
      env _ rows (by simp [opsAt]) assumptions.initialClaim
    sumcheck := childSpec_of_rows "piccs.v1_1.sumcheck_chain" _ _
      env _ rows (by simp [opsAt]) assumptions.sumcheck
    eval_K := childSpec_of_rows "piccs.v1_1.eval_K_terminal" _ _
      env _ rows (by simp [opsAt]) assumptions.eval_K
    eval_A := childSpec_of_rows "piccs.v1_1.eval_A_terminal" _ _
      env _ rows (by simp [opsAt]) assumptions.eval_A
    ccs := childSpec_of_rows "piccs.v1_1.ccs_terminal" _ _
      env _ rows (by simp [opsAt]) assumptions.ccs
    norm := childSpec_of_rows "piccs.v1_1.norm_terminal" _ _
      env _ rows (by simp [opsAt]) assumptions.norm
    finalIdentity := childSpec_of_rows "piccs.v1_1.final_identity" _ _
      env _ rows (by simp [opsAt]) assumptions.finalIdentity
    outputBinding := childSpec_of_rows "piccs.v1_1.output_binding" _ _
      env _ rows (by simp [opsAt]) assumptions.outputBinding }

/-- Mechanical coverage of the exact production PiCCS relation. Every
shared equality is derived from the parent carrier; no transcript state or
challenge is supplied as a premise. -/
theorem spec_implies_phaseHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (specification : SpecHolds relation interface offset env) :
    PhaseHolds relation ajtai interface offset env template := by
  let shared := atOffset interface offset
  let running := evalRunning interface offset env
  let fresh := evalFresh interface offset env
  let proof := evalProof relation interface offset env template
  let context := ChallengeDerivation.productionContext
    relation ajtai running fresh
  have statementCoverage := StatementBinding.spec_implies_keyStatement
    relation ajtai running fresh (statementBindingInterface shared)
      (statementBindingOffset offset) env specification.statementBinding
  have statementState := StatementAbsorption.spec_implies_keyInitialState
    relation ajtai (statementAbsorptionInterface shared)
      (statementAbsorptionOffset interface offset) env
      specification.statementAbsorption
  dsimp only at statementState
  rw [ProductionKey.key_oracle_eq relation ajtai] at statementState
  have challengeCoverage :=
    ChallengeDerivation.spec_implies_derivePreSumcheck
      (challengeInterface shared offset) (challengeOffset interface offset) env
      context (by
        simpa [shared, running, fresh, context, challengeInterface,
          statementAbsorptionInterface, atOffset, evalRunning, evalFresh]
          using statementState) specification.challenge
  have keyChallenges :=
    ChallengeDerivation.spec_implies_keyExecution_challenges
      relation ajtai running fresh proof (challengeInterface shared offset)
      (challengeOffset interface offset) env (by
        simpa [shared, running, fresh, context, challengeInterface,
          statementAbsorptionInterface, atOffset, evalRunning, evalFresh]
          using statementState) specification.challenge
  have roundCoverage := RoundTranscript.spec_implies_keyExecution_rounds
    relation ajtai running fresh proof (roundTranscriptInterface shared)
      (roundTranscriptOffset interface offset) env (by
        simpa [shared, context, challengeInterface,
          roundTranscriptInterface, atOffset] using challengeCoverage.2.2)
      (by
        intro roundIndex
        rfl)
      specification.roundTranscript
  have initialEq := InitialClaim.spec_implies_keyInitial
    relation ajtai running fresh proof (initialClaimInterface shared)
      (initialClaimOffset interface offset) env (by
        simpa [shared, initialClaimInterface, challengeInterface, atOffset]
          using keyChallenges.2)
      (by
        intro coordinate
        rfl)
      (by
        intro coordinate
        rfl)
      specification.initialClaim
  have evalKEq := EvalKTerminal.spec_implies_keyPadAtMessage
    relation ajtai running fresh proof (evalKInterface shared)
      (evalKOffset interface offset) env (by
        simpa [shared, evalKInterface, roundTranscriptInterface, roundPoint,
          atOffset] using roundCoverage.1)
      (by rfl) (by
        simpa [shared, evalKInterface, challengeInterface, atOffset]
          using keyChallenges.2)
      (by
        intro coordinate
        rfl)
      specification.eval_K
  have evalAEq := EvalATerminal.spec_implies_keyMatrixAtMessage
    relation ajtai running fresh proof (evalAInterface shared)
      (evalAOffset interface offset) env (by
        simpa [shared, evalAInterface, roundTranscriptInterface, roundPoint,
          atOffset] using roundCoverage.1)
      (by rfl) (by
        simpa [shared, evalAInterface, challengeInterface, atOffset]
          using keyChallenges.2)
      (by
        intro coordinate
        rfl)
      specification.eval_A
  have ccsEq := CcsTerminal.spec_implies_keyCcsAtMessage
    relation ajtai running fresh proof (ccsInterface relation shared)
      (ccsOffset interface offset) env (by
        intro matrix
        rfl)
      specification.ccs
  have normEq := NormTerminal.spec_implies_keyNormAtMessage
    relation ajtai running fresh proof (normInterface relation shared)
      (normOffset relation interface offset) env (by
        simpa [shared, normInterface, challengeInterface, atOffset]
          using keyChallenges.2)
      (by
        intro source
        rfl)
      specification.norm
  have alphaInterfaceEq : PointEquality.Owned.evalRightPoint
      (FinalIdentity.pointInterfaceAt (finalIdentityInterface relation shared)
        (finalIdentityOffset relation interface offset))
      (finalIdentityOffset relation interface offset) env =
      ChallengeDerivation.evalAlpha (challengeInterface shared offset)
        (challengeOffset interface offset) env := by
    apply cubePoint_eq_of_coordinates
    simpa [shared, PointEquality.Owned.evalRightPoint,
      FinalIdentity.pointInterfaceAt, finalIdentityInterface,
      challengeAlpha, challengeInterface, atOffset] using
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.evalAlpha_coordinates
          (challengeInterface shared offset)
            (challengeOffset interface offset) env).symm
  have gammaInterfaceEq :
      ((finalIdentityInterface relation shared).gamma
        (finalIdentityOffset relation interface offset)).eval env =
      ChallengeDerivation.evalGamma (challengeInterface shared offset)
        (challengeOffset interface offset) env := by
    have startEq : challengeStart shared =
        challengeOffset interface offset := by
      simpa [shared] using challengeStart_atOffset interface offset
    rw [ChallengeDerivation.evalGamma_eq]
    change (ChallengeDerivation.gamma
      (challengeInterface shared shared.baseOffset)
        (challengeStart shared)).eval env =
      (ChallengeDerivation.gamma
        (challengeInterface shared offset)
          (challengeOffset interface offset)).eval env
    have baseEq : shared.baseOffset = offset := rfl
    rw [baseEq, startEq]
  have terminalEq := FinalIdentity.spec_implies_keyTerminal
    relation ajtai running fresh proof (finalIdentityInterface relation shared)
      (finalIdentityOffset relation interface offset) env (by
        simpa [shared, finalIdentityInterface, roundTranscriptInterface,
          roundPoint, atOffset] using roundCoverage.1)
      (by
        exact alphaInterfaceEq.trans keyChallenges.1)
      (by
        exact gammaInterfaceEq.trans keyChallenges.2)
      (by
        change (EvalKTerminal.output (evalKInterface shared)
          (evalKStart shared)).eval env = _
        have startEq : evalKStart shared = evalKOffset interface offset := by
          simpa [shared] using evalKStart_atOffset interface offset
        rw [startEq]
        exact evalKEq)
      (by
        change (EvalATerminal.output (evalAInterface shared)
          (evalAStart shared)).eval env = _
        have startEq : evalAStart shared = evalAOffset interface offset := by
          simpa [shared] using evalAStart_atOffset interface offset
        rw [startEq]
        exact evalAEq)
      (by
        change (CcsTerminal.output relation (ccsInterface relation shared)
          (ccsStart shared)).eval env = _
        have startEq : ccsStart shared = ccsOffset interface offset := by
          simpa [shared] using ccsStart_atOffset interface offset
        rw [startEq]
        exact ccsEq)
      (by
        change (NormTerminal.output (normInterface relation shared)
          (normStart shared)).eval env = _
        have startEq : normStart shared =
            normOffset relation interface offset := by
          simpa [shared] using normStart_atOffset relation interface offset
        rw [startEq]
        exact normEq)
      specification.finalIdentity
  have sumcheckRoundPointEq : SumcheckChain.evalRoundPoint
      (sumcheckInterface shared) (sumcheckOffset interface offset) env =
      RoundTranscript.evalRoundPoint (roundTranscriptInterface shared)
        (roundTranscriptOffset interface offset) env := by
    apply cubePoint_eq_of_coordinates
    change (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex =>
          ((roundTranscriptRound shared (sumcheckOffset interface offset)
            roundIndex).challenge).eval env) =
      (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex =>
          (RoundTranscript.challenge (roundTranscriptInterface shared)
            (roundTranscriptOffset interface offset) roundIndex).eval env)
    apply List.map_congr_left
    intro roundIndex _
    have startEq : roundTranscriptStart shared =
        roundTranscriptOffset interface offset := by
      simpa [shared] using roundTranscriptStart_atOffset interface offset
    change (RoundTranscript.challenge (roundTranscriptInterface shared)
      (roundTranscriptStart shared) roundIndex).eval env =
        (RoundTranscript.challenge (roundTranscriptInterface shared)
          (roundTranscriptOffset interface offset) roundIndex).eval env
    rw [startEq]
  have chain := SumcheckChain.spec_implies_keyChain
    relation ajtai running fresh proof (sumcheckInterface shared)
      (sumcheckOffset interface offset) env (by
        change (InitialClaim.output (initialClaimInterface shared)
          (initialClaimStart shared)).eval env = _
        have startEq : initialClaimStart shared =
            initialClaimOffset interface offset := by
          simpa [shared] using initialClaimStart_atOffset interface offset
        rw [startEq]
        exact initialEq)
      (by
        intro roundIndex
        rfl)
      (by
        exact sumcheckRoundPointEq.trans roundCoverage.1)
      (by
        change (SumcheckChain.output (sumcheckInterface shared)
          (sumcheckStart shared)).eval env = _ at terminalEq
        have startEq : sumcheckStart shared =
            sumcheckOffset interface offset := by
          simpa [shared] using sumcheckStart_atOffset interface offset
        rw [startEq] at terminalEq
        exact terminalEq)
      specification.sumcheck
  have coverage :
      NightstreamFPrime.Spec.Folding.PiCCS.v1_1.Coverage
        (ProductionKey.key relation ajtai) running fresh proof := {
    transcript := (ProductionKey.key relation ajtai
      ).piCcsExecution_coins_eq_derive running fresh proof
    input_eval_K := by
      intro coordinate
      exact congrFun statementCoverage.eval_K coordinate
    input_eval_A := by
      intro coordinate
      exact congrFun statementCoverage.eval_A coordinate
    output_eval_K := OutputBinding.key_output_eval_K
      relation ajtai running fresh proof
    output_eval_A := OutputBinding.key_output_eval_A
      relation ajtai running fresh proof
    chain := by
      simpa [ChallengeDerivation.productionContext] using chain
  }
  have outgoing := OutputBinding.spec_implies_keyOutgoingState
    relation ajtai running fresh proof (outputBindingInterface shared)
      (outputBindingOffset relation interface offset) env (by
        simpa [shared, outputBindingInterface, roundTranscriptInterface,
          atOffset] using roundCoverage.2)
      (by
        intro source coefficient
        rfl)
      (by
        intro source matrix coefficient
        rfl)
      specification.outputBinding
  refine {
    accepted := (NightstreamFPrime.Spec.Folding.PiCCS.v1_1.accepted_iff_coverage
      (ProductionKey.key relation ajtai) running fresh proof).mpr coverage
    outgoingState := ?_ }
  simpa [shared, running, fresh, proof, evalProof, evalOutput,
    outputBindingFinalState, outputBindingInterface, atOffset] using outgoing

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal
