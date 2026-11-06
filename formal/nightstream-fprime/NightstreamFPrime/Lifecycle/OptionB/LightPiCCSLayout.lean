import NightstreamFPrime.Lifecycle.OptionB.LightPiCCSPhase
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal

/-! The light Π_CCS layout (CW-2.5d, d2). The eleven leaf gadgets of `LightPiCCSPhase` run one
after another from a parent offset, and each reads its inputs from the external words or from
the outputs of earlier gadgets. `soundness` gives `LightPiCCSPhase.SpecHolds` from the rows, and
`wired` gives `LightPiCCSPhase.Wired` from the external words alone. The statement absorption is
the duplex gadget on the light fold domain; the evaluation terminals and the final identity are
today's gadgets at the light shape; the CCS terminal is today's gadget, which is the same for
every production relation; and the norm terminal reads the constant 0 for the running source
(`Light.PiCCS.maskMessage`). -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightPiCCSLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

/-- The external words of the light Π_CCS: the running claim's point and evaluation family, the
fresh instance, the SumCheck round messages and the Π_CCS output of both sources. -/
structure Interface (logicalWidth degreeBound : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  runningPoint : Nat → Fin LightShape.cubeVariables → KExpr
  runningEvaluation : Nat → StatementAbsorption.EvaluationExpr
  fresh : Nat → StatementAbsorption.FreshExpr logicalWidth publicFits
  round : Nat → Fin productionShape.cubeVariables → RoundTranscript.Message degreeBound
  output : Nat → OutputBinding.ShapeOutputExpr LightShape

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (interface : Interface logicalWidth degreeBound publicFits) (offset : Nat)

/-- A running input that the statement absorption never reads: `publicInputBlocks` reads only
the fresh instance. -/
def unusedRunning : StatementAbsorption.RunningExpr logicalWidth publicFits where
  point := fun _ => KExpr.zero
  commitment := fun _ _ _ => 0
  publicInput := fun _ _ => 0
  evaluation := fun _ => ⟨fun _ => KExpr.zero, fun _ _ => KExpr.zero⟩

def statementInterface : StatementAbsorption.Interface logicalWidth publicFits where
  running := fun _ => unusedRunning
  fresh := fun _ => interface.fresh offset

def statementDuplex : Duplex.Formal.Owned.Interface where
  initial := fun _ => Hash.zeroE
  actions := StatementAbsorption.domainActions lightFoldDomainChunk
    (statementInterface interface offset)

def statementCircuit : FormalCircuit :=
  Duplex.Formal.Owned.circuit (statementDuplex interface offset)

def statementFinal : Layer.EState :=
  Duplex.Formal.Owned.output (statementDuplex interface offset) offset

def challengeOffset : Nat :=
  Formal.nextOffset (statementCircuit interface offset) offset

def challengeInterface : ChallengeDerivation.Interface where
  initialState := fun _ => statementFinal interface offset

def challengeCircuit : FormalCircuit :=
  ChallengeDerivation.circuit (challengeInterface interface offset)

def gamma : KExpr :=
  ChallengeDerivation.gamma (challengeInterface interface offset)
    (challengeOffset interface offset)

def roundsOffset : Nat :=
  Formal.nextOffset (challengeCircuit interface offset) (challengeOffset interface offset)

def roundsInterface : RoundTranscript.Interface degreeBound where
  initialState := fun _ => ChallengeDerivation.finalState (challengeInterface interface offset)
    (challengeOffset interface offset)
  round := fun _ => interface.round offset

def roundsCircuit : FormalCircuit :=
  RoundTranscript.circuit (roundsInterface interface offset)

def roundPoint (coordinate : Fin productionShape.cubeVariables) : KExpr :=
  RoundTranscript.challenge (roundsInterface interface offset) (roundsOffset interface offset)
    coordinate

def initialOffset : Nat :=
  Formal.nextOffset (roundsCircuit interface offset) (roundsOffset interface offset)

def initialInterface : InitialClaim.ShapeInterface LightShape where
  gamma := fun _ => gamma interface offset
  eval_K := fun _ coordinate =>
    (interface.runningEvaluation offset).eval_K coordinate.coefficient
  eval_A := fun _ coordinate =>
    (interface.runningEvaluation offset).eval_A coordinate.matrix coordinate.coefficient

def initialCircuit : FormalCircuit :=
  InitialClaim.circuit (initialInterface interface offset)

def sumcheckOffset : Nat :=
  Formal.nextOffset (initialCircuit interface offset) (initialOffset interface offset)

def sumcheckInterface : SumcheckChain.Interface degreeBound where
  initial := fun _ => InitialClaim.output (initialInterface interface offset)
    (initialOffset interface offset)
  round := fun _ => RoundTranscript.round (roundsInterface interface offset)
    (roundsOffset interface offset)

def sumcheckCircuit : FormalCircuit :=
  SumcheckChain.circuit (sumcheckInterface interface offset)

def evalKOffset : Nat :=
  Formal.nextOffset (sumcheckCircuit interface offset) (sumcheckOffset interface offset)

def evalKInterface : EvalKTerminal.ShapeInterface LightShape where
  roundPoint := fun _ => roundPoint interface offset
  priorPoint := fun _ => interface.runningPoint offset
  gamma := fun _ => gamma interface offset
  outputEval_K := fun _ coordinate =>
    (interface.output offset).padCoordinate (runningSourceIndex coordinate.running)
      coordinate.coefficient

def evalKCircuit : FormalCircuit :=
  PointWeightedHorner.Owned.circuit (EvalKTerminal.coreInterface (evalKInterface interface offset))
    (by decide)

def evalAOffset : Nat :=
  Formal.nextOffset (evalKCircuit interface offset) (evalKOffset interface offset)

def evalAInterface : EvalATerminal.ShapeInterface LightShape where
  roundPoint := fun _ => roundPoint interface offset
  priorPoint := fun _ => interface.runningPoint offset
  gamma := fun _ => gamma interface offset
  outputEval_A := fun _ coordinate =>
    (interface.output offset).matrixCoordinate (runningSourceIndex coordinate.running)
      coordinate.matrix coordinate.coefficient

def evalACircuit : FormalCircuit :=
  PointWeightedHorner.Owned.circuit (EvalATerminal.coreInterface (evalAInterface interface offset))
    (by decide)

def ccsOffset : Nat :=
  Formal.nextOffset (evalACircuit interface offset) (evalAOffset interface offset)

def ccsInterface : CcsTerminal.Interface where
  freshMatrix := fun _ matrix =>
    (interface.output offset).matrixCoordinate (freshSourceIndex LightPiCCS.lightFreshIndex)
      matrix (Formal.constantCoefficient relation)

def ccsCircuit : FormalCircuit :=
  CcsTerminal.circuit relation (ccsInterface relation interface offset)

def normOffset : Nat :=
  Formal.nextOffset (ccsCircuit relation interface offset) (ccsOffset interface offset)

/-- The norm terminal reads the fresh source's assignment and the constant 0 for the running
source, as the light check masks it. -/
def normInterface : NormTerminal.ShapeInterface LightShape where
  gamma := fun _ => gamma interface offset
  sourceAssignment := fun _ source =>
    if source.val < LightShape.freshCount then
      (interface.output offset).padCoordinate source (Formal.constantCoefficient relation)
    else KExpr.zero

def normCircuit : FormalCircuit :=
  NormTerminal.circuit (normInterface relation interface offset)

def finalIdentityOffset : Nat :=
  Formal.nextOffset (normCircuit relation interface offset) (normOffset relation interface offset)

def finalIdentityInterface : FinalIdentity.ShapeInterface LightShape where
  roundPoint := fun _ => roundPoint interface offset
  alpha := fun _ => ChallengeDerivation.alpha (challengeInterface interface offset)
    (challengeOffset interface offset)
  gamma := fun _ => gamma interface offset
  eval_K := fun _ => EvalKTerminal.output (evalKInterface interface offset)
    (evalKOffset interface offset)
  eval_A := fun _ => EvalATerminal.output (evalAInterface interface offset)
    (evalAOffset interface offset)
  ccs := fun _ => CcsTerminal.output relation (ccsInterface relation interface offset)
    (ccsOffset interface offset)
  norm := fun _ => NormTerminal.output (normInterface relation interface offset)
    (normOffset relation interface offset)
  terminal := fun _ => SumcheckChain.output (sumcheckInterface interface offset)
    (sumcheckOffset interface offset)

def finalIdentityCircuit : FormalCircuit :=
  FinalIdentity.shapeCircuit (finalIdentityInterface relation interface offset) (by decide)

def outputOffset : Nat :=
  Formal.nextOffset (finalIdentityCircuit relation interface offset)
    (finalIdentityOffset relation interface offset)

def outputInterface : OutputBinding.ShapeInterface LightShape where
  roundPoint := fun _ => roundPoint interface offset
  initialState := fun _ => RoundTranscript.finalState (roundsInterface interface offset)
    (roundsOffset interface offset)
  output := fun _ => interface.output offset

def outputCircuit : FormalCircuit :=
  OutputBinding.circuit (outputInterface interface offset)

def finalOffset : Nat :=
  Formal.nextOffset (outputCircuit interface offset) (outputOffset relation interface offset)

/-- The leaves and offsets of the light Π_CCS phase that this layout places. -/
def leaves : LightPiCCSPhase.Leaves logicalWidth degreeBound publicFits where
  statement := statementInterface interface offset
  statementOffset := offset
  statementFinal := statementFinal interface offset
  challenge := challengeInterface interface offset
  challengeOffset := challengeOffset interface offset
  rounds := roundsInterface interface offset
  roundsOffset := roundsOffset interface offset
  initial := initialInterface interface offset
  initialOffset := initialOffset interface offset
  sumcheck := sumcheckInterface interface offset
  sumcheckOffset := sumcheckOffset interface offset
  evalK := evalKInterface interface offset
  evalKOffset := evalKOffset interface offset
  evalA := evalAInterface interface offset
  evalAOffset := evalAOffset interface offset
  ccs := ccsInterface relation interface offset
  ccsOffset := ccsOffset interface offset
  norm := normInterface relation interface offset
  normOffset := normOffset relation interface offset
  finalIdentity := finalIdentityInterface relation interface offset
  finalIdentityOffset := finalIdentityOffset relation interface offset
  output := outputInterface interface offset
  outputOffset := outputOffset relation interface offset

def opsAt : List Op :=
  [Formal.childOp "light.piccs.statement_absorption" (statementCircuit interface offset) offset,
    Formal.childOp "light.piccs.challenge_derivation" (challengeCircuit interface offset)
      (challengeOffset interface offset),
    Formal.childOp "light.piccs.round_transcript" (roundsCircuit interface offset)
      (roundsOffset interface offset),
    Formal.childOp "light.piccs.initial_claim" (initialCircuit interface offset)
      (initialOffset interface offset),
    Formal.childOp "light.piccs.sumcheck_chain" (sumcheckCircuit interface offset)
      (sumcheckOffset interface offset),
    Formal.childOp "light.piccs.eval_K_terminal" (evalKCircuit interface offset)
      (evalKOffset interface offset),
    Formal.childOp "light.piccs.eval_A_terminal" (evalACircuit interface offset)
      (evalAOffset interface offset),
    Formal.childOp "light.piccs.ccs_terminal" (ccsCircuit relation interface offset)
      (ccsOffset interface offset),
    Formal.childOp "light.piccs.norm_terminal" (normCircuit relation interface offset)
      (normOffset relation interface offset),
    Formal.childOp "light.piccs.final_identity" (finalIdentityCircuit relation interface offset)
      (finalIdentityOffset relation interface offset),
    Formal.childOp "light.piccs.output_binding" (outputCircuit interface offset)
      (outputOffset relation interface offset)]

def main : Circuit Unit :=
  fun offset => ((), finalOffset relation interface offset, opsAt relation interface offset)

/-- Each child's input scoping, which its rows need to give its spec. -/
structure Assumptions (env : Env) : Prop where
  statement : (statementCircuit interface offset).assumptions offset env
  challenge : (challengeCircuit interface offset).assumptions (challengeOffset interface offset) env
  rounds : (roundsCircuit interface offset).assumptions (roundsOffset interface offset) env
  initial : (initialCircuit interface offset).assumptions (initialOffset interface offset) env
  sumcheck : (sumcheckCircuit interface offset).assumptions (sumcheckOffset interface offset) env
  evalK : (evalKCircuit interface offset).assumptions (evalKOffset interface offset) env
  evalA : (evalACircuit interface offset).assumptions (evalAOffset interface offset) env
  ccs : (ccsCircuit relation interface offset).assumptions (ccsOffset interface offset) env
  norm : (normCircuit relation interface offset).assumptions
    (normOffset relation interface offset) env
  finalIdentity : (finalIdentityCircuit relation interface offset).assumptions
    (finalIdentityOffset relation interface offset) env
  output : (outputCircuit interface offset).assumptions (outputOffset relation interface offset)
    env

private theorem childSpec_of_rows (name : String) (child : FormalCircuit)
    (childOffset : Nat) (env : Env) (ops : List Op) (rows : holds env ops)
    (member : Formal.childOp name child childOffset ∈ ops)
    (assumptions : child.assumptions childOffset env) :
    child.spec childOffset env := by
  have callHolds := rows (Formal.childOp name child childOffset) member
  change child.assumptions childOffset env → child.spec childOffset env at callHolds
  exact callHolds assumptions

/-- The rows give each leaf's spec. The CCS terminal is the same for every production relation,
so the leaves hold for the relation of any circuit key. -/
theorem soundness (interface : Interface logicalWidth 8 publicFits) (env : Env)
    (assumptions : Assumptions relation interface offset env)
    (rows : holds env (Circuit.ops (main relation interface) offset))
    (target : ProductionKey.LogicalRelation logicalWidth publicFits) :
    LightPiCCSPhase.SpecHolds target (leaves relation interface offset) env := by
  change holds env (opsAt relation interface offset) at rows
  have statementSpec := childSpec_of_rows "light.piccs.statement_absorption"
    (statementCircuit interface offset) offset env _ rows (by simp [opsAt])
    assumptions.statement
  have challengeSpec := childSpec_of_rows "light.piccs.challenge_derivation"
    (challengeCircuit interface offset) (challengeOffset interface offset) env _ rows
    (by simp [opsAt])
    assumptions.challenge
  have roundsSpec := childSpec_of_rows "light.piccs.round_transcript"
    (roundsCircuit interface offset) (roundsOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.rounds
  have initialSpec := childSpec_of_rows "light.piccs.initial_claim"
    (initialCircuit interface offset) (initialOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.initial
  have sumcheckSpec := childSpec_of_rows "light.piccs.sumcheck_chain"
    (sumcheckCircuit interface offset) (sumcheckOffset interface offset) env _ rows
    (by simp [opsAt])
    assumptions.sumcheck
  have evalKSpec := childSpec_of_rows "light.piccs.eval_K_terminal"
    (evalKCircuit interface offset) (evalKOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.evalK
  have evalASpec := childSpec_of_rows "light.piccs.eval_A_terminal"
    (evalACircuit interface offset) (evalAOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.evalA
  have ccsSpec := childSpec_of_rows "light.piccs.ccs_terminal"
    (ccsCircuit relation interface offset) (ccsOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.ccs
  have normSpec := childSpec_of_rows "light.piccs.norm_terminal"
    (normCircuit relation interface offset)
        (normOffset relation interface offset) env _ rows (by simp [opsAt])
    assumptions.norm
  have finalIdentitySpec := childSpec_of_rows "light.piccs.final_identity"
    (finalIdentityCircuit relation interface offset)
        (finalIdentityOffset relation interface offset) env _ rows (by simp [opsAt])
    assumptions.finalIdentity
  have outputSpec := childSpec_of_rows "light.piccs.output_binding"
    (outputCircuit interface offset)
        (outputOffset relation interface offset) env _ rows (by simp [opsAt])
    assumptions.output
  exact ⟨statementSpec, challengeSpec, roundsSpec, initialSpec, sumcheckSpec, evalKSpec,
    evalASpec, ccsSpec, normSpec, finalIdentitySpec, outputSpec⟩

/-- The running source's index in the light shape, its only running claim. -/
def runningIndex : Fin LightShape.runningCount := ⟨0, by decide⟩

private theorem runningIndex_eq (index : Fin LightShape.runningCount) : index = runningIndex :=
  Fin.ext (by
    have bound := index.isLt
    change index.val < 1 at bound
    change index.val = 0
    omega)

/-- The external words carry the light fold's running claim, fresh instance and proof. -/
structure Represents
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (proof : Spec.Folding.Light.Proof LightShape 8)
    (interface : Interface logicalWidth 8 publicFits)
    (offset : Nat) (env : Env) : Prop where
  fresh : StatementAbsorption.evalFresh (interface.fresh offset) env =
    ⟨fresh.commitments, fresh.publicInputs⟩
  runningPoint : (canonicalFinIndices LightShape.cubeVariables).map
      (fun coordinate => (interface.runningPoint offset coordinate).eval env) =
    running.point.coordinates
  runningPad : ∀ coefficient,
    ((interface.runningEvaluation offset).eval_K coefficient).eval env =
      (running.evaluations runningIndex).pad coefficient
  runningMatrix : ∀ matrix coefficient,
    ((interface.runningEvaluation offset).eval_A matrix coefficient).eval env =
      (running.evaluations runningIndex).matrix matrix coefficient
  rounds : ∀ roundIndex,
    (interface.round offset roundIndex).semanticPolynomial env = proof.piCcsRounds roundIndex
  outputPad : ∀ source coefficient,
    ((interface.output offset).padCoordinate source coefficient).eval env =
      proof.piCcsOutput.padCoordinate source coefficient
  outputMatrix : ∀ source matrix coefficient,
    ((interface.output offset).matrixCoordinate source matrix coefficient).eval env =
      proof.piCcsOutput.matrixCoordinate source matrix coefficient

private theorem point_eq {count : Nat} {left right : CubePoint K count}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The layout's wires: the external words alone give `LightPiCCSPhase.Wired` for the relation
of any circuit key. -/
theorem wired (target : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (proof : Spec.Folding.Light.Proof LightShape 8)
    (interface : Interface logicalWidth 8 publicFits)
    (env : Env) (represents : Represents running fresh proof interface offset env) :
    LightPiCCSPhase.Wired target ajtai running fresh proof (leaves relation interface offset)
      env where
  statementFresh := represents.fresh
  challengeStart := rfl
  roundsStart := by
    simp only [leaves, roundsInterface, RoundTranscript.evalState, ChallengeDerivation.evalState]
  roundsMessages := represents.rounds
  initialGamma := rfl
  initialPad := fun coordinate => by
    change ((interface.runningEvaluation offset).eval_K coordinate.coefficient).eval env =
      (running.evaluations coordinate.running).pad coordinate.coefficient
    rw [runningIndex_eq coordinate.running]
    exact represents.runningPad coordinate.coefficient
  initialMatrix := fun coordinate => by
    change ((interface.runningEvaluation offset).eval_A coordinate.matrix
        coordinate.coefficient).eval env =
      (running.evaluations coordinate.running).matrix coordinate.matrix coordinate.coefficient
    rw [runningIndex_eq coordinate.running]
    exact represents.runningMatrix coordinate.matrix coordinate.coefficient
  evalKRoundPoint := rfl
  evalKPriorPoint := point_eq represents.runningPoint
  evalKGamma := rfl
  evalKOutput := fun coordinate => represents.outputPad _ _
  evalARoundPoint := rfl
  evalAPriorPoint := point_eq represents.runningPoint
  evalAGamma := rfl
  evalAOutput := fun coordinate => represents.outputMatrix _ _ _
  ccsFresh := fun matrix => represents.outputMatrix _ _ _
  normGamma := rfl
  normSource := fun source => by
    by_cases freshSource : source.val < LightShape.freshCount <;>
      simp only [leaves, normInterface, LightPiCCSPhase.lightMessage,
        Spec.Folding.Light.PiCCS.maskMessage, freshSource, ↓reduceIte]
    · exact represents.outputPad _ _
    · rfl
  finalRoundPoint := rfl
  finalAlpha := by
    rw [ChallengeDerivation.evalAlpha_coordinates]
    simp only [PointEquality.Owned.evalRightPoint, canonicalFinIndices, List.map_ofFn]
    rfl
  finalGamma := rfl
  finalPad := rfl
  finalMatrix := rfl
  finalCcs := rfl
  finalNorm := rfl
  sumcheckInitial := rfl
  sumcheckRounds := represents.rounds
  sumcheckPoint := rfl
  sumcheckTerminal := rfl
  outputStart := by simp only [leaves, outputInterface, RoundTranscript.evalState]
  outputPad := represents.outputPad
  outputMatrix := represents.outputMatrix

/-- The layout's rows and external words give the light Π_CCS phase: the light check accepts,
with the light key's SumCheck point and outgoing transcript state. -/
theorem phaseHolds (interface : Interface logicalWidth 8 publicFits) (env : Env)
    (assumptions : Assumptions relation interface offset env)
    (rows : holds env (Circuit.ops (main relation interface) offset))
    (target : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (proof : Spec.Folding.Light.Proof LightShape 8)
    (represents : Represents running fresh proof interface offset env) :
    LightPiCCSPhase.PhaseHolds target ajtai running fresh proof (leaves relation interface offset)
      env :=
  LightPiCCSPhase.spec_implies_phaseHolds target ajtai running fresh proof
    (leaves relation interface offset) env
    (soundness relation offset interface env assumptions rows target)
    (wired relation offset target ajtai running fresh proof interface env represents)

end NightstreamFPrime.Lifecycle.OptionB.LightPiCCSLayout
