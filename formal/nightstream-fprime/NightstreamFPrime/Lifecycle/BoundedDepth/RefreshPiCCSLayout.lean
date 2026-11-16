import NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCSPhase
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal
import NightstreamFPrime.Gadgets.Poseidon2.CleanChildren
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.CcsTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.EvalATerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.EvalKTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.FinalIdentity
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.InitialClaim
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.NormTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.SumcheckChain

/-! The refresh Π_CCS layout (CW-3.4a). The eleven leaf gadgets of `RefreshPiCCSPhase` run one
after another from a parent offset, and each reads its inputs from the external words or from
the outputs of earlier gadgets. `soundness` gives `RefreshPiCCSPhase.SpecHolds` from the rows,
and `wired` gives `RefreshPiCCSPhase.Wired` from the external words alone. The statement
absorption is the duplex gadget on `RefreshPiCCS.statementActions`; every other leaf is today's
production-shape gadget, as in `PiCCS.v1_2.Formal`, and the norm terminal reads every source. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCSLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

/-- The external words of the refresh Π_CCS: the 14 children (the slot's point, the proof's
commitments and evaluation families, and the Π_DEC digit words), the fresh instance, the
SumCheck round messages and the Π_CCS output of every source. -/
structure Interface (logicalWidth degreeBound : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  running : Nat → StatementAbsorption.RunningExpr logicalWidth publicFits
  fresh : Nat → StatementAbsorption.FreshExpr logicalWidth publicFits
  round : Nat → Fin productionShape.cubeVariables → RoundTranscript.Message degreeBound
  output : Nat → OutputBinding.OutputExpr

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (interface : Interface logicalWidth degreeBound publicFits) (offset : Nat)

def statementInterface : StatementAbsorption.Interface logicalWidth publicFits where
  running := fun _ => interface.running offset
  fresh := fun _ => interface.fresh offset

def statementDuplex : Duplex.Formal.Owned.Interface where
  initial := fun _ => Hash.zeroE
  actions := RefreshPiCCS.statementActions (statementInterface interface offset)

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

def initialInterface : InitialClaim.Interface where
  gamma := fun _ => gamma interface offset
  eval_K := fun _ coordinate =>
    ((interface.running offset).evaluation coordinate.running).eval_K coordinate.coefficient
  eval_A := fun _ coordinate =>
    ((interface.running offset).evaluation coordinate.running).eval_A coordinate.matrix
      coordinate.coefficient

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

def evalKInterface : EvalKTerminal.Interface where
  roundPoint := fun _ => roundPoint interface offset
  priorPoint := fun _ => (interface.running offset).point
  gamma := fun _ => gamma interface offset
  outputEval_K := fun _ coordinate =>
    (interface.output offset).padCoordinate (runningSourceIndex coordinate.running)
      coordinate.coefficient

def evalKCircuit : FormalCircuit :=
  EvalKTerminal.circuit (evalKInterface interface offset)

def evalAOffset : Nat :=
  Formal.nextOffset (evalKCircuit interface offset) (evalKOffset interface offset)

def evalAInterface : EvalATerminal.Interface where
  roundPoint := fun _ => roundPoint interface offset
  priorPoint := fun _ => (interface.running offset).point
  gamma := fun _ => gamma interface offset
  outputEval_A := fun _ coordinate =>
    (interface.output offset).matrixCoordinate (runningSourceIndex coordinate.running)
      coordinate.matrix coordinate.coefficient

def evalACircuit : FormalCircuit :=
  EvalATerminal.circuit (evalAInterface interface offset)

def ccsOffset : Nat :=
  Formal.nextOffset (evalACircuit interface offset) (evalAOffset interface offset)

def ccsInterface : CcsTerminal.Interface where
  freshMatrix := fun _ matrix =>
    (interface.output offset).matrixCoordinate (freshSourceIndex RefreshPiCCSPhase.freshIndex)
      matrix (Formal.constantCoefficient relation)

def ccsCircuit : FormalCircuit :=
  CcsTerminal.circuit relation (ccsInterface relation interface offset)

def normOffset : Nat :=
  Formal.nextOffset (ccsCircuit relation interface offset) (ccsOffset interface offset)

/-- The norm terminal reads every source's assignment: `Refresh.verify` runs today's check. -/
def normInterface : NormTerminal.Interface where
  gamma := fun _ => gamma interface offset
  sourceAssignment := fun _ source =>
    (interface.output offset).padCoordinate source (Formal.constantCoefficient relation)

def normCircuit : FormalCircuit :=
  NormTerminal.circuit (normInterface relation interface offset)

def finalIdentityOffset : Nat :=
  Formal.nextOffset (normCircuit relation interface offset) (normOffset relation interface offset)

def finalIdentityInterface : FinalIdentity.Interface where
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
  FinalIdentity.circuit (finalIdentityInterface relation interface offset)

def outputOffset : Nat :=
  Formal.nextOffset (finalIdentityCircuit relation interface offset)
    (finalIdentityOffset relation interface offset)

def outputInterface : OutputBinding.Interface where
  roundPoint := fun _ => roundPoint interface offset
  initialState := fun _ => RoundTranscript.finalState (roundsInterface interface offset)
    (roundsOffset interface offset)
  output := fun _ => interface.output offset

def outputCircuit : FormalCircuit :=
  OutputBinding.circuit (outputInterface interface offset)

def finalOffset : Nat :=
  Formal.nextOffset (outputCircuit interface offset) (outputOffset relation interface offset)

/-- The leaves and offsets of the refresh Π_CCS phase that this layout places. -/
def leaves : RefreshPiCCSPhase.Leaves logicalWidth degreeBound publicFits where
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
  [Formal.childOp "refresh.piccs.statement_absorption" (statementCircuit interface offset) offset,
    Formal.childOp "refresh.piccs.challenge_derivation" (challengeCircuit interface offset)
      (challengeOffset interface offset),
    Formal.childOp "refresh.piccs.round_transcript" (roundsCircuit interface offset)
      (roundsOffset interface offset),
    Formal.childOp "refresh.piccs.initial_claim" (initialCircuit interface offset)
      (initialOffset interface offset),
    Formal.childOp "refresh.piccs.sumcheck_chain" (sumcheckCircuit interface offset)
      (sumcheckOffset interface offset),
    Formal.childOp "refresh.piccs.eval_K_terminal" (evalKCircuit interface offset)
      (evalKOffset interface offset),
    Formal.childOp "refresh.piccs.eval_A_terminal" (evalACircuit interface offset)
      (evalAOffset interface offset),
    Formal.childOp "refresh.piccs.ccs_terminal" (ccsCircuit relation interface offset)
      (ccsOffset interface offset),
    Formal.childOp "refresh.piccs.norm_terminal" (normCircuit relation interface offset)
      (normOffset relation interface offset),
    Formal.childOp "refresh.piccs.final_identity" (finalIdentityCircuit relation interface offset)
      (finalIdentityOffset relation interface offset),
    Formal.childOp "refresh.piccs.output_binding" (outputCircuit interface offset)
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
    RefreshPiCCSPhase.SpecHolds target (leaves relation interface offset) env := by
  change holds env (opsAt relation interface offset) at rows
  have statementSpec := childSpec_of_rows "refresh.piccs.statement_absorption"
    (statementCircuit interface offset) offset env _ rows (by simp [opsAt])
    assumptions.statement
  have challengeSpec := childSpec_of_rows "refresh.piccs.challenge_derivation"
    (challengeCircuit interface offset) (challengeOffset interface offset) env _ rows
    (by simp [opsAt])
    assumptions.challenge
  have roundsSpec := childSpec_of_rows "refresh.piccs.round_transcript"
    (roundsCircuit interface offset) (roundsOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.rounds
  have initialSpec := childSpec_of_rows "refresh.piccs.initial_claim"
    (initialCircuit interface offset) (initialOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.initial
  have sumcheckSpec := childSpec_of_rows "refresh.piccs.sumcheck_chain"
    (sumcheckCircuit interface offset) (sumcheckOffset interface offset) env _ rows
    (by simp [opsAt])
    assumptions.sumcheck
  have evalKSpec := childSpec_of_rows "refresh.piccs.eval_K_terminal"
    (evalKCircuit interface offset) (evalKOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.evalK
  have evalASpec := childSpec_of_rows "refresh.piccs.eval_A_terminal"
    (evalACircuit interface offset) (evalAOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.evalA
  have ccsSpec := childSpec_of_rows "refresh.piccs.ccs_terminal"
    (ccsCircuit relation interface offset) (ccsOffset interface offset) env _ rows (by simp [opsAt])
    assumptions.ccs
  have normSpec := childSpec_of_rows "refresh.piccs.norm_terminal"
    (normCircuit relation interface offset)
        (normOffset relation interface offset) env _ rows (by simp [opsAt])
    assumptions.norm
  have finalIdentitySpec := childSpec_of_rows "refresh.piccs.final_identity"
    (finalIdentityCircuit relation interface offset)
        (finalIdentityOffset relation interface offset) env _ rows (by simp [opsAt])
    assumptions.finalIdentity
  have outputSpec := childSpec_of_rows "refresh.piccs.output_binding"
    (outputCircuit interface offset)
        (outputOffset relation interface offset) env _ rows (by simp [opsAt])
    assumptions.output
  exact ⟨statementSpec, challengeSpec, roundsSpec, initialSpec, sumcheckSpec, evalKSpec,
    evalASpec, ccsSpec, normSpec, finalIdentitySpec, outputSpec⟩

/-- The external words carry the refresh fold's children, fresh instance and proof. -/
structure Represents (children : Children logicalWidth publicFits)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape)
    (proof : Spec.Folding.Nifs.PaperNonInteractive.Proof K PaperAlgebra.Commitment
      productionShape 8)
    (interface : Interface logicalWidth 8 publicFits) (offset : Nat) (env : Env) : Prop where
  fresh : StatementAbsorption.evalFresh (interface.fresh offset) env = fresh
  point : (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate => ((interface.running offset).point coordinate).eval env) =
    children.point.coordinates
  commitments : ∀ child row lane,
    ((interface.running offset).commitment child row lane).eval env =
      children.commitments child row lane
  evaluations : ∀ child,
    StatementAbsorption.evalEvaluation ((interface.running offset).evaluation child) env =
      children.evaluations child
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

/-- The layout's wires: the external words alone give `RefreshPiCCSPhase.Wired` for the relation
of any circuit key. -/
theorem wired (target : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (children : Children logicalWidth publicFits)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape)
    (proof : Spec.Folding.Nifs.PaperNonInteractive.Proof K PaperAlgebra.Commitment
      productionShape 8)
    (interface : Interface logicalWidth 8 publicFits)
    (env : Env) (represents : Represents children fresh proof interface offset env) :
    RefreshPiCCSPhase.Wired target ajtai children fresh proof (leaves relation interface offset)
      env where
  statementFresh := represents.fresh
  statementCommitments := represents.commitments
  statementEvaluations := represents.evaluations
  challengeStart := rfl
  roundsStart := by
    simp only [leaves, roundsInterface, RoundTranscript.evalState, ChallengeDerivation.evalState]
  roundsMessages := represents.rounds
  initialGamma := rfl
  initialPad := fun coordinate => by
    change (((interface.running offset).evaluation coordinate.running).eval_K
        coordinate.coefficient).eval env =
      (children.evaluations coordinate.running).pad coordinate.coefficient
    rw [← represents.evaluations coordinate.running]
    rfl
  initialMatrix := fun coordinate => by
    change (((interface.running offset).evaluation coordinate.running).eval_A coordinate.matrix
        coordinate.coefficient).eval env =
      (children.evaluations coordinate.running).matrix coordinate.matrix coordinate.coefficient
    rw [← represents.evaluations coordinate.running]
    rfl
  evalKRoundPoint := rfl
  evalKPriorPoint := point_eq represents.point
  evalKGamma := rfl
  evalKOutput := fun coordinate => represents.outputPad _ _
  evalARoundPoint := rfl
  evalAPriorPoint := point_eq represents.point
  evalAGamma := rfl
  evalAOutput := fun coordinate => represents.outputMatrix _ _ _
  ccsFresh := fun matrix => represents.outputMatrix _ _ _
  normGamma := rfl
  normSource := fun source => represents.outputPad _ _
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

/-- The layout's rows and external words give the refresh Π_CCS phase: the refresh key's check
accepts, with its SumCheck point and outgoing transcript state. -/
theorem phaseHolds (interface : Interface logicalWidth 8 publicFits) (env : Env)
    (assumptions : Assumptions relation interface offset env)
    (rows : holds env (Circuit.ops (main relation interface) offset))
    (target : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (children : Children logicalWidth publicFits)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape)
    (proof : Spec.Folding.Nifs.PaperNonInteractive.Proof K PaperAlgebra.Commitment
      productionShape 8)
    (represents : Represents children fresh proof interface offset env) :
    RefreshPiCCSPhase.PhaseHolds target ajtai children fresh proof
      (leaves relation interface offset) env :=
  RefreshPiCCSPhase.spec_implies_phaseHolds target ajtai children fresh proof
    (leaves relation interface offset) env
    (soundness relation offset interface env assumptions rows target)
    (wired relation offset target ajtai children fresh proof interface env represents)

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

open NightstreamFPrime.Gadgets.Polynomial

/-- The external words of the refresh Π_CCS. The words that a transcript absorbs (the fresh
instance, the children's commitments and evaluations, the round messages and the Π_CCS output)
are affine and live before the layout; the children's point is live. -/
structure InputsLive (live : Nat → Prop) : Prop where
  freshCommitment : ∀ index row lane,
    Duplex.Formal.WordFacts live offset ((interface.fresh offset).commitment index row lane)
  freshPublicInput : ∀ index column,
    Duplex.Formal.WordFacts live offset ((interface.fresh offset).publicInput index column)
  round : ∀ roundIndex coefficient,
    Duplex.Formal.WordFacts live offset
        ((interface.round offset roundIndex).coefficient coefficient).c0 ∧
      Duplex.Formal.WordFacts live offset
        ((interface.round offset roundIndex).coefficient coefficient).c1
  runningCommitment : ∀ child row lane,
    Duplex.Formal.WordFacts live offset ((interface.running offset).commitment child row lane)
  runningEvalK : ∀ child coefficient,
    Duplex.Formal.WordFacts live offset
        (((interface.running offset).evaluation child).eval_K coefficient).c0 ∧
      Duplex.Formal.WordFacts live offset
        (((interface.running offset).evaluation child).eval_K coefficient).c1
  runningEvalA : ∀ child matrix coefficient,
    Duplex.Formal.WordFacts live offset
        (((interface.running offset).evaluation child).eval_A matrix coefficient).c0 ∧
      Duplex.Formal.WordFacts live offset
        (((interface.running offset).evaluation child).eval_A matrix coefficient).c1
  runningPoint : ∀ coordinate,
    ((interface.running offset).point coordinate).c0.VarsSatisfy live ∧
      ((interface.running offset).point coordinate).c1.VarsSatisfy live
  outputPad : ∀ source coefficient,
    Duplex.Formal.WordFacts live offset
        ((interface.output offset).padCoordinate source coefficient).c0 ∧
      Duplex.Formal.WordFacts live offset
        ((interface.output offset).padCoordinate source coefficient).c1
  outputMatrix : ∀ source matrix coefficient,
    Duplex.Formal.WordFacts live offset
        ((interface.output offset).matrixCoordinate source matrix coefficient).c0 ∧
      Duplex.Formal.WordFacts live offset
        ((interface.output offset).matrixCoordinate source matrix coefficient).c1

theorem placed :
    Placed offset (localLength (Circuit.ops (main relation interface) offset))
      (Circuit.ops (main relation interface) offset) :=
  Placed.child_cons rfl (Duplex.Formal.Owned.placed _ _) <|
  Placed.child_cons rfl (ChallengeDerivation.placed _ _) <|
  Placed.child_cons rfl (RoundTranscript.placed _ _) <|
  Placed.child_cons rfl (.of_marks_nil (InitialClaim.marks_nil _ _)) <|
  Placed.child_cons rfl (.of_marks_nil (SumcheckChain.marks_nil _ _)) <|
  Placed.child_cons rfl (.of_marks_nil (EvalKTerminal.marks_nil _ _)) <|
  Placed.child_cons rfl (.of_marks_nil (EvalATerminal.marks_nil _ _)) <|
  Placed.child_cons rfl (.of_marks_nil (CcsTerminal.marks_nil _ _ _)) <|
  Placed.child_cons rfl (.of_marks_nil (NormTerminal.marks_nil _ _)) <|
  Placed.child_cons rfl (.of_marks_nil (FinalIdentity.marks_nil _ _)) <|
  Placed.child_cons rfl (OutputBinding.placed _ _) <|
  Placed.nil _

/-- The refresh Π_CCS's rows: its transcripts' permutations and rows that read the external words,
the earlier gadgets' outputs and the layout's own locals. The SumCheck point is affine and live
before the initial claim, and the transcript state after the output binding is live. -/
theorem clean {live : Nat → Prop} (inputs : InputsLive interface offset live)
    (own : OwnLive live offset (localLength (Circuit.ops (main relation interface) offset))
      (Circuit.ops (main relation interface) offset)) :
    Clean live (Circuit.ops (main relation interface) offset) ∧
      (∀ coordinate,
        Duplex.Formal.WordFacts live (initialOffset interface offset)
            (roundPoint interface offset coordinate).c0 ∧
          Duplex.Formal.WordFacts live (initialOffset interface offset)
            (roundPoint interface offset coordinate).c1) ∧
      Duplex.Formal.StateFacts live (finalOffset relation interface offset)
        (OutputBinding.finalState (outputInterface interface offset)
          (outputOffset relation interface offset)) := by
  have outputTail := Placed.child_cons (name := "refresh.piccs.output_binding") rfl
    (OutputBinding.placed (outputInterface interface offset)
      (outputOffset relation interface offset)) (Placed.nil _)
  have finalTail := Placed.child_cons (name := "refresh.piccs.final_identity") rfl
    (.of_marks_nil (FinalIdentity.marks_nil (finalIdentityInterface relation interface offset)
      (finalIdentityOffset relation interface offset))) outputTail
  have normTail := Placed.child_cons (name := "refresh.piccs.norm_terminal") rfl
    (.of_marks_nil (NormTerminal.marks_nil (normInterface relation interface offset)
      (normOffset relation interface offset))) finalTail
  have ccsTail := Placed.child_cons (name := "refresh.piccs.ccs_terminal") rfl
    (.of_marks_nil (CcsTerminal.marks_nil relation (ccsInterface relation interface offset)
      (ccsOffset interface offset))) normTail
  have evalATail := Placed.child_cons (name := "refresh.piccs.eval_A_terminal") rfl
    (.of_marks_nil (EvalATerminal.marks_nil (evalAInterface interface offset)
      (evalAOffset interface offset))) ccsTail
  have evalKTail := Placed.child_cons (name := "refresh.piccs.eval_K_terminal") rfl
    (.of_marks_nil (EvalKTerminal.marks_nil (evalKInterface interface offset)
      (evalKOffset interface offset))) evalATail
  have sumcheckTail := Placed.child_cons (name := "refresh.piccs.sumcheck_chain") rfl
    (.of_marks_nil (SumcheckChain.marks_nil (sumcheckInterface interface offset)
      (sumcheckOffset interface offset))) evalKTail
  have initialTail := Placed.child_cons (name := "refresh.piccs.initial_claim") rfl
    (.of_marks_nil (InitialClaim.marks_nil (initialInterface interface offset)
      (initialOffset interface offset))) sumcheckTail
  have roundsTail := Placed.child_cons (name := "refresh.piccs.round_transcript") rfl
    (RoundTranscript.placed (roundsInterface interface offset) (roundsOffset interface offset))
    initialTail
  have challengeTail := Placed.child_cons (name := "refresh.piccs.challenge_derivation") rfl
    (ChallengeDerivation.placed (challengeInterface interface offset)
      (challengeOffset interface offset)) roundsTail
  obtain ⟨statementOwn, afterStatement⟩ :=
    own.child_cons rfl (Duplex.Formal.Owned.placed _ _) challengeTail
  obtain ⟨challengeOwn, afterChallenge⟩ :=
    afterStatement.child_cons rfl (ChallengeDerivation.placed _ _) roundsTail
  obtain ⟨roundsOwn, afterRounds⟩ :=
    afterChallenge.child_cons rfl (RoundTranscript.placed _ _) initialTail
  obtain ⟨initialOwn, afterInitial⟩ :=
    afterRounds.child_cons rfl (.of_marks_nil (InitialClaim.marks_nil _ _)) sumcheckTail
  obtain ⟨sumcheckOwn, afterSumcheck⟩ :=
    afterInitial.child_cons rfl (.of_marks_nil (SumcheckChain.marks_nil _ _)) evalKTail
  obtain ⟨evalKOwn, afterEvalK⟩ :=
    afterSumcheck.child_cons rfl (.of_marks_nil (EvalKTerminal.marks_nil _ _)) evalATail
  obtain ⟨evalAOwn, afterEvalA⟩ :=
    afterEvalK.child_cons rfl (.of_marks_nil (EvalATerminal.marks_nil _ _)) ccsTail
  obtain ⟨ccsOwn, afterCcs⟩ :=
    afterEvalA.child_cons rfl (.of_marks_nil (CcsTerminal.marks_nil _ _ _)) normTail
  obtain ⟨normOwn, afterNorm⟩ :=
    afterCcs.child_cons rfl (.of_marks_nil (NormTerminal.marks_nil _ _)) finalTail
  obtain ⟨finalOwn, afterFinal⟩ :=
    afterNorm.child_cons rfl (.of_marks_nil (FinalIdentity.marks_nil _ _)) outputTail
  obtain ⟨outputOwn, -⟩ := afterFinal.child_cons rfl (OutputBinding.placed _ _) (Placed.nil _)
  -- The statement absorption and the challenges.
  have statementWords := RefreshPiCCS.statementActions_facts (statementInterface interface offset)
    offset inputs.freshCommitment inputs.freshPublicInput inputs.runningCommitment
    inputs.runningEvalK inputs.runningEvalA
  obtain ⟨statementClean, statementState⟩ := Duplex.Formal.Owned.output_facts
    (statementDuplex interface offset) offset (Duplex.Formal.zeroE_stateFacts live offset)
    (fun word member => (statementWords word member).1)
    (fun word member => (statementWords word member).2)
    (fun word member => by
      rw [statementDuplex, RefreshPiCCS.statementActions_expected_nil] at member
      cases member)
    statementOwn
  obtain ⟨challengeClean, alphaLive, gammaLive, challengeState⟩ := ChallengeDerivation.clean
    (challengeInterface interface offset) (challengeOffset interface offset)
    (by dsimp only [challengeInterface]; exact statementState) challengeOwn
  have roundsAfter : offset ≤ roundsOffset interface offset := by
    unfold roundsOffset challengeOffset Formal.nextOffset
    omega
  have roundsStart : Duplex.Formal.StateFacts live (roundsOffset interface offset)
      ((roundsInterface interface offset).initialState (roundsOffset interface offset)) := by
    dsimp only [roundsInterface]
    exact challengeState
  obtain ⟨roundsClean, roundPointLive, roundsState⟩ := RoundTranscript.clean
    (roundsInterface interface offset) (roundsOffset interface offset) roundsStart
    (fun roundIndex coefficient => ⟨(inputs.round roundIndex coefficient).1.mono roundsAfter,
      (inputs.round roundIndex coefficient).2.mono roundsAfter⟩) roundsOwn
  have roundPointFacts : ∀ coordinate,
      Duplex.Formal.WordFacts live (initialOffset interface offset)
          (roundPoint interface offset coordinate).c0 ∧
        Duplex.Formal.WordFacts live (initialOffset interface offset)
          (roundPoint interface offset coordinate).c1 :=
    RoundTranscript.challenge_facts (roundsInterface interface offset)
      (roundsOffset interface offset) roundsStart
      (fun roundIndex coefficient => ⟨(inputs.round roundIndex coefficient).1.mono roundsAfter,
        (inputs.round roundIndex coefficient).2.mono roundsAfter⟩) roundsOwn
  -- The claims and the terminals.
  have runningEvalK : ∀ coordinate,
      Horner.KSupported ((initialInterface interface offset).eval_K
        (initialOffset interface offset) coordinate) live := fun _ =>
    ⟨(inputs.runningEvalK _ _).1.varsSatisfy, (inputs.runningEvalK _ _).2.varsSatisfy⟩
  have runningEvalA : ∀ coordinate,
      Horner.KSupported ((initialInterface interface offset).eval_A
        (initialOffset interface offset) coordinate) live := fun _ =>
    ⟨(inputs.runningEvalA _ _ _).1.varsSatisfy, (inputs.runningEvalA _ _ _).2.varsSatisfy⟩
  have initialLocals := OwnLive.of_marks_nil (InitialClaim.marks_nil _ _) initialOwn
  have initialGamma : Horner.KSupported
      ((initialInterface interface offset).gamma (initialOffset interface offset)) live := by
    dsimp only [initialInterface]
    exact gammaLive
  have initialFree := InitialClaim.free (initialInterface interface offset)
    (initialOffset interface offset) initialGamma runningEvalK
    runningEvalA initialLocals
  have initialLive := InitialClaim.output_live (initialInterface interface offset)
    (initialOffset interface offset) initialGamma runningEvalK
    runningEvalA initialLocals
  have sumcheckInitial : SumcheckChain.KSupported
      ((sumcheckInterface interface offset).initial (sumcheckOffset interface offset)) live := by
    dsimp only [sumcheckInterface]
    exact initialLive
  have sumcheckRounds : ∀ roundIndex, SumcheckChain.RoundSupported
      ((sumcheckInterface interface offset).round (sumcheckOffset interface offset) roundIndex)
      live := by
    intro roundIndex
    dsimp only [sumcheckInterface]
    exact ⟨fun coefficient => ⟨(inputs.round roundIndex coefficient).1.varsSatisfy,
      (inputs.round roundIndex coefficient).2.varsSatisfy⟩, roundPointLive roundIndex⟩
  have sumcheckLocals := OwnLive.of_marks_nil (SumcheckChain.marks_nil _ _) sumcheckOwn
  have sumcheckFree := SumcheckChain.free (sumcheckInterface interface offset)
    (sumcheckOffset interface offset) sumcheckInitial sumcheckRounds sumcheckLocals
  have terminalLive := SumcheckChain.output_live (sumcheckInterface interface offset)
    (sumcheckOffset interface offset) sumcheckInitial sumcheckRounds sumcheckLocals
  have padLive : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).c0.VarsSatisfy live ∧
        ((interface.output offset).padCoordinate source coefficient).c1.VarsSatisfy live :=
    fun source coefficient => ⟨(inputs.outputPad source coefficient).1.varsSatisfy,
      (inputs.outputPad source coefficient).2.varsSatisfy⟩
  have matrixLive : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix coefficient).c0.VarsSatisfy live ∧
        ((interface.output offset).matrixCoordinate source matrix coefficient).c1.VarsSatisfy
          live :=
    fun source matrix coefficient =>
      ⟨(inputs.outputMatrix source matrix coefficient).1.varsSatisfy,
        (inputs.outputMatrix source matrix coefficient).2.varsSatisfy⟩
  have evalKPoint : ∀ coordinate, Horner.KSupported
      ((evalKInterface interface offset).roundPoint (evalKOffset interface offset) coordinate)
      live := by
    dsimp only [evalKInterface]
    exact roundPointLive
  have evalKGamma : Horner.KSupported
      ((evalKInterface interface offset).gamma (evalKOffset interface offset)) live := by
    dsimp only [evalKInterface]
    exact gammaLive
  have evalKLocals := OwnLive.of_marks_nil (EvalKTerminal.marks_nil _ _) evalKOwn
  have evalKFree := EvalKTerminal.free (evalKInterface interface offset)
    (evalKOffset interface offset) evalKPoint inputs.runningPoint evalKGamma
    (fun _ => padLive _ _) evalKLocals
  have evalKLive := EvalKTerminal.output_live (evalKInterface interface offset)
    (evalKOffset interface offset) evalKPoint inputs.runningPoint evalKGamma
    (fun _ => padLive _ _) evalKLocals
  have evalAPoint : ∀ coordinate, Horner.KSupported
      ((evalAInterface interface offset).roundPoint (evalAOffset interface offset) coordinate)
      live := by
    dsimp only [evalAInterface]
    exact roundPointLive
  have evalAGamma : Horner.KSupported
      ((evalAInterface interface offset).gamma (evalAOffset interface offset)) live := by
    dsimp only [evalAInterface]
    exact gammaLive
  have evalALocals := OwnLive.of_marks_nil (EvalATerminal.marks_nil _ _) evalAOwn
  have evalAFree := EvalATerminal.free (evalAInterface interface offset)
    (evalAOffset interface offset) evalAPoint inputs.runningPoint evalAGamma
    (fun _ => matrixLive _ _ _) evalALocals
  have evalALive := EvalATerminal.output_live (evalAInterface interface offset)
    (evalAOffset interface offset) evalAPoint inputs.runningPoint evalAGamma
    (fun _ => matrixLive _ _ _) evalALocals
  have ccsLocals := OwnLive.of_marks_nil (CcsTerminal.marks_nil _ _ _) ccsOwn
  have ccsFree := CcsTerminal.free relation (ccsInterface relation interface offset)
    (ccsOffset interface offset) (fun _ => matrixLive _ _ _) ccsLocals
  have ccsLive := CcsTerminal.output_live relation (ccsInterface relation interface offset)
    (ccsOffset interface offset) ccsLocals
  have normGamma : Horner.KSupported
      ((normInterface relation interface offset).gamma (normOffset relation interface offset))
      live := by
    dsimp only [normInterface]
    exact gammaLive
  have normSources : ∀ source, Horner.KSupported
      ((normInterface relation interface offset).sourceAssignment
        (normOffset relation interface offset) source) live := by
    intro source
    dsimp only [normInterface]
    exact padLive _ _
  have normLocals := OwnLive.of_marks_nil (NormTerminal.marks_nil _ _) normOwn
  have normFree := NormTerminal.free (normInterface relation interface offset)
    (normOffset relation interface offset) normGamma normSources normLocals
  have normLive := NormTerminal.output_live (normInterface relation interface offset)
    (normOffset relation interface offset) normGamma normSources normLocals
  have finalLocals := OwnLive.of_marks_nil (FinalIdentity.marks_nil _ _) finalOwn
  have finalFree := FinalIdentity.free (finalIdentityInterface relation interface offset)
    (finalIdentityOffset relation interface offset)
    (by dsimp only [finalIdentityInterface]; exact roundPointLive)
    (by dsimp only [finalIdentityInterface]; exact alphaLive)
    (by dsimp only [finalIdentityInterface]; exact gammaLive)
    (by dsimp only [finalIdentityInterface]; exact evalKLive)
    (by dsimp only [finalIdentityInterface]; exact evalALive)
    (by dsimp only [finalIdentityInterface]; exact ccsLive)
    (by dsimp only [finalIdentityInterface]; exact normLive)
    (by dsimp only [finalIdentityInterface]; exact terminalLive) finalLocals
  -- The output binding.
  have outputAfterRounds :
      initialOffset interface offset ≤ outputOffset relation interface offset := by
    unfold outputOffset finalIdentityOffset normOffset ccsOffset evalAOffset evalKOffset
      sumcheckOffset Formal.nextOffset
    omega
  have outputAfter : offset ≤ outputOffset relation interface offset :=
    Nat.le_trans (Nat.le_trans roundsAfter (Nat.le_add_right _ _)) outputAfterRounds
  have outputStart : Duplex.Formal.StateFacts live (outputOffset relation interface offset)
      ((outputInterface interface offset).initialState
        (outputOffset relation interface offset)) := by
    dsimp only [outputInterface]
    exact roundsState.mono outputAfterRounds
  obtain ⟨outputClean, outputState⟩ := OutputBinding.clean (outputInterface interface offset)
    (outputOffset relation interface offset) outputStart
    (fun source coefficient => ⟨(inputs.outputPad source coefficient).1.mono outputAfter,
      (inputs.outputPad source coefficient).2.mono outputAfter⟩)
    (fun source matrix coefficient =>
      ⟨(inputs.outputMatrix source matrix coefficient).1.mono outputAfter,
        (inputs.outputMatrix source matrix coefficient).2.mono outputAfter⟩) outputOwn
  refine ⟨?_, roundPointFacts, outputState⟩
  exact Clean.child_cons statementClean <| Clean.child_cons challengeClean <|
    Clean.child_cons roundsClean <|
    Clean.child_cons (.of_free (InitialClaim.marks_nil _ _) initialFree) <|
    Clean.child_cons (.of_free (SumcheckChain.marks_nil _ _) sumcheckFree) <|
    Clean.child_cons (.of_free (EvalKTerminal.marks_nil _ _) evalKFree) <|
    Clean.child_cons (.of_free (EvalATerminal.marks_nil _ _) evalAFree) <|
    Clean.child_cons (.of_free (CcsTerminal.marks_nil _ _ _) ccsFree) <|
    Clean.child_cons (.of_free (NormTerminal.marks_nil _ _) normFree) <|
    Clean.child_cons (.of_free (FinalIdentity.marks_nil _ _) finalFree) <|
    Clean.child_cons outputClean Clean.nil

end NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCSLayout
