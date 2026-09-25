import NightstreamFPrime.Export.Stage1.SecurityInstance
import NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
import NightstreamFPrime.Lifecycle.Nifs.ClaimCheck
import NightstreamFPrime.Lifecycle.Nifs.SupportedContinuation
import NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic

/-!
Selected PiRLC/PiDEC continuation checks for SuperNeo v1.1 Section 7.5 and
Appendix B.4. Each raw reply is checked against its verifier-computed parent
and all sixteen exact child claims. Stored binary recomposition and the
parent CE check use the existing production key and relation.

The adversary supplies its private tape law and resumed calls. Check and
storage clocks, their stated bounds and the original call/check moment stay
explicit. No checker correctness, opening validity, or FS transfer is assumed.
These are declared clocks; no machine-time refinement is claimed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NifsExtractionProvider

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open _root_.NightstreamFPrime.Lifecycle
open StrongReduction
open PiRLC.CoordinateForkLaw PiRLC.PaperForkExtraction
open PiRLC.PaperForkExtractionWork (Result)
open PiRLC.CoordinateCheckedCalls (CheckResult)

variable (inst : SecurityInstance)

abbrev rlc := PaperAlgebra.piRlcAlgebra inst.ajtai
abbrev dec := PaperAlgebra.piDecAlgebra inst.ajtai
abbrev publicSplit := PaperAlgebra.publicInputSplit inst.ajtai
abbrev evaluationArity := PaperAlgebra.evaluationArity inst.ajtai
abbrev Assignment := PaperAlgebra.Assignment
  (logicalWidth := inst.logicalWidth) (publicFits := inst.publicFits)
abbrev Batch := InputBatch (PaperAlgebra.Structure inst.logicalWidth)
  (PaperAlgebra.PublicInput (logicalWidth := inst.logicalWidth) (publicFits := inst.publicFits))
  PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment
  productionGlobalParams PaperProfile.arity
abbrev Coins := Fin PaperProfile.arity.total → Challenge (rlc inst)
abbrev Reply := PaperWeakSuffix.Reply (Assignment inst) PaperAlgebra.Evaluation
  PaperAlgebra.Commitment productionGlobalParams
abbrev Children := Fin productionGlobalParams.k → Assignment inst
abbrev ParentResponse := Response (Assignment inst) RingF productionGlobalParams PaperProfile.arity
abbrev CheckClock := Coins inst → Reply inst → Nat
abbrev StorageClock := Children inst → Nat
abbrev ParentClock := ParentResponse inst → Nat

private def storeChildren {shape : Phi81Relation.Shape}
    (assignments : Fin productionGlobalParams.k → Phi81Relation.Assignment shape) :
    Vector (StoredAssignmentArithmetic.StoredAssignment shape.carrierWidth)
      productionGlobalParams.k :=
  Vector.ofFn fun child => Vector.ofFn (assignments child)

private theorem storedFamily_view {Value : Type*} {count width : Nat}
    (assignments : Fin count → Fin width → Value) :
    (fun child => ((Vector.ofFn fun index => Vector.ofFn (assignments index)).get child).get) =
      assignments := by
  funext child column
  simp [Vector.get]

private theorem storeChildren_view {shape : Phi81Relation.Shape}
    (assignments : Fin productionGlobalParams.k → Phi81Relation.Assignment shape) :
    (fun child => StoredAssignmentArithmetic.view ((storeChildren assignments).get child)) =
      assignments := by
  dsimp only [storeChildren, StoredAssignmentArithmetic.view]
  exact storedFamily_view assignments

private theorem storedRecompose_value {shape : Phi81Relation.Shape}
    (assignments : Fin productionGlobalParams.k → Phi81Relation.Assignment shape) :
    StoredAssignmentArithmetic.view
        (StoredAssignmentArithmetic.recompose (storeChildren assignments)).value =
      Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment assignments := by
  rw [StoredAssignmentArithmetic.recompose_value, storeChildren_view,
    Phi81Relation.EvaluationHomomorphism.PiDEC.raw_recomposeAssignment_eq]

private theorem storedRecompose_work_le {width : Nat}
    (assignments : Vector (StoredAssignmentArithmetic.StoredAssignment width)
      productionGlobalParams.k) :
    (StoredAssignmentArithmetic.recompose assignments).work ≤ 116 * width + 613 := by
  have bound := StoredAssignmentArithmetic.recompose_work_le assignments
  norm_num [productionGlobalParams] at bound
  omega

/-- Recompose the same sixteen assignments, retaining the existing stored
arithmetic clock and the caller's function-to-array storage clock. -/
def recompose (storageClock : StorageClock inst) (assignments : Children inst) : Result (Assignment inst) :=
  let parent := StoredAssignmentArithmetic.recompose (storeChildren assignments)
  ⟨StoredAssignmentArithmetic.view parent.value, parent.work + storageClock assignments⟩

/-- Stored recomposition returns the production PiDEC parent assignment. -/
theorem recompose_value (storageClock : StorageClock inst) (assignments : Children inst) :
    (recompose inst storageClock assignments).value = (dec inst).recomposeAssignment assignments := by
  exact storedRecompose_value assignments

/-- The existing arithmetic clock plus bounded storage gives this declared
recomposition cost; the bound makes no machine-runtime claim. -/
theorem recompose_work_le (storageClock : StorageClock inst) (storageBound : Nat)
    (bounded : ∀ assignments, storageClock assignments ≤ storageBound)
    (assignments : Children inst) :
    (recompose inst storageClock assignments).work ≤
      116 * (PaperAlgebra.FullShape inst.logicalWidth inst.publicFits).carrierWidth + 613 + storageBound := by
  exact Nat.add_le_add (storedRecompose_work_le (storeChildren assignments)) (bounded assignments)

private def suffixCheck (batch : Batch inst) (vector : Coins inst) (reply : Reply inst) : Bool :=
  let attempt := PaperWeakSuffix.attempt (rlc inst) batch vector reply
  letI := PaperAlgebra.piDecDecision inst.ajtai attempt
  decide (PiDEC.PaperVerifier.Accepted (dec inst) (publicSplit inst)
    (evaluationArity inst) attempt) &&
    (List.finRange productionGlobalParams.k).all fun child =>
      Nifs.ClaimCheck.check inst.ajtai
        (PiDEC.PaperVerifier.children (publicSplit inst) attempt child)
        (reply.assignments child)

/-- The public attempt and all child openings are checked before the actual
stored recomposition can be returned. No reply field supplies the parent. -/
def suffixProgram (batch : Batch inst) (checkClock : CheckClock inst) (storageClock : StorageClock inst) :
    PaperWeakSuffix.Program (arity := PaperProfile.arity) (rlc inst) where
  check := fun vector reply => ⟨suffixCheck inst batch vector reply, checkClock vector reply⟩
  recompose := recompose inst storageClock

/-- Exact suffix value correctness for every challenge vector and raw reply,
including rejection of an invalid public attempt or any invalid child. -/
theorem suffixProgram_correct (batch : Batch inst) (checkClock : CheckClock inst)
    (storageClock : StorageClock inst) :
    PaperWeakSuffix.Correct (rlc inst) batch (dec inst) (publicSplit inst)
      (evaluationArity inst) (suffixProgram inst batch checkClock storageClock) := by
  constructor
  · intro vector reply
    simp only [suffixProgram, suffixCheck, Bool.and_eq_true, decide_eq_true_eq,
      List.all_eq_true, List.mem_finRange, forall_const, Nifs.ClaimCheck.check_eq_true_iff]
  · exact recompose_value inst storageClock

/-- The coordinate search checks the exact computed CE parent, including its
stage-dependent norm, public input, commitment, Pad and matrix evaluations. -/
def parentChecker (batch : Batch inst) (parentClock : ParentClock inst)
    (response : ParentResponse inst) : CheckResult :=
  ⟨Nifs.ClaimCheck.check inst.ajtai (response.output (rlc inst) batch)
    response.assignment, parentClock response⟩

/-- The executable parent check accepts exactly the existing CE success predicate. -/
theorem parentChecker_spec (batch : Batch inst) (parentClock : ParentClock inst)
    (response : ParentResponse inst) :
    (parentChecker inst batch parentClock response).accepted = true ↔
      response.Success (PaperAlgebra.semantics inst.ajtai) productionGlobalParams (rlc inst) batch := by
  dsimp only [parentChecker, Response.Success]
  exact Nifs.ClaimCheck.check_eq_true_iff
    (logicalWidth := inst.logicalWidth) (publicFits := inst.publicFits) inst.ajtai
    (response.output (rlc inst) batch) response.assignment

/-- Wrap any actual resumed adversary call with the selected checks. Only
clock bounds and the original call/check first moment are hypotheses. -/
def algorithm {Tape : Type*} (batch : Batch inst) (tapes : PMF Tape)
    (rawCall : PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) (rlc inst))
    (checkClock : CheckClock inst) (storageClock : StorageClock inst) (parentClock : ParentClock inst)
    (storageBound : Nat) (storageBounded : ∀ assignments, storageClock assignments ≤ storageBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (PaperWeakOracle.baseWork (rlc inst)
        (suffixProgram inst batch checkClock storageClock) rawCall vector tape : ℝ)) :
    PaperWeakAlgorithm.Algorithm Tape (rlc inst) batch (dec inst)
      (publicSplit inst) (evaluationArity inst) where
  tapes := tapes
  rawCall := rawCall
  suffixProgram := suffixProgram inst batch checkClock storageClock
  suffixCorrect := suffixProgram_correct inst batch checkClock storageClock
  recomposeBound := 116 * (PaperAlgebra.FullShape inst.logicalWidth inst.publicFits).carrierWidth +
      613 + storageBound
  recomposeBounded := recompose_work_le inst storageClock storageBound storageBounded
  baseSummable := baseSummable
  parentChecker := parentChecker inst batch parentClock
  parentChecker_spec := parentChecker_spec inst batch parentClock

private theorem inputBatch_ext
    {Structure PublicInput Point Evaluation Commitment : Type*}
    {params : GlobalParams} {arity : BatchArity params}
    (left right : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
    (systems : left.system = right.system) (points : left.point = right.point)
    (inputs : left.inputs = right.inputs)
    (counts : left.evaluationCount = right.evaluationCount) : left = right := by
  cases left
  cases right
  cases systems
  cases points
  cases inputs
  cases counts
  rfl

private theorem relationSource_eq {width : Nat}
    {fits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth width}
    (selectedRelation : ProductionKey.LogicalRelation width fits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := width) (publicFits := fits)) :
    Lifecycle.PiRLC.v1_1.InputBinding.relationSource selectedRelation =
      (ProductionKey.key selectedRelation ajtai).relationSource := by
  rfl

section Provider

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input) (contexts : PMF Context)
  (firstPhase : Context → InteractivePrefix.Prover State productionShape 9)

/-- The literal receipt supplies the new point and all 17 evaluation claims.
Commitments, public inputs and the relation come from the selected statement.
The empty certificate is only a batch view; this constructor makes no
acceptance claim and does not alter the checked prefix receipt. -/
def batchAt (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) : Batch inst where
  system := Lifecycle.PiRLC.v1_1.InputBinding.relationSource inst.relation
  point := coins.roundPoint
  inputs := fun coordinate =>
    (PiCCSStoredWitnessCheck.statement inst (inputs context)).publicOutput
      { coins := coins, response := { rounds := ⟨[]⟩, fullOutput := output } }
      (Fin.cast (by rfl) coordinate)
  sameSystem := fun _ => rfl
  samePoint := fun _ => rfl
  evaluationCount := 1
  evaluationsSize := fun _ => rfl

/-- This computable batch is the exact one expected by the existing
supported continuation, for every public receipt. -/
theorem batchAt_eq (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) :
    batchAt inst inputs context coins output =
      Nifs.WeakExtraction.batchForOutput inst.relation inst.ajtai
        (inst.running (inputs context))
        (inst.fresh (inputs context)) coins output := by
  apply inputBatch_ext
  · dsimp only [batchAt, Nifs.WeakExtraction.batchForOutput,
      PaperStrongInterface.piRlcBatchForProbe]
    exact relationSource_eq inst.relation inst.ajtai
  · rfl
  · funext coordinate
    simp only [batchAt, Nifs.WeakExtraction.batchForOutput,
      PaperStrongInterface.piRlcBatchForProbe, PiCCSStoredWitnessCheck.statement_eq_key]
    rfl
  · rfl

/-- Construct the selected checked continuation at one literal receipt.
The call, tape and clocks are supplied independently of any context PMF.
The exact batch conversion supplies the existing continuation type. -/
def continuationAt (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape)
    (tapes : PMF Tape)
    (rawCall : PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) (rlc inst))
    (checkClock : CheckClock inst) (storageClock : StorageClock inst) (parentClock : ParentClock inst)
    (storageBound : Nat) (storageBounded : ∀ assignments, storageClock assignments ≤ storageBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (PaperWeakOracle.baseWork (rlc inst)
        (suffixProgram inst (batchAt inst inputs context coins output) checkClock storageClock)
        rawCall vector tape : ℝ)) :
    Nifs.WeakExtraction.Continuation Tape inst.relation inst.ajtai
      (inst.running (inputs context)) (inst.fresh (inputs context)) coins output :=
  Eq.mp (congrArg
    (fun batch : Batch inst => PaperWeakAlgorithm.Algorithm Tape (rlc inst) batch (dec inst)
        (publicSplit inst) (evaluationArity inst))
    (batchAt_eq inst inputs context coins output))
    (algorithm inst (batchAt inst inputs context coins output) tapes rawCall checkClock storageClock
      parentClock storageBound storageBounded baseSummable)

/-- Each positive checked receipt keeps its captured state, raw call and tape
law. The adapter supplies all semantic checks and their correctness itself. -/
def provider
    (tapes : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → PMF Tape)
    (rawCall : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state →
        PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) (rlc inst))
    (checkClock : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → CheckClock inst)
    (storageClock : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → StorageClock inst)
    (parentClock : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → ParentClock inst)
    (storageBound : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → Nat)
    (storageBounded : ∀ context coins output state support assignments,
      storageClock context coins output state support assignments ≤
        storageBound context coins output state support)
    (baseSummable : ∀ context coins output state support vector, Summable fun tape =>
      (tapes context coins output state support tape).toReal *
        (PaperWeakOracle.baseWork (rlc inst)
          (suffixProgram inst (batchAt inst inputs context coins output)
            (checkClock context coins output state support)
            (storageClock context coins output state support))
          (rawCall context coins output state support) vector tape : ℝ)) :
    Nifs.SupportedContinuation.Provider Tape inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context)) contexts firstPhase :=
  fun context coins output state support =>
    continuationAt inst inputs context coins output
        (tapes context coins output state support) (rawCall context coins output state support)
        (checkClock context coins output state support) (storageClock context coins output state support)
        (parentClock context coins output state support) (storageBound context coins output state support)
        (storageBounded context coins output state support) (baseSummable context coins output state support)

end Provider

end NightstreamFPrime.Export.Stage1.NifsExtractionProvider
