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
open PiDECInputCheck (relation logicalWidth publicFits)
open Poseidon2HashChainV1Setup (productionAjtaiKey)

abbrev rlc := PaperAlgebra.piRlcAlgebra productionAjtaiKey
abbrev dec := PaperAlgebra.piDecAlgebra productionAjtaiKey
abbrev publicSplit := PaperAlgebra.publicInputSplit productionAjtaiKey
abbrev evaluationArity := PaperAlgebra.evaluationArity productionAjtaiKey
abbrev Assignment := PaperAlgebra.Assignment
  (logicalWidth := logicalWidth) (publicFits := publicFits)
abbrev Batch := InputBatch (PaperAlgebra.Structure logicalWidth)
  (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
  PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment
  productionGlobalParams PaperProfile.arity
abbrev Coins := Fin PaperProfile.arity.total → Challenge rlc
abbrev Reply := PaperWeakSuffix.Reply Assignment PaperAlgebra.Evaluation
  PaperAlgebra.Commitment productionGlobalParams
abbrev Children := Fin productionGlobalParams.k → Assignment
abbrev ParentResponse := Response Assignment RingF productionGlobalParams PaperProfile.arity
abbrev CheckClock := Coins → Reply → Nat
abbrev StorageClock := Children → Nat
abbrev ParentClock := ParentResponse → Nat

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
def recompose (storageClock : StorageClock) (assignments : Children) : Result Assignment :=
  let parent := StoredAssignmentArithmetic.recompose (storeChildren assignments)
  ⟨StoredAssignmentArithmetic.view parent.value, parent.work + storageClock assignments⟩

/-- Stored recomposition returns the production PiDEC parent assignment. -/
theorem recompose_value (storageClock : StorageClock) (assignments : Children) :
    (recompose storageClock assignments).value = dec.recomposeAssignment assignments := by
  exact storedRecompose_value assignments

/-- The existing arithmetic clock plus bounded storage gives this declared
recomposition cost; the bound makes no machine-runtime claim. -/
theorem recompose_work_le (storageClock : StorageClock) (storageBound : Nat)
    (bounded : ∀ assignments, storageClock assignments ≤ storageBound)
    (assignments : Children) :
    (recompose storageClock assignments).work ≤
      116 * (PaperAlgebra.FullShape logicalWidth publicFits).carrierWidth + 613 + storageBound := by
  exact Nat.add_le_add (storedRecompose_work_le (storeChildren assignments)) (bounded assignments)

private def suffixCheck (batch : Batch) (vector : Coins) (reply : Reply) : Bool :=
  let attempt := PaperWeakSuffix.attempt rlc batch vector reply
  letI := PaperAlgebra.piDecDecision productionAjtaiKey attempt
  decide (PiDEC.PaperVerifier.Accepted dec publicSplit
    evaluationArity attempt) &&
    (List.finRange productionGlobalParams.k).all fun child =>
      Nifs.ClaimCheck.check productionAjtaiKey
        (PiDEC.PaperVerifier.children publicSplit attempt child)
        (reply.assignments child)

/-- The public attempt and all child openings are checked before the actual
stored recomposition can be returned. No reply field supplies the parent. -/
def suffixProgram (batch : Batch) (checkClock : CheckClock) (storageClock : StorageClock) :
    PaperWeakSuffix.Program (arity := PaperProfile.arity) rlc where
  check := fun vector reply => ⟨suffixCheck batch vector reply, checkClock vector reply⟩
  recompose := recompose storageClock

/-- Exact suffix value correctness for every challenge vector and raw reply,
including rejection of an invalid public attempt or any invalid child. -/
theorem suffixProgram_correct (batch : Batch) (checkClock : CheckClock) (storageClock : StorageClock) :
    PaperWeakSuffix.Correct rlc batch dec publicSplit
      evaluationArity (suffixProgram batch checkClock storageClock) := by
  constructor
  · intro vector reply
    simp only [suffixProgram, suffixCheck, Bool.and_eq_true, decide_eq_true_eq,
      List.all_eq_true, List.mem_finRange, forall_const, Nifs.ClaimCheck.check_eq_true_iff]
  · exact recompose_value storageClock

/-- The coordinate search checks the exact computed CE parent, including its
stage-dependent norm, public input, commitment, Pad and matrix evaluations. -/
def parentChecker (batch : Batch) (parentClock : ParentClock) (response : ParentResponse) : CheckResult :=
  ⟨Nifs.ClaimCheck.check productionAjtaiKey (response.output rlc batch)
    response.assignment, parentClock response⟩

/-- The executable parent check accepts exactly the existing CE success predicate. -/
theorem parentChecker_spec (batch : Batch) (parentClock : ParentClock) (response : ParentResponse) :
    (parentChecker batch parentClock response).accepted = true ↔
      response.Success (PaperAlgebra.semantics productionAjtaiKey) productionGlobalParams rlc batch := by
  dsimp only [parentChecker, Response.Success]
  exact Nifs.ClaimCheck.check_eq_true_iff
    (logicalWidth := logicalWidth) (publicFits := publicFits) productionAjtaiKey
    (response.output rlc batch) response.assignment

/-- Wrap any actual resumed adversary call with the selected checks. Only
clock bounds and the original call/check first moment are hypotheses. -/
def algorithm {Tape : Type*} (batch : Batch) (tapes : PMF Tape)
    (rawCall : PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) rlc)
    (checkClock : CheckClock) (storageClock : StorageClock) (parentClock : ParentClock)
    (storageBound : Nat) (storageBounded : ∀ assignments, storageClock assignments ≤ storageBound)
    (baseSummable : ∀ vector, Summable fun tape =>
      (tapes tape).toReal * (PaperWeakOracle.baseWork rlc
        (suffixProgram batch checkClock storageClock) rawCall vector tape : ℝ)) :
    PaperWeakAlgorithm.Algorithm Tape rlc batch dec
      publicSplit evaluationArity where
  tapes := tapes
  rawCall := rawCall
  suffixProgram := suffixProgram batch checkClock storageClock
  suffixCorrect := suffixProgram_correct batch checkClock storageClock
  recomposeBound := 116 * (PaperAlgebra.FullShape logicalWidth publicFits).carrierWidth + 613 + storageBound
  recomposeBounded := recompose_work_le storageClock storageBound storageBounded
  baseSummable := baseSummable
  parentChecker := parentChecker batch parentClock
  parentChecker_spec := parentChecker_spec batch parentClock

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
    (output : FullOutputCoordinates.FullOutput K productionShape) : Batch where
  system := Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation
  point := coins.roundPoint
  inputs := fun coordinate =>
    (PiCCSStoredWitnessCheck.statement (inputs context)).publicOutput
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
    batchAt inputs context coins output =
      Nifs.WeakExtraction.batchForOutput relation productionAjtaiKey
        (PiCCSInputCheck.running (inputs context))
        (PiCCSInputCheck.fresh (inputs context)) coins output := by
  apply inputBatch_ext
  · dsimp only [batchAt, Nifs.WeakExtraction.batchForOutput,
      PaperStrongInterface.piRlcBatchForProbe]
    exact relationSource_eq relation productionAjtaiKey
  · rfl
  · funext coordinate
    simp only [batchAt, Nifs.WeakExtraction.batchForOutput,
      PaperStrongInterface.piRlcBatchForProbe, PiCCSStoredWitnessCheck.statement_eq_key]
    rfl
  · rfl

/-- Each positive checked receipt keeps its captured state, raw call and tape
law. The adapter supplies all semantic checks and their correctness itself. -/
def provider
    (tapes : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → PMF Tape)
    (rawCall : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state →
        PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) rlc)
    (checkClock : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → CheckClock)
    (storageClock : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → StorageClock)
    (parentClock : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → ParentClock)
    (storageBound : ∀ context coins output state,
      Nifs.SupportedContinuation.Supported contexts firstPhase context coins output state → Nat)
    (storageBounded : ∀ context coins output state support assignments,
      storageClock context coins output state support assignments ≤
        storageBound context coins output state support)
    (baseSummable : ∀ context coins output state support vector, Summable fun tape =>
      (tapes context coins output state support tape).toReal *
        (PaperWeakOracle.baseWork rlc
          (suffixProgram (batchAt inputs context coins output)
            (checkClock context coins output state support)
            (storageClock context coins output state support))
          (rawCall context coins output state support) vector tape : ℝ)) :
    Nifs.SupportedContinuation.Provider Tape relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context)) contexts firstPhase :=
  fun context coins output state support =>
    Eq.mp (congrArg
      (fun batch : Batch => PaperWeakAlgorithm.Algorithm Tape rlc batch dec publicSplit evaluationArity)
      (batchAt_eq inputs context coins output))
      (algorithm (batchAt inputs context coins output)
        (tapes context coins output state support) (rawCall context coins output state support)
        (checkClock context coins output state support) (storageClock context coins output state support)
        (parentClock context coins output state support) (storageBound context coins output state support)
        (storageBounded context coins output state support) (baseSummable context coins output state support))

end Provider

end NightstreamFPrime.Export.Stage1.NifsExtractionProvider
