import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.Baseline

/-!
Removal witness for `Pi_CCS` carried-evaluation truth.

Assurance tier: model-level.

Owns: one explicit nonzero prior claim over zero matrices/assignments, the
matching public running CE arrays, exact `Pi_DEC` recomposition of those arrays
into a checked incoming parent, weakened-plan acceptance, and independent
semantic rejection.

Does not own: SumCheck or Fiat--Shamir soundness, commitment binding security,
Rust/R1CS refinement, physical rows, costs, or row removal.

Emits constraints: no.

Authority boundary: the counterexample deliberately remains self-consistent
at the public `Pi_DEC` boundary. It fails only because the claimed coefficient
is compared with the evaluation derived from authoritative matrices and
assignments. A re-digested or recomposed false claim is not truth.

| Family | Stage path | Counterexample | Lean owner |
|---|---|---|---|
| paper relation | `fprime.active.nifs.pi_ccs.relation.carried_evaluations.necessity` | set coordinate `(0,0,0)` to one, recompose public parent evaluations | `carriedEvaluations_necessary` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

/-- First running/matrix/lane carried coordinate. -/
def forgedCarriedCoordinate : CarriedCoordinate Sources.shape.paperShape where
  running := ⟨0, by decide⟩
  matrix := ⟨0, by decide⟩
  coefficient := ⟨0, by decide⟩

/-- One false public claim; every other claim remains zero. -/
def forgedClaim (coordinate : CarriedCoordinate Sources.shape.paperShape) : K :=
  if coordinate = forgedCarriedCoordinate then K.one else K.zero

/-- Preserve matrices, assignments, and prior point; change only the public
carried-claim function. -/
def forgedCarriedData : PiCCS.SplitNc.Sources.Data Sources.shape :=
  { Sources.data with claimedCoefficient := forgedClaim }

@[simp] theorem forgedCarriedData_freshAssignment
    (source : Fin Sources.shape.freshCount) :
    forgedCarriedData.freshAssignment source =
      Sources.data.freshAssignment source := by
  rfl

@[simp] theorem forgedCarriedData_runningAssignment
    (source : Fin Sources.shape.runningCount) :
    forgedCarriedData.runningAssignments source =
      Sources.data.runningAssignments source := by
  rfl

theorem forgedCarried_runningZero
    (source : Fin Sources.shape.runningCount) :
    forgedCarriedData.runningAssignments source = Context.zeroAssignment := by
  rfl

theorem forgedCarried_freshCcsHolds :
    Semantics.Paper.FreshCcsHolds forgedCarriedData := by
  simpa [forgedCarriedData] using Sources.freshCcsHolds

theorem forgedCarried_allSourceNormsHold :
    Semantics.Paper.AllSourceNormsHold forgedCarriedData := by
  simpa [forgedCarriedData] using Sources.allSourceNormsHold

theorem forgedCarried_claim_one :
    forgedCarriedData.carriedData.claimedCoefficient
      forgedCarriedCoordinate = K.one := by
  simp [forgedCarriedData, PiCCS.SplitNc.Sources.Data.carriedData, forgedClaim]

theorem forgedCarried_computed_zero :
    CarriedEvaluationResidual.computedCoefficient
        ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
        forgedCarriedData.carriedData forgedCarriedCoordinate = K.zero := by
  apply
    CarriedEvaluationResidual.computedCoefficient_eq_zero_of_assignment_zero
      ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws K.embed
      ConcreteCarrier.embed_zero forgedCarriedData.carriedData
      forgedCarriedCoordinate
  exact forgedCarried_runningZero forgedCarriedCoordinate.running

/-- The explicit nonzero claim cannot equal the independently derived zero
matrix evaluation. -/
theorem forgedCarried_not_evaluations :
    ¬Semantics.Paper.CarriedEvaluationsHold forgedCarriedData := by
  intro holds
  have selected := holds forgedCarriedCoordinate
  unfold CarriedEvaluationResidual.EvaluationClaimHolds at selected
  have impossible : K.one = K.zero :=
    forgedCarried_claim_one.symm.trans
      (selected.trans forgedCarried_computed_zero)
  exact (by decide : K.one ≠ K.zero) impossible

/-! ## Public source product and incoming recomposition -/

/-- Running CE statements carry exactly the forged public prior-claim arrays;
all other fields remain the baseline children. -/
def forgedCarriedChildren (child : Fin productionGlobalParams.k) :
    Phi81Relation.CEStatement
      (RelationShape Sources.shape Context.publicRingColumns Context.publicFits)
      (CommitmentValue Context.verifierRows) :=
  { Context.children child with
    evaluations := InputAuthority.priorEvaluations forgedCarriedData child }

/-- Parent evaluation array is computed by exact verifier-owned radix
recomposition of the forged children. -/
def forgedCarriedParent :
    Phi81Relation.CEStatement
      (RelationShape Sources.shape Context.publicRingColumns Context.publicFits)
      (CommitmentValue Context.verifierRows) :=
  { Context.parent with
    evaluations :=
      (ConcretePhi81.decAlgebra Context.key).recomposeEvaluations
        (fun child => (forgedCarriedChildren child).evaluations) }

/-- Preserve the fresh source and replace only the running evaluation arrays. -/
def forgedCarriedInput :
    SourceProduct Sources.shape Context.publicRingColumns Context.publicFits
      (CommitmentValue Context.verifierRows) productionGlobalParams
      FixedActive.arity :=
  { Context.context.input with running := forgedCarriedChildren }

/-- Complete context whose polynomial input, public source product, and
incoming parent all carry the same forged claim family. -/
def forgedCarriedContext :
    FixedActive.Context Sources.shape Unit Context.publicRingColumns
      Context.publicFits Context.verifierRows :=
  { Context.context with
    input := forgedCarriedInput
    runningParent := some forgedCarriedParent
    piCcsInput := PublicInput.ofSources forgedCarriedData }

@[simp] theorem forgedCarriedContext_input_fresh
    (source : Fin FixedActive.arity.freshCount) :
    forgedCarriedContext.input.fresh source = Context.freshStatement := by
  rfl

@[simp] theorem forgedCarriedContext_input_running
    (source : Fin (FixedActive.arity.mode.count productionGlobalParams)) :
    forgedCarriedContext.input.running source = forgedCarriedChildren source := by
  rfl

/-- The public source product is bound field-for-field to the forged source
data even though the carried semantic equation is false. -/
theorem forgedCarried_sourceBound :
    InputAuthority.BoundToSources Context.publicRingColumns Context.publicFits
      (ConcretePhi81.commit Context.key) forgedCarriedData Context.alignment
      forgedCarriedContext.input := by
  refine { fresh := ?_, running := ?_ }
  · intro source
    have original := Context.sourceBound.fresh source
    refine {
      constraintSystem := ?_
      commitment := ?_
      publicInput := ?_
      stage := ?_
    }
    · rw [forgedCarriedContext_input_fresh]
      exact original.constraintSystem
    · rw [forgedCarriedContext_input_fresh,
        forgedCarriedData_freshAssignment]
      exact original.commitment
    · rw [forgedCarriedContext_input_fresh,
        forgedCarriedData_freshAssignment]
      exact original.publicInput
    · rw [forgedCarriedContext_input_fresh]
      exact original.stage
  · intro source
    have original := Context.sourceBound.running source
    refine {
      constraintSystem := ?_
      commitment := ?_
      publicInput := ?_
      point := ?_
      evaluations := ?_
      stage := ?_
    }
    · rw [forgedCarriedContext_input_running]
      exact original.constraintSystem
    · rw [forgedCarriedContext_input_running,
        forgedCarriedData_runningAssignment]
      exact original.commitment
    · rw [forgedCarriedContext_input_running,
        forgedCarriedData_runningAssignment]
      exact original.publicInput
    · rw [forgedCarriedContext_input_running]
      exact original.point
    · rfl
    · rw [forgedCarriedContext_input_running]
      exact original.stage

/-- Exact public `Pi_DEC` acceptance of the recomposed forged evaluation
arrays. -/
theorem forgedCarried_piDecAccepted :
    PiDEC.Accepted (ConcretePhi81.decAlgebra Context.key)
      (ConcretePhi81.RunningAuthority.attempt forgedCarriedContext
        FixedActive.arity_mode forgedCarriedParent) := by
  have baselineComplete :=
    PiDEC.complete (ConcretePhi81.semantics Context.key)
      productionGlobalParams (ConcretePhi81.decAlgebra Context.key)
      Context.parent Context.zeroAssignment rfl Context.parentHolds
  have baselineAccepted := baselineComplete.1
  refine {
    parentCombined := rfl
    childFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    samePoint := fun _ => rfl
    commitmentEquation := ?_
    publicInputEquation := ?_
    evaluationEquation := rfl
  }
  · change Context.parent.commitment =
      (ConcretePhi81.decAlgebra Context.key).recomposeCommitment
        (fun child => (Context.children child).commitment)
    simpa only [Context.children] using baselineAccepted.commitmentEquation
  · change Context.parent.publicInput =
      (ConcretePhi81.decAlgebra Context.key).recomposePublicInput
        (fun child => (Context.children child).publicInput)
    simpa only [Context.children] using baselineAccepted.publicInputEquation

/-- Strict incoming `Pi_DEC` accepts because the forged parent evaluation is
exactly the recomposition of the forged child arrays; all other recomposed
fields are unchanged from the checked baseline. -/
theorem forgedCarried_runningAccepted :
    ConcretePhi81.RunningAuthority.Accepted forgedCarriedContext := by
  exact .active {
    active := FixedActive.arity_mode
    parent := forgedCarriedParent
    parentBound := rfl
    piDec := forgedCarried_piDecAccepted
  }

def forgedCarriedWitness : SemanticFold.Witness forgedCarriedContext where
  point := Context.zeroPoint Sources.shape.rowVariables
  challenges := Context.zeroChallenges

def forgedCarriedCandidate : BaselineCandidate := {
  context := forgedCarriedContext
  data := forgedCarriedData
  point := forgedCarriedWitness.point
  challenges := forgedCarriedWitness.challenges
  parent := SemanticFold.parentOf forgedCarriedContext forgedCarriedData
    forgedCarriedWitness
  children := SemanticFold.childrenOf forgedCarriedContext forgedCarriedData
    forgedCarriedWitness
}

theorem forgedCarried_challengesValid :
    SemanticFold.ChallengesValid forgedCarriedContext forgedCarriedWitness := by
  intro coordinate
  simpa [forgedCarriedContext] using
    (Context.samplerBound ()).challengeValid coordinate

/-- Every retained leaf accepts the self-consistently recomposed false carried
claim. -/
theorem forgedCarriedWeakened :
    CheckPlan.Accepts baselineSemantics
      (CheckPlan.without SemanticFold.ObligationPlan.checks
        .carriedEvaluations)
      forgedCarriedCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact forgedCarried_freshCcsHolds
  | allSourceNorm => exact forgedCarried_allSourceNormsHold
  | carriedEvaluations => exact (retained rfl).elim
  | polynomialInput => rfl
  | sourceProduct => exact forgedCarried_sourceBound
  | incomingAuthority => exact forgedCarried_runningAccepted
  | challengeStrongSet => exact forgedCarried_challengesValid
  | parentExact => rfl
  | childrenExact => rfl

theorem forgedCarriedRejected :
    ¬baselineTarget forgedCarriedCandidate := by
  intro realized
  exact forgedCarried_not_evaluations realized.paper.2.2

/-- Closed inclusion-necessity of binding every carried claim to its
independently derived matrix evaluation. -/
theorem carriedEvaluations_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .carriedEvaluations :=
  ⟨forgedCarriedCandidate, forgedCarriedWeakened, forgedCarriedRejected⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality
