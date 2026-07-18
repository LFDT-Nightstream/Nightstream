import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs.CanonicalContext

/-!
Removal witnesses for independent `Pi_CCS` paper-relation obligations.

Assurance tier: model-level.

Owns: explicit source-data counterexamples for fresh CCS truth and strict
all-source norm, their complete bound NIFS contexts, weakened-plan acceptance,
and rejection by the independent semantic target.

Does not own: carried-evaluation necessity, SumCheck/transcript refinement,
Rust/R1CS correspondence, physical rows, costs, security reduction, or row
removal.

Emits constraints: no.

Authority boundary: malformed relations and assignments are defined directly
in the independent source language. `CanonicalContext` computes their public
statements and checked parent; no circuit or accepted verifier execution
defines whether the paper relation holds.

| Family | Stage path | Counterexample | Lean owner |
|---|---|---|---|
| paper relation | `fprime.active.nifs.pi_ccs.relation.fresh_ccs.necessity` | replace the constraint polynomial by constant one | `freshCcs_necessary` |
| paper relation | `fprime.active.nifs.pi_ccs.relation.all_source_norm.necessity` | use a magnitude-two fresh assignment under the zero polynomial | `allSourceNorm_necessary` |
| paper relation | `fprime.active.nifs.pi_ccs.relation.carried_evaluations.necessity` | owned by the sibling family module | `PiCcs.CarriedEvaluations` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

/-! ## Fresh-CCS removal witness -/

/-- Constant-one monomial at the exact three-matrix profile. -/
def falseCcsMonomial :
    CCSResidualTable.Monomial F Sources.shape.matrixCount where
  coefficient := 1
  exponents := fun _ => 0

/-- Constant-one relation polynomial, which cannot vanish at any row. -/
def falseCcsPolynomial :
    CCSResidualTable.ConstraintPolynomial F Sources.shape.matrixCount where
  degreeBound := 1
  terms := [falseCcsMonomial]
  termsBelowDegree := by
    intro term member
    simp only [List.mem_singleton] at member
    subst term
    decide

/-- Preserve every independent source field except the relation polynomial. -/
def falseCcsData : PiCCS.SplitNc.Sources.Data Sources.shape :=
  { Sources.data with constraintPolynomial := falseCcsPolynomial }

theorem falseCcs_runningZero
    (source : Fin Sources.shape.runningCount) :
    falseCcsData.runningAssignments source = Context.zeroAssignment := by
  rfl

theorem falseCcs_allSourceNormsHold :
    Semantics.Paper.AllSourceNormsHold falseCcsData := by
  simpa [falseCcsData] using Sources.allSourceNormsHold

theorem falseCcs_carriedEvaluationsHold :
    Semantics.Paper.CarriedEvaluationsHold falseCcsData := by
  simpa [falseCcsData] using Sources.carriedEvaluationsHold

/-- Constant one contradicts the required zero residual at the first source
and first Boolean row. -/
theorem falseCcs_not_freshCcs :
    ¬Semantics.Paper.FreshCcsHolds falseCcsData := by
  intro holds
  have atSource := holds PiCcs.CanonicalContext.firstFresh
  have atVertex := atSource (BooleanVertex.cons false BooleanVertex.nil)
  change (1 : F) = 0 at atVertex
  exact (by decide : (1 : F) ≠ 0) atVertex

def falseCcsContext := PiCcs.CanonicalContext.context falseCcsData

def falseCcsWitness : SemanticFold.Witness falseCcsContext where
  point := baselineWitness.point
  challenges := baselineWitness.challenges

def falseCcsCandidate : BaselineCandidate := {
  context := falseCcsContext
  data := falseCcsData
  point := falseCcsWitness.point
  challenges := falseCcsWitness.challenges
  parent := SemanticFold.parentOf falseCcsContext falseCcsData falseCcsWitness
  children :=
    SemanticFold.childrenOf falseCcsContext falseCcsData falseCcsWitness
}

theorem falseCcs_challengesValid :
    SemanticFold.ChallengesValid falseCcsContext falseCcsWitness := by
  intro coordinate
  simpa [falseCcsContext, PiCcs.CanonicalContext.context] using
    (Context.samplerBound ()).challengeValid coordinate

/-- Every retained leaf accepts the constant-one relation mutation. -/
theorem falseCcsWeakened :
    CheckPlan.Accepts baselineSemantics
      (CheckPlan.without SemanticFold.ObligationPlan.checks .freshCcs)
      falseCcsCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact (retained rfl).elim
  | allSourceNorm => exact falseCcs_allSourceNormsHold
  | carriedEvaluations => exact falseCcs_carriedEvaluationsHold
  | polynomialInput => rfl
  | sourceProduct =>
      exact PiCcs.CanonicalContext.sourceBound falseCcsData falseCcs_runningZero
        falseCcs_carriedEvaluationsHold
  | incomingAuthority =>
      exact PiCcs.CanonicalContext.runningAccepted falseCcsData
  | challengeStrongSet => exact falseCcs_challengesValid
  | parentExact => rfl
  | childrenExact => rfl

theorem falseCcsRejected : ¬baselineTarget falseCcsCandidate := by
  intro realized
  exact falseCcs_not_freshCcs realized.paper.1

/-- Closed inclusion-necessity of fresh CCS relation truth. -/
theorem freshCcs_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .freshCcs :=
  ⟨falseCcsCandidate, falseCcsWeakened, falseCcsRejected⟩

/-! ## All-source-norm removal witness -/

/-- Empty sparse polynomial: the fresh CCS residual is identically zero. -/
def zeroCcsPolynomial :
    CCSResidualTable.ConstraintPolynomial F Sources.shape.matrixCount where
  degreeBound := 1
  terms := []
  termsBelowDegree := by simp

/-- Magnitude-two fresh assignment at the first logical coordinate. -/
def highNormFreshAssignment :
    PaperLinearAlgebra.Assignment F Sources.shape.logicalWidth :=
  fun column => if column.val = 0 then 2 else 0

/-- Combine the magnitude-two fresh assignment with an identically zero CCS
polynomial; all running data and carried claims remain unchanged. -/
def highNormData : PiCCS.SplitNc.Sources.Data Sources.shape :=
  { Sources.data with
    constraintPolynomial := zeroCcsPolynomial
    freshAssignments := fun _ => highNormFreshAssignment }

theorem highNorm_runningZero
    (source : Fin Sources.shape.runningCount) :
    highNormData.runningAssignments source = Context.zeroAssignment := by
  rfl

theorem highNorm_freshCcsHolds :
    Semantics.Paper.FreshCcsHolds highNormData := by
  intro source vertex
  rfl

theorem highNorm_carriedEvaluationsHold :
    Semantics.Paper.CarriedEvaluationsHold highNormData := by
  simpa [highNormData] using Sources.carriedEvaluationsHold

/-- The first complete-carrier coordinate has centered magnitude exactly two,
violating the strict paper bound. -/
theorem highNorm_not_allSourceNorms :
    ¬Semantics.Paper.AllSourceNormsHold highNormData := by
  intro holds
  have atCoordinate := holds
    (PiCCS.SplitNc.Sources.Data.freshIndex
      PiCcs.CanonicalContext.firstFresh)
    (⟨0, by decide⟩ : Fin Sources.shape.carrierWidth)
  change centeredMagnitude (2 : F) < 2 at atCoordinate
  exact (by decide : ¬centeredMagnitude (2 : F) < 2) atCoordinate

def highNormContext := PiCcs.CanonicalContext.context highNormData

def highNormWitness : SemanticFold.Witness highNormContext where
  point := baselineWitness.point
  challenges := baselineWitness.challenges

def highNormCandidate : BaselineCandidate := {
  context := highNormContext
  data := highNormData
  point := highNormWitness.point
  challenges := highNormWitness.challenges
  parent := SemanticFold.parentOf highNormContext highNormData highNormWitness
  children :=
    SemanticFold.childrenOf highNormContext highNormData highNormWitness
}

theorem highNorm_challengesValid :
    SemanticFold.ChallengesValid highNormContext highNormWitness := by
  intro coordinate
  simpa [highNormContext, PiCcs.CanonicalContext.context] using
    (Context.samplerBound ()).challengeValid coordinate

/-- Every retained leaf accepts the magnitude-two source mutation. -/
theorem highNormWeakened :
    CheckPlan.Accepts baselineSemantics
      (CheckPlan.without SemanticFold.ObligationPlan.checks .allSourceNorm)
      highNormCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact highNorm_freshCcsHolds
  | allSourceNorm => exact (retained rfl).elim
  | carriedEvaluations => exact highNorm_carriedEvaluationsHold
  | polynomialInput => rfl
  | sourceProduct =>
      exact PiCcs.CanonicalContext.sourceBound highNormData highNorm_runningZero
        highNorm_carriedEvaluationsHold
  | incomingAuthority =>
      exact PiCcs.CanonicalContext.runningAccepted highNormData
  | challengeStrongSet => exact highNorm_challengesValid
  | parentExact => rfl
  | childrenExact => rfl

theorem highNormRejected : ¬baselineTarget highNormCandidate := by
  intro realized
  exact highNorm_not_allSourceNorms realized.paper.2.1

/-- Closed inclusion-necessity of the strict all-source norm obligation. -/
theorem allSourceNorm_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .allSourceNorm :=
  ⟨highNormCandidate, highNormWeakened, highNormRejected⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality
