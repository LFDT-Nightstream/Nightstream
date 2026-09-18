import Nightstream.SuperNeo.Concrete.Relation
import Nightstream.SuperNeo.Concrete.Parameters

/-!
Model-level witnesses and adversarial cases for `REL-CCS`, `REL-CE`,
`REL-CONCRETE`, and `PARAM-GLOBAL`.
-/

namespace NightstreamTests.ConcreteRelations

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete

def toyStructure : Structure where
  matrices := [[[1, 0], [0, 1]]]
  polynomial := [⟨1, [1]⟩]
  rows := 2
  columns := 2
  pointDimension := 0

def toyContext : Context where
  publicWidth := 1
  ajtaiKey := []

def toyAssignment : Assignment := [0, 0]

theorem toyStructure_wellFormed : toyStructure.WellFormed := by
  simp [Structure.WellFormed, MatrixWellFormed, toyStructure, ringDegree]

theorem toyAssignment_satisfies : ccsSatisfied toyStructure toyAssignment := by
  refine ⟨toyStructure_wellFormed, rfl, ?_⟩
  intro row hrow
  change row < 2 at hrow
  have hcases : row = 0 ∨ row = 1 := by omega
  rcases hcases with rfl | rfl <;>
    decide

theorem toyAssignment_norm : normBounded 2 toyAssignment := by
  simp [normBounded, toyAssignment, centeredMagnitude, goldilocksModulus]

theorem toyPoint_valid : evaluationPointValid toyStructure [] :=
  ⟨toyStructure_wellFormed, rfl⟩

example : CCS.Holds (relationSemantics toyContext) productionGlobalParams
    (canonicalCCSStatement toyContext toyStructure .fresh toyAssignment)
    toyAssignment :=
  canonicalCCS_holds toyContext productionGlobalParams toyStructure .fresh
    toyAssignment toyAssignment_norm toyAssignment_satisfies

example : CE.Holds (relationSemantics toyContext) productionGlobalParams
    (canonicalCEStatement toyContext toyStructure .fresh [] toyAssignment)
    toyAssignment :=
  canonicalCE_holds toyContext productionGlobalParams toyStructure .fresh []
    toyAssignment toyAssignment_norm toyPoint_valid

def wrongPublicStatement : CCSStatement :=
  { canonicalCCSStatement toyContext toyStructure .fresh toyAssignment with
    publicInput := [1] }

example : ¬ CCS.Holds (relationSemantics toyContext) productionGlobalParams
    wrongPublicStatement toyAssignment := by
  apply ccs_rejects_wrong_public_input
  decide

def wrongCommitmentStatement : CCSStatement :=
  { canonicalCCSStatement toyContext toyStructure .fresh toyAssignment with
    commitment := [ringFZero] }

example : ¬ CCS.Holds (relationSemantics toyContext) productionGlobalParams
    wrongCommitmentStatement toyAssignment := by
  apply ccs_rejects_wrong_commitment
  simp [wrongCommitmentStatement, toyContext, ajtaiCommit]

/-- Regression for the missing CE shape check: even a prover who supplies the
evaluator's matching output for a one-coordinate point cannot use it against a
structure whose verifier-owned point dimension is zero. -/
def invalidPointStatement : CEStatement :=
  canonicalCEStatement toyContext toyStructure .fresh [K.zero] toyAssignment

example : ¬ CE.Holds (relationSemantics toyContext) productionGlobalParams
    invalidPointStatement toyAssignment := by
  apply ce_rejects_invalid_point
  simp [invalidPointStatement, canonicalCEStatement, evaluationPointValid,
    toyStructure]

def wrongEvaluationStatement : CEStatement :=
  { canonicalCEStatement toyContext toyStructure .fresh [] toyAssignment with
    evaluations := #[] }

example : ¬ CE.Holds (relationSemantics toyContext) productionGlobalParams
    wrongEvaluationStatement toyAssignment := by
  apply ce_rejects_wrong_evaluations
  intro h
  have hsize := congrArg Array.size h
  simp [wrongEvaluationStatement, canonicalCEStatement, matrixEvaluations,
    toyStructure] at hsize

def minusOne : F := ⟨goldilocksModulus - 1, by decide⟩

example : centeredMagnitude minusOne = 1 := by decide

example : normBounded 2 [1, minusOne] := by
  intro x hx
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl
  · decide
  · decide

/-- `X²⁷ · X²⁷ = -X²⁷ - 1` in `F[X]/(X⁵⁴ + X²⁷ + 1)`. -/
example : ringFMul (ringFMonomial 27 1) (ringFMonomial 27 1) ⟨0, by decide⟩ =
    minusOne := by decide

example : ringFMul (ringFMonomial 27 1) (ringFMonomial 27 1) ⟨27, by decide⟩ =
    minusOne := by decide

example : (packAssignment ([1, 2] : Assignment)).length = 1 := by decide

example : (packAssignment ([1, 2] : Assignment)).head (by decide) ⟨0, by decide⟩ = 1 := by
  decide

example : (61 + 14) * 216 < 16384 :=
  production_allows_every_advertised_batch (by decide)

/-- The cap is tight for the selected constants: admitting one more fresh
instance would violate Definition 14. -/
example : ¬ (62 + 14) * 216 < 16384 := by decide

example {fresh : Nat} (hFresh : fresh ≤ 61) :
    (fresh + 14) * 216 < 16384 :=
  production_allows_every_advertised_batch hFresh

end NightstreamTests.ConcreteRelations
