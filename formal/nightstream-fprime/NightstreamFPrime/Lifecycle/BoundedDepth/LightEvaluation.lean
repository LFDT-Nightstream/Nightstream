import NightstreamFPrime.Lifecycle.BoundedDepth.Keys
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.RingKCombination
import NightstreamFPrime.Spec.Folding.Light.RlcModule

/-! The light Π_RLC evaluation family `y = y_running + ρ · y_fresh` (Cyclo, ePrint 2026/359,
Figure 3, step 6, on SuperNeo v1.2 Section 7.4, verifier Step 1) as one `CombinationStep` with
the running family as the prior. Block 0 is `Eval_K` (the pad family) and block `m + 1` is
`Eval_A` of matrix `m`; each `K` value is two field cells, so the step has
5 × 54 × 2 = 540 rows, against 1,836 + 7,344 = 9,180 for today's combinations
(`EvalKCombination`, `EvalACombination`). `output_eq` gives the
evaluation family of the production light key's Π_RLC output. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.LightEvaluation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

abbrev blockCount : Nat := LightShape.matrixCount + 1

abbrev cellCount : Nat := RingKCombination.cellCount

/-- An evaluation family as ring blocks: the pad family, then one block per matrix. -/
def familyBlocks (family : EvaluationFamily K LightShape) (block : Fin blockCount) : RingK :=
  if first : block.val = 0 then family.pad
  else family.matrix ⟨block.val - 1, by
    have bound := block.isLt
    simp only [blockCount] at bound
    omega⟩

structure Interface where
  challenge : Nat → Fin ringDegree → Expr
  running : Nat → Fin blockCount → Fin ringDegree → KExpr
  fresh : Nat → Fin blockCount → Fin ringDegree → KExpr

def stepInterface (interface : Interface) : CombinationStep.Interface blockCount cellCount where
  challenge := interface.challenge
  prior := fun offset block lane cell =>
    RingKCombination.expressionCell cell (interface.running offset block lane)
  value := fun offset block lane cell =>
    RingKCombination.expressionCell cell (interface.fresh offset block lane)

def output (offset : Nat) (block : Fin blockCount) (lane : Fin ringDegree) : KExpr :=
  ⟨CombinationStep.output offset (CombinationStep.indexOf block lane RingKCombination.c0Cell),
    CombinationStep.output offset (CombinationStep.indexOf block lane RingKCombination.c1Cell)⟩

def evalOutput (offset : Nat) (env : Env) : Fin blockCount → RingK :=
  fun block lane => (output offset block lane).eval env

def evalChallenge (interface : Interface) (offset : Nat) (env : Env) : RingF :=
  fun lane => (interface.challenge offset lane).eval env

def evalRunning (interface : Interface) (offset : Nat) (env : Env) : Fin blockCount → RingK :=
  fun block lane => (interface.running offset block lane).eval env

def evalFresh (interface : Interface) (offset : Nat) (env : Env) : Fin blockCount → RingK :=
  fun block lane => (interface.fresh offset block lane).eval env

theorem rowCount : CombinationStep.privateCount blockCount cellCount = 540 := by
  rfl

private theorem expressionCell_eval (env : Env) (cell : Fin cellCount) (value : KExpr) :
    (RingKCombination.expressionCell cell value).eval env =
      RingKCombination.kCell cell (value.eval env) := by
  unfold RingKCombination.expressionCell RingKCombination.kCell
  split <;> rfl

/-- The rows compute `y_running + ρ · y_fresh` in every block. -/
theorem output_combination (interface : Interface) (offset : Nat) (env : Env)
    (spec : CombinationStep.SpecHolds (stepInterface interface) offset env) :
    evalOutput offset env = fun block =>
      ringKAdd (evalRunning interface offset env block)
        (ringKMul (RingKAction.embedChallenge (evalChallenge interface offset env))
          (evalFresh interface offset env block)) := by
  funext block lane
  have cellResult := fun cell : Fin cellCount => spec (CombinationStep.indexOf block lane cell)
  simp only [CombinationStep.blockOf_indexOf, CombinationStep.laneOf_indexOf,
    CombinationStep.cellOf_indexOf, stepInterface, expressionCell_eval] at cellResult
  have product := fun cell : Fin cellCount =>
    RingKCombination.ringKMul_cell cell (evalChallenge interface offset env)
      (evalFresh interface offset env block) lane
  have low := (cellResult RingKCombination.c0Cell).trans
    (congrArg (_ + ·) (product RingKCombination.c0Cell))
  have high := (cellResult RingKCombination.c1Cell).trans
    (congrArg (_ + ·) (product RingKCombination.c1Cell))
  simp only [RingKCombination.kCell, RingKCombination.c0Cell, RingKCombination.c1Cell] at low high
  simp only [evalOutput, output, KExpr.eval, ringKAdd, K.add]
  exact congrArg₂ K.mk low high

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- With the circuit's words for `ρ` and both families, the rows give the first evaluation family
of the production light key's Π_RLC output. -/
theorem output_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface) (offset : Nat) (env : Env)
    (spec : CombinationStep.SpecHolds (stepInterface interface) offset env)
    (system : RelationSource LightShape (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (point : CubePoint K LightShape.cubeVariables)
    (fresh running : CE.Instance (RelationSource LightShape
      (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)))
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      (CubePoint K LightShape.cubeVariables) (EvaluationFamily K LightShape)
      PaperAlgebra.Commitment)
    (size : fresh.evaluations.size = running.evaluations.size)
    (challenge : RingF) (challengeWords : evalChallenge interface offset env = challenge)
    (runningWords : evalRunning interface offset env =
      familyBlocks (running.evaluations.getD 0 (lightExtraction ajtai).evaluationModule.zero))
    (freshWords : evalFresh interface offset env =
      familyBlocks (fresh.evaluations.getD 0 (lightExtraction ajtai).evaluationModule.zero)) :
    evalOutput offset env =
      familyBlocks ((Folding.Light.Rlc.output (keys relation ajtai).light.piRlcAlgebra
        (keys relation ajtai).light.one challenge system point fresh running).evaluations.getD 0
          (lightExtraction ajtai).evaluationModule.zero) := by
  have combined := Folding.Light.Rlc.output_evaluation (lightExtraction ajtai) challenge system
    point fresh running size 0
  rw [output_combination interface offset env spec, challengeWords, runningWords, freshWords]
  change _ = familyBlocks ((Folding.Light.Rlc.output
    (HyperNova.Construction2.BoundedDepthKeys.algebraToLight (PaperAlgebra.piRlcAlgebra ajtai))
    (lightExtraction ajtai).ring.one challenge system point fresh running).evaluations.getD 0
      (lightExtraction ajtai).evaluationModule.zero)
  rw [combined]
  funext block
  unfold familyBlocks
  by_cases first : block.val = 0 <;> simp only [first, ↓reduceDIte] <;> rfl

end NightstreamFPrime.Lifecycle.BoundedDepth.LightEvaluation
