import NightstreamFPrime.Lifecycle.BoundedDepth.Keys
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.PublicInputCombination
import NightstreamFPrime.Spec.Folding.Light.RlcModule

/-! The light Π_RLC public input `x = x_running + ρ · x_fresh` under the coefficient embedding
(Cyclo, ePrint 2026/359, Figure 3, step 6, on SuperNeo v1.2 Section 7.4, verifier Step 1) as one
`CombinationStep` over the five public ring columns, with the running public input as the prior:
5 × 54 = 270 rows, against 4,590 for today's 17-source combination
(`PublicInputCombination.logicalRowCount_eq`). `output_eq` gives the public
input of `Light.Rlc.output` for the production light key. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.LightPublicInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

abbrev blockCount : Nat := publicRingColumns

abbrev cellCount : Nat := 1

def cell : Fin cellCount := ⟨0, by decide⟩

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

structure Interface (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) where
  challenge : Nat → Fin ringDegree → Expr
  running : Nat → Fin (FullShape logicalWidth publicFits).publicWidth → Expr
  fresh : Nat → Fin (FullShape logicalWidth publicFits).publicWidth → Expr

def stepInterface (interface : Interface logicalWidth publicFits) :
    CombinationStep.Interface blockCount cellCount where
  challenge := interface.challenge
  prior := fun offset block lane _ =>
    interface.running offset (PublicInputCombination.publicColumn block lane)
  value := fun offset block lane _ =>
    interface.fresh offset (PublicInputCombination.publicColumn block lane)

def output (offset : Nat) (column : Fin (FullShape logicalWidth publicFits).publicWidth) : Expr :=
  CombinationStep.output offset (CombinationStep.indexOf
    (PiRLCAlgebra.PublicInput.publicBlockIndex (FullShape logicalWidth publicFits) column)
    (PiRLCAlgebra.PublicInput.publicLaneIndex column) cell)

def evalOutput (offset : Nat) (env : Env) :
    PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun column => (output (publicFits := publicFits) offset column).eval env

def evalChallenge (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    RingF :=
  fun lane => (interface.challenge offset lane).eval env

def evalRunning (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun column => (interface.running offset column).eval env

def evalFresh (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun column => (interface.fresh offset column).eval env

theorem rowCount : CombinationStep.privateCount blockCount cellCount = 270 := by
  norm_num [CombinationStep.privateCount, publicRingColumns, ringDegree]

/-- The rows compute `x_running + ρ · x_fresh`. -/
theorem output_combination (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (spec : CombinationStep.SpecHolds (stepInterface interface) offset env) :
    evalOutput offset env =
      PiRLCAlgebra.PublicInput.publicAdd (evalRunning interface offset env)
        (PiRLCAlgebra.PublicInput.publicAct (evalChallenge interface offset env)
          (evalFresh interface offset env)) := by
  funext column
  have columnResult := spec (CombinationStep.indexOf
    (PiRLCAlgebra.PublicInput.publicBlockIndex (FullShape logicalWidth publicFits) column)
    (PiRLCAlgebra.PublicInput.publicLaneIndex column) cell)
  simp only [CombinationStep.blockOf_indexOf, CombinationStep.laneOf_indexOf,
    CombinationStep.cellOf_indexOf, stepInterface,
    PublicInputCombination.publicColumn_decode] at columnResult
  exact columnResult

/-- The production Π_RLC public input of `ρ · x_fresh + 1 · x_running`. -/
theorem combine_eq (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (challenge : RingF)
    (fresh running : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (piRlcAlgebra ajtai).combinePublicInput
        (Folding.Light.Rlc.pair challenge ringFOne) (Folding.Light.Rlc.pair fresh running) =
      PiRLCAlgebra.PublicInput.publicAdd running
        (PiRLCAlgebra.PublicInput.publicAct challenge fresh) := by
  rw [(PaperExtractionAlgebra.extractionAlgebra ajtai).combinePublicInput_eq]
  exact Folding.Light.Rlc.linearCombination_pair_eq
    (PaperExtractionAlgebra.extractionAlgebra ajtai).publicInputLaws challenge fresh running

/-- With the circuit's words for `ρ` and both public inputs, the rows give the public input of
the production light key's Π_RLC output. -/
theorem output_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (spec : CombinationStep.SpecHolds (stepInterface interface) offset env)
    (system : StrongReduction.RelationSource BoundedDepth.LightShape
      (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (point : CubePoint K BoundedDepth.LightShape.cubeVariables)
    (fresh running : CE.Instance (StrongReduction.RelationSource BoundedDepth.LightShape
      (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)))
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      (CubePoint K BoundedDepth.LightShape.cubeVariables)
      (StrongReduction.EvaluationFamily K BoundedDepth.LightShape) PaperAlgebra.Commitment)
    (challenge : RingF) (challengeWords : evalChallenge interface offset env = challenge)
    (runningWords : evalRunning interface offset env = running.publicInput)
    (freshWords : evalFresh interface offset env = fresh.publicInput) :
    evalOutput offset env =
      (Folding.Light.Rlc.output (BoundedDepth.keys relation ajtai).light.piRlcAlgebra
        (BoundedDepth.keys relation ajtai).light.one challenge system point fresh
        running).publicInput := by
  rw [output_combination interface offset env spec, challengeWords, runningWords, freshWords]
  exact (combine_eq ajtai challenge fresh.publicInput running.publicInput).symm

end NightstreamFPrime.Lifecycle.BoundedDepth.LightPublicInput
