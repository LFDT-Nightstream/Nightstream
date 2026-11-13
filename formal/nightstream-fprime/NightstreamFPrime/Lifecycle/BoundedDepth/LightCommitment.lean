import NightstreamFPrime.Lifecycle.BoundedDepth.Keys
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.CombinationStep
import NightstreamFPrime.Spec.Folding.Light.RlcModule

/-! The light Π_RLC commitment `c = c_running + ρ · c_fresh` (Cyclo, ePrint 2026/359, Figure 3,
step 6, on SuperNeo v1.2 Section 7.4, verifier Step 1) as one `CombinationStep` with the
running commitment as the prior: 18 × 64 = 1,152 rows, against 19,584 for today's 17-source
combination (`CommitmentCombination.logicalRowCount_eq`). `output_eq` gives the commitment of
`Light.Rlc.output` for the production light key. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.LightCommitment

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.RingRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

def blockCount : Nat := productionProfile.commitmentWidth

abbrev cellCount : Nat := 1

def cell : Fin cellCount := ⟨0, by decide⟩

structure Interface where
  challenge : Nat → Fin ringDegree → Expr
  running : Nat → Fin blockCount → Fin ringDegree → Expr
  fresh : Nat → Fin blockCount → Fin ringDegree → Expr

def stepInterface (interface : Interface) : CombinationStep.Interface blockCount cellCount where
  challenge := interface.challenge
  prior := fun offset block lane _ => interface.running offset block lane
  value := fun offset block lane _ => interface.fresh offset block lane

def output (offset : Nat) (row : Fin blockCount) (lane : Fin ringDegree) : Expr :=
  CombinationStep.output offset (CombinationStep.indexOf row lane cell)

def evalOutput (offset : Nat) (env : Env) : PiRLCAlgebra.Commitment.Value blockCount :=
  fun row lane => (output offset row lane).eval env

def evalChallenge (interface : Interface) (offset : Nat) (env : Env) : RingF :=
  fun lane => (interface.challenge offset lane).eval env

def evalRunning (interface : Interface) (offset : Nat) (env : Env) :
    PiRLCAlgebra.Commitment.Value blockCount :=
  fun row lane => (interface.running offset row lane).eval env

def evalFresh (interface : Interface) (offset : Nat) (env : Env) :
    PiRLCAlgebra.Commitment.Value blockCount :=
  fun row lane => (interface.fresh offset row lane).eval env

theorem rowCount : CombinationStep.privateCount blockCount cellCount = 1152 := by
  norm_num [CombinationStep.privateCount, blockCount, productionProfile, ringDegree]

/-- The rows compute `c_running + ρ · c_fresh`. -/
theorem output_combination (interface : Interface) (offset : Nat) (env : Env)
    (spec : CombinationStep.SpecHolds (stepInterface interface) offset env) :
    evalOutput offset env =
      PiRLCAlgebra.Commitment.commitmentAdd (evalRunning interface offset env)
        (PiRLCAlgebra.Commitment.commitmentAct (evalChallenge interface offset env)
          (evalFresh interface offset env)) := by
  funext row lane
  have rowResult := spec (CombinationStep.indexOf row lane cell)
  simp only [CombinationStep.blockOf_indexOf, CombinationStep.laneOf_indexOf,
    CombinationStep.cellOf_indexOf] at rowResult
  exact rowResult

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- The production Π_RLC commitment of `ρ · c_fresh + 1 · c_running`. -/
theorem combine_eq (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (challenge : RingF) (fresh running : PaperAlgebra.Commitment) :
    (PaperAlgebra.piRlcAlgebra ajtai).combineCommitment
        (Folding.Light.Rlc.pair challenge ringFOne) (Folding.Light.Rlc.pair fresh running) =
      PiRLCAlgebra.Commitment.commitmentAdd running
        (PiRLCAlgebra.Commitment.commitmentAct challenge fresh) := by
  rw [(PaperExtractionAlgebra.extractionAlgebra ajtai).combineCommitment_eq]
  exact Folding.Light.Rlc.linearCombination_pair_eq
    (PaperExtractionAlgebra.extractionAlgebra ajtai).commitmentLaws challenge fresh running

/-- With the circuit's words for `ρ` and both commitments, the rows give the commitment of the
production light key's Π_RLC output. -/
theorem output_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface) (offset : Nat) (env : Env)
    (spec : CombinationStep.SpecHolds (stepInterface interface) offset env)
    (system : StrongReduction.RelationSource BoundedDepth.LightShape
      (RingCarrierLayout.carrierWidth logicalWidth)
      (ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth logicalWidth)))
    (point : CubePoint K BoundedDepth.LightShape.cubeVariables)
    (fresh running : CE.Instance (StrongReduction.RelationSource BoundedDepth.LightShape
      (RingCarrierLayout.carrierWidth logicalWidth)
      (ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth logicalWidth)))
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      (CubePoint K BoundedDepth.LightShape.cubeVariables)
      (StrongReduction.EvaluationFamily K BoundedDepth.LightShape) PaperAlgebra.Commitment)
    (challenge : RingF) (challengeWords : evalChallenge interface offset env = challenge)
    (runningWords : evalRunning interface offset env = running.commitment)
    (freshWords : evalFresh interface offset env = fresh.commitment) :
    evalOutput offset env =
      (Folding.Light.Rlc.output (BoundedDepth.keys relation ajtai).light.piRlcAlgebra
        (BoundedDepth.keys relation ajtai).light.one challenge system point fresh
        running).commitment := by
  rw [output_combination interface offset env spec, challengeWords, runningWords, freshWords]
  exact (combine_eq ajtai challenge fresh.commitment running.commitment).symm

end NightstreamFPrime.Lifecycle.BoundedDepth.LightCommitment
