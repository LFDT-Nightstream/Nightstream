import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightSchedule
import NightstreamFPrime.Lifecycle.RandomOracle.Test
import NightstreamFPrime.Spec.Folding.Light.ErrorBudget
import NightstreamFPrime.Spec.Folding.Light.RoundByRound

/-! The light fold's Π_CCS test error in the random-oracle model: `Test.test_error_le` on the
light schedule. The running claim and the witness that opens the output claims are fixed; the
adversary outputs the fresh claim and the light proof. A false acceptance (`FalseAcceptance`)
is the light Π_CCS check on the oracle coins with the output claims open at the ambient bound
and false source claims (`¬ Light.PiCCS.SemanticTruth`). `Light.PiCCS.falseAcceptance_splits`
turns it into a hit (`falseAcceptance_hits`), so its probability is at most
`(Q + 1) · 524/q²` (`test_error_le`, `testError_eq`). The light transcript has
`28 + 1 + 28 + 1 = 58` challenges (`challenges_length`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightTest

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

/-- The light shape is the shape of the light error budget at the production cube. -/
theorem lightShape_eq : LightShape = Light.ErrorBudget.shape 28 :=
  rfl

/-- SuperNeo v1.2 Appendix B.2, equation (14), at the light shape: 524 over `q²`. -/
theorem testError_eq :
    IndependentExecution.testError LightShape 8 = (524 : ℝ) / (fieldModulus : ℝ) ^ 2 := by
  rw [lightShape_eq]
  exact Light.ErrorBudget.production_testError

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

theorem challenges_length :
    (Test.challenges (LightSchedule.schedule logicalWidth publicFits)).length = 58 := by
  rw [Test.challenges_length]
  rfl

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) LightShape)
  (witness : OutputWitness LightShape (RingCarrierLayout.carrierWidth logicalWidth))

local notation "Width" => ProductionKey.degreeBound relation
local notation "LightStatement" =>
  Schedule.Statement (LightSchedule.schedule logicalWidth publicFits)
local notation "Oracle" =>
  Test.Point (LightSchedule.schedule logicalWidth publicFits)
    (ProductionKey.degreeBound relation) → Test.Answer

/-- The light Π_CCS protocol data of the fresh claim and `running`, read through `witness`. -/
noncomputable def source (fresh : LightStatement) :
    ProtocolPolynomial.Data K (LightSchedule.schedule logicalWidth publicFits).shape :=
  Light.PiCCS.protocolData baseOps K.embed
    ((((keys relation ajtai).light.statement running fresh).sourceConnectedInputs
      witness).toUnifiedInputs baseOps)

theorem source_degreeCovers (fresh : LightStatement) :
    (source relation ajtai running witness fresh).toVerifierInput.sumcheckDegreeBound ≤ Width :=
  le_of_eq ((keys relation ajtai).light.statement_sumcheckDegreeBound_eq running fresh)

/-- The light probe with the Π_CCS coins read from `oracle`. -/
noncomputable def oracleProbe (oracle : Oracle) (fresh : LightStatement)
    (proof : Light.Proof LightShape Width) : Probe K LightShape :=
  { (keys relation ajtai).light.probe running fresh proof with
    coins := Test.coins oracle fresh (LightSchedule.messages proof) }

/-- The light Π_CCS accepts on the oracle coins, `witness` opens the output claims at the
ambient bound, and the source claims are false. -/
def FalseAcceptance (oracle : Oracle) (fresh : LightStatement)
    (proof : Light.Proof LightShape Width) : Prop :=
  Light.PiCCS.Accepted extensionOps K.embed ((keys relation ajtai).light.statement running fresh)
      Width (oracleProbe relation ajtai running oracle fresh proof) ∧
    AmbientOutputHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai)
      productionGlobalParams ((keys relation ajtai).light.statement running fresh)
      (oracleProbe relation ajtai running oracle fresh proof) witness ∧
    ¬ Light.PiCCS.SemanticTruth
      ((((keys relation ajtai).light.statement running fresh).sourceConnectedInputs
        witness).toUnifiedInputs baseOps) baseOps extensionOps K.embed

theorem falseAcceptance_hits (oracle : Oracle) (fresh : LightStatement)
    (proof : Light.Proof LightShape Width)
    (false_ : FalseAcceptance relation ajtai running witness oracle fresh proof) :
    Test.Hits (Test.bad (source relation ajtai running witness)) oracle fresh
      (LightSchedule.messages proof) := by
  obtain ⟨accepted, ambient, invalid⟩ := false_
  exact Test.splits_hits (source relation ajtai running witness) oracle fresh
    (LightSchedule.messages proof) (oracleProbe relation ajtai running oracle fresh proof) rfl rfl
    (Light.PiCCS.falseAcceptance_splits (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((keys relation ajtai).light.statement running fresh) (keys relation ajtai).light.constantLaw
      Width (source_degreeCovers relation ajtai running witness fresh)
      (oracleProbe relation ajtai running oracle fresh proof) witness invalid ambient accepted)

/-- The test step for the light fold: an adversary with at most `Q` queries makes a light Π_CCS false
acceptance with probability at most `(Q + 1) · 524/q²`. -/
theorem test_error_le {Output : Type}
    (adversary : OracleComp
      (Test.Point (LightSchedule.schedule logicalWidth publicFits) Width) Test.Answer Output)
    {queries : Nat} (bounded : adversary.QueryBound queries)
    (fresh : Output → LightStatement) (proof : Output → Light.Proof LightShape Width) :
    𝔼 oracle, (if FalseAcceptance relation ajtai running witness oracle
        (fresh (adversary.run oracle)) (proof (adversary.run oracle)) then (1 : ℝ) else 0) ≤
      (queries + 1) * ((524 : ℝ) / (fieldModulus : ℝ) ^ 2) := by
  refine (Test.test_error_le (source relation ajtai running witness)
    (source_degreeCovers relation ajtai running witness) adversary bounded fresh
    (fun output => LightSchedule.messages (proof output))
    (fun oracle => FalseAcceptance relation ajtai running witness oracle
      (fresh (adversary.run oracle)) (proof (adversary.run oracle)))
    fun oracle false_ => falseAcceptance_hits relation ajtai running witness oracle _ _
      false_).trans (le_of_eq ?_)
  rw [← testError_eq]
  rfl

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightTest
