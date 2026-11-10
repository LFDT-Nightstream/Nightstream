import NightstreamFPrime.Lifecycle.Nifs.VerifierErrorBudget
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshSchedule
import NightstreamFPrime.Lifecycle.RandomOracle.Test

/-! The refresh fold's Π_CCS test error in the random-oracle model: `Test.test_error_le` on the
refresh schedule. The input accumulator and the witness that opens the output claims are
fixed; the adversary outputs the fresh claim and the refresh proof. The Π_CCS input is the
fresh claim and the children that the proof's split gives for the accumulator
(`Refresh.children`), and the schedule statement is the fresh claim with that split. A false
acceptance (`FalseAcceptance`) is the refresh Π_CCS check on the oracle coins with the output
claims open at the ambient bound and false source claims (`¬ SourceHolds`, `b = 2` on all 17
sources). `RoundByRound.falseAcceptance_splits` turns it into a hit (`falseAcceptance_hits`), so
its probability is at most `(Q + 1) · 4589/q²` (`test_error_le`,
`Nifs.VerifierErrorBudget.test_error_eq`). The refresh transcript has `28 + 1 + 28 + 17 = 74`
challenges (`challenges_length`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshTest

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

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

theorem challenges_length :
    (Test.challenges (RefreshSchedule.schedule logicalWidth publicFits)).length = 74 := by
  rw [Test.challenges_length]
  rfl

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (accumulator : Refresh.Accumulator PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)))
  (witness : OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth))

local notation "Width" => ProductionKey.degreeBound relation
local notation "RefreshStatement" =>
  Schedule.Statement (RefreshSchedule.schedule logicalWidth publicFits)
local notation "RefreshProof" => Refresh.Proof PaperAlgebra.Commitment productionShape
  (ProductionKey.degreeBound relation)
local notation "Oracle" =>
  Test.Point (RefreshSchedule.schedule logicalWidth publicFits)
    (ProductionKey.degreeBound relation) → Test.Answer

/-- The refresh Π_CCS protocol data of the fresh claim and the split's children, read through
`witness`. -/
noncomputable def source (statement : RefreshStatement) :
    ProtocolPolynomial.Data K (RefreshSchedule.schedule logicalWidth publicFits).shape :=
  ((refreshKey relation ajtai).statement
      (Refresh.children (refreshKey relation ajtai) accumulator statement.2)
      statement.1).sourceProtocolData K.embed witness

theorem source_degreeCovers (statement : RefreshStatement) :
    (source relation ajtai accumulator witness statement).toVerifierInput.sumcheckDegreeBound ≤
      Width :=
  (refreshKey relation ajtai).statement_sumcheckDegreeBound_le _ statement.1

/-- The refresh Π_CCS probe with the coins read from `oracle`. -/
noncomputable def oracleProbe (oracle : Oracle)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : RefreshProof) : Probe K productionShape :=
  { (refreshKey relation ajtai).piCcsProbe
      (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh proof.nifs with
    coins := Test.coins oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs) }

/-- The refresh Π_CCS accepts on the oracle coins, `witness` opens the output claims at the
ambient bound, and the source claims are false. -/
def FalseAcceptance (oracle : Oracle)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : RefreshProof) : Prop :=
  (oracleProbe relation ajtai accumulator oracle fresh proof).FixedWidthAccepted extensionOps
      K.embed ((refreshKey relation ajtai).statement
        (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh) Width ∧
    AmbientOutputHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai)
      productionGlobalParams ((refreshKey relation ajtai).statement
        (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh)
      (oracleProbe relation ajtai accumulator oracle fresh proof) witness ∧
    ¬ SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((refreshKey relation ajtai).statement
        (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh) witness

theorem falseAcceptance_hits (oracle : Oracle)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : RefreshProof)
    (false_ : FalseAcceptance relation ajtai accumulator witness oracle fresh proof) :
    Test.Hits (Test.bad (source relation ajtai accumulator witness)) oracle (fresh, proof.split)
      (RefreshSchedule.messages proof.nifs) := by
  obtain ⟨accepted, ambient, invalid⟩ := false_
  exact Test.splits_hits (source relation ajtai accumulator witness) oracle (fresh, proof.split)
    (RefreshSchedule.messages proof.nifs)
    (oracleProbe relation ajtai accumulator oracle fresh proof) rfl
    (Spec.Folding.Nifs.PaperNonInteractive.piCcsProbe_rounds _ _ fresh proof.nifs)
    (RoundByRound.falseAcceptance_splits (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      rfl ((refreshKey relation ajtai).statement
        (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh)
      (refreshKey relation ajtai).constantLaw
      (source_degreeCovers relation ajtai accumulator witness (fresh, proof.split)) witness
      invalid _ ambient accepted)

/-- The test step for the refresh fold: an adversary with at most `Q` queries makes a refresh Π_CCS
false acceptance with probability at most `(Q + 1) · 4589/q²`. -/
theorem test_error_le {Output : Type}
    (adversary : OracleComp
      (Test.Point (RefreshSchedule.schedule logicalWidth publicFits) Width) Test.Answer Output)
    {queries : Nat} (bounded : adversary.QueryBound queries)
    (fresh : Output → Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Output → RefreshProof) :
    𝔼 oracle, (if FalseAcceptance relation ajtai accumulator witness oracle
        (fresh (adversary.run oracle)) (proof (adversary.run oracle)) then (1 : ℝ) else 0) ≤
      (queries + 1) * ((4589 : ℝ) / (goldilocksModulus : ℝ) ^ 2) := by
  refine (Test.test_error_le (source relation ajtai accumulator witness)
    (source_degreeCovers relation ajtai accumulator witness) adversary bounded
    (fun output => (fresh output, (proof output).split))
    (fun output => RefreshSchedule.messages (proof output).nifs)
    (fun oracle => FalseAcceptance relation ajtai accumulator witness oracle
      (fresh (adversary.run oracle)) (proof (adversary.run oracle)))
    fun oracle false_ => falseAcceptance_hits relation ajtai accumulator witness oracle _ _
      false_).trans (le_of_eq ?_)
  rw [← Nifs.VerifierErrorBudget.test_error_eq]
  rfl

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshTest
