import NightstreamFPrime.Lifecycle.RandomOracleUniqueness
import NightstreamFPrime.Lifecycle.Nifs.BindingBridge

/-!
Owns the binding step of Lemma 6 (ROM_KNOWLEDGE_SOUNDNESS.md): the event that
`RandomOracleUniqueness.collisionChance` counts is an MSIS break of the same
Ajtai key.

Inputs: two valid runs (`RandomOracleUniqueness.Valid`) that output the same
fresh statement, and whose rerun keeps the running statement but extracts a
different witness (`Collides`).

Outputs:
- `collides_relaxedBindingCollision`: the two complete `Π_RLC` forks give a
  `(2B, C)`-relaxed binding collision on one input commitment, as SuperNeo
  v1.2 Appendix B states. The collision data are the two forks' challenge and
  response differences (`PaperForkBinding.collisionAt`);
- `collides_shortKernel`: that collision gives a nonzero integer kernel
  vector of the same key with every coordinate below `8TB`
  (`Binding.relaxedBindingCollision_to_shortKernel`);
- `rerun_shortKernel`: the same for every rerun that `collisionChance`
  counts.

The extracted witnesses satisfy only the corrected ambient relation, so the
step uses relaxed binding, not ordinary binding. Does not own: the hardness of
the resulting MSIS instance or the running time of the reduction.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open NightstreamFPrime.Lifecycle.RandomOracleExtraction
open NightstreamFPrime.Lifecycle.RandomOracleUniqueness
open _root_.NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra
open StrongReduction ConcreteCarrier

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  {Output : Type}
  (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
    Answer Output)
  (claim : Output → Claim relation)

local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer
local notation "Retries" => Fin Nifs.PaperProfile.arity.total → Answer

/-- The `Π_RLC` input commitments of a probe's batch are the statement's
commitments: they do not depend on the probe. -/
theorem probe_phi
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (probe probe' : Probe K productionShape) :
    PiRLC.phi (Nifs.PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai)
        running fresh probe).inputs =
      PiRLC.phi (Nifs.PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai)
        running fresh probe').inputs := by
  funext coordinate
  rfl

/-- The oracle batch's input commitments depend only on the two statements,
not on the oracle or the proof. -/
theorem batch_phi (oracle other : Oracle)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof proof' : Proof (ProductionKey.degreeBound relation)) :
    PiRLC.phi (batch relation ajtai oracle running fresh proof).inputs =
      PiRLC.phi (batch relation ajtai other running fresh proof').inputs :=
  probe_phi relation ajtai running fresh _ _

/-- A rerun that keeps the running statement and the fresh statement but
extracts a different witness exposes a relaxed binding collision on one
input commitment of the base run. -/
theorem collides_relaxedBindingCollision {oracle other : Oracle} {retries otherRetries : Retries}
    (valid : Valid relation ajtai adversary claim oracle retries)
    (otherValid : Valid relation ajtai adversary claim other otherRetries)
    (sameFresh : (claimed relation adversary claim other).fresh =
      (claimed relation adversary claim oracle).fresh)
    (collides : Collides relation ajtai adversary claim oracle retries other otherRetries) :
    ∃ coordinate, Nonempty (PiRLC.RelaxedBindingCollision
      (ProductionKey.key relation ajtai).piRlcSemantics (ProductionKey.key relation ajtai).params
      (Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
        (rows := productionProfile.commitmentWidth))
      ((batch relation ajtai oracle (claimed relation adversary claim oracle).running
        (claimed relation adversary claim oracle).fresh
        (claimed relation adversary claim oracle).proof).inputs coordinate).commitment) := by
  have samePhi : PiRLC.phi (batch relation ajtai oracle (claimed relation adversary claim oracle).running
        (claimed relation adversary claim oracle).fresh (claimed relation adversary claim oracle).proof).inputs =
      PiRLC.phi (batch relation ajtai other (claimed relation adversary claim other).running
        (claimed relation adversary claim other).fresh (claimed relation adversary claim other).proof).inputs := by
    rw [collides.1, sameFresh]
    exact batch_phi relation ajtai _ _ _ _ _ _
  rcases PiRLC.PaperForkBinding.two_forks_unique_or_collision
      (PaperExtractionAlgebra.extractionAlgebra ajtai)
      (Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
        (rows := productionProfile.commitmentWidth))
      (Nifs.BindingBridge.compatible relation ajtai)
      (ForkStrongSet.strongSetUnits Phi81StrongSet.lowNormInvertibility)
      _ _ (completeFork relation ajtai adversary claim oracle retries valid.1 valid.2)
      (completeFork relation ajtai adversary claim other otherRetries otherValid.1 otherValid.2)
      samePhi with equal | collision
  · exfalso
    apply collides.2
    unfold witnessOf
    rw [dif_pos valid, dif_pos otherValid]
    exact congrArg (fun assignments => some (Nifs.PaperStrongInterface.outputWitnessOfAssignments
      (ProductionKey.key relation ajtai) assignments)) equal.symm
  · exact collision

/-- The binding reduction's collision event is a short kernel vector of the
same key: every coordinate is below `8TB = 113246208`. -/
theorem collides_shortKernel {oracle other : Oracle} {retries otherRetries : Retries}
    (valid : Valid relation ajtai adversary claim oracle retries)
    (otherValid : Valid relation ajtai adversary claim other otherRetries)
    (sameFresh : (claimed relation adversary claim other).fresh =
      (claimed relation adversary claim oracle).fresh)
    (collides : Collides relation ajtai adversary claim oracle retries other otherRetries) :
    Nonempty (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound) := by
  obtain ⟨_, ⟨collision⟩⟩ := collides_relaxedBindingCollision relation ajtai adversary claim
    valid otherValid sameFresh collides
  exact ⟨Binding.relaxedBindingCollision_to_shortKernel ajtai _ {
    delta₁ := collision.delta₁
    delta₂ := collision.delta₂
    opening₁ := collision.opening₁
    opening₂ := collision.opening₂
    delta₁Valid := collision.delta₁Valid
    delta₂Valid := collision.delta₂Valid
    firstEquation := collision.firstEquation
    secondEquation := collision.secondEquation
    firstNorm := collision.firstNorm
    secondNorm := collision.secondNorm
    crossDifferent := collision.crossDifferent }⟩

/-- Every rerun that `collisionChance` counts gives a short kernel vector: the
rerun from the fork context forks at the same index, so it outputs the base
fresh statement (`fresh_eq_of_fork`). -/
theorem rerun_shortKernel {oracle : Oracle} {retries otherRetries : Retries} (fresh : Oracle)
    (valid : Valid relation ajtai adversary claim oracle retries)
    (forked : forkIndex relation adversary claim
      (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
        oracle fresh) = forkIndex relation adversary claim oracle)
    (otherValid : Valid relation ajtai adversary claim
      (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
        oracle fresh) otherRetries)
    (collides : Collides relation ajtai adversary claim oracle retries
      (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
        oracle fresh) otherRetries) :
    Nonempty (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound) :=
  collides_shortKernel relation ajtai adversary claim valid otherValid
    (fresh_eq_of_fork relation adversary claim oracle fresh forked) collides

end NightstreamFPrime.Lifecycle.RandomOracleBinding
