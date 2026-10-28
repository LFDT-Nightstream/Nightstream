import NightstreamFPrime.Lifecycle.RandomOracleUniqueness
import NightstreamFPrime.Lifecycle.Nifs.BindingBridge

/-!
Owns the binding reduction of Lemma 6 (SECURITY_MODEL.md): a
function of the two runs that `RandomOracleUniqueness.collisionChance`
compares, which returns a short kernel vector of the same Ajtai key.

Inputs: a base run and a rerun that are `PairedOf`: both extract, and the rerun
outputs the base run's running and fresh statements.

Outputs:
- `rerunKernelOf`: the reduction on the claims of two runs and their retries,
  with no adversary; `rerunKernel` is it at the adversary's own claims. At
  the first coordinate where the two complete `Π_RLC` forks
  extract different assignments, the `(2B, C)`-relaxed binding collision of
  the forks' challenge and response differences (`PaperForkBinding.collisionAt`,
  SuperNeo v1.2 Appendix B), as a nonzero integer kernel vector with every
  coordinate below `8TB` (`Binding.relaxedBindingCollision_to_shortKernel`);
  `none` when the runs are not paired or extract the same assignments;
- `rerunKernel_isSome`: every pair of runs that `collisionChance` counts makes
  `rerunKernel` return a vector;
- `kernelChance`, `collisionChance_le_kernelChance`: the chance that the
  reduction returns a vector, under the law of `collisionChance`, bounds
  `collisionChance`.

The vector is a function of the two runs. The definition is `noncomputable`:
it decides the run conditions classically and takes each strong-set inverse,
which is unique, with `Classical.choose` (`ForkStrongSet.strongSetUnits`).
The extracted witnesses satisfy only the corrected ambient relation, so the
step uses relaxed binding, not ordinary binding. Does not own: the hardness of
the resulting MSIS instance, or the reduction as an algorithm with a strict
budget of runs (`RandomOracleReduction`).
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleBinding

open scoped BigOperators

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

attribute [local instance low] Classical.propDecidable

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

theorem batch_phi (oracle other : Oracle)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof proof' : Proof (ProductionKey.degreeBound relation)) :
    PiRLC.phi (batch relation ajtai oracle running fresh proof).inputs =
      PiRLC.phi (batch relation ajtai other running fresh proof').inputs :=
  probe_phi relation ajtai running fresh _ _


def PairedOf (fork other : ForkClaims relation) : Prop :=
  ValidOf relation ajtai fork ∧ ValidOf relation ajtai other ∧
    other.base.running = fork.base.running ∧ other.base.fresh = fork.base.fresh

theorem paired_phi {fork other : ForkClaims relation} (paired : PairedOf relation ajtai fork other) :
    PiRLC.phi (batch relation ajtai fork.oracle fork.base.running fork.base.fresh fork.base.proof).inputs =
      PiRLC.phi (batch relation ajtai other.oracle other.base.running other.base.fresh other.base.proof).inputs := by
  rw [paired.2.2.1, paired.2.2.2]
  exact batch_phi relation ajtai _ _ _ _ _ _

noncomputable def forkAssignments {fork : ForkClaims relation} (valid : ValidOf relation ajtai fork) :=
  PiRLC.PaperForkExtraction.extractedAssignment (PaperExtractionAlgebra.extractionAlgebra ajtai)
    (ForkStrongSet.strongSetUnits Phi81StrongSet.lowNormInvertibility)
    (completeForkOf relation ajtai fork valid.1 valid.2)

noncomputable def kernelOf {commitment}
    (collision : PiRLC.RelaxedBindingCollision
      (ProductionKey.key relation ajtai).piRlcSemantics (ProductionKey.key relation ajtai).params
      (Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
        (rows := productionProfile.commitmentWidth)) commitment) :
    Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound :=
  Binding.relaxedBindingCollision_to_shortKernel ajtai _ {
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
    crossDifferent := collision.crossDifferent }

noncomputable def rerunKernelOf (fork other : ForkClaims relation) :
    Option (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound) :=
  if paired : PairedOf relation ajtai fork other then
    if different : ∃ coordinate, forkAssignments relation ajtai paired.1 coordinate ≠
        forkAssignments relation ajtai paired.2.1 coordinate then
      some (kernelOf relation ajtai (PiRLC.PaperForkBinding.collisionAt
        (PaperExtractionAlgebra.extractionAlgebra ajtai)
        (Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
          (rows := productionProfile.commitmentWidth))
        (Nifs.BindingBridge.compatible relation ajtai)
        (ForkStrongSet.strongSetUnits Phi81StrongSet.lowNormInvertibility) _ _
        (completeForkOf relation ajtai fork paired.1.1 paired.1.2)
        (completeForkOf relation ajtai other paired.2.1.1 paired.2.1.2)
        (paired_phi relation ajtai paired) (Fin.find _ different)
        (Fin.find_spec different)))
    else none
  else none

noncomputable def rerunKernel (oracle : Oracle) (retries : Retries) (other : Oracle)
    (otherRetries : Retries) :
    Option (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound) :=
  rerunKernelOf relation ajtai (forkClaimsOf relation adversary claim oracle retries)
    (forkClaimsOf relation adversary claim other otherRetries)

theorem rerunKernel_isSome {oracle other : Oracle} {retries otherRetries : Retries}
    (valid : Valid relation ajtai adversary claim oracle retries)
    (otherValid : Valid relation ajtai adversary claim other otherRetries)
    (sameFresh : (claimed relation adversary claim other).fresh =
      (claimed relation adversary claim oracle).fresh)
    (collides : Collides relation ajtai adversary claim oracle retries other otherRetries) :
    (rerunKernel relation ajtai adversary claim oracle retries other otherRetries).isSome := by
  have paired : PairedOf relation ajtai (forkClaimsOf relation adversary claim oracle retries)
      (forkClaimsOf relation adversary claim other otherRetries) :=
    ⟨valid, otherValid, collides.1, sameFresh⟩
  have different : ∃ coordinate, forkAssignments relation ajtai paired.1 coordinate ≠
      forkAssignments relation ajtai paired.2.1 coordinate := by
    by_contra same
    simp only [not_exists, not_not] at same
    apply collides.2
    unfold witnessOf
    rw [dif_pos otherValid, dif_pos valid]
    exact congrArg (fun assignments => some (Nifs.PaperStrongInterface.outputWitnessOfAssignments
      (ProductionKey.key relation ajtai) assignments)) (funext same).symm
  unfold rerunKernel rerunKernelOf
  rw [dif_pos paired, dif_pos different]
  rfl

noncomputable def kernelChance : ℝ :=
  reductionChance relation ajtai adversary claim fun oracle retries other otherRetries =>
    (rerunKernel relation ajtai adversary claim oracle retries other otherRetries).isSome

theorem collisionChance_le_kernelChance :
    collisionChance relation ajtai adversary claim ≤ kernelChance relation ajtai adversary claim := by
  unfold collisionChance kernelChance reductionChance
  refine Finset.expect_le_expect fun oracle _ => Finset.sum_le_sum fun retries _ =>
    mul_le_mul_of_nonneg_left ?_ (weight_nonnegative relation ajtai adversary claim oracle retries)
  split_ifs with valid
  · exact retryChance_mono relation ajtai adversary claim _ oracle
      fun fresh otherRetries forked otherValid collides =>
        rerunKernel_isSome relation ajtai adversary claim valid otherValid
          (fresh_eq_of_fork relation adversary claim oracle fresh forked) collides
  · exact le_rfl

end NightstreamFPrime.Lifecycle.RandomOracleBinding
