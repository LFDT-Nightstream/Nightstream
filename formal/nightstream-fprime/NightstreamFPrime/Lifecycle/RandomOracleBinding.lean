import NightstreamFPrime.Lifecycle.RandomOracleUniqueness
import NightstreamFPrime.Lifecycle.Nifs.BindingBridge

/-!
Owns the binding reduction of Lemma 6 (ROM_KNOWLEDGE_SOUNDNESS.md): a
function of the two runs that `RandomOracleUniqueness.collisionChance`
compares, which returns a short kernel vector of the same Ajtai key.

Inputs: a base run and a rerun that are `Paired`: both extract, and the rerun
outputs the base run's running and fresh statements.

Outputs:
- `rerunKernel`: at the first coordinate where the two complete `Π_RLC` forks
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

The vector is computed from the two runs; no step chooses it from a proof that
one exists. The extracted witnesses satisfy only the corrected ambient
relation, so the step uses relaxed binding, not ordinary binding. Does not
own: the hardness of the resulting MSIS instance or the running time of the
reduction.
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


/-! ## The binding reduction -/

/-- The two runs that the binding reduction compares: both extract, and the
rerun outputs the base run's running and fresh statements. -/
def Paired (oracle : Oracle) (retries : Retries) (other : Oracle) (otherRetries : Retries) : Prop :=
  Valid relation ajtai adversary claim oracle retries ∧
    Valid relation ajtai adversary claim other otherRetries ∧
    (claimed relation adversary claim other).running = (claimed relation adversary claim oracle).running ∧
    (claimed relation adversary claim other).fresh = (claimed relation adversary claim oracle).fresh

/-- Paired runs have the same `Π_RLC` input commitments. -/
theorem paired_phi {oracle other : Oracle} {retries otherRetries : Retries}
    (paired : Paired relation ajtai adversary claim oracle retries other otherRetries) :
    PiRLC.phi (batch relation ajtai oracle (claimed relation adversary claim oracle).running
        (claimed relation adversary claim oracle).fresh (claimed relation adversary claim oracle).proof).inputs =
      PiRLC.phi (batch relation ajtai other (claimed relation adversary claim other).running
        (claimed relation adversary claim other).fresh (claimed relation adversary claim other).proof).inputs := by
  rw [paired.2.2.1, paired.2.2.2]
  exact batch_phi relation ajtai _ _ _ _ _ _

/-- The `Π_RLC` assignments that a valid run's complete fork extracts, one for
each coordinate. -/
noncomputable def forkAssignments {oracle : Oracle} {retries : Retries}
    (valid : Valid relation ajtai adversary claim oracle retries) :=
  PiRLC.PaperForkExtraction.extractedAssignment (PaperExtractionAlgebra.extractionAlgebra ajtai)
    (ForkStrongSet.strongSetUnits Phi81StrongSet.lowNormInvertibility)
    (completeFork relation ajtai adversary claim oracle retries valid.1 valid.2)

/-- A relaxed collision under the key's `Π_RLC` semantics, as a short kernel
vector of the same key. -/
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

/-- The binding reduction. For paired runs whose forks extract different
assignments, the short kernel vector of the relaxed collision at the first
coordinate where the assignments differ; otherwise `none`. -/
noncomputable def rerunKernel (oracle : Oracle) (retries : Retries) (other : Oracle)
    (otherRetries : Retries) :
    Option (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound) :=
  if paired : Paired relation ajtai adversary claim oracle retries other otherRetries then
    if different : ∃ coordinate, forkAssignments relation ajtai adversary claim paired.1 coordinate ≠
        forkAssignments relation ajtai adversary claim paired.2.1 coordinate then
      some (kernelOf relation ajtai (PiRLC.PaperForkBinding.collisionAt
        (PaperExtractionAlgebra.extractionAlgebra ajtai)
        (Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
          (rows := productionProfile.commitmentWidth))
        (Nifs.BindingBridge.compatible relation ajtai)
        (ForkStrongSet.strongSetUnits Phi81StrongSet.lowNormInvertibility) _ _
        (completeFork relation ajtai adversary claim oracle retries paired.1.1 paired.1.2)
        (completeFork relation ajtai adversary claim other otherRetries paired.2.1.1 paired.2.1.2)
        (paired_phi relation ajtai adversary claim paired) (Fin.find _ different)
        (Fin.find_spec different)))
    else none
  else none

/-- Every pair of runs that `collisionChance` counts makes the reduction
return a vector: the extracted witnesses differ, so the forks' extracted
assignments differ at some coordinate. -/
theorem rerunKernel_isSome {oracle other : Oracle} {retries otherRetries : Retries}
    (valid : Valid relation ajtai adversary claim oracle retries)
    (otherValid : Valid relation ajtai adversary claim other otherRetries)
    (sameFresh : (claimed relation adversary claim other).fresh =
      (claimed relation adversary claim oracle).fresh)
    (collides : Collides relation ajtai adversary claim oracle retries other otherRetries) :
    (rerunKernel relation ajtai adversary claim oracle retries other otherRetries).isSome := by
  have paired : Paired relation ajtai adversary claim oracle retries other otherRetries :=
    ⟨valid, otherValid, collides.1, sameFresh⟩
  have different : ∃ coordinate, forkAssignments relation ajtai adversary claim paired.1 coordinate ≠
      forkAssignments relation ajtai adversary claim paired.2.1 coordinate := by
    by_contra same
    simp only [not_exists, not_not] at same
    apply collides.2
    unfold witnessOf
    rw [dif_pos otherValid, dif_pos valid]
    exact congrArg (fun assignments => some (Nifs.PaperStrongInterface.outputWitnessOfAssignments
      (ProductionKey.key relation ajtai) assignments)) (funext same).symm
  unfold rerunKernel
  rw [dif_pos paired, dif_pos different]
  rfl

/-- The chance that the reduction returns a vector, under the law of
`collisionChance`: the base run extracts, and its rerun from the same context
makes `rerunKernel` return a vector. -/
noncomputable def kernelChance : ℝ :=
  𝔼 oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
    (if Valid relation ajtai adversary claim oracle retries then
      retryChance relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle
        (fun other otherRetries =>
          (rerunKernel relation ajtai adversary claim oracle retries other otherRetries).isSome)
    else 0)

/-- The binding term of the knowledge error is at most the reduction's chance. -/
theorem collisionChance_le_kernelChance :
    collisionChance relation ajtai adversary claim ≤ kernelChance relation ajtai adversary claim := by
  unfold collisionChance kernelChance
  refine Finset.expect_le_expect fun oracle _ => Finset.sum_le_sum fun retries _ =>
    mul_le_mul_of_nonneg_left ?_ (weight_nonnegative relation ajtai adversary claim oracle retries)
  split_ifs with valid
  · exact retryChance_mono relation ajtai adversary claim _ oracle
      fun fresh otherRetries forked otherValid collides =>
        rerunKernel_isSome relation ajtai adversary claim valid otherValid
          (fresh_eq_of_fork relation adversary claim oracle fresh forked) collides
  · exact le_rfl

theorem kernelChance_nonnegative : 0 ≤ kernelChance relation ajtai adversary claim :=
  Finset.expect_nonneg fun oracle _ => Finset.sum_nonneg fun retries _ =>
    mul_nonneg (weight_nonnegative relation ajtai adversary claim oracle retries) (by
      split_ifs
      · exact retryChance_nonnegative relation ajtai adversary claim _ oracle _
      · exact le_rfl)

/-- The reduction's chance is a probability. -/
theorem kernelChance_le_one : kernelChance relation ajtai adversary claim ≤ 1 := by
  unfold kernelChance
  calc
    _ ≤ 𝔼 _oracle : Oracle, (1 : ℝ) := Finset.expect_le_expect fun oracle _ =>
        (Finset.sum_le_sum fun retries _ => mul_le_of_le_one_right
          (weight_nonnegative relation ajtai adversary claim oracle retries) (by
            split_ifs
            · exact retryChance_le_one relation ajtai adversary claim _ oracle _
            · exact zero_le_one)).trans (retryWeight_sum_le_one _)
    _ = 1 := Finset.expect_const Finset.univ_nonempty _

end NightstreamFPrime.Lifecycle.RandomOracleBinding
