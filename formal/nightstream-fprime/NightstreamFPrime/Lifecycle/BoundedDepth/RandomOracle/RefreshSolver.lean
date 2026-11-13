import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshUniqueness
import NightstreamFPrime.Lifecycle.RandomOracle.Solver
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.ChainBinding

/-! The refresh fold's MSIS solver in the random-oracle model: `Solver` on the refresh game
`RefreshFork.game`, with the binding step of `RefreshUniqueness` as the kernel.

The solver sees the adversary only through its run records, so its kernel reads a pair as a
view (`Reduction.View`). `forkOf` is `RefreshFork.fork` on a valid view: the base response and
the 17 retry responses on the base batch (SuperNeo v1.2 Appendix B.3), with each retry's opening
from `retryHoldsOf`, the view form of `RefreshFork.retry_holds`. Every response opens at the
chain bound (`responsesBoundedOf`). `kernelOf` is `RefreshUniqueness.rerunKernel` on two views,
and on the views of an adversary the two are equal by definition (`rerunKernel_eq`). So
`RefreshUniqueness.kernelChance` is the reduction chance of `kernelOf` (`kernelChance_eq`).

`kernelChance_le_solver`: with at most `Q` queries, `budget · (kernelChance − solverChance) ≤
35Q + 653`, the work bound of the one-circuit solver (`RandomOracleSolver.kernelChance_le_solver`).
The work is `Solver.work` with 17 `ρ` coordinates and 74 challenges:
`1 + 2 · 17 · (Q + 17) + (Q + 74)` (`work_eq`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshSolver

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction (extractedAssignment)
open NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork (normFreeLaws)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt)

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

variable (logicalWidth publicFits) in
/-- A view of the refresh game: a base run and its 17 retries. -/
abbrev View :=
  Reduction.View (RefreshSchedule.schedule logicalWidth publicFits) 8
    (Attempt logicalWidth publicFits)

section Fork

variable {key : CircuitKey logicalWidth publicFits} {view : View logicalWidth publicFits}

/-- `RefreshFork.retry_holds` on a view: a valid retry at `index` opens the parent of the base
run's batch at the retry's challenges. -/
theorem retryHoldsOf {index : Fin (RefreshSchedule.schedule logicalWidth publicFits).rhoCount}
    (valid : Reduction.ForkValidOf (RefreshFork.game ajtai key) view index) :
    Light.Rlc.Holds (refreshKey key.relation ajtai).piRlcSemantics (view.forks index).bound
      (RefreshFork.parentFor key.relation ajtai view.oracle
        (toClaim (key.system ajtai) view.base.link.start.1) view.base.link.fold.fresh
        (RefreshFork.proofOf view.base.link.fold.proof)
        (Fork.rho (Reduction.forkedOf (RefreshFork.game ajtai key) view index)
          ((RefreshFork.game ajtai key).statement (view.forks index))
          ((RefreshFork.game ajtai key).messages (view.forks index))))
      (view.forks index).witness := by
  have holds := (RefreshFork.succeeds_holds ajtai valid.1.1).2.2.2
  obtain ⟨statementEq, messagesEq⟩ := valid.identifies
  have freshEq : (view.forks index).link.fold.fresh = view.base.link.fold.fresh :=
    congrArg Prod.fst statementEq
  have splitEq : (RefreshFork.proofOf (view.forks index).link.fold.proof).split =
      (RefreshFork.proofOf view.base.link.fold.proof).split :=
    congrArg Prod.snd statementEq
  have messagesEq' :
      RefreshSchedule.messages (RefreshFork.proofOf (view.forks index).link.fold.proof).nifs =
        RefreshSchedule.messages (RefreshFork.proofOf view.base.link.fold.proof).nifs :=
    messagesEq
  have startEq : (view.forks index).link.start = view.base.link.start := valid.2.2
  have batchEq : RefreshFork.batch key.relation ajtai
      (Reduction.forkedOf (RefreshFork.game ajtai key) view index)
      (toClaim (key.system ajtai) (view.forks index).link.start.1)
      (view.forks index).link.fold.fresh
      (RefreshFork.proofOf (view.forks index).link.fold.proof) =
      RefreshFork.batch key.relation ajtai view.oracle
        (toClaim (key.system ajtai) view.base.link.start.1) view.base.link.fold.fresh
        (RefreshFork.proofOf view.base.link.fold.proof) := by
    rw [RefreshFork.batch_congr key.relation ajtai _ _ _ splitEq messagesEq', freshEq, startEq]
    exact RefreshFork.batch_update key.relation ajtai view.oracle
      (toClaim (key.system ajtai) view.base.link.start.1) view.base.link.fold.fresh
      (RefreshFork.proofOf view.base.link.fold.proof) index (view.retries index)
  unfold RefreshFork.parentFor at holds ⊢
  rw [batchEq] at holds
  exact holds

/-- `RefreshFork.fork` on a valid view: the complete fork without norms on the base batch. -/
noncomputable def forkOf (valid : Reduction.ValidOf (RefreshFork.game ajtai key) view) :
    RefreshFork.RlcFork ajtai key view.oracle view.base where
  base := ⟨Fork.rho view.oracle ((RefreshFork.game ajtai key).statement view.base)
      ((RefreshFork.game ajtai key).messages view.base),
    view.base.witness⟩
  forks index :=
    ⟨Fork.rho (Reduction.forkedOf (RefreshFork.game ajtai key) view index)
        ((RefreshFork.game ajtai key).statement (view.forks index))
        ((RefreshFork.game ajtai key).messages (view.forks index)),
      (view.forks index).witness⟩
  baseSuccess :=
    let holds := (RefreshFork.succeeds_holds ajtai valid.1).2.2.2
    PiRLC.NormFreeFork.success_at _ _ holds.1 holds.2.1 holds.2.2
  forkSuccess index :=
    let holds := retryHoldsOf ajtai (valid.2 index)
    PiRLC.NormFreeFork.success_at _ _ holds.1 holds.2.1 holds.2.2
  baseStrong _ := Fork.readRho_valid _
  forkStrong _ _ := Fork.readRho_valid _
  agreeExcept index _ different := (valid.2 index).agree different
  changed index := (valid.2 index).changed

/-- `RefreshUniqueness.responsesBounded` on a valid view: every response opens at the chain
bound. -/
theorem responsesBoundedOf (valid : Reduction.ValidOf (RefreshFork.game ajtai key) view) :
    PiRLC.ChainBinding.ResponsesBounded
      (BoundedDepthReverse.chainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)
      (forkOf ajtai valid) := by
  refine ⟨fun column => Nat.lt_of_lt_of_le
    ((RefreshFork.succeeds_holds ajtai valid.1).2.2.2.1.2.2 column) valid.1.2.2.2.2.2,
    fun index column => ?_⟩
  exact Nat.lt_of_lt_of_le
    ((RefreshFork.succeeds_holds ajtai (valid.2 index).1.1).2.2.2.1.2.2 column)
    (valid.2 index).1.1.2.2.2.2.2

end Fork

section Kernel

variable (key : CircuitKey logicalWidth publicFits)

/-- `RefreshUniqueness.rerunKernel` on two views: when the extracted witnesses differ at a
coordinate, `PiRLC.ChainBinding.forkCollision` at the chain bound. -/
noncomputable def kernelOf (view other : View logicalWidth publicFits) :
    Option (RingRelation.PiRLCAlgebra.Binding.ShortKernelVector ajtai
      (Folding.Schedule.msisChainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)) :=
  if paired : Reduction.ValidOf (RefreshFork.game ajtai key) view ∧
      Reduction.ValidOf (RefreshFork.game ajtai key) other ∧
      (RefreshFork.game ajtai key).statement other.base =
        (RefreshFork.game ajtai key).statement view.base ∧
      other.base.link.start = view.base.link.start then
    if different : ∃ coordinate,
        extractedAssignment (normFreeLaws (PaperExtractionAlgebra.extractionAlgebra ajtai))
            RefreshFork.strongSet (forkOf ajtai paired.1) coordinate ≠
          extractedAssignment (normFreeLaws (PaperExtractionAlgebra.extractionAlgebra ajtai))
            RefreshFork.strongSet (forkOf ajtai paired.2.1) coordinate then
      some (RingRelation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel ajtai _ le_rfl
        (UniquenessLink.ajtaiCollision ajtai (PiRLC.ChainBinding.forkCollision
          (PaperExtractionAlgebra.extractionAlgebra ajtai) _
          (Nifs.BindingBridge.compatible key.relation ajtai) RefreshFork.strongSet
          (fun left right => Nifs.BindingBridge.differenceBounded key.relation ajtai _ left right)
          (forkOf ajtai paired.1) (forkOf ajtai paired.2.1)
          (responsesBoundedOf ajtai paired.1) (responsesBoundedOf ajtai paired.2.1)
          (RefreshUniqueness.batch_phi ajtai
            (congrArg (fun start => toClaim (key.system ajtai) start.1) paired.2.2.2.symm)
            (congrArg Prod.fst paired.2.2.1.symm) (congrArg Prod.snd paired.2.2.1.symm))
          (Classical.choose different) (Classical.choose_spec different))))
    else none
  else none

variable (adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8)
  Answer (Attempt logicalWidth publicFits))

/-- On an adversary's views, `kernelOf` is `RefreshUniqueness.rerunKernel`. -/
theorem rerunKernel_eq (oracle : RefreshFork.Oracle logicalWidth publicFits)
    (retries : RefreshUniqueness.Retries logicalWidth publicFits)
    (other : RefreshFork.Oracle logicalWidth publicFits)
    (otherRetries : RefreshUniqueness.Retries logicalWidth publicFits) :
    RefreshUniqueness.rerunKernel ajtai key adversary oracle retries other otherRetries =
      kernelOf ajtai key (Reduction.viewOf (RefreshFork.game ajtai key) adversary oracle retries)
        (Reduction.viewOf (RefreshFork.game ajtai key) adversary other otherRetries) :=
  rfl

/-- The refresh kernel chance is the reduction chance of `kernelOf`. -/
theorem kernelChance_eq :
    RefreshUniqueness.kernelChance ajtai key adversary =
      Uniqueness.reductionChance (RefreshFork.game ajtai key) adversary
        fun oracle retries other otherRetries =>
          (kernelOf ajtai key
            (Reduction.viewOf (RefreshFork.game ajtai key) adversary oracle retries)
            (Reduction.viewOf (RefreshFork.game ajtai key) adversary other otherRetries)).isSome :=
  rfl

/-- The refresh MSIS solver: `Reduction.solve` with kernel `kernelOf`. -/
noncomputable def solve (budget : ℕ)
    (tape : Fin budget → RefreshFork.Oracle logicalWidth publicFits) :
    Option (RingRelation.PiRLCAlgebra.Binding.ShortKernelVector ajtai
      (Folding.Schedule.msisChainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)) :=
  Reduction.solve (RefreshFork.game ajtai key) (kernelOf ajtai key) adversary budget tape

/-- The chance that the refresh solver outputs a kernel vector with `budget` adversary runs. -/
noncomputable def solverChance (budget : ℕ) : ℝ :=
  Solver.solverChance (RefreshFork.game ajtai key) adversary (kernelOf ajtai key) budget

theorem work_eq (queries : Nat) :
    Solver.work (RefreshSchedule.schedule logicalWidth publicFits) queries =
      35 * queries + 653 := by
  unfold Solver.work
  rw [RefreshTest.challenges_length]
  show 1 + 2 * 17 * (queries + 17) + (queries + 74) = 35 * queries + 653
  omega

/-- The refresh kernel chance is at most the refresh solver's chance plus
`(35Q + 653) / budget`. -/
theorem kernelChance_le_solver {queries : Nat} (bounded : adversary.QueryBound queries)
    (budget : ℕ) :
    budget * (RefreshUniqueness.kernelChance ajtai key adversary -
        solverChance ajtai key adversary budget) ≤ 35 * queries + 653 := by
  have bound := Solver.reductionChance_le_solver (RefreshFork.game ajtai key) adversary
    (kernelOf ajtai key) bounded budget
  rw [work_eq] at bound
  push_cast at bound
  exact bound

end Kernel

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshSolver
