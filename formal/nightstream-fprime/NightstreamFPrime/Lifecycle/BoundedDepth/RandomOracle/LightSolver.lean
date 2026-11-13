import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightUniqueness
import NightstreamFPrime.Lifecycle.RandomOracle.Solver
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.ChainBinding

/-! The light fold's MSIS solver in the random-oracle model: `Solver` on the light game
`LightFork.game`, with the binding step of `LightUniqueness` as the kernel.

The solver sees the adversary only through its run records, so its kernel reads a pair as a
view (`Reduction.View`). `boundedForkOf` is `LightUniqueness.boundedFork` on a valid view: the
base output's witness and the retry output's witness give a light Π_RLC fork at the chain
bound; the retry's opening is `retryHoldsOf`, the view form of `LightFork.retry_holds`.
`kernelOf` is `LightUniqueness.rerunKernel` on two views, and on the views of an adversary the
two are equal by definition (`rerunKernel_eq`). So `LightUniqueness.kernelChance` is the
reduction chance of `kernelOf` (`kernelChance_eq`).

`kernelChance_le_solver`: with at most `Q` queries, `budget · (kernelChance − solverChance) ≤
3Q + 61`. The solver (`Reduction.solve`) outputs a `ShortKernelVector` of the Ajtai key at
`msisChainBound p cap`, runs the adversary at most `budget` times, and is black-box
(`Reduction.solve_congr`). The work is `Solver.work` with one `ρ` coordinate and 58 challenges:
`1 + 2 · (Q + 1) + (Q + 58)` (`work_eq`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightSolver

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthFolds (freshToLight)
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
/-- A view of the light game: a base run and its one retry. -/
abbrev View :=
  Reduction.View (LightSchedule.schedule logicalWidth publicFits) 8 (Attempt logicalWidth publicFits)

section Fork

variable {key : CircuitKey logicalWidth publicFits} {view : View logicalWidth publicFits}

/-- `LightFork.retry_holds` on a view: a valid retry's witness opens the light Π_RLC output on
the base run's inputs at the retry's answer. -/
theorem retryHoldsOf
    (valid : Reduction.ForkValidOf (LightFork.game ajtai key) view LightFork.first) :
    Light.Rlc.Holds (keys key.relation ajtai).light.piRlcSemantics
      (view.forks LightFork.first).bound
      (LightFork.outputFor key.relation ajtai view.oracle
        (LightFork.runningOf view.base.link.start.1) (freshToLight view.base.link.fold.fresh)
        (LightFork.proofOf view.base.link.fold.proof)
        (Fork.readRho (view.retries LightFork.first)))
      (view.forks LightFork.first).witness := by
  have holds := (LightFork.succeeds_holds ajtai valid.1.1).2.2
  have identifies := Fork.rhoPoint_identifies LightFork.first valid.1.2
  have statementEq : freshToLight (view.forks LightFork.first).link.fold.fresh =
      freshToLight view.base.link.fold.fresh :=
    identifies.1
  have proofEq : LightFork.proofOf (view.forks LightFork.first).link.fold.proof =
      LightFork.proofOf view.base.link.fold.proof :=
    congrArg (fun messages : (LightSchedule.schedule logicalWidth publicFits).Messages 8 =>
      (⟨messages.rounds, messages.output⟩ : Light.Proof LightShape 8)) identifies.2
  have startEq : (view.forks LightFork.first).link.start = view.base.link.start := valid.2.2
  rw [statementEq, proofEq, startEq] at holds
  have outputEq : LightFork.outputFor key.relation ajtai
      (Reduction.forkedOf (LightFork.game ajtai key) view LightFork.first)
      (LightFork.runningOf view.base.link.start.1) (freshToLight view.base.link.fold.fresh)
      (LightFork.proofOf view.base.link.fold.proof)
      (LightFork.challenge (Reduction.forkedOf (LightFork.game ajtai key) view LightFork.first)
        (freshToLight view.base.link.fold.fresh) (LightFork.proofOf view.base.link.fold.proof)) =
      LightFork.outputFor key.relation ajtai view.oracle
        (LightFork.runningOf view.base.link.start.1) (freshToLight view.base.link.fold.fresh)
        (LightFork.proofOf view.base.link.fold.proof)
        (Fork.readRho (view.retries LightFork.first)) :=
    (LightFork.outputFor_update key.relation ajtai view.oracle
      (LightFork.runningOf view.base.link.start.1) (freshToLight view.base.link.fold.fresh)
      (LightFork.proofOf view.base.link.fold.proof) (view.retries LightFork.first) _).trans
      (congrArg (LightFork.outputFor key.relation ajtai view.oracle
        (LightFork.runningOf view.base.link.start.1) (freshToLight view.base.link.fold.fresh)
        (LightFork.proofOf view.base.link.fold.proof))
        (LightFork.challenge_update view.oracle (freshToLight view.base.link.fold.fresh)
          (LightFork.proofOf view.base.link.fold.proof) (view.retries LightFork.first)))
  rw [outputEq] at holds
  exact holds

/-- `LightUniqueness.boundedFork` on a valid view: the fork at the chain bound. -/
noncomputable def boundedForkOf (valid : Reduction.ValidOf (LightFork.game ajtai key) view) :
    Light.Rlc.Fork (keys key.relation ajtai).light.piRlcAlgebra (keys key.relation ajtai).light.one
      (BoundedDepthReverse.chainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)
      (keys key.relation ajtai).light.relationSource
      (LightTest.oracleProbe key.relation ajtai (LightFork.runningOf view.base.link.start.1)
        view.oracle (freshToLight view.base.link.fold.fresh)
        (LightFork.proofOf view.base.link.fold.proof)).coins.roundPoint
      (LightFork.outputs key.relation ajtai view.oracle (LightFork.runningOf view.base.link.start.1)
        (freshToLight view.base.link.fold.fresh) (LightFork.proofOf view.base.link.fold.proof)
        (keys key.relation ajtai).light.freshSource)
      (LightFork.outputs key.relation ajtai view.oracle (LightFork.runningOf view.base.link.start.1)
        (freshToLight view.base.link.fold.fresh) (LightFork.proofOf view.base.link.fold.proof)
        (keys key.relation ajtai).light.runningSource) where
  baseChallenge := LightFork.challenge view.oracle (freshToLight view.base.link.fold.fresh)
    (LightFork.proofOf view.base.link.fold.proof)
  forkChallenge := Fork.readRho (view.retries LightFork.first)
  baseAssignment := view.base.witness
  forkAssignment := (view.forks LightFork.first).witness
  baseValid := Fork.readRho_valid _
  forkValid := Fork.readRho_valid _
  changed same := (valid.2 LightFork.first).2.1 ((Fork.readRho_eq_iff _ _).mp same).symm
  baseHolds := LightUniqueness.lightHolds_mono ajtai (LightFork.succeeds_holds ajtai valid.1).2.2
    valid.1.2.2.2.2.2
  forkHolds := LightUniqueness.lightHolds_mono ajtai (retryHoldsOf ajtai (valid.2 LightFork.first))
    (valid.2 LightFork.first).1.1.2.2.2.2.2

end Fork

section Kernel

variable (key : CircuitKey logicalWidth publicFits)

/-- `LightUniqueness.rerunKernel` on two views: different fresh witnesses through
`Light.Rlc.collisionAbove`, else different running witnesses through
`Light.Rlc.runningCollisionAbove`, both on the forks at the chain bound. -/
noncomputable def kernelOf (view other : View logicalWidth publicFits) :
    Option (RingRelation.PiRLCAlgebra.Binding.ShortKernelVector ajtai
      (Folding.Schedule.msisChainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)) :=
  if paired : Reduction.ValidOf (LightFork.game ajtai key) view ∧
      Reduction.ValidOf (LightFork.game ajtai key) other ∧
      freshToLight other.base.link.fold.fresh = freshToLight view.base.link.fold.fresh ∧
      other.base.link.start = view.base.link.start then
    have freshCommitment :
        (LightFork.outputs key.relation ajtai view.oracle
          (LightFork.runningOf view.base.link.start.1) (freshToLight view.base.link.fold.fresh)
          (LightFork.proofOf view.base.link.fold.proof)
          (keys key.relation ajtai).light.freshSource).commitment =
        (LightFork.outputs key.relation ajtai other.oracle
          (LightFork.runningOf other.base.link.start.1) (freshToLight other.base.link.fold.fresh)
          (LightFork.proofOf other.base.link.fold.proof)
          (keys key.relation ajtai).light.freshSource).commitment :=
      LightUniqueness.outputs_commitment_eq ajtai
        (congrArg (fun start => LightFork.runningOf start.1) paired.2.2.2.symm) paired.2.2.1.symm
    have runningCommitment :
        (LightFork.outputs key.relation ajtai view.oracle
          (LightFork.runningOf view.base.link.start.1) (freshToLight view.base.link.fold.fresh)
          (LightFork.proofOf view.base.link.fold.proof)
          (keys key.relation ajtai).light.runningSource).commitment =
        (LightFork.outputs key.relation ajtai other.oracle
          (LightFork.runningOf other.base.link.start.1) (freshToLight other.base.link.fold.fresh)
          (LightFork.proofOf other.base.link.fold.proof)
          (keys key.relation ajtai).light.runningSource).commitment :=
      LightUniqueness.outputs_commitment_eq ajtai
        (congrArg (fun start => LightFork.runningOf start.1) paired.2.2.2.symm) paired.2.2.1.symm
    if freshDifferent :
        (boundedForkOf ajtai paired.1).freshAssignment (lightExtraction ajtai)
            LightFork.strongSet ≠
          (boundedForkOf ajtai paired.2.1).freshAssignment (lightExtraction ajtai)
            LightFork.strongSet then
      some (RingRelation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel ajtai _ le_rfl
        (LightUniqueness.lightCollision ajtai (relation := key.relation)
          (Light.Rlc.collisionAbove (lightExtraction ajtai) _
          (LightUniqueness.compatible ajtai key.relation) LightFork.strongSet rfl
          (fun left right => Nifs.BindingBridge.differenceBounded key.relation ajtai _ left right)
          (boundedForkOf ajtai paired.1) (boundedForkOf ajtai paired.2.1)
          freshCommitment freshDifferent)))
    else if runningDifferent :
        (boundedForkOf ajtai paired.1).runningAssignment (lightExtraction ajtai)
            LightFork.strongSet ≠
          (boundedForkOf ajtai paired.2.1).runningAssignment (lightExtraction ajtai)
            LightFork.strongSet then
      some (RingRelation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel ajtai _ le_rfl
        (LightUniqueness.lightCollision ajtai (relation := key.relation)
          (Light.Rlc.runningCollisionAbove (lightExtraction ajtai) _
          (LightUniqueness.compatible ajtai key.relation) LightFork.strongSet rfl
          (fun left right => Nifs.BindingBridge.differenceBounded key.relation ajtai _ left right)
          (boundedForkOf ajtai paired.1) (boundedForkOf ajtai paired.2.1)
          freshCommitment runningCommitment (not_not.mp freshDifferent) runningDifferent)))
    else none
  else none

variable (adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
  (Attempt logicalWidth publicFits))

/-- On an adversary's views, `kernelOf` is `LightUniqueness.rerunKernel`. -/
theorem rerunKernel_eq (oracle : LightFork.Oracle logicalWidth publicFits)
    (retries : LightUniqueness.Retries logicalWidth publicFits)
    (other : LightFork.Oracle logicalWidth publicFits)
    (otherRetries : LightUniqueness.Retries logicalWidth publicFits) :
    LightUniqueness.rerunKernel ajtai key adversary oracle retries other otherRetries =
      kernelOf ajtai key (Reduction.viewOf (LightFork.game ajtai key) adversary oracle retries)
        (Reduction.viewOf (LightFork.game ajtai key) adversary other otherRetries) :=
  rfl

/-- The light kernel chance is the reduction chance of `kernelOf`. -/
theorem kernelChance_eq :
    LightUniqueness.kernelChance ajtai key adversary =
      Uniqueness.reductionChance (LightFork.game ajtai key) adversary
        fun oracle retries other otherRetries =>
          (kernelOf ajtai key (Reduction.viewOf (LightFork.game ajtai key) adversary oracle retries)
            (Reduction.viewOf (LightFork.game ajtai key) adversary other otherRetries)).isSome :=
  rfl

/-- The light MSIS solver: `Reduction.solve` with kernel `kernelOf`. -/
noncomputable def solve (budget : ℕ)
    (tape : Fin budget → LightFork.Oracle logicalWidth publicFits) :
    Option (RingRelation.PiRLCAlgebra.Binding.ShortKernelVector ajtai
      (Folding.Schedule.msisChainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)) :=
  Reduction.solve (LightFork.game ajtai key) (kernelOf ajtai key) adversary budget tape

/-- The chance that the light solver outputs a kernel vector with `budget` adversary runs. -/
noncomputable def solverChance (budget : ℕ) : ℝ :=
  Solver.solverChance (LightFork.game ajtai key) adversary (kernelOf ajtai key) budget

theorem work_eq (queries : Nat) :
    Solver.work (LightSchedule.schedule logicalWidth publicFits) queries = 3 * queries + 61 := by
  unfold Solver.work
  rw [LightTest.challenges_length]
  show 1 + 2 * 1 * (queries + 1) + (queries + 58) = 3 * queries + 61
  omega

/-- The light kernel chance is at most the light solver's chance plus `(3Q + 61) / budget`. -/
theorem kernelChance_le_solver {queries : Nat} (bounded : adversary.QueryBound queries)
    (budget : ℕ) :
    budget * (LightUniqueness.kernelChance ajtai key adversary -
        solverChance ajtai key adversary budget) ≤ 3 * queries + 61 := by
  have bound := Solver.reductionChance_le_solver (LightFork.game ajtai key) adversary
    (kernelOf ajtai key) bounded budget
  rw [work_eq] at bound
  push_cast at bound
  exact bound

end Kernel

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightSolver
