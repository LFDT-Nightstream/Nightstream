import NightstreamFPrime.Lifecycle.RandomOracle.Fork
import NightstreamFPrime.Lifecycle.OptionB.PriorLink

/-! The part of the option-B fork step that the light and refresh folds share. An adversary
outputs a NIFS attempt of option B (`Attempt`): a prior state, the slot it starts from, a fold,
a bound and a witness. The attempt succeeds (`Succeeds`) in the shape of the walk's
`BoundedDepthReverse.Succeeds`: the prior link (`OptionB.PriorLink`), the bounded-depth verifier
with fold verifiers `folds`, and the output claim open at the bound (`SlotHolds`, the walk's
`ClaimHolds`). The fork game of one fold (`game`) fixes the fold's circuit key and reads its
verifiers from the oracle.

The transcripts absorb the fresh instance and not the start slot. So a retry that moves the
start slot has a fresh instance of the base run and a different slot, and `priorLink_binds`
turns it into a collision of the two prior states (`mismatchChance_le`). `fork_failure_le` is
`Fork.fork_failure_le` with that collision chance (`collisionChance`) in place of the
mismatch chance. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- An option-B NIFS attempt: `BoundedDepthReverse.Attempt` at option B's types. -/
abbrev Attempt (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  BoundedDepthReverse.Attempt (PriorPreimage logicalWidth publicFits)
    (CircuitKey logicalWidth publicFits) (SlotClaim logicalWidth publicFits)
    (StepFresh logicalWidth publicFits) StepProof
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))

abbrev Folds (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  BoundedDepth.Folds (CircuitKey logicalWidth publicFits) (SlotClaim logicalWidth publicFits)
    (StepFresh logicalWidth publicFits) StepProof

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- A slot claim opens at norm `bound` under the relation of `key`, with true evaluations. -/
def SlotHolds (key : CircuitKey logicalWidth publicFits) (bound : Nat)
    (claim : SlotClaim logicalWidth publicFits)
    (witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Prop :=
  Spec.Folding.Light.Rlc.Holds (PaperAlgebra.semantics ajtai) bound
    (toClaim (key.system ajtai) claim) witness

/-- The NIFS success event of option B with the fold verifiers `folds`: the prior link, the
bounded-depth verifier with the light cap, and the output claim open at a bound within the
chain bound. -/
def Succeeds (folds : Folds logicalWidth publicFits) (attempt : Attempt logicalWidth publicFits) :
    Prop :=
  BoundedDepthReverse.Succeeds productionGlobalParams Folding.Schedule.productionLightFoldCap
    folds PriorLink (SlotHolds ajtai) attempt

/-- Two successful attempts with one fresh instance start from one slot, or their prior states
collide (`priorLink_binds`). -/
theorem start_eq_or_collision {folds folds' : Folds logicalWidth publicFits}
    {attempt attempt' : Attempt logicalWidth publicFits}
    (succeeds : Succeeds ajtai folds attempt) (succeeds' : Succeeds ajtai folds' attempt')
    (sameFresh : attempt.link.fold.fresh = attempt'.link.fold.fresh) :
    attempt.link.start = attempt'.link.start ∨
      StateCollision (toState attempt.link.prior) (toState attempt'.link.prior) :=
  priorLink_binds succeeds.1 succeeds'.1 sameFresh

section Game

variable {schedule : Schedule} (key : CircuitKey logicalWidth publicFits)
  (folds : (Point schedule 8 → Answer) → Folds logicalWidth publicFits)
  (statement : Attempt logicalWidth publicFits → schedule.Statement)
  (messages : Attempt logicalWidth publicFits → schedule.Messages 8)

/-- The fork game of one option-B fold with circuit key `key`: the fold verifiers read from the
oracle, the transcript's statement and messages of an attempt, and its start slot. -/
def game :
    Fork.Game schedule 8 (Attempt logicalWidth publicFits) (SlotData logicalWidth publicFits) where
  statement := statement
  messages := messages
  start attempt := attempt.link.start
  Succeeds oracle attempt := attempt.link.fold.key = key ∧ Succeeds ajtai (folds oracle) attempt

variable (adversary : OracleComp (Point schedule 8) Answer (Attempt logicalWidth publicFits))

/-- The chance that a retry at `index` gives a collision of the base and retry prior states. -/
noncomputable def collisionChance (index : Fin schedule.rhoCount)
    (oracle : Point schedule 8 → Answer) : ℝ :=
  Fork.retryChance (game ajtai key folds statement messages) adversary (fun answer =>
    StateCollision (toState (adversary.run oracle).link.prior)
      (toState (adversary.run (Function.update oracle
        (Fork.rhoPoint (game ajtai key folds statement messages) adversary index oracle)
        answer)).link.prior)) index oracle

variable {statement}

/-- A retry that moves the start slot gives a prior-state collision, when the statement fixes
the fresh instance. -/
theorem mismatchChance_le
    (freshOf : ∀ {attempt attempt'}, statement attempt = statement attempt' →
      attempt.link.fold.fresh = attempt'.link.fold.fresh)
    {oracle : Point schedule 8 → Answer}
    (succeeds : Fork.Succeeds (game ajtai key folds statement messages) adversary oracle)
    (index : Fin schedule.rhoCount) :
    Fork.mismatchChance (game ajtai key folds statement messages) adversary index oracle ≤
      collisionChance ajtai key folds statement messages adversary index oracle := by
  apply Fork.retryChance_mono
  intro answer retry moved
  obtain ⟨⟨_, retrySucceeds⟩, samePoint⟩ := retry
  have sameStatement := (Fork.rhoPoint_identifies index samePoint).1
  rcases start_eq_or_collision ajtai succeeds.2 retrySucceeds (freshOf sameStatement.symm) with
    same | collision
  · exact absurd same.symm moved
  · exact collision

/-- `Fork.fork_failure_le` for an option-B fold: a successful run whose retry family is invalid
has probability at most `rhoCount · (Q + rhoCount) · sampleError` plus the expected chance of a
prior-state collision between the base run and a retry. -/
theorem fork_failure_le
    (freshOf : ∀ {attempt attempt'}, statement attempt = statement attempt' →
      attempt.link.fold.fresh = attempt'.link.fold.fresh)
    {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Fork.Succeeds (game ajtai key folds statement messages) adversary oracle then
        ∑ retries, retryWeight
            (Fork.retrySets (game ajtai key folds statement messages) adversary oracle) retries *
          (if ∀ index, Fork.ForkValid (game ajtai key folds statement messages) adversary oracle
            retries index then 0 else 1)
      else 0) ≤
      schedule.rhoCount * (((queries + schedule.rhoCount : Nat) : ℝ) * Fork.sampleError) +
        𝔼 oracle, (if Fork.Succeeds (game ajtai key folds statement messages) adversary oracle
          then ∑ index, collisionChance ajtai key folds statement messages adversary index oracle
          else 0) :=
  Fork.fork_failure_le_of_mismatch _ adversary bounded _ fun _ succeeds index =>
    mismatchChance_le ajtai key folds messages adversary freshOf succeeds index

end Game

end NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink
