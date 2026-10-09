import NightstreamFPrime.Lifecycle.RandomOracleKnowledge
import NightstreamFPrime.Lifecycle.RandomOracleBinding
import NightstreamFPrime.Lifecycle.RandomOracleFidelity
import NightstreamFPrime.Export.Stage1.NifsRealSuccess
import NightstreamFPrime.Layout.Stage1.PiCCSSecurity

/-!
Owns the random-oracle knowledge theorem with the prior-state link: the
verifier accepts a claim only when the adversary's prior preimage links the
running statement to the absorbed digest (owner decision 2026-10-06).

Inputs: an oracle adversary whose output gives a claim and a prior preimage,
and the verifier-owned context digest.

Outputs:
- `linkedClaim`: the claim whose `linked` check is `PriorLink`;
- `succeeds_iff_realSuccess`: at an oracle that answers the claim's
  challenges as the sponge does, the linked success event is
  `NifsRealSuccess.RealSuccess`;
- `link_collision`: two linked claims with one fresh statement and different
  running statements give a `StateHashCollision`;
- `mismatch_collision`, `moves_collision`: every `Π_RLC` retry (Lemma 5) and
  every binding-reduction rerun (Lemma 6) that changes the running statement
  gives such a collision;
- `knowledge_error_le_linked`: the knowledge theorem with those two error
  terms replaced by state-hash collision events (`hashMismatchChance`,
  `hashRunningChance`). The third named term, `collisionChance`, is at most
  the chance that the binding reduction returns a short kernel vector of the
  key (`RandomOracleBinding.collisionChance_le_kernelChance`).

Does not own: the hardness of the state hash or of MSIS.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RandomOracleLink

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
open NightstreamFPrime.Lifecycle.RandomOracleKnowledge
open NightstreamFPrime.Layout.Stage1.PiCCSSecurity

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Two linked preimages for one fresh statement and different running
statements collide: their hashes both equal the absorbed prior digest. -/
theorem link_collision {left right : Lifecycle.HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {running running' : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)} {context : KeyDigest}
    (link : PriorLink left running fresh context) (link' : PriorLink right running' fresh context)
    (moved : running' ≠ running) :
    StateHashCollision left right := by
  rcases stateHash_identifies_statement_or_collision left right link.wellFormed link'.wellFormed
      (link.digest.symm.trans link'.digest) with same | collision
  · exfalso
    apply moved
    calc running' = right.running functionIndex := link'.running_eq.symm
      _ = left.running functionIndex := by rw [same]
      _ = running := link.running_eq
  · exact collision

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  {Output : Type}
  (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
    Answer Output)
  (claim : Output → Claim relation) (prior : Output → Lifecycle.HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) (contextDigest : KeyDigest)

local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer
local notation "Retries" => Fin Nifs.PaperProfile.arity.total → Answer

/-- The claim with the verifier's prior-state check as its `linked` field. -/
def linkedClaim (output : Output) : Claim relation :=
  { claim output with
    linked := PriorLink (prior output) (claim output).running (claim output).fresh contextDigest }

local notation "Linked" => linkedClaim relation claim prior contextDigest

/-- A claim passes the `linked` check exactly when its prior preimage links it. -/
theorem linked_iff (oracle : Oracle) :
    (claimed relation adversary Linked oracle).linked ↔
      PriorLink (prior (adversary.run oracle)) (claimed relation adversary Linked oracle).running
        (claimed relation adversary Linked oracle).fresh contextDigest :=
  Iff.rfl

/-- The adversary's output as a real-success output. -/
def realOutput (output : Output) : NifsRealSuccess.RealOutput relation :=
  ⟨prior output, (claim output).proof, (claim output).children⟩

/-- At an oracle that answers the claim's challenges as the sponge does, the
linked success event is the real success event: `PriorLink`, acceptance by
`PaperNonInteractive.verify`, and openings of the exact returned children. -/
theorem succeeds_iff_realSuccess (oracle : Oracle)
    (deployed : RandomOracleFidelity.Deployed oracle (claimed relation adversary Linked oracle).fresh (claimed relation adversary Linked oracle).proof) :
    Succeeds relation ajtai adversary Linked oracle ↔
      NifsRealSuccess.RealSuccess relation ajtai contextDigest (claimed relation adversary Linked oracle).running (claimed relation adversary Linked oracle).fresh
        (some (realOutput relation claim prior (adversary.run oracle))) := by
  have attemptEq := RandomOracleFidelity.attempt_eq relation ajtai (claimed relation adversary Linked oracle).running deployed
  have opens (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (verified : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
        (claimed relation adversary Linked oracle).running (claimed relation adversary Linked oracle).fresh (claimed relation adversary Linked oracle).proof = some result)
      (child : Fin (ProductionKey.key relation ajtai).params.k) :=
    PiDEC.OutputWitnessConsumer.runningStatement_eq_child (ProductionKey.key relation ajtai)
      (claimed relation adversary Linked oracle).running (claimed relation adversary Linked oracle).fresh (claimed relation adversary Linked oracle).proof result _ attemptEq verified child
  unfold Succeeds
  rw [RandomOracleFidelity.accepts_iff_verify relation ajtai _ _ deployed]
  constructor
  · rintro ⟨⟨⟨result, verified⟩, holds⟩, link⟩
    refine ⟨link, result, _, verified, attemptEq, fun child => ?_⟩
    have opened := holds (Fin.cast (ProductionKey.key relation ajtai).outputCount_eq.symm child)
    rw [← opens result verified] at opened
    exact opened
  · rintro ⟨link, result, returned, verified, returnedEq, holds⟩
    cases Option.some.inj (returnedEq.symm.trans attemptEq)
    refine ⟨⟨⟨result, verified⟩, fun child => ?_⟩, link⟩
    rw [← opens result verified]
    exact holds _

/-- A `Π_RLC` retry that changes the running statement is a state-hash
collision: the retry hits the same oracle point, so it outputs the same fresh
statement (`rhoPoint_identifies`). -/
theorem mismatch_collision {oracle : Oracle} {index : Fin Nifs.PaperProfile.arity.total}
    {answer : Answer}
    (succeeds : Succeeds relation ajtai adversary Linked oracle)
    (retry : answer ∈ retrySets relation ajtai adversary Linked oracle index)
    (moved : (claimed relation adversary Linked
        (Function.update oracle (rhoPoint relation adversary Linked index oracle) answer)).running ≠
      (claimed relation adversary Linked oracle).running) :
    StateHashCollision (prior (adversary.run oracle))
      (prior (adversary.run (Function.update oracle (rhoPoint relation adversary Linked index oracle) answer))) := by
  have sameFresh := (rhoPoint_identifies index retry.2).1
  have link := (linked_iff relation adversary claim prior contextDigest oracle).mp succeeds.2
  have link' := (linked_iff relation adversary claim prior contextDigest _).mp retry.1.2
  rw [sameFresh] at link'
  exact link_collision link link' moved

/-- A binding-reduction rerun that changes the running statement is a
state-hash collision: the rerun forks at the same index, so it outputs the
same fresh statement (`fresh_eq_of_fork`). -/
theorem moves_collision {oracle : Oracle} {retries otherRetries : Retries} (fresh : Oracle)
    (valid : Valid relation ajtai adversary Linked oracle retries)
    (forked : forkIndex relation adversary Linked
      (overlay (context relation adversary Linked (forkIndex relation adversary Linked oracle) oracle)
        oracle fresh) = forkIndex relation adversary Linked oracle)
    (otherValid : Valid relation ajtai adversary Linked
      (overlay (context relation adversary Linked (forkIndex relation adversary Linked oracle) oracle)
        oracle fresh) otherRetries)
    (moved : Moves relation adversary Linked oracle
      (overlay (context relation adversary Linked (forkIndex relation adversary Linked oracle) oracle)
        oracle fresh) otherRetries) :
    StateHashCollision (prior (adversary.run oracle))
      (prior (adversary.run (overlay (context relation adversary Linked
        (forkIndex relation adversary Linked oracle) oracle) oracle fresh))) := by
  have sameFresh := fresh_eq_of_fork relation adversary Linked oracle fresh forked
  have link := (linked_iff relation adversary claim prior contextDigest oracle).mp valid.1.2
  have link' := (linked_iff relation adversary claim prior contextDigest _).mp otherValid.1.2
  rw [sameFresh] at link'
  exact link_collision link link' moved

/-! ## The knowledge theorem with collision events -/

/-- The chance that a retry at `index` gives a state-hash collision with the
base run. -/
noncomputable def hashMismatchChance (index : Fin Nifs.PaperProfile.arity.total) (oracle : Oracle) : ℝ :=
  mass (retrySets relation ajtai adversary Linked oracle index ∩
      {answer | StateHashCollision (prior (adversary.run oracle))
        (prior (adversary.run (Function.update oracle
          (rhoPoint relation adversary Linked index oracle) answer)))}) /
    mass (retrySets relation ajtai adversary Linked oracle index)

/-- The binding reduction's chance that a rerun gives a state-hash collision
with the base run. -/
noncomputable def hashRunningChance : ℝ :=
  𝔼 oracle, ∑ retries, weight relation ajtai adversary Linked oracle retries *
    (if Valid relation ajtai adversary Linked oracle retries then
      retryChance relation ajtai adversary Linked (forkIndex relation adversary Linked oracle) oracle
        (fun other _ => StateHashCollision (prior (adversary.run oracle)) (prior (adversary.run other)))
    else 0)

theorem mismatchChance_le {oracle : Oracle} (succeeds : Succeeds relation ajtai adversary Linked oracle)
    (index : Fin Nifs.PaperProfile.arity.total) :
    mismatchChance relation ajtai adversary Linked index oracle ≤
      hashMismatchChance relation ajtai adversary claim prior contextDigest index oracle := by
  unfold mismatchChance hashMismatchChance
  apply div_le_div_of_nonneg_right _ (mass_nonnegative _)
  apply mass_mono
  rintro answer ⟨retry, moved⟩
  exact ⟨retry, mismatch_collision relation ajtai adversary claim prior contextDigest succeeds retry moved⟩

theorem runningChance_le :
    runningChance relation ajtai adversary Linked ≤
      hashRunningChance relation ajtai adversary claim prior contextDigest := by
  unfold runningChance hashRunningChance
  refine Finset.expect_le_expect fun oracle _ => Finset.sum_le_sum fun retries _ =>
    mul_le_mul_of_nonneg_left ?_ (weight_nonnegative relation ajtai adversary Linked oracle retries)
  split_ifs with valid
  · exact retryChance_mono relation ajtai adversary Linked _ oracle
      fun fresh _ forked otherValid moved =>
        moves_collision relation ajtai adversary claim prior contextDigest fresh valid forked otherValid moved
  · exact le_rfl

/-- The knowledge error with the prior-state link: the statistical part, the
state-hash collision chances of the retries and reruns, and the binding
reduction's collision chance. -/
noncomputable def linkedKnowledgeError (queries : Nat) : ℝ :=
  statisticalError queries +
  (𝔼 oracle, (if Succeeds relation ajtai adversary Linked oracle then
      ∑ index, hashMismatchChance relation ajtai adversary claim prior contextDigest index oracle else 0) +
    collisionChance relation ajtai adversary Linked +
    hashRunningChance relation ajtai adversary claim prior contextDigest)

/-- **ROM knowledge soundness with the prior-state link.** The extractor
succeeds with probability at least the linked acceptance probability minus
the statistical error, two state-hash collision chances and the binding
reduction's collision chance. -/
theorem knowledge_error_le_linked {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Succeeds relation ajtai adversary Linked oracle then (1 : ℝ) else 0) ≤
      𝔼 oracle, ∑ retries, weight relation ajtai adversary Linked oracle retries *
          (if Extracts relation ajtai adversary Linked oracle retries then 1 else 0) +
        linkedKnowledgeError relation ajtai adversary claim prior contextDigest queries := by
  have bound : knowledgeError relation ajtai adversary Linked queries ≤
      linkedKnowledgeError relation ajtai adversary claim prior contextDigest queries := by
    unfold knowledgeError linkedKnowledgeError
    refine add_le_add le_rfl (add_le_add (add_le_add (Finset.expect_le_expect fun oracle _ => ?_) le_rfl)
      (runningChance_le relation ajtai adversary claim prior contextDigest))
    by_cases succeeds : Succeeds relation ajtai adversary Linked oracle
    · simp only [if_pos succeeds]
      exact Finset.sum_le_sum fun index _ =>
        mismatchChance_le relation ajtai adversary claim prior contextDigest succeeds index
    · simp only [if_neg succeeds, le_refl]
  exact (knowledge_error_le relation ajtai adversary Linked bounded).trans (add_le_add le_rfl bound)

end NightstreamFPrime.Export.Stage1.RandomOracleLink
