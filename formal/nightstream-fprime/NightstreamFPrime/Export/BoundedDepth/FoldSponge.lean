import NightstreamFPrime.Export.DomainSeparation
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.Fidelity
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightKnowledge
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge

/-! `[DSFS]` for the light and refresh fold transcripts (`SECURITY_MODEL.md`, adversary 3).

The random-oracle model of a fold is a uniform function: a point is the list of sponge
calls before a read (`Lifecycle.RandomOracle.Test.point`), and its answer is the rate lanes of
the state after them (`Lifecycle.RandomOracle.Test.Answer`). A `K` coin of slot `s` reads lanes
`2s` and `2s + 1` of the answer, and a `Π_RLC` draw reads lanes 0 to 4, so two reads of one state
read one answer, as in the sponge. The conditions of `[DSFS]` for a fold (`Conditions`) are:
the model reads each address as the sponge does at the rate lanes (`Fidelity.DeployedOn` of
`Fidelity.spongeOracle`), distinct challenges of one transcript have distinct addresses, equal
calls fix the absorbed data, the calls start with the fold's domain, and that domain is
separated from every other use of the table. `conditions` proves them for every schedule.

The event that moves is the failure event: a run succeeds and the extractor, with the retries of
the retry law, does not extract (`LightKnowledge.failureChance`, `RefreshKnowledge.failureChance`).
Under the premise `Spec.Sponge.DuplexTransfer` of that event, its chance for a prover with
`calls` permutation calls is at most the knowledge error of a model adversary with `calls`
queries plus `duplexError calls` (`light_knowledge_le`, `refresh_knowledge_le`). The step circuits recompute the challenges with
the permutation, so these bounds are for a verifier outside a circuit. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FoldSponge

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt)
open NightstreamFPrime.Export.DomainSeparation

section Schedule

variable (schedule : Schedule) {width : Nat}

/-- The conditions of `[DSFS]` for a fold transcript with the row `use` of the table. -/
structure Conditions (use : Use) : Prop where
  reads : Fidelity.DeployedOn (Fidelity.spongeOracle (schedule := schedule) (width := width))
  addresses : ∀ (statement : schedule.Statement) (messages : schedule.Messages width),
    Function.Injective (schedule.challengeAddress statement messages)
  identify : ∀ {statement statement' : schedule.Statement}
    {messages messages' : schedule.Messages width} (challenge : schedule.Challenge),
    schedule.proverCalls statement messages challenge =
        schedule.proverCalls statement' messages' challenge →
      schedule.AgreeOnAbsorbed statement statement' messages messages' challenge
  owned : ∀ (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge),
      use.tag <+: ((schedule.challengeCalls statement messages challenge).map List.ofFn).flatten
  separated : ∀ other ∈ uses, other ≠ use → Separated use other

/-- Every fold schedule whose domain is a row of the table meets the conditions. -/
theorem conditions {use : Use} (tag : use.tag = schedule.domainChunk) (member : use ∈ uses) :
    Conditions schedule (width := width) use where
  reads := Fidelity.spongeOracle_deployed
  addresses statement messages _ _ same := schedule.challengeAddress_injective statement messages same
  identify challenge same := schedule.proverCalls_identify challenge same
  owned statement messages challenge := tag ▸ fold_owns schedule statement messages challenge
  separated _ other distinct := separated_of_mem member other distinct

end Schedule

/-- The light fold's row of the table. -/
def lightUse : Use := ⟨"light fold", lightFoldDomainChunk⟩

/-- The refresh fold's row of the table. -/
def refreshUse : Use := ⟨"refresh fold", refreshFoldDomainChunk⟩

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (key : CircuitKey logicalWidth publicFits)

theorem light_conditions :
    Conditions (LightSchedule.schedule logicalWidth publicFits) (width := 8) lightUse :=
  conditions _ rfl (by simp [lightUse, uses])

theorem refresh_conditions :
    Conditions (RefreshSchedule.schedule logicalWidth publicFits) (width := 8) refreshUse :=
  conditions _ rfl (by simp [refreshUse, uses])

/-- A light adversary with at most `queries` oracle queries. -/
abbrev LightAdversary (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth)
    (queries : ℕ) :=
  {adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits) // adversary.QueryBound queries}

/-- A refresh adversary with at most `queries` oracle queries. -/
abbrev RefreshAdversary (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth)
    (queries : ℕ) :=
  {adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits) // adversary.QueryBound queries}

/-- Adversary 3 for the light fold in the ideal-permutation model: under
`Spec.Sponge.DuplexTransfer` of the failure event with the light conditions, the chance
`winsOf world` of the event for each `world` with `calls` permutation calls is at most the
knowledge error of a model adversary with `calls` queries plus `duplexError calls`. -/
theorem light_knowledge_le {World : Type} (winsOf : World → ℝ) (calls : ℕ)
    (premise : Sponge.DuplexTransfer
      (Conditions (LightSchedule.schedule logicalWidth publicFits) (width := 8) lightUse) winsOf
      (fun queries (adversary : LightAdversary logicalWidth publicFits queries) =>
        LightKnowledge.failureChance ajtai key adversary.1) calls)
    (world : World) :
    ∃ adversary : LightAdversary logicalWidth publicFits calls,
      winsOf world ≤ LightKnowledge.knowledgeError ajtai key adversary.1 calls +
        Sponge.duplexError calls := by
  obtain ⟨adversary, wins⟩ := premise light_conditions world
  exact ⟨adversary, by linarith [LightKnowledge.failureChance_le ajtai key adversary.1 adversary.2]⟩

/-- Adversary 3 for the refresh fold, as `light_knowledge_le`. -/
theorem refresh_knowledge_le {World : Type} (winsOf : World → ℝ) (calls : ℕ)
    (premise : Sponge.DuplexTransfer
      (Conditions (RefreshSchedule.schedule logicalWidth publicFits) (width := 8) refreshUse)
      winsOf (fun queries (adversary : RefreshAdversary logicalWidth publicFits queries) =>
        RefreshKnowledge.failureChance ajtai key adversary.1) calls)
    (world : World) :
    ∃ adversary : RefreshAdversary logicalWidth publicFits calls,
      winsOf world ≤ RefreshKnowledge.knowledgeError ajtai key adversary.1 calls +
        Sponge.duplexError calls := by
  obtain ⟨adversary, wins⟩ := premise refresh_conditions world
  exact ⟨adversary,
    by linarith [RefreshKnowledge.failureChance_le ajtai key adversary.1 adversary.2]⟩

end NightstreamFPrime.Export.BoundedDepth.FoldSponge
