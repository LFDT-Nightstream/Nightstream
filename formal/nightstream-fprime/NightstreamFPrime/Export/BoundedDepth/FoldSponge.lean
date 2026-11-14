import NightstreamFPrime.Export.DomainSeparation
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightKnowledge
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge

/-! `[SPONGE]` for the light and refresh fold transcripts (`SECURITY_MODEL.md`, adversary 3).

The random-oracle model of a fold reads a challenge at a point: the sponge calls before the read
and a lane pair (`Lifecycle.RandomOracle.Test.point`). A `K` coin reads lanes `2s` and `2s + 1`
of its slot `s`; a `Π_RLC` draw reads lanes 0 to 4. The model's answers at distinct points are
independent, so the sponge meets the model only if two reads that share calls read disjoint
lanes. The conditions of `[SPONGE]` for a fold (`Conditions`) are: distinct challenges of one
transcript have distinct addresses, equal calls fix the absorbed data, a draw never shares its
calls with a coin of slot other than 0 (`rho_ne`, because the draw's calls are longer), the
calls start with the fold's domain, and that domain is separated from every other use of the
table. `conditions` proves them for every schedule, and `light_knowledge_le` and
`refresh_knowledge_le` move the knowledge bound of the model to the ideal-permutation model plus
`spongeError`, under the premise `Spec.Sponge.Transfer`. The step circuits recompute the
challenges with the permutation, so these bounds are for a verifier outside a circuit. -/

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
open NightstreamFPrime.Lifecycle.RandomOracle.Sponge (slotOf)
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt)
open NightstreamFPrime.Export.DomainSeparation

attribute [local instance low] Classical.propDecidable

section Schedule

variable (schedule : Schedule) {width : Nat}

/-- A draw's calls hold the statement calls, the squeezes before the rounds and every round. -/
theorem rho_length_ge (statement : schedule.Statement) (messages : schedule.Messages width)
    (index : Fin schedule.rhoCount) :
    (schedule.statementCalls statement).length +
        (schedule.shape.cubeVariables + 1) / Transcript.coinsPerChunk +
          schedule.shape.cubeVariables * Schedule.roundCallCount width ≤
      (schedule.challengeCalls statement messages (.rho index)).length := by
  simp only [Schedule.challengeCalls, Schedule.proverCalls, Schedule.outputCalls,
    Schedule.roundsCalls, Schedule.preRoundCalls, List.length_append, List.length_replicate,
    flatMap_length_constant _ _ _ (schedule.roundCalls_length messages),
    canonicalFinIndices_length]
  omega

/-- A `Π_RLC` draw never shares its calls with a coin whose slot is not 0. -/
theorem rho_ne (statement statement' : schedule.Statement)
    (messages messages' : schedule.Messages width) (index : Fin schedule.rhoCount)
    (challenge : schedule.Challenge) (slot : challenge.slot ≠ slotOf 0) :
    schedule.challengeCalls statement messages (.rho index) ≠
      schedule.challengeCalls statement' messages' challenge := by
  intro same
  have lengths := congrArg List.length same
  have bound := rho_length_ge schedule statement messages index
  have statements := schedule.statementCalls_length statement statement'
  have rounds := Schedule.roundCallCount_pos width
  have six : Transcript.coinsPerChunk = 6 := rfl
  cases challenge with
  | alpha coordinate =>
      have coin : (schedule.challengeCalls statement' messages' (.alpha coordinate)).length =
          (schedule.statementCalls statement').length + coordinate.val / Transcript.coinsPerChunk := by
        simp only [Schedule.challengeCalls, Schedule.proverCalls, Schedule.fixedCalls,
          List.length_append, List.length_replicate]
      rw [coin] at lengths
      have positive : 1 ≤ schedule.shape.cubeVariables * Schedule.roundCallCount width :=
        Nat.mul_pos (Nat.lt_of_le_of_lt (Nat.zero_le _) coordinate.isLt) rounds
      have inside := coordinate.isLt
      rw [six] at bound lengths
      omega
  | gamma =>
      have cube : schedule.shape.cubeVariables ≠ 0 := by
        intro zero
        exact slot (by simp [Schedule.Challenge.slot, zero])
      have coin : (schedule.challengeCalls statement' messages' .gamma).length =
          (schedule.statementCalls statement').length +
            schedule.shape.cubeVariables / Transcript.coinsPerChunk := by
        simp only [Schedule.challengeCalls, Schedule.proverCalls, Schedule.fixedCalls,
          List.length_append, List.length_replicate]
      rw [coin] at lengths
      have positive : 1 ≤ schedule.shape.cubeVariables * Schedule.roundCallCount width :=
        Nat.mul_pos (Nat.pos_of_ne_zero cube) rounds
      rw [six] at bound lengths
      omega
  | round _ => exact slot rfl
  | rho _ => exact slot rfl

/-- The conditions of `[SPONGE]` for a fold transcript with the row `use` of the table. -/
structure Conditions (use : Use) : Prop where
  addresses : ∀ (statement : schedule.Statement) (messages : schedule.Messages width),
    Function.Injective (schedule.challengeAddress statement messages)
  identify : ∀ {statement statement' : schedule.Statement}
    {messages messages' : schedule.Messages width} (challenge : schedule.Challenge),
    schedule.proverCalls statement messages challenge =
        schedule.proverCalls statement' messages' challenge →
      schedule.AgreeOnAbsorbed statement statement' messages messages' challenge
  lanes : ∀ (statement statement' : schedule.Statement)
    (messages messages' : schedule.Messages width) (index : Fin schedule.rhoCount)
    (challenge : schedule.Challenge), challenge.slot ≠ slotOf 0 →
      schedule.challengeCalls statement messages (.rho index) ≠
        schedule.challengeCalls statement' messages' challenge
  owned : ∀ (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge),
      use.tag <+: ((schedule.challengeCalls statement messages challenge).map List.ofFn).flatten
  separated : ∀ other ∈ uses, other ≠ use → Separated use other

private theorem separated_symm {use use' : Use} (separated : Separated use use') :
    Separated use' use :=
  fun blocks blocks' first second equal => separated blocks' blocks second first equal.symm

instance : Std.Symm Separated := ⟨fun _ _ => separated_symm⟩

/-- Every fold schedule whose domain is a row of the table meets the conditions. -/
theorem conditions {use : Use} (tag : use.tag = schedule.domainChunk) (member : use ∈ uses) :
    Conditions schedule (width := width) use where
  addresses statement messages _ _ same := schedule.challengeAddress_injective statement messages same
  identify challenge same := schedule.proverCalls_identify challenge same
  lanes := rho_ne schedule
  owned statement messages challenge := tag ▸ fold_owns schedule statement messages challenge
  separated _ other distinct := separated.forall member other distinct.symm

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

/-- The light model's wins: success without extraction, less the knowledge error past the
statistical error (the collision and kernel chances). -/
noncomputable def lightWins {queries : ℕ} (adversary : LightAdversary logicalWidth publicFits queries) :
    ℝ :=
  𝔼 oracle, (if Fork.Succeeds (LightFork.game ajtai key) adversary.1 oracle then (1 : ℝ) else 0) -
    (𝔼 oracle, ∑ retries,
        Uniqueness.weight (LightFork.game ajtai key) adversary.1 oracle retries *
          (if LightKnowledge.Extracts ajtai key adversary.1 oracle retries then 1 else 0) +
      LightKnowledge.knowledgeError ajtai key adversary.1 queries) +
    LightKnowledge.statisticalError queries

/-- The refresh model's wins, as `lightWins`. -/
noncomputable def refreshWins {queries : ℕ}
    (adversary : RefreshAdversary logicalWidth publicFits queries) : ℝ :=
  𝔼 oracle, (if Fork.Succeeds (RefreshFork.game ajtai key) adversary.1 oracle then (1 : ℝ) else 0) -
    (𝔼 oracle, ∑ retries,
        Uniqueness.weight (RefreshFork.game ajtai key) adversary.1 oracle retries *
          (if RefreshKnowledge.Extracts ajtai key adversary.1 oracle retries then 1 else 0) +
      RefreshKnowledge.knowledgeError ajtai key adversary.1 queries) +
    RefreshKnowledge.statisticalError queries

/-- Adversary 3 for the light fold in the ideal-permutation model: under `Spec.Sponge.Transfer`
with the light conditions, the wins of each `world` are at most the statistical error plus
`spongeError calls`. -/
theorem light_knowledge_le {queries : ℕ} {World : Type} (winsOf : World → ℝ) (calls : ℕ)
    (premise : Sponge.Transfer
      (Conditions (LightSchedule.schedule logicalWidth publicFits) (width := 8) lightUse) winsOf
      (lightWins ajtai key (queries := queries)) calls)
    (world : World) :
    winsOf world ≤ LightKnowledge.statisticalError queries + Sponge.spongeError calls :=
  Sponge.transfer_le premise light_conditions (fun adversary => by
    unfold lightWins
    linarith [LightKnowledge.knowledge_error_le ajtai key adversary.1 adversary.2]) world

/-- Adversary 3 for the refresh fold, as `light_knowledge_le`. -/
theorem refresh_knowledge_le {queries : ℕ} {World : Type} (winsOf : World → ℝ) (calls : ℕ)
    (premise : Sponge.Transfer
      (Conditions (RefreshSchedule.schedule logicalWidth publicFits) (width := 8) refreshUse)
      winsOf (refreshWins ajtai key (queries := queries)) calls)
    (world : World) :
    winsOf world ≤ RefreshKnowledge.statisticalError queries + Sponge.spongeError calls :=
  Sponge.transfer_le premise refresh_conditions (fun adversary => by
    unfold refreshWins
    linarith [RefreshKnowledge.knowledge_error_le ajtai key adversary.1 adversary.2]) world

end NightstreamFPrime.Export.BoundedDepth.FoldSponge
