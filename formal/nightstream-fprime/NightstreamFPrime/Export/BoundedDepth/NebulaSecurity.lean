import NightstreamFPrime.Export.BoundedDepth.IvcSecurity
import NightstreamFPrime.Lifecycle.Nebula.EndToEnd
import NightstreamFPrime.Lifecycle.Nebula.EtaRound
import NightstreamFPrime.Lifecycle.Nebula.MemoryProgram

/-! The end-to-end bound of a memory application on the bounded-depth scheme (`SECURITY_MODEL.md`, adversary 5),
on the joint law of the memory statement that the IVC adversary claims and the reverse
extractor's sample.

* `experiment`: the extracted history is the steps that the walk decodes (`IvcWalk.trail`), so
  every event reads data of the sample.
* `stage1Extraction`: the extraction premise of `Lifecycle.Nebula.EndToEnd` for the memory
  program, from `IvcWalk.covered`. Its failure is a collision or a failed stage among the walk's
  first visits, which `ReverseStages.bad_le` bounds.
* `MemoryRound`: the memory round (`SECURITY_MODEL.md`, trust item 6) in the form of HyperNova
  errata Assumption 1, for the stages that the walk admits. It is a premise.
* `romMemoryRound`: the same event in the random-oracle model of `η`, bounded for every `η`
  sampler by `(queries + 1) · 2^−109.91` plus a Poseidon2 collision term
  (`Lifecycle.Nebula.EtaRound`). It does not discharge `MemoryRound`, whose `η` is Poseidon2.
* `bad_le`: on the claim that the adversary's tape gives, the bad mass is at most the walk's
  collision and stage terms, the state-link and chain collision masses, and the memory-round
  error. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.NebulaSecurity

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper (TerminalStatement)
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.Nebula
open NightstreamFPrime.Lifecycle.Nebula.EndToEnd (MemoryStatement Experiment Stage1Extraction History)
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.IvcWalk
open NightstreamFPrime.Export.BoundedDepth.IvcSecurity (walkAdversary)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- State `i` of the history that decoded steps give, the steps listed from the last back: the
input state of step `i`, or the statement's current state after the last step. -/
def states (statement : TerminalStatement AppState) (steps : List (Decoded logicalWidth publicFits))
    (i : ℕ) : AppState :=
  (steps.reverse[i]?.map fun decoded => decoded.input.zi).getD statement.zi

/-- Witness `i` of the history that decoded steps give. -/
def witnesses (steps : List (Decoded logicalWidth publicFits)) (i : ℕ) : AppWitness :=
  (steps.reverse[i]?.map fun decoded => decoded.input.witness).getD []

theorem leads_length (program : Lifecycle.Stage1.Application.Program) :
    ∀ (statement : TerminalStatement AppState) (steps : List (Decoded logicalWidth publicFits)),
      Leads program statement steps → steps.length = statement.iteration
  | _, [], ⟨zero, _⟩ => zero.symm
  | _, _ :: older, ⟨_, _, _, iterationEq, earlier⟩ => by
      rw [List.length_cons, leads_length program _ older earlier]
      exact iterationEq

/-- Decoded steps that lead to `statement` give a step history of `statement`. -/
theorem history_of_leads (program : Lifecycle.Stage1.Application.Program) :
    ∀ (statement : TerminalStatement AppState) (steps : List (Decoded logicalWidth publicFits)),
      Leads program statement steps →
        History program statement (states statement steps) (witnesses steps)
  | statement, [], ⟨zero, same⟩ => by
      refine ⟨?_, ?_, fun i below => absurd below (by omega)⟩
      · simpa [states] using same
      · simp [states]
  | statement, decoded :: older, ⟨valid, stepped, z0Eq, iterationEq, earlier⟩ => by
      obtain ⟨start, finish, steps⟩ := history_of_leads program _ older earlier
      have length := leads_length program _ older earlier
      dsimp only at length start finish steps
      have inside : ∀ i, i < older.length →
          (decoded :: older).reverse[i]? = older.reverse[i]? := fun i below => by
        rw [List.reverse_cons, List.getElem?_append_left (by simpa using below)]
      have last : (decoded :: older).reverse[older.length]? = some decoded := by
        rw [List.reverse_cons, List.getElem?_append_right (by simp)]
        simp
      have past : (decoded :: older).reverse[older.length + 1]? = none := by
        rw [List.getElem?_eq_none_iff]
        simp
      have sameStates : ∀ i, i ≤ older.length →
          states statement (decoded :: older) i =
            states ⟨decoded.input.iteration, decoded.input.z0, decoded.input.zi⟩ older i := by
        intro i atMost
        rcases Nat.lt_or_eq_of_le atMost with below | equal
        · have reversed : i < older.reverse.length := by simpa using below
          simp only [states, inside i below, List.getElem?_eq_getElem reversed, Option.map_some,
            Option.getD_some]
        · subst equal
          simp only [states, last, Option.map_some, Option.getD_some]
          rw [List.getElem?_eq_none_iff.mpr (by simp)]
          rfl
      have sameWitnesses : ∀ i, i < older.length →
          witnesses (decoded :: older) i = witnesses older i := fun i below => by
        simp only [witnesses, inside i below]
      refine ⟨?_, ?_, fun i below => ?_⟩
      · rw [sameStates 0 (Nat.zero_le _), start, z0Eq]
      · rw [← iterationEq, ← length]
        simp only [states, past, Option.map_none, Option.getD_none]
      · rw [← iterationEq, ← length] at below
        rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ below) with before | equal
        · rw [sameStates i before.le, sameStates (i + 1) before, sameWitnesses i before]
          exact steps i (length ▸ before)
        · subst equal
          have atLast : states statement (decoded :: older) older.length = decoded.input.zi := by
            simp only [states, last, Option.map_some, Option.getD_some]
          have witnessLast : witnesses (decoded :: older) older.length = decoded.input.witness := by
            simp only [witnesses, last, Option.map_some, Option.getD_some]
          have afterLast : states statement (decoded :: older) (older.length + 1) = statement.zi := by
            simp only [states, past, Option.map_none, Option.getD_none]
          rw [atLast, witnessLast, afterLast]
          exact ⟨valid, stepped.symm⟩

/-- A sample: the claimed memory statement, the walk's start, and the extractor's answers. -/
abbrev Sample (width : ℕ) (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  MemoryStatement width × Visit logicalWidth publicFits × List (Answer logicalWidth publicFits)

variable {p : Plan} (app : PortApp p.toGeometry)
  {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  {step : AppState → AppWitness → AppState}
  {cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : Exports relation ajtai circuitKeys step cubeFits)

/-- The extracted history is the steps that the walk decodes. -/
def experiment : Experiment app.width (Sample app.width logicalWidth publicFits) where
  statement s := s.2.1.statement
  memory s := s.1
  states s := states s.2.1.statement (trail ajtai circuitKeys exports s.2.1 s.2.2)
  witness s := witnesses (trail ajtai circuitKeys exports s.2.1 s.2.2)

/-- The terminal accepts the sample's statement and proof, the walk starts at bound `B`, and the
answers cover the statement's iteration. On the samples of `ClaimedStage.law` for the walk's
stage `depth`, the last two hold when the adversary's iteration is at most `depth`. -/
def Accepted (control : AppState → AppWitness → Fin 2) (cap : Nat)
    (s : Sample app.width logicalWidth publicFits) : Prop :=
  Terminal.HoldsFor cap ajtai circuitKeys step control s.2.1.statement s.2.1.proof ∧
    s.2.1.bounds = (fun _ => productionGlobalParams.bigB) ∧
    s.2.1.statement.iteration ≤ s.2.2.length

/-- A collision or a failed stage at one of the walk's first visits. -/
def Failure (start : Visit logicalWidth publicFits) (answers : List (Answer logicalWidth publicFits)) :
    Prop :=
  ∃ j < answers.length,
    WalkCollision ajtai circuitKeys exports
        (visitAfter (back ajtai circuitKeys exports) start (answers.take j)) ∨
      StageFails (attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports)
        (WalkSucceeds ajtai) (WalkExtracted ajtai) start answers j

/-- The extraction premise of `Lifecycle.Nebula.EndToEnd` for the memory program, from the walk. -/
theorem stage1Extraction (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (programEq : exports.program = MemoryApp.program p.toGeometry app)
    (control : AppState → AppWitness → Fin 2) (cap : Nat) :
    Stage1Extraction (MemoryApp.program p.toGeometry app) (experiment app ajtai circuitKeys exports)
      (Accepted app ajtai circuitKeys (step := step) control cap)
      (fun s => Failure ajtai circuitKeys exports s.2.1 s.2.2) := by
  refine ⟨fun ⟨_, ⟨statement, proof, bounds⟩, answers⟩ ⟨accepted, start, depth⟩ => ?_⟩
  dsimp only at start depth accepted
  subst start
  by_cases leads : Leads exports.program statement
      (trail ajtai circuitKeys exports (initial (statement, proof)) answers)
  · left
    rw [← programEq]
    exact history_of_leads exports.program _ _ leads
  · right
    exact covered ajtai circuitKeys exports keysShaped control cap (statement, proof) answers depth
      accepted leads

/-- A reverse stage of the walk, with a memory statement that its tape gives. -/
structure ClaimedStage where
  stage : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits)
  claim : stage.Tape → MemoryStatement app.width

/-- The joint law of the claim, the walk's start and the answers. -/
noncomputable def ClaimedStage.law (c : ClaimedStage app (logicalWidth := logicalWidth)
    (publicFits := publicFits)) : PMF (Sample app.width logicalWidth publicFits) :=
  c.stage.tape.map fun tape => (c.claim tape, c.stage.input tape, c.stage.results tape)

/-- The memory round (`SECURITY_MODEL.md`, trust item 6), in the form of HyperNova errata
Assumption 1: for every claimed stage whose stage is admitted, the mass of samples whose
extracted run the model accepts with a claimed segment that passes the product test with
unbalanced multisets is at most `error c`. The relation computes `η` with Poseidon2, so no
random-oracle model covers it. -/
def MemoryRound
    (StageAdmitted : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits) → Prop)
    (error : ClaimedStage app (logicalWidth := logicalWidth) (publicFits := publicFits) → ℝ) :
    Prop :=
  ∀ c : ClaimedStage app, StageAdmitted c.stage →
    (c.law.toOuterMeasure
      {s | (experiment app ajtai circuitKeys exports).Unbalanced app s}).toReal ≤ error c

/-- The memory round in the random-oracle model of `η` (CW-10.8): for an `η` sampler of samples
with `queries` queries and a secure geometry, the chance over the sampler's coins and a uniform
oracle that the extracted run is unbalanced with the oracle's challenges is at most
`(queries + 1) · 2^−109.91` plus the sampler's retry collision term. At the Poseidon2 oracle the
event is `Unbalanced` (`EtaRound.unbalancedIn_deployed`), but no result bounds it there, so
`MemoryRound` stays a premise. -/
theorem romMemoryRound (valid : p.Valid) (secure : p.Secure) {queries : ℕ}
    (sampler : EtaRound.EtaSampler (Sample app.width logicalWidth publicFits) queries) :
    sampler.unbalancedChance app (experiment app ajtai circuitKeys exports) ≤
      ((queries + 1 : ℕ) : ℝ) * (2 : ℝ) ^ (-(10991 / 100 : ℝ)) +
        sampler.retryCollision app (experiment app ajtai circuitKeys exports) :=
  sampler.unbalancedChance_le_secure valid secure

variable {Admitted : NifsAdversary (WalkAttempt logicalWidth publicFits) → Prop}
  {StageAdmitted : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits) → Prop}
  {Efficient : (adversary : NifsAdversary (WalkAttempt logicalWidth publicFits)) →
    NifsExtractor (Answer logicalWidth publicFits) adversary → Prop}
  {error : NifsAdversary (WalkAttempt logicalWidth publicFits) → ℝ}
  (assumption : Assumption1 (WalkSucceeds ajtai) (WalkExtracted ajtai) Admitted Efficient error)
  (closed : Closed (attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports) Admitted
    StageAdmitted Efficient)
  (adversary : IvcAdversary (IvcSecurity.Output logicalWidth publicFits))
  (admitted : StageAdmitted (Stage.start (walkAdversary adversary)))

local notation "stages" => reverseStages assumption closed (walkAdversary adversary) admitted

/-- The bound of adversary 5 for a memory application on the bounded-depth scheme: on the joint law of the
memory statement that the adversary claims from its own tape and the walk's sample, the mass of
samples that the memory verifier accepts but whose extracted history does not attest the claim
is at most the walk's collision and stage terms, the state-link and chain collision masses of
the extracted runs, and the memory-round error of the last stage. -/
theorem bad_le (valid : p.Valid) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (programEq : exports.program = MemoryApp.program p.toGeometry app)
    (control : AppState → AppWitness → Fin 2) (cap depth : ℕ)
    (claim : adversary.Tape → MemoryStatement app.width)
    {memoryError : ClaimedStage app (logicalWidth := logicalWidth) (publicFits := publicFits) → ℝ}
    (round : MemoryRound app ajtai circuitKeys exports StageAdmitted memoryError) :
    let c : ClaimedStage app :=
      ⟨(stages depth).1, claim ∘ origin assumption closed (walkAdversary adversary) admitted depth⟩
    (c.law.toOuterMeasure {s | (experiment app ajtai circuitKeys exports).Bad app
        (Accepted app ajtai circuitKeys (step := step) control cap) s}).toReal ≤
      ∑ j : Fin depth,
          (((stages j.val).1.tape.toOuterMeasure
              {tape | WalkCollision ajtai circuitKeys exports
                (visitAfter (back ajtai circuitKeys exports) ((stages j.val).1.input tape)
                  ((stages j.val).1.results tape))}).toReal +
            error ((stages j.val).1.nifs (attemptOf ajtai circuitKeys exports)
              (back ajtai circuitKeys exports))) +
        (c.law.toOuterMeasure
          {s | (experiment app ajtai circuitKeys exports).StateLinkCollision app s}).toReal +
        (c.law.toOuterMeasure
          {s | (experiment app ajtai circuitKeys exports).ChainCollision app s}).toReal +
        memoryError c := by
  intro c
  have cover := EndToEnd.bad_le_sum valid
    (stage1Extraction app ajtai circuitKeys exports keysShaped programEq control cap) c.law
  have failure : c.law.toOuterMeasure {s | Failure ajtai circuitKeys exports s.2.1 s.2.2} =
      (stages depth).1.tape.toOuterMeasure
        {tape | Failure ajtai circuitKeys exports ((stages depth).1.input tape)
          ((stages depth).1.results tape)} := by
    rw [ClaimedStage.law, PMF.toOuterMeasure_map_apply]
    rfl
  have walk := NightstreamFPrime.Spec.HyperNova.ReverseStages.bad_le assumption closed (walkAdversary adversary) admitted depth
    (fun input results => Failure ajtai circuitKeys exports input results)
    (WalkCollision ajtai circuitKeys exports) fun _ _ results length found => by
      obtain ⟨j, inside, event⟩ := found
      exact ⟨⟨j, length ▸ inside⟩, event⟩
  rw [← failure] at walk
  have finite (event : Set (Sample app.width logicalWidth publicFits)) :
      c.law.toOuterMeasure event ≠ ∞ := event_ne_top c.law event
  have real := ENNReal.toReal_mono
    (ENNReal.add_ne_top.mpr ⟨ENNReal.add_ne_top.mpr ⟨ENNReal.add_ne_top.mpr
      ⟨finite _, finite _⟩, finite _⟩, finite _⟩) cover
  rw [ENNReal.toReal_add (ENNReal.add_ne_top.mpr ⟨ENNReal.add_ne_top.mpr ⟨finite _, finite _⟩,
      finite _⟩) (finite _),
    ENNReal.toReal_add (ENNReal.add_ne_top.mpr ⟨finite _, finite _⟩) (finite _),
    ENNReal.toReal_add (finite _) (finite _)] at real
  refine real.trans ?_
  gcongr
  exact round c (stages depth).2

end NightstreamFPrime.Export.BoundedDepth.NebulaSecurity
