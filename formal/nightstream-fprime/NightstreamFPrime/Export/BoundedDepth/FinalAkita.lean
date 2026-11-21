import NightstreamFPrime.Export.BoundedDepth.FinalSplit
import NightstreamFPrime.Spec.Compression.Akita.Extract.ComposeRoot
import NightstreamFPrime.Spec.Compression.Akita.Extract.Decoders
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tower

/-! The final verifier over the Akita opening of the key polynomial, `W` and `V` (SPEC.md,
Algorithm 10; Akita, ePrint 2026/1983, revision of 2026-10-15, Theorem 10.17, Proposition 10.44
and Theorem 10.46).

The layout of the argument is a function of `pc` and of the depths of the slots. The stages of
the transcript are a function of the layout. So the result is for the final proofs of one
`shape`: its `pc` and its depths.

- `First` has the values of the first message: the statement, the slot claims and the fresh
  commitment. `First.claims` adds `pc` and the depths of the shape.
- `compose` is the data `Compose` of the final verifier: `lead` is Π_CCS on the fresh instance
  after the checks of steps 1 to 5 (`Setup.lead`), the claims of the argument are the slot claims
  and the output claim of the run, and the test of a table `W` is `Verifies` for the proof
  `Setup.envelope` that `W` gives. `compose_sized` has the conditions `Compose.Sized`.
- `final_suffix`: the stages of the final verifier over an Akita opening are a suffix from the
  stage 0. The Akita opening is the hypothesis `later`: a suffix from the stage `Compose.root`
  whose first certificates satisfy `OpensAt`.
- `final_extract_sound`: an output of the extractor gives a proof with witnesses that `Verifies`
  accepts for the statement of the first message, or a solution of Module-SIS (`Solved`) for a
  matrix of the Akita opening or for the key commitment.
- `final_extraction_le`: the verifier accepts and the extractor returns no output with
  probability at most `(Q + 1)·(Setup.runCost + ε_Akita) + (Q + 1)·(2 + loops)/budget`.
- `breakChance` is the probability that the extractor returns a solution of Module-SIS, and
  `final_sound` is the bound for "the verifier accepts and no proof with witnesses satisfies
  `Verifies`".
- `three_sound` is `final_sound` for the opening with a batching root, a committed fold with
  three opening groups and a last packing fold (`composeRootChain_suffix`).
- `tower_extract_sound`, `tower_extraction_le`, `tower_runs_le` and `tower_sound` are
  `final_extract_sound`, `final_extraction_le`, `final_runs_le` and `final_sound` for the opening
  with a batching root, a fold with three opening groups, a packing level, four trace levels and
  the terminal (`liftTower_suffix`), under the verifier `towerAccepts`. `K` has the basis `(1, U)`
  with `U² = 2` (`Degree2 K`).
- `mixed_ready`, `mixed_extract_sound`, `mixed_extraction_le`, `mixed_runs_le` and `mixed_sound`
  are these theorems for the schedule with the Euclidean route at the levels 3 to 5 and at the
  terminal (`mixedTower_suffix`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalAkita

open scoped BigOperators
open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier (Mode Claims Premises)

attribute [local instance low] Classical.propDecidable

noncomputable local instance : Algebra Zq K := inferInstanceAs (Algebra F K)

noncomputable local instance : Degree2 K :=
  ⟨AlgEquiv.ofRingEquiv (f := carrierEquiv.ringEquiv) fun _ => rfl⟩

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode}

/-- The values of the first message of a final proof with steps: the statement, the claim of
each slot and the commitment of the fresh instance. -/
structure First (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  statement : TerminalStatement AppState
  slots : Fin 2 → SlotClaim logicalWidth publicFits
  commitment : PaperAlgebra.Commitment

/-- The values of a final proof before `W`, with `pc` and the depths of `shape`. -/
def First.claims (first : First logicalWidth publicFits) (shape : Claims logicalWidth publicFits) :
    Claims logicalWidth publicFits :=
  ⟨shape.pc, fun s => (first.slots s, (shape.running s).2), first.commitment⟩

variable (setup : FinalVerifier.Setup mode logicalWidth publicFits) (packageWidths : Fin 2 → ℕ)
  (step : AppState → AppWitness → AppState) (cap : ℕ) (control : AppState → AppWitness → Fin 2)
  (shape : Claims logicalWidth publicFits) {B : Batch (Fin 3)} {Wire Coord : Type} {total : ℕ}
  (read : Reader Wire) (head : Wire → First logicalWidth publicFits)
  (bound : Prefix Wire (Option Coord → K) total → (g : Fin 3) → (B.shape g).Payload)

/-- The first message has a statement for which a proof with witnesses satisfies `Verifies`. -/
def Proved (wire : Wire) : Prop :=
  ∃ proof, Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
    (head wire).statement proof

/-- The data of the final verifier over an Akita opening of the groups `B`, for the final proofs
with `pc` and the depths of `shape`. The stages from 0 are Π_CCS on the fresh instance, the
lookup challenges and the argument. `head` decodes the first message, and `bound anchor g` is
the commitment of the group `g` that a stage input fixes. -/
noncomputable def compose :
    Compose (E := K) (setup.params shape) B Wire Coord (Probe K CompressionFresh.oneShape)
      total where
  start := 0
  read := read
  leadCosts := Compression.Fresh.costs CompressionFresh.oneShape 8
  lead wire W := setup.lead (head wire).statement ((head wire).claims shape) W
  statement wire probe := setup.claimsOf (head wire).statement ((head wire).claims shape) probe
  good wire W := Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
    (head wire).statement (setup.envelope (head wire).statement ((head wire).claims shape) W)
  bound := bound

variable [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]

/-- The conditions `Compose.Sized` of the final verifier. `premises`: the premises of
`FinalVerifier.soundness`. `laid`: the shape passes the layout of step 5. `lookupsFit` and
`challengesFit`: an answer of the oracle has a slot for each lookup challenge and for each
challenge after `V` (`exists_fair_base`, `exists_fair_challenges`). `cube` and `groupsFit`: an
answer has a slot for each coordinate of `α` and three slots. `early`: the commitments of the key
polynomial and of `W` are fixed by the first message. `fits`: the stages before the Akita opening
are stages of the transcript. -/
theorem compose_sized (premises : Premises setup packageWidths) (laid : setup.Laid shape)
    (lookupsFit : (setup.params shape).lookups ≤ Fintype.card Coord)
    (challengesFit : 4 + ((setup.params shape).lookups + ((setup.params shape).claims +
      (setup.params shape).claims + (setup.params shape).spanVars)) ≤ Fintype.card Coord)
    (cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord)
    (groupsFit : 3 ≤ Fintype.card Coord)
    (early : ∀ anchor g, g ≠ 2 → bound anchor g = bound (anchor.take 0) g)
    (fits : (compose setup packageWidths step cap control shape read head bound).root
      (setup.verifier shape) ≤ total) :
    (compose setup packageWidths step cap control shape read head bound).Sized
      (setup.verifier shape) := by
  obtain ⟨wf, counts, matrix, columns, rows, key_fits⟩ := FinalVerifier.laid_facts setup laid
  exact
    { wf, counts, matrix, columns, rows, key_fits
      lookups := exists_fair_base (by simpa using lookupsFit)
      challenges := exists_fair_challenges _ challengesFit
      column := (le_of_eq (rfl : (setup.verifier shape).columnVars = 2)).trans
        ((Nat.le_succ 2).trans groupsFit)
      lead_sound := fun _ _ refuted =>
        FinalVerifier.lead_sound setup step cap control premises refuted
      lead_staged := fun _ W =>
        FinalVerifier.lead_staged setup (exists_fair_slots cube) exists_fair_slot _ _ W _
      lead_bare := fun _ W other => FinalVerifier.lead_bare setup _ _ W other
      lead_nonnegative := Compression.Fresh.costs_nonnegative _ _
      early, fits }

section Chain

variable {Inner BreakT CandT KeyT : Type} {accept : Prefix Wire (Option Coord → K) total → Prop}
  {body : Structure (Prefix Wire (Option Coord → K) total) (Option Coord) K (ℕ → Wire) Inner BreakT
    CandT KeyT
    (Option B.Weak)}
  {key : (g : Fin 3) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin 3 → ℕ}
  {commit : Wire → (g : Fin 3) → (B.shape g).Payload} {value : Wire → Fin 3 → K}
  {place : Wire → Fin 3 → List K} {error : ℝ} {loops tree : ℕ}
  (embed : B.LevelBreak key openKey
    (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) → BreakT)

local notation "C" => compose setup packageWidths step cap control shape read head bound

local notation "vf" => FinalVerifier.Setup.verifier setup shape

local notation "whole" => Compose.chain C vf accept body key openKey mass embed

variable (body key) in
/-- The output of the extractor is a solution of Module-SIS: for the key commitment, from the
decommitment of a certificate and the registered decommitment `honest` (`tableCollision`), or a
witness of `R_bind` of the Akita opening `body`. -/
def Solved (honest : Certified (B.shape 0)) (innerBound : ℕ) (out : Inner ⊕ BreakT) : Prop :=
  Sum.elim
    (fun certificate => ∃ ω, body.witness certificate = some ω ∧
      (tableCollision (key 0) innerBound (ω.certified 0) honest).isSome)
    body.breaks out

/-- The stages of the final verifier over an Akita opening are a suffix from the stage 0 (Akita
Theorem 10.45). The stage errors are `Setup.runCost` and the stage errors `error` of the Akita
opening. `ran`: the verifier accepts only with the conditions `Compose.Ran`. `embedded`: `embed`
maps a solution of Module-SIS for a matrix of a group to a witness of `R_bind` of the Akita
opening. `later`: the stages of the Akita opening are a suffix whose first certificates satisfy
`OpensAt`. `sized`: the conditions `Compose.Sized`, which `compose_sized` gives. -/
theorem final_suffix (sized : Compose.Sized C vf)
    (ran : ∀ point, accept point → Compose.Ran C vf commit value place point)
    (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
    (later : Tier.Suffix accept (Compose.root C vf)
      (OpensAt (Compose.root C vf) commit value place key mass) body error loops tree) :
    Tier.Suffix accept 0 (Compose.Extracted C key mass) whole (setup.runCost shape + error)
      (2 + loops) (4 * tree) := by
  have chain := Compose.suffix sized ran embed embedded later
  exact chain

variable (prover : OracleComp (Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))

/-- An output of the extractor for the final verifier over an Akita opening (Akita
Theorem 10.17). The verifier accepts the transcript of the prover. The output gives a proof with
witnesses that `Verifies` accepts for the statement of the first message of the prover, or it is
a solution of Module-SIS (`Solved`). `registered`: `honest` is a certified decommitment of the
key commitment to the table of the key polynomial. `covers`: `innerBound` is the bound of
Akita Corollary 10.25 for `honest` and a decommitment of a certificate. -/
theorem final_extract_sound (sized : Compose.Sized C vf)
    (ran : ∀ point, accept point → Compose.Ran C vf commit value place point)
    (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
    (later : Tier.Suffix accept (Compose.root C vf)
      (OpensAt (Compose.root C vf) commit value place key mass) body error loops tree)
    {honest : Certified (B.shape 0)}
    {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    (budget : ℕ) (table : Prefix Wire (Option Coord → K) total → Option Coord → K)
    (tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget (Structure.protocol
      whole).stages)
    (out : Inner ⊕ BreakT)
    (result : Checkpoint.extract whole prover budget table tape = some out) :
    accept ((Structure.protocol whole).final prover table) ∧
      (Proved setup packageWidths step cap control head
          (((Structure.protocol whole).input prover table 0).wire 0) ∨
        Solved body key honest innerBound out) := by
  obtain ⟨accepted, extracted⟩ := (final_suffix setup packageWidths step cap control shape read
    head bound embed sized ran embedded later).extract_sound rfl prover budget table tape out
    result
  refine ⟨accepted, ?_⟩
  rcases out with certificate | broken
  · rcases Compose.Extracted.relation_or_collision extracted (registered _) covers with
      good | collision
    · exact Or.inl ⟨_, good⟩
    · exact Or.inr collision
  · exact Or.inr extracted

variable [Finite Wire]

/-- Akita Theorem 10.17 for the final verifier over an Akita opening. For each prover with at
most `Q` queries to the oracle, the probability that the verifier accepts and the extractor
returns no output is at most `(Q + 1)·(runCost + ε) + (Q + 1)·(2 + loops)/budget`.
`Setup.runCost` is the sum of the bounds of the draws of Π_CCS on the fresh instance and of the
argument, and `ε` and `loops` are the stage errors and the retry loops of the Akita opening. -/
theorem final_extraction_le (sized : Compose.Sized C vf)
    (ran : ∀ point, accept point → Compose.Ran C vf commit value place point)
    (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
    (later : Tier.Suffix accept (Compose.root C vf)
      (OpensAt (Compose.root C vf) commit value place key mass) body error loops tree)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (if accept ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + error) + (queries + 1) * (2 + loops : ℕ) / budget :=
  (final_suffix setup packageWidths step cap control shape read head bound embed sized ran
    embedded later).extraction_le rfl prover bounded positive

/-- The expected number of runs of the prover in the extraction is at most
`K̂ + (Q + 1)·(K̂ − 1)` with `K̂ = 4·tree` for the tree size `tree` of the Akita opening (Akita
Theorem 10.17, the cost). -/
theorem final_runs_le (sized : Compose.Sized C vf)
    (ran : ∀ point, accept point → Compose.Ran C vf commit value place point)
    (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
    (later : Tier.Suffix accept (Compose.root C vf)
      (OpensAt (Compose.root C vf) commit value place key mass) body error loops tree)
    {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (extractRuns whole prover budget table tape : ℝ) ≤
      (4 * tree : ℕ) + (queries + 1) * ((4 * tree : ℕ) - 1) :=
  (final_suffix setup packageWidths step cap control shape read head bound embed sized ran
    embedded later).runs_le rfl prover bounded budget

variable (accept body key openKey mass) in
/-- The probability that the extractor returns a solution of Module-SIS (`Solved`): the term of
the Module-SIS assumption in the soundness of the final verifier. The extractor is the solver. -/
noncomputable def breakChance (honest : Certified (B.shape 0)) (innerBound budget : ℕ) : ℝ :=
  𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
      (Structure.protocol whole).stages,
    if ∃ out, Checkpoint.extract whole prover budget table tape = some out ∧
      Solved body key honest innerBound out then (1 : ℝ) else 0

/-- Soundness of the final verifier with Akita openings. For each prover with at most `Q`
queries to the oracle, the probability that the verifier accepts and no proof with witnesses
satisfies `Verifies` for the statement of the first message is at most
`(Q + 1)·(runCost + ε) + (Q + 1)·(2 + loops)/budget + breakChance`. The first two terms are those
of `final_extraction_le`, and `breakChance` is the probability that the extractor returns a
solution of Module-SIS. `registered` and `covers` are those of `final_extract_sound`. -/
theorem final_sound (sized : Compose.Sized C vf)
    (ran : ∀ point, accept point → Compose.Ran C vf commit value place point)
    (embedded : ∀ solution, solution.isSome → body.breaks (embed solution))
    (later : Tier.Suffix accept (Compose.root C vf)
      (OpensAt (Compose.root C vf) commit value place key mass) body error loops tree)
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 _tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (if accept ((Structure.protocol whole).final prover table) ∧
            ¬ Proved setup packageWidths step cap control head
              (((Structure.protocol whole).input prover table 0).wire 0) then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + error) + (queries + 1) * (2 + loops : ℕ) / budget +
        breakChance setup packageWidths step cap control shape read head bound accept body key
          openKey mass embed prover honest innerBound budget := by
  have missed := final_extraction_le setup packageWidths step cap control shape read head bound
    embed prover sized ran embedded later bounded positive
  refine le_trans ?_ (add_le_add missed le_rfl)
  unfold breakChance
  simp only [← Finset.expect_add_distrib]
  refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun tape _ => ?_
  have cases : accept ((Structure.protocol whole).final prover table) ∧
      ¬ Proved setup packageWidths step cap control head
        (((Structure.protocol whole).input prover table 0).wire 0) →
      (accept ((Structure.protocol whole).final prover table) ∧
        Checkpoint.extract whole prover budget table tape = none) ∨
      ∃ out, Checkpoint.extract whole prover budget table tape = some out ∧
        Solved body key honest innerBound out := by
    intro event
    rcases result : Checkpoint.extract whole prover budget table tape with _ | out
    · exact Or.inl ⟨event.1, rfl⟩
    · exact Or.inr ⟨out, rfl, ((final_extract_sound setup packageWidths step cap control shape
        read head bound embed prover sized ran embedded later registered covers budget table tape
        out result).2).resolve_left event.2⟩
  by_cases event : accept ((Structure.protocol whole).final prover table) ∧
      ¬ Proved setup packageWidths step cap control head
        (((Structure.protocol whole).input prover table 0).wire 0)
  · rcases cases event with none | solved
    · rw [ite_eq_left event, ite_eq_left none]
      split_ifs <;> norm_num
    · rw [ite_eq_left event, ite_eq_left solved]
      split_ifs <;> norm_num
  · rw [ite_eq_right event]
    split_ifs <;> norm_num

end Chain

section Three

variable {S' : Compression.Akita.Shape} {O' : OpeningShape S'} {fold' : Descriptor}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq K} {key : (g : Fin 3) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim K} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin 3) → K → Option (Rq (B.carrierDegree g))} {mass : Fin 3 → ℕ} {base : ℕ}
  {β' : Module.Basis (Fin O'.planes) Zq K} {key' : Key S'} {openKey' : OpeningKey O'}
  {claim' : Wire → Claim O' K} {μ' : ℕ} {fits' : O'.cells fold' ≤ 2 ^ μ'} {batch' : ℕ}
  {rowsFit' : O'.rowCount + 1 ≤ 2 ^ batch'} {block' : Fin S'.blocks → Coord}
  {decode' : K → Option (Rq O'.carrierDegree)} {weight' : ℕ}
  {commit : Wire → (g : Fin 3) → (B.shape g).Payload} {value : Wire → Fin 3 → K}
  {place : Wire → Fin 3 → List K} {next : Wire → S'.Payload} {witnessOf : Wire → Witness O' fold'}

local notation "C" => compose setup packageWidths step cap control shape read head bound

local notation "vf" => FinalVerifier.Setup.verifier setup shape

local notation "first" =>
  committedBatch β key openKey claim μ fits batch rowsFit block decode (Compose.read C) mass base
    (Compose.root C vf + 1)

local notation "second" =>
  clearFold β' key' openKey' claim' μ' fits' batch' rowsFit' block' decode' (Compose.read C)
    weight' (Tier.finish first)

local notation "accepts" => composeAccepts C vf first second commit value place next witnessOf

local notation "opening" => composeRoot C vf first second commit value place next witnessOf

local notation "whole" => Compose.chain C vf accepts opening key openKey mass Sum.inl

/-- `final_sound` for the Akita opening with a batching root, a committed fold with three opening
groups and a last packing fold. `ε_Akita` is `1/|K|` for the weights of the batching root and the
stage errors of the two levels. `inside`: the stages are stages of the transcript. The conditions
on the Akita opening are those of `composeRootChain_suffix`, and the other hypotheses are those
of `final_sound`. -/
theorem three_sound [Finite Wire] {peak : Fin 3 → ℕ} {peak' : ℕ}
    (premises : Premises setup packageWidths) (laid : setup.Laid shape)
    (lookupsFit : (setup.params shape).lookups ≤ Fintype.card Coord)
    (challengesFit : 4 + ((setup.params shape).lookups + ((setup.params shape).claims +
      (setup.params shape).claims + (setup.params shape).spanVars)) ≤ Fintype.card Coord)
    (cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord)
    (early : ∀ anchor g, g ≠ 2 → bound anchor g = bound (anchor.take 0) g)
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (groupsFit : 3 ≤ Fintype.card Coord) (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (inside : Tier.finish second ≤ total)
    (prover : OracleComp (Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 _tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
            ¬ Proved setup packageWidths step cap control head
              (((Structure.protocol whole).input prover table 0).wire 0) then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + (1 / Fintype.card K / 2 ^ Root.rootBits 3 +
          (Tier.levelError first + Tier.levelError second))) +
        (queries + 1) * (2 + (1 + (∑ g, Compression.Akita.Shape.blocks (Batch.shape B g) + 1 +
          (Compression.Akita.Shape.blocks S' + 1))) : ℕ) / budget +
        breakChance setup packageWidths step cap control shape read head bound accepts opening
          key openKey mass Sum.inl prover honest innerBound budget :=
  final_sound setup packageWidths step cap control shape read head bound Sum.inl prover
    (compose_sized setup packageWidths step cap control shape read head bound premises laid
      lookupsFit challengesFit cube groupsFit early
      (by
        have ends := Tier.finish_eq first
        have ends' := Tier.finish_eq second
        have starts : Tier.start first = Compose.root C vf + 1 := rfl
        have starts' : Tier.start second = Tier.finish first := rfl
        omega))
    (fun _ accepted => accepted.1) (fun _ found => found)
    (composeRootChain_suffix injective units sparse response uniform small varsFit batchFit
      groupsFit injective' units' sparse' base' small' varsFit' batchFit' inside)
    registered covers bounded positive

end Three

section Tower

variable {S₁ : Compression.Akita.Shape} {O₁ : OpeningShape S₁} {fold₁ : Descriptor}
  {S₂ : Compression.Akita.Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor}
  {S₃ : Compression.Akita.Shape} {O₃ : OpeningShape S₃} {fold₃ : Descriptor}
  {S₄ : Compression.Akita.Shape} {O₄ : OpeningShape S₄} {fold₄ : Descriptor}
  {S₅ : Compression.Akita.Shape} {O₅ : OpeningShape S₅} {fold₅ : Descriptor}
  {S₆ : Compression.Akita.Shape}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq K} {key : (g : Fin 3) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim K} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin 3) → K → Option (Rq (B.carrierDegree g))} {mass : Fin 3 → ℕ}
  {β₁ : Module.Basis (Fin O₁.planes) Zq K} {key₁ : Key S₁} {openKey₁ : OpeningKey O₁}
  {claim₁ : Wire → Claim O₁ K} {μ₁ : ℕ} {fits₁ : O₁.cells fold₁ ≤ 2 ^ μ₁} {digits₁ : ℕ}
  {room₁ : O₁.quotientRoom fold₁ digits₁ ≤ 2 ^ μ₁} {batch₁ : ℕ}
  {rowsFit₁ : O₁.rowCount + 1 ≤ 2 ^ batch₁} {block₁ : Fin S₁.blocks → Coord}
  {decode₁ : K → Option (Rq O₁.carrierDegree)} {weight₁ ν₁ : ℕ}
  {key₂ : Key S₂} {openKey₂ : OpeningKey O₂} {claim₂ : Wire → TraceClaim O₂ K} {μ₂ : ℕ}
  {fits₂ : O₂.cells fold₂ ≤ 2 ^ μ₂} {batch₂ : ℕ}
  {rowsFit₂ : O₂.traceRowCount + 1 ≤ 2 ^ batch₂} {block₂ : Fin S₂.blocks → Coord}
  {decode₂ : K → Option (Rq S₂.innerDegree)} {weight₂ ν₂ m₂ : ℕ}
  {key₃ : Key S₃} {openKey₃ : OpeningKey O₃} {claim₃ : Wire → TraceClaim O₃ K} {μ₃ : ℕ}
  {fits₃ : O₃.cells fold₃ ≤ 2 ^ μ₃} {batch₃ : ℕ}
  {rowsFit₃ : O₃.traceRowCount + 1 ≤ 2 ^ batch₃} {block₃ : Fin S₃.blocks → Coord}
  {decode₃ : K → Option (Rq S₃.innerDegree)} {weight₃ ν₃ m₃ : ℕ}
  {key₄ : Key S₄} {openKey₄ : OpeningKey O₄} {claim₄ : Wire → TraceClaim O₄ K} {μ₄ : ℕ}
  {fits₄ : O₄.cells fold₄ ≤ 2 ^ μ₄} {batch₄ : ℕ}
  {rowsFit₄ : O₄.traceRowCount + 1 ≤ 2 ^ batch₄} {block₄ : Fin S₄.blocks → Coord}
  {decode₄ : K → Option (Rq S₄.innerDegree)} {weight₄ ν₄ m₄ : ℕ}
  {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {claim₅ : Wire → TraceClaim O₅ K} {μ₅ : ℕ}
  {fits₅ : O₅.cells fold₅ ≤ 2 ^ μ₅} {batch₅ : ℕ}
  {rowsFit₅ : O₅.traceRowCount + 1 ≤ 2 ^ batch₅} {block₅ : Fin S₅.blocks → Coord}
  {decode₅ : K → Option (Rq S₅.innerDegree)} {weight₅ ν₅ m₅ : ℕ}
  {L : Terminal S₆ K Wire Coord K total} {m₆ : ℕ}
  {commit : Wire → (g : Fin 3) → (B.shape g).Payload} {value : Wire → Fin 3 → K}
  {place : Wire → Fin 3 → List K} {quarter₂ : 4 * m₂ = S₂.innerDegree}
  {quarter₃ : 4 * m₃ = S₃.innerDegree} {quarter₄ : 4 * m₄ = S₄.innerDegree}
  {quarter₅ : 4 * m₅ = S₅.innerDegree} {quarter₆ : 4 * m₆ = S₆.innerDegree}
  {commit₁ : Wire → S₁.Payload} {commit₂ : Wire → S₂.Payload} {commit₃ : Wire → S₃.Payload}
  {commit₄ : Wire → S₄.Payload} {commit₅ : Wire → S₅.Payload}
  {stateOf : Wire → Fin S₆.blocks → Fin S₆.innerRank → Rq S₆.innerDegree}

local notation "C" => compose setup packageWidths step cap control shape read head bound

local notation "vf" => FinalVerifier.Setup.verifier setup shape

local notation "zeroth" =>
  liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode
    (Compose.read C) mass (Compose.root C vf + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ claim₁ μ₁ fits₁ digits₁ room₁ batch₁ rowsFit₁ block₁
    decode₁ (Compose.read C) weight₁ (Tier.finish zeroth)) ν₁

local notation "second" =>
  Tier.tensored (committedTrace (fold := fold₂) key₂ openKey₂ claim₂ μ₂ fits₂ batch₂ rowsFit₂
    block₂ decode₂ (Compose.read C) weight₂ (Tier.finish first)) ν₂

local notation "third" =>
  Tier.tensored (committedTrace (fold := fold₃) key₃ openKey₃ claim₃ μ₃ fits₃ batch₃ rowsFit₃
    block₃ decode₃ (Compose.read C) weight₃ (Tier.finish second)) ν₃

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := fold₄) key₄ openKey₄ claim₄ μ₄ fits₄ batch₄ rowsFit₄
    block₄ decode₄ (Compose.read C) weight₄ (Tier.finish third)) ν₄

local notation "fifth" =>
  Tier.tensored (committedTrace (fold := fold₅) key₅ openKey₅ claim₅ μ₅ fits₅ batch₅ rowsFit₅
    block₅ decode₅ (Compose.read C) weight₅ (Tier.finish fourth)) ν₅

local notation "coefficient" => Route.coefficient

local notation "accepts" =>
  towerAccepts C vf coefficient coefficient coefficient coefficient zeroth first second third
    fourth fifth L commit value place quarter₂ quarter₃ quarter₄ quarter₅ quarter₆ commit₁ commit₂
    commit₃ commit₄ commit₅ stateOf

local notation "opening" =>
  towerRoot coefficient coefficient coefficient coefficient accepts zeroth first second third
    fourth fifth L m₂ m₃ m₄ m₅ m₆ (Compose.root C vf) (Compose.read C) commit value place

local notation "whole" => Compose.chain C vf accepts opening key openKey mass Sum.inl

local notation "towerError" =>
  (1 / (Fintype.card K : ℝ) / 2 ^ Root.rootBits 3 + (Tier.levelError zeroth +
    (Tier.levelError first +
      (Tier.levelError second +
        (Tier.levelError third +
          (Tier.levelError fourth +
            (Tier.levelError fifth +
              Terminal.termError L)))))))

local notation "towerLoops" =>
  (1 + (∑ g, Compression.Akita.Shape.blocks (Batch.shape B g) + 1 +
    (Compression.Akita.Shape.blocks S₁ + 1 + (Compression.Akita.Shape.blocks S₂ + 1 +
      (Compression.Akita.Shape.blocks S₃ + 1 + (Compression.Akita.Shape.blocks S₄ + 1 +
        (Compression.Akita.Shape.blocks S₅ + 1 + Compression.Akita.Shape.blocks S₆)))))))

local notation "towerTree" =>
  (2 * (2 * (∑ g, Compression.Akita.Shape.blocks (Batch.shape B g) + 1) *
    (2 * (Compression.Akita.Shape.blocks S₁ + 1) * (2 * (Compression.Akita.Shape.blocks S₂ + 1) *
      (2 * (Compression.Akita.Shape.blocks S₃ + 1) *
        (2 * (Compression.Akita.Shape.blocks S₄ + 1) *
          (2 * (Compression.Akita.Shape.blocks S₅ + 1) *
            (Compression.Akita.Shape.blocks S₆ + 1))))))))

variable {peak : Fin 3 → ℕ} {peakL : ℕ}
  (premises : Premises setup packageWidths) (laid : setup.Laid shape)
  (lookupsFit : (setup.params shape).lookups ≤ Fintype.card Coord)
  (challengesFit : 4 + ((setup.params shape).lookups + ((setup.params shape).claims +
    (setup.params shape).claims + (setup.params shape).spanVars)) ≤ Fintype.card Coord)
  (cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord)
  (early : ∀ anchor g, g ≠ 2 → bound anchor g = bound (anchor.take 0) g)
  (injective : Function.Injective block)
  (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
  (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
    Sparse (mass g) (peak g) challenge)
  (response : ∀ g, 2 ≤ (B.fold g).responseBase)
  (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
  (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
  (groupsFit : 3 ≤ Fintype.card Coord)
  (sized₁ : LevelSized block₁ decode₁ weight₁ fold₁.responseBase S₁.certifiedBase μ₁ batch₁)
  (sized₂ : LevelSized block₂ decode₂ weight₂ fold₂.responseBase S₂.certifiedBase μ₂ batch₂)
  (sized₃ : LevelSized block₃ decode₃ weight₃ fold₃.responseBase S₃.certifiedBase μ₃ batch₃)
  (sized₄ : LevelSized block₄ decode₄ weight₄ fold₄.responseBase S₄.certifiedBase μ₄ batch₄)
  (sized₅ : LevelSized block₅ decode₅ weight₅ fold₅.responseBase S₅.certifiedBase μ₅ batch₅)
  (inside : Terminal.start L < total)
  (injectiveL : Function.Injective (Terminal.block L))
  (unitsL : UnitDifferences (Terminal.family L))
  (sparseL : ∀ challenge ∈ Terminal.family L, Sparse (Terminal.weight L) peakL challenge)

include premises laid lookupsFit challengesFit cube early injective units sparse response
  uniform small varsFit batchFit groupsFit sized₁ sized₂ sized₃ sized₄ sized₅ inside injectiveL
  unitsL sparseL

/-- The conditions `Compose.Sized` of the final verifier and the suffix of the opening with a
batching root, a fold with three opening groups (`liftBatch`), a packing level (`liftFold` with
the tensor reduction), four trace levels (`committedTrace` with the tensor reduction) and the
terminal `L`, under the verifier `towerAccepts`: `compose_sized` and `liftTower_suffix`. The first
group of hypotheses is that of `compose_sized`, and the others are those of `liftTower_suffix`
on the parameters. `next`: the fold challenge of the terminal is the stage after the stages of
the level 5. -/
theorem tower_ready (next : Terminal.start L = Tier.finish fifth) :
    Compose.Sized C vf ∧
      Tier.Suffix accepts (Compose.root C vf)
        (OpensAt (Compose.root C vf) commit value place key mass) opening towerError towerLoops
        towerTree := by
  refine ⟨compose_sized setup packageWidths step cap control shape read head bound premises laid
    lookupsFit challengesFit cube groupsFit early ?_, ?_⟩
  · have ends₀ := Tier.finish_eq zeroth
    have ends₁ := Tier.finish_eq first
    have ends₂ := Tier.finish_eq second
    have ends₃ := Tier.finish_eq third
    have ends₄ := Tier.finish_eq fourth
    have ends₅ := Tier.finish_eq fifth
    have starts₀ : Tier.start zeroth = Compose.root C vf + 1 := rfl
    have starts₁ : Tier.start first = Tier.finish zeroth := rfl
    have starts₂ : Tier.start second = Tier.finish first := rfl
    have starts₃ : Tier.start third = Tier.finish second := rfl
    have starts₄ : Tier.start fourth = Tier.finish third := rfl
    have starts₅ : Tier.start fifth = Tier.finish fourth := rfl
    omega
  · exact liftTower_suffix injective units sparse response uniform small varsFit batchFit
      groupsFit sized₁ sized₂ sized₃ sized₄ sized₅ next inside quarter₂ quarter₃ quarter₄ quarter₅
      quarter₆ injectiveL unitsL sparseL commit₁ commit₂ commit₃ commit₄ commit₅ stateOf
      (fun _ accepted => ⟨accepted.2.1, accepted.2.2.1⟩) (fun _ accepted => accepted.2.2.2.1)
      (fun _ accepted => accepted.2.2.2.2.1) (fun _ accepted => accepted.2.2.2.2.2.1)
      (fun _ accepted => accepted.2.2.2.2.2.2.1) (fun _ accepted => accepted.2.2.2.2.2.2.2.1)
      (fun _ accepted => accepted.2.2.2.2.2.2.2.2.1)
      (fun _ accepted => accepted.2.2.2.2.2.2.2.2.2)

local notation "ready" next:max =>
  tower_ready setup packageWidths step cap control shape read head bound
    (quarter₂ := quarter₂) (quarter₃ := quarter₃) (quarter₄ := quarter₄) (quarter₅ := quarter₅)
    (quarter₆ := quarter₆) (commit₁ := commit₁) (commit₂ := commit₂) (commit₃ := commit₃)
    (commit₄ := commit₄) (commit₅ := commit₅) (stateOf := stateOf) (commit := commit)
    (value := value) (place := place) premises laid lookupsFit challengesFit cube early injective
    units sparse response uniform small varsFit batchFit groupsFit sized₁ sized₂ sized₃ sized₄
    sized₅ inside injectiveL unitsL sparseL next

variable (prover : OracleComp (Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))

/-- `final_extract_sound` for the opening with the tower of `tower_ready` (Akita Theorem 10.17):
an output of the extractor gives a proof with witnesses that `Verifies` accepts for the statement
of the first message, or a solution of Module-SIS (`Solved`) for the key commitment, for a matrix
of a group, of a level, or for `A_term`. The hypotheses are those of `tower_ready`, with
`registered` and `covers` of `final_extract_sound`. -/
theorem tower_extract_sound (next : Terminal.start L = Tier.finish fifth)
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    (budget : ℕ) (table : Prefix Wire (Option Coord → K) total → Option Coord → K)
    (tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget (Structure.protocol
      whole).stages)
    (out : _) (result : Checkpoint.extract whole prover budget table tape = some out) :
    accepts ((Structure.protocol whole).final prover table) ∧
      (Proved setup packageWidths step cap control head
          (((Structure.protocol whole).input prover table 0).wire 0) ∨
        Solved opening key honest innerBound out) :=
  final_extract_sound setup packageWidths step cap control shape read head bound Sum.inl prover
    (ready next).1 (fun _ accepted => accepted.1) (fun _ found => found) (ready next).2
    registered covers budget table tape out result

variable [Finite Wire]

/-- `final_extraction_le` for the opening with the tower of `tower_ready` (Akita Theorem 10.17):
the verifier accepts and the extractor returns no output with probability at most
`(Q + 1)·(runCost + ε) + (Q + 1)·(2 + loops)/budget`. `ε` is `1/|K|` for the weights of the
batching root, `Σ_b p_b + Σ_j ε_j` for each level and `B_term·p_term` for the terminal (206), and
`loops` is one for the root, `B_ℓ + 1` for each level and `B_term` for the terminal. -/
theorem tower_extraction_le (next : Terminal.start L = Tier.finish fifth) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + towerError) +
        (queries + 1) * (2 + towerLoops : ℕ) / budget :=
  final_extraction_le setup packageWidths step cap control shape read head bound Sum.inl prover
    (ready next).1 (fun _ accepted => accepted.1) (fun _ found => found) (ready next).2 bounded
    positive

/-- `final_runs_le` for the opening with the tower of `tower_ready`: the expected number of runs
of the prover in the extraction is at most `K̂ + (Q + 1)·(K̂ − 1)` with `K̂ = 4·tree`, and `tree`
is the product of 2 for the root, `2·(B_ℓ + 1)` for each level and `B_term + 1` for the
terminal (Akita Theorem 10.17, the cost). -/
theorem tower_runs_le (next : Terminal.start L = Tier.finish fifth) {queries : ℕ}
    (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (extractRuns whole prover budget table tape : ℝ) ≤
      (4 * towerTree : ℕ) + (queries + 1) * ((4 * towerTree : ℕ) - 1) :=
  final_runs_le setup packageWidths step cap control shape read head bound Sum.inl prover
    (ready next).1 (fun _ accepted => accepted.1) (fun _ found => found) (ready next).2 bounded
    budget

/-- Soundness of the final verifier over the opening with the tower of `tower_ready`
(`final_sound`). For each prover with at most `Q` queries to the oracle, the probability that the
verifier `towerAccepts` accepts and no proof with witnesses satisfies `Verifies` for the statement
of the first message is at most
`(Q + 1)·(runCost + ε) + (Q + 1)·(2 + loops)/budget + breakChance`, with `ε` and `loops` of
`tower_extraction_le`. `breakChance` is the probability that the extractor returns a solution of
Module-SIS for the key commitment, for a matrix of a group, of a level, or for `A_term`. -/
theorem tower_sound (next : Terminal.start L = Tier.finish fifth)
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 _tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
            ¬ Proved setup packageWidths step cap control head
              (((Structure.protocol whole).input prover table 0).wire 0) then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + towerError) +
        (queries + 1) * (2 + towerLoops : ℕ) / budget +
        breakChance setup packageWidths step cap control shape read head bound accepts opening
          key openKey mass Sum.inl prover honest innerBound budget :=
  final_sound setup packageWidths step cap control shape read head bound Sum.inl prover
    (ready next).1 (fun _ accepted => accepted.1) (fun _ found => found) (ready next).2 registered
    covers bounded positive

end Tower

section Mixed

variable {S₁ : Compression.Akita.Shape} {O₁ : OpeningShape S₁} {fold₁ : Descriptor}
  {S₂ : Compression.Akita.Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor}
  {S₃ : Compression.Akita.Shape} {O₃ : OpeningShape S₃} {fold₃ : Descriptor}
  {S₄ : Compression.Akita.Shape} {O₄ : OpeningShape S₄} {fold₄ : Descriptor}
  {S₅ : Compression.Akita.Shape} {O₅ : OpeningShape S₅} {fold₅ : Descriptor}
  {S₆ : Compression.Akita.Shape}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq K} {key : (g : Fin 3) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim K} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin 3) → K → Option (Rq (B.carrierDegree g))} {mass : Fin 3 → ℕ}
  {β₁ : Module.Basis (Fin O₁.planes) Zq K} {key₁ : Key S₁} {openKey₁ : OpeningKey O₁}
  {claim₁ : Wire → Claim O₁ K} {μ₁ : ℕ} {fits₁ : O₁.cells fold₁ ≤ 2 ^ μ₁} {digits₁ : ℕ}
  {room₁ : O₁.quotientRoom fold₁ digits₁ ≤ 2 ^ μ₁} {batch₁ : ℕ}
  {rowsFit₁ : O₁.rowCount + 1 ≤ 2 ^ batch₁} {block₁ : Fin S₁.blocks → Coord}
  {decode₁ : K → Option (Rq O₁.carrierDegree)} {weight₁ ν₁ : ℕ}
  {key₂ : Key S₂} {openKey₂ : OpeningKey O₂} {claim₂ : Wire → TraceClaim O₂ K} {μ₂ : ℕ}
  {fits₂ : O₂.cells fold₂ ≤ 2 ^ μ₂} {batch₂ : ℕ}
  {rowsFit₂ : O₂.traceRowCount + 1 ≤ 2 ^ batch₂} {block₂ : Fin S₂.blocks → Coord}
  {decode₂ : K → Option (Rq S₂.innerDegree)} {weight₂ ν₂ m₂ : ℕ}
  {key₃ : Key S₃} {openKey₃ : OpeningKey O₃} {claim₃ : Wire → TraceClaim O₃ K} {μ₃ : ℕ}
  {fits₃ : O₃.cells fold₃ ≤ 2 ^ μ₃} {batch₃ : ℕ}
  {rowsFit₃ : O₃.traceRowCount + 1 ≤ 2 ^ batch₃} {block₃ : Fin S₃.blocks → Coord}
  {decode₃ : K → Option (Rq S₃.innerDegree)} {weight₃ ν₃ m₃ : ℕ}
  {key₄ : Key S₄} {openKey₄ : OpeningKey O₄} {claim₄ : Wire → TraceClaim O₄ K} {μ₄ : ℕ}
  {fits₄ : O₄.cells fold₄ ≤ 2 ^ μ₄} {batch₄ : ℕ}
  {rowsFit₄ : O₄.traceRowCount + 1 ≤ 2 ^ batch₄} {block₄ : Fin S₄.blocks → Coord}
  {decode₄ : K → Option (Rq S₄.innerDegree)} {weight₄ ν₄ m₄ : ℕ}
  {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {claim₅ : Wire → TraceClaim O₅ K} {μ₅ : ℕ}
  {fits₅ : O₅.cells fold₅ ≤ 2 ^ μ₅} {batch₅ : ℕ}
  {rowsFit₅ : O₅.traceRowCount + 1 ≤ 2 ^ batch₅} {block₅ : Fin S₅.blocks → Coord}
  {decode₅ : K → Option (Rq S₅.innerDegree)} {weight₅ ν₅ m₅ : ℕ}
  {operator₃ limit₃ operator₄ limit₄ operator₅ limit₅ operatorL limitL : ℕ}
  {L : Terminal S₆ K Wire Coord K total (.euclidean operatorL limitL)} {m₆ : ℕ}
  {commit : Wire → (g : Fin 3) → (B.shape g).Payload} {value : Wire → Fin 3 → K}
  {place : Wire → Fin 3 → List K} {quarter₂ : 4 * m₂ = S₂.innerDegree}
  {quarter₃ : 4 * m₃ = S₃.innerDegree} {quarter₄ : 4 * m₄ = S₄.innerDegree}
  {quarter₅ : 4 * m₅ = S₅.innerDegree} {quarter₆ : 4 * m₆ = S₆.innerDegree}
  {commit₁ : Wire → S₁.Payload} {commit₂ : Wire → S₂.Payload} {commit₃ : Wire → S₃.Payload}
  {commit₄ : Wire → S₄.Payload} {commit₅ : Wire → S₅.Payload}
  {stateOf : Wire → Fin S₆.blocks → Fin S₆.innerRank → Rq S₆.innerDegree}

local notation "C" => compose setup packageWidths step cap control shape read head bound

local notation "vf" => FinalVerifier.Setup.verifier setup shape

local notation "zeroth" =>
  liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode
    (Compose.read C) mass (Compose.root C vf + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ claim₁ μ₁ fits₁ digits₁ room₁ batch₁ rowsFit₁ block₁
    decode₁ (Compose.read C) weight₁ (Tier.finish zeroth)) ν₁

local notation "second" =>
  Tier.tensored (committedTrace (fold := fold₂) key₂ openKey₂ claim₂ μ₂ fits₂ batch₂ rowsFit₂
    block₂ decode₂ (Compose.read C) weight₂ (Tier.finish first)) ν₂

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := fold₃) key₃ openKey₃ claim₃ μ₃ fits₃ batch₃ rowsFit₃
    block₃ decode₃ (Compose.read C) weight₃ operator₃ limit₃ (Tier.finish second)) ν₃

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := fold₄) key₄ openKey₄ claim₄ μ₄ fits₄ batch₄ rowsFit₄
    block₄ decode₄ (Compose.read C) weight₄ operator₄ limit₄ (Tier.finish third)) ν₄

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := fold₅) key₅ openKey₅ claim₅ μ₅ fits₅ batch₅ rowsFit₅
    block₅ decode₅ (Compose.read C) weight₅ operator₅ limit₅ (Tier.finish fourth)) ν₅

local notation "accepts" =>
  towerAccepts (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) C vf
    Route.coefficient (Route.euclidean operator₃ limit₃) (Route.euclidean operator₄ limit₄)
    (Route.euclidean operator₅ limit₅) zeroth first second third fourth fifth L commit value place
    quarter₂ quarter₃ quarter₄ quarter₅ quarter₆ commit₁ commit₂ commit₃ commit₄ commit₅ stateOf

local notation "opening" =>
  towerRoot (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) Route.coefficient
    (Route.euclidean operator₃ limit₃) (Route.euclidean operator₄ limit₄)
    (Route.euclidean operator₅ limit₅) accepts zeroth first second third fourth fifth L m₂ m₃ m₄
    m₅ m₆ (Compose.root C vf) (Compose.read C) commit value place

local notation "whole" => Compose.chain C vf accepts opening key openKey mass Sum.inl

local notation "towerError" =>
  (1 / (Fintype.card K : ℝ) / 2 ^ Root.rootBits 3 + (Tier.levelError zeroth +
    (Tier.levelError first +
      (Tier.levelError second +
        (Tier.levelError third +
          (Tier.levelError fourth +
            (Tier.levelError fifth +
              Terminal.termError L)))))))

local notation "towerLoops" =>
  (1 + (∑ g, Compression.Akita.Shape.blocks (Batch.shape B g) + 1 +
    (Compression.Akita.Shape.blocks S₁ + 1 + (Compression.Akita.Shape.blocks S₂ + 1 +
      (Compression.Akita.Shape.blocks S₃ + 1 + (Compression.Akita.Shape.blocks S₄ + 1 +
        (Compression.Akita.Shape.blocks S₅ + 1 + Compression.Akita.Shape.blocks S₆)))))))

local notation "towerTree" =>
  (2 * (2 * (∑ g, Compression.Akita.Shape.blocks (Batch.shape B g) + 1) *
    (2 * (Compression.Akita.Shape.blocks S₁ + 1) * (2 * (Compression.Akita.Shape.blocks S₂ + 1) *
      (2 * (Compression.Akita.Shape.blocks S₃ + 1) *
        (2 * (Compression.Akita.Shape.blocks S₄ + 1) *
          (2 * (Compression.Akita.Shape.blocks S₅ + 1) *
            (Compression.Akita.Shape.blocks S₆ + 1))))))))

variable {peak : Fin 3 → ℕ}
  (premises : Premises setup packageWidths) (laid : setup.Laid shape)
  (lookupsFit : (setup.params shape).lookups ≤ Fintype.card Coord)
  (challengesFit : 4 + ((setup.params shape).lookups + ((setup.params shape).claims +
    (setup.params shape).claims + (setup.params shape).spanVars)) ≤ Fintype.card Coord)
  (cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord)
  (early : ∀ anchor g, g ≠ 2 → bound anchor g = bound (anchor.take 0) g)
  (injective : Function.Injective block)
  (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
  (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
    Sparse (mass g) (peak g) challenge)
  (response : ∀ g, 2 ≤ (B.fold g).responseBase)
  (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
  (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
  (groupsFit : 3 ≤ Fintype.card Coord)
  (sized₁ : LevelSized block₁ decode₁ weight₁ fold₁.responseBase S₁.certifiedBase μ₁ batch₁)
  (sized₂ : LevelSized block₂ decode₂ weight₂ fold₂.responseBase S₂.certifiedBase μ₂ batch₂)
  (sized₃ : LevelSized block₃ decode₃ weight₃ fold₃.responseBase S₃.certifiedBase μ₃ batch₃)
  (sized₄ : LevelSized block₄ decode₄ weight₄ fold₄.responseBase S₄.certifiedBase μ₄ batch₄)
  (sized₅ : LevelSized block₅ decode₅ weight₅ fold₅.responseBase S₅.certifiedBase μ₅ batch₅)
  (normed₃ : NormSized S₃ fold₃ decode₃ operator₃ limit₃ μ₃)
  (normed₄ : NormSized S₄ fold₄ decode₄ operator₄ limit₄ μ₄)
  (normed₅ : NormSized S₅ fold₅ decode₅ operator₅ limit₅ μ₅)
  (inside : Terminal.start L < total)
  (injectiveL : Function.Injective (Terminal.block L))
  (unitsL : UnitDifferences (Terminal.family L))
  (boundedL : ∀ challenge ∈ Terminal.family L, MulBound operatorL challenge)

include premises laid lookupsFit challengesFit cube early injective units sparse response
  uniform small varsFit batchFit groupsFit sized₁ sized₂ sized₃ sized₄ sized₅ normed₃ normed₄
  normed₅ inside injectiveL unitsL boundedL

/-- `tower_ready` for the schedule with the Euclidean route at the levels 3 to 5 and at the
terminal (Akita Section 6.2 and Corollary 10.35 (ii)): the conditions `Compose.Sized` of the final
verifier and the suffix of the opening with a batching root, `liftBatch`, `liftFold` and
`committedTrace` with the tensor reduction, three levels `committedTraceSq` with the tensor
reduction and the terminal `L` on the Euclidean route, under the verifier `towerAccepts` with
these routes: `compose_sized` and `mixedTower_suffix`. `normed₃` to `normed₅` and `boundedL` are
the hypotheses of `mixedTower_suffix` on the Euclidean route. `next`: the fold challenge of the
terminal is the stage after the stages of the level 5. -/
theorem mixed_ready (next : Terminal.start L = Tier.finish fifth) :
    Compose.Sized C vf ∧
      Tier.Suffix accepts (Compose.root C vf)
        (OpensAt (Compose.root C vf) commit value place key mass) opening towerError towerLoops
        towerTree := by
  refine ⟨compose_sized setup packageWidths step cap control shape read head bound premises laid
    lookupsFit challengesFit cube groupsFit early ?_, ?_⟩
  · have ends₀ := Tier.finish_eq zeroth
    have ends₁ := Tier.finish_eq first
    have ends₂ := Tier.finish_eq second
    have ends₃ := Tier.finish_eq third
    have ends₄ := Tier.finish_eq fourth
    have ends₅ := Tier.finish_eq fifth
    have starts₀ : Tier.start zeroth = Compose.root C vf + 1 := rfl
    have starts₁ : Tier.start first = Tier.finish zeroth := rfl
    have starts₂ : Tier.start second = Tier.finish first := rfl
    have starts₃ : Tier.start third = Tier.finish second := rfl
    have starts₄ : Tier.start fourth = Tier.finish third := rfl
    have starts₅ : Tier.start fifth = Tier.finish fourth := rfl
    omega
  · exact mixedTower_suffix injective units sparse response uniform small varsFit batchFit
      groupsFit sized₁ sized₂ sized₃ sized₄ sized₅ normed₃ normed₄ normed₅ next inside quarter₂
      quarter₃ quarter₄ quarter₅ quarter₆ injectiveL unitsL boundedL commit₁ commit₂ commit₃
      commit₄ commit₅ stateOf
      (fun _ accepted => ⟨accepted.2.1, accepted.2.2.1⟩) (fun _ accepted => accepted.2.2.2.1)
      (fun _ accepted => accepted.2.2.2.2.1) (fun _ accepted => accepted.2.2.2.2.2.1)
      (fun _ accepted => accepted.2.2.2.2.2.2.1) (fun _ accepted => accepted.2.2.2.2.2.2.2.1)
      (fun _ accepted => accepted.2.2.2.2.2.2.2.2.1)
      (fun _ accepted => accepted.2.2.2.2.2.2.2.2.2)

local notation "ready" next:max =>
  mixed_ready setup packageWidths step cap control shape read head bound
    (quarter₂ := quarter₂) (quarter₃ := quarter₃) (quarter₄ := quarter₄) (quarter₅ := quarter₅)
    (quarter₆ := quarter₆) (commit₁ := commit₁) (commit₂ := commit₂) (commit₃ := commit₃)
    (commit₄ := commit₄) (commit₅ := commit₅) (stateOf := stateOf) (commit := commit)
    (value := value) (place := place) premises laid lookupsFit challengesFit cube early injective
    units sparse response uniform small varsFit batchFit groupsFit sized₁ sized₂ sized₃ sized₄
    sized₅ normed₃ normed₄ normed₅ inside injectiveL unitsL boundedL next

variable (prover : OracleComp (Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))

/-- `final_extract_sound` for the opening with the tower of `mixed_ready` (Akita Theorem 10.17):
an output of the extractor gives a proof with witnesses that `Verifies` accepts for the statement
of the first message, or a solution of Module-SIS (`Solved`) for the key commitment, for a matrix
of a group, of a level, or for `A_term`. A solution for `A` at the levels 3 to 5 and for `A_term`
is in the Euclidean norm. The hypotheses are those of `mixed_ready`, with `registered` and
`covers` of `final_extract_sound`. -/
theorem mixed_extract_sound (next : Terminal.start L = Tier.finish fifth)
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    (budget : ℕ) (table : Prefix Wire (Option Coord → K) total → Option Coord → K)
    (tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget (Structure.protocol
      whole).stages)
    (out : _) (result : Checkpoint.extract whole prover budget table tape = some out) :
    accepts ((Structure.protocol whole).final prover table) ∧
      (Proved setup packageWidths step cap control head
          (((Structure.protocol whole).input prover table 0).wire 0) ∨
        Solved opening key honest innerBound out) :=
  final_extract_sound setup packageWidths step cap control shape read head bound Sum.inl prover
    (ready next).1 (fun _ accepted => accepted.1) (fun _ found => found) (ready next).2
    registered covers budget table tape out result

variable [Finite Wire]

/-- `final_extraction_le` for the opening with the tower of `mixed_ready` (Akita Theorem 10.17):
the verifier accepts and the extractor returns no output with probability at most
`(Q + 1)·(runCost + ε) + (Q + 1)·(2 + loops)/budget`. `ε` is `1/|K|` for the weights of the
batching root, `Σ_b p_b + Σ_j ε_j` for each level and `B_term·p_term` for the terminal (206), and
`loops` is one for the root, `B_ℓ + 1` for each level and `B_term` for the terminal. The `ε_j` of
the levels 3 to 5 are the bounds `foldCostsSq` of the checks with the direct norm check. -/
theorem mixed_extraction_le (next : Terminal.start L = Tier.finish fifth) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + towerError) +
        (queries + 1) * (2 + towerLoops : ℕ) / budget :=
  final_extraction_le setup packageWidths step cap control shape read head bound Sum.inl prover
    (ready next).1 (fun _ accepted => accepted.1) (fun _ found => found) (ready next).2 bounded
    positive

/-- `final_runs_le` for the opening with the tower of `mixed_ready`: the expected number of runs
of the prover in the extraction is at most `K̂ + (Q + 1)·(K̂ − 1)` with `K̂ = 4·tree`, and `tree`
is the product of 2 for the root, `2·(B_ℓ + 1)` for each level and `B_term + 1` for the
terminal (Akita Theorem 10.17, the cost). -/
theorem mixed_runs_le (next : Terminal.start L = Tier.finish fifth) {queries : ℕ}
    (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (extractRuns whole prover budget table tape : ℝ) ≤
      (4 * towerTree : ℕ) + (queries + 1) * ((4 * towerTree : ℕ) - 1) :=
  final_runs_le setup packageWidths step cap control shape read head bound Sum.inl prover
    (ready next).1 (fun _ accepted => accepted.1) (fun _ found => found) (ready next).2 bounded
    budget

/-- Soundness of the final verifier over the opening with the tower of `mixed_ready`
(`final_sound`). For each prover with at most `Q` queries to the oracle, the probability that the
verifier `towerAccepts` accepts and no proof with witnesses satisfies `Verifies` for the statement
of the first message is at most
`(Q + 1)·(runCost + ε) + (Q + 1)·(2 + loops)/budget + breakChance`, with `ε` and `loops` of
`mixed_extraction_le`. `breakChance` is the probability that the extractor returns a solution of
Module-SIS for the key commitment, for a matrix of a group, of a level, or for `A_term`. -/
theorem mixed_sound (next : Terminal.start L = Tier.finish fifth)
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 _tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
            ¬ Proved setup packageWidths step cap control head
              (((Structure.protocol whole).input prover table 0).wire 0) then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + towerError) +
        (queries + 1) * (2 + towerLoops : ℕ) / budget +
        breakChance setup packageWidths step cap control shape read head bound accepts opening
          key openKey mass Sum.inl prover honest innerBound budget :=
  final_sound setup packageWidths step cap control shape read head bound Sum.inl prover
    (ready next).1 (fun _ accepted => accepted.1) (fun _ found => found) (ready next).2 registered
    covers bounded positive

end Mixed

end NightstreamFPrime.Export.BoundedDepth.FinalAkita
