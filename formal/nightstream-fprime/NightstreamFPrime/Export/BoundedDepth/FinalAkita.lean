import NightstreamFPrime.Export.BoundedDepth.FinalSplit
import NightstreamFPrime.Spec.Compression.Akita.Extract.ComposeRoot

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
- `final_suffix`: the stages of the final verifier over a batching root, a committed fold with
  three opening groups and a last packing fold are a suffix from the stage 0.
- `final_extract_sound`: an output of the extractor gives a proof with witnesses that `Verifies`
  accepts for the statement of the first message, or a solution of Module-SIS (`Solved`) for a
  matrix of a level or for the key commitment.
- `final_extraction_le`: the verifier accepts and the extractor returns no output with
  probability at most `(Q + 1)·(Setup.runCost + ε_Akita) + (Q + 1)·loops/budget`.
- `breakChance` is the probability that the extractor returns a solution of Module-SIS, and
  `final_sound` is the bound for "the verifier accepts and no proof with witnesses satisfies
  `Verifies`". -/

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
  (bound : Prefix Wire (Coord → K) total → (g : Fin 3) → (B.shape g).Payload)

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
`FinalVerifier.soundness`. `laid`: the shape passes the layout of step 5. `lookups` and
`challenges`: an answer of the oracle gives uniform lookup challenges and uniform challenges
after `V`. `cube` and `groupsFit`: an answer has a slot for each coordinate of `α` and three
slots. `early`: the commitments of the key polynomial and of `W` are fixed by the first message.
`fits`: the stages before the Akita opening are stages of the transcript. -/
theorem compose_sized (premises : Premises setup packageWidths) (laid : setup.Laid shape)
    (lookups : ∃ decode : (Coord → K) → Fin (setup.params shape).lookups → F, Fair decode)
    (challenges : ∃ decode : (Coord → K) → Compression.Verifier.Challenges (setup.params shape),
      Fair decode)
    (cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord)
    (groupsFit : 3 ≤ Fintype.card Coord)
    (early : ∀ anchor g, g ≠ 2 → bound anchor g = bound (anchor.take 0) g)
    (fits : (compose setup packageWidths step cap control shape read head bound).root
      (setup.verifier shape) ≤ total) :
    (compose setup packageWidths step cap control shape read head bound).Sized
      (setup.verifier shape) := by
  obtain ⟨wf, counts, matrix, columns, rows, key_fits⟩ := FinalVerifier.laid_facts setup laid
  exact
    { wf, counts, matrix, columns, rows, key_fits, lookups, challenges
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

variable {S' : Compression.Akita.Shape} {O' : OpeningShape S'} {fold' : Descriptor}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq K} {key : (g : Fin 3) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim K} {μ : ℕ} {fits : B.cells < 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin 3) → K → Option (Rq (B.carrierDegree g))} {mass : Fin 3 → ℕ} {base : ℕ}
  {β' : Module.Basis (Fin O'.planes) Zq K} {key' : Key S'} {openKey' : OpeningKey O'}
  {claim' : Wire → Claim O' K} {μ' : ℕ} {fits' : O'.cells fold' < 2 ^ μ'} {batch' : ℕ}
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

local notation "whole" =>
  Compose.chain C vf accepts (composeRoot C vf first second commit value place next witnessOf)
    key openKey mass Sum.inl

local notation "akitaError" =>
  (1 / Fintype.card K + (∑ b, Tier.chance first b + List.sum (Tier.costs first) +
    (∑ b, Tier.chance second b + List.sum (Tier.costs second))))

local notation "loops" =>
  2 + (1 + (∑ g, Compression.Akita.Shape.blocks (Batch.shape B g) + 1 +
    (Compression.Akita.Shape.blocks S' + 1)))

local notation "tree" =>
  4 * (2 * (2 * (∑ g, Compression.Akita.Shape.blocks (Batch.shape B g) + 1) *
    (2 * (Compression.Akita.Shape.blocks S' + 1))))

/-- The output of the extractor is a solution of Module-SIS: for the key commitment, from the
decommitment of a certificate and the registered decommitment `honest` (`tableCollision`), or
for a matrix of the first level or of the last level of the Akita opening. -/
def Solved (honest : Certified (B.shape 0)) (innerBound : ℕ)
    (out : Opened B.Weak (Opened (Weak O') (Prefix Wire (Coord → K) total)) ⊕
      (B.LevelBreak key openKey
          (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) ⊕
        (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕ Unit))) : Prop :=
  Sum.elim
    (fun certificate => ∃ ω, certificate.opening = some ω ∧
      (tableCollision (key 0) innerBound (ω.certified 0) honest).isSome)
    (Sum.elim (fun solution => solution.isSome)
      (Sum.elim (fun solution => solution.isSome) fun _ => False))
    out

variable {peak : Fin 3 → ℕ} {peak' : ℕ} (premises : Premises setup packageWidths)
  (laid : setup.Laid shape)
  (lookups : ∃ decode : (Coord → K) → Fin (setup.params shape).lookups → F, Fair decode)
  (challenges : ∃ decode : (Coord → K) → Compression.Verifier.Challenges (setup.params shape),
    Fair decode)
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

include premises laid lookups challenges cube early injective units sparse response uniform small
  varsFit batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit'

omit premises laid lookups challenges cube early injective units sparse response uniform small
  varsFit batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit'
  [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
theorem root_le (inside : Tier.finish second ≤ total) : Compose.root C vf ≤ total := by
  have ends := Tier.finish_eq first
  have ends' := Tier.finish_eq second
  have starts : Tier.start first = Compose.root C vf + 1 := rfl
  have starts' : Tier.start second = Tier.finish first := rfl
  omega

/-- The stages of the final verifier over a batching root, a committed fold with three opening
groups and a last packing fold are a suffix from the stage 0 (Akita Theorem 10.45). The stage
errors are `Setup.runCost` and the stage errors of the Akita opening. The conditions on the
Akita opening are those of `composeChain_suffix`. -/
theorem final_suffix (inside : Tier.finish second ≤ total) :
    Tier.Suffix accepts 0 (Compose.Extracted C key mass) whole
      (setup.runCost shape + akitaError) loops tree := by
  have sized := compose_sized setup packageWidths step cap control shape read head bound premises
    laid lookups challenges cube groupsFit early
    (root_le setup packageWidths step cap control shape read head bound inside)
  have chain := composeChain_suffix (commit := commit) (value := value) (place := place)
    (next := next) (witnessOf := witnessOf) sized injective units sparse response uniform small
    varsFit batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit' inside
  exact chain

variable (prover : OracleComp (Prefix Wire (Coord → K) total) (Coord → K) (ℕ → Wire))

/-- An output of the extractor for the final verifier over the Akita opening (Akita
Theorem 10.17). The verifier accepts the transcript of the prover. The output gives a proof with
witnesses that `Verifies` accepts for the statement of the first message of the prover, or it is
a solution of Module-SIS (`Solved`). `registered`: `honest` is a certified decommitment of the
key commitment to the table of the key polynomial. `covers`: `innerBound` is the bound of
Akita Corollary 10.25 for `honest` and a decommitment of a certificate. -/
theorem final_extract_sound (inside : Tier.finish second ≤ total)
    {honest : Certified (B.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    (budget : ℕ) (table : Prefix Wire (Coord → K) total → Coord → K)
    (tape : Tape (Prefix Wire (Coord → K) total) Coord K budget (Structure.protocol whole).stages)
    (out : Opened B.Weak (Opened (Weak O') (Prefix Wire (Coord → K) total)) ⊕
      (B.LevelBreak key openKey
          (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) ⊕
        (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕ Unit)))
    (result : Checkpoint.extract whole prover budget table tape = some out) :
    accepts ((Structure.protocol whole).final prover table) ∧
      (Proved setup packageWidths step cap control head
          (((Structure.protocol whole).input prover table 0).wire 0) ∨
        Solved honest innerBound out) := by
  obtain ⟨accepted, extracted⟩ := (final_suffix setup packageWidths step cap control shape read
    head bound premises laid lookups challenges cube early injective units sparse response uniform
    small varsFit batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit'
    inside (commit := commit) (value := value) (place := place) (next := next)
    (witnessOf := witnessOf)).extract_sound rfl prover budget table tape out result
  refine ⟨accepted, ?_⟩
  rcases out with certificate | broken
  · rcases Compose.Extracted.relation_or_collision extracted (registered _) covers with
      good | collision
    · exact Or.inl ⟨_, good⟩
    · exact Or.inr collision
  · exact Or.inr extracted

variable [Finite Wire]

/-- Akita Theorem 10.17 for the final verifier over the Akita opening. For each prover with at
most `Q` queries to the oracle, the probability that the verifier accepts and the extractor
returns no output is at most `(Q + 1)·(runCost + ε) + (Q + 1)·loops/budget`. `Setup.runCost` is
the sum of the bounds of the draws of Π_CCS on the fresh instance and of the argument, and `ε`
is `1/|K|` for the weights of the batching root and the stage errors of the two levels. -/
theorem final_extraction_le (inside : Tier.finish second ≤ total) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → K) total) Coord K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + akitaError) +
        (queries + 1) * (loops : ℕ) / budget :=
  (final_suffix setup packageWidths step cap control shape read head bound premises laid lookups
    challenges cube early injective units sparse response uniform small varsFit batchFit groupsFit
    injective' units' sparse' base' small' varsFit' batchFit' inside (commit := commit)
    (value := value) (place := place) (next := next) (witnessOf := witnessOf)).extraction_le rfl
    prover bounded positive

/-- The expected number of runs of the prover in the extraction is at most
`K̂ + (Q + 1)·(K̂ − 1)` with the tree size `K̂` of `final_suffix` (Akita Theorem 10.17, the
cost). -/
theorem final_runs_le (inside : Tier.finish second ≤ total) {queries : ℕ}
    (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → K) total) Coord K budget
          (Structure.protocol whole).stages,
        (extractRuns whole prover budget table tape : ℝ) ≤
      (tree : ℕ) + (queries + 1) * ((tree : ℕ) - 1) :=
  (final_suffix setup packageWidths step cap control shape read head bound premises laid lookups
    challenges cube early injective units sparse response uniform small varsFit batchFit groupsFit
    injective' units' sparse' base' small' varsFit' batchFit' inside (commit := commit)
    (value := value) (place := place) (next := next) (witnessOf := witnessOf)).runs_le rfl prover
    bounded budget

omit premises laid lookups challenges cube early injective units sparse response uniform small
  varsFit batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit' in
/-- The probability that the extractor returns a solution of Module-SIS (`Solved`): the term of
the Module-SIS assumption in the soundness of the final verifier. The extractor is the solver. -/
noncomputable def breakChance (honest : Certified (B.shape 0)) (innerBound budget : ℕ) : ℝ :=
  𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → K) total) Coord K budget
      (Structure.protocol whole).stages,
    if ∃ out, Checkpoint.extract whole prover budget table tape = some out ∧
      Solved honest innerBound out then (1 : ℝ) else 0

/-- Soundness of the final verifier with Akita openings. For each prover with at most `Q`
queries to the oracle, the probability that the verifier accepts and no proof with witnesses
satisfies `Verifies` for the statement of the first message is at most
`(Q + 1)·(runCost + ε) + (Q + 1)·loops/budget + breakChance`. The first two terms are those of
`final_extraction_le`, and `breakChance` is the probability that the extractor returns a solution
of Module-SIS. `registered` and `covers` are those of `final_extract_sound`. -/
theorem final_sound (inside : Tier.finish second ≤ total) {honest : Certified (B.shape 0)}
    {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (B.fold 0).sourceBound (B.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 _tape : Tape (Prefix Wire (Coord → K) total) Coord K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
            ¬ Proved setup packageWidths step cap control head
              (((Structure.protocol whole).input prover table 0).wire 0) then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + akitaError) +
        (queries + 1) * (loops : ℕ) / budget +
        breakChance setup packageWidths step cap control shape read head bound
          (commit := commit) (value := value) (place := place) (next := next)
          (witnessOf := witnessOf) (β := β) (β' := β') (key := key) (key' := key')
          (openKey := openKey) (openKey' := openKey') (claim := claim) (claim' := claim')
          (fits := fits) (fits' := fits') (rowsFit := rowsFit) (rowsFit' := rowsFit')
          (block := block) (block' := block') (decode := decode) (decode' := decode')
          (mass := mass) (weight' := weight') (base := base) prover honest innerBound budget := by
  have missed := final_extraction_le setup packageWidths step cap control shape read head bound
    premises laid lookups challenges cube early injective units sparse response uniform small
    varsFit batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit'
    (commit := commit) (value := value) (place := place) (next := next) (witnessOf := witnessOf)
    prover inside bounded positive
  refine le_trans ?_ (add_le_add missed le_rfl)
  unfold breakChance
  simp only [← Finset.expect_add_distrib]
  refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun tape _ => ?_
  have cases : accepts ((Structure.protocol whole).final prover table) ∧
      ¬ Proved setup packageWidths step cap control head
        (((Structure.protocol whole).input prover table 0).wire 0) →
      (accepts ((Structure.protocol whole).final prover table) ∧
        Checkpoint.extract whole prover budget table tape = none) ∨
      ∃ out, Checkpoint.extract whole prover budget table tape = some out ∧
        Solved honest innerBound out := by
    intro event
    rcases result : Checkpoint.extract whole prover budget table tape with _ | out
    · exact Or.inl ⟨event.1, rfl⟩
    · exact Or.inr ⟨out, rfl, ((final_extract_sound setup packageWidths step cap control shape
        read head bound premises laid lookups challenges cube early injective units sparse
        response uniform small varsFit batchFit groupsFit injective' units' sparse' base' small'
        varsFit' batchFit' prover inside registered covers budget table tape out
        result).2).resolve_left event.2⟩
  by_cases event : accepts ((Structure.protocol whole).final prover table) ∧
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

end NightstreamFPrime.Export.BoundedDepth.FinalAkita
