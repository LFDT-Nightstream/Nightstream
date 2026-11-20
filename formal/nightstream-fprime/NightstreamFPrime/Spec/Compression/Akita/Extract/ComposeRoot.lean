import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchChain
import NightstreamFPrime.Spec.Compression.Akita.Extract.Compose

/-! The compression argument over an Akita opening of three groups with a batching root, one fold
with opening groups and a last packing fold (Akita, ePrint 2026/1983, revision of 2026-10-15,
Theorem 10.17, Proposition 10.44 and Theorem 10.46, "Batching root"; SPEC.md Algorithm 10).

- `composeAccepts` is the verifier: the conditions `Compose.Ran` of the compression argument, and
  the conditions of `rootAccepts` with the batching root at the stage `Compose.root`.
- `composeRoot` is the checkpoint structure of the Akita opening: the batching root over the fold
  `T` with opening groups over the last level `N`. `composeRoot_suffix` states that it is a
  suffix from the stage `Compose.root`, and `compose_suffix` puts the two checkpoints of the
  compression argument before it.
- `composeChain_suffix`, `compose_extract_sound`, `compose_extraction_le` and `compose_runs_le`
  are the result for `committedBatch` over `clearFold`. The hypotheses are conditions on the
  parameters only. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Root

variable {B : Batch (Fin 3)} {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor}
  {E Wire Coord Lead Result' Source' : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  [Fintype Coord] [DecidableEq Coord] {total : ℕ} {P : Verifier.Params Zq E}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin 3) → Key (B.shape g)}
  {openKey : B.OpenKey} {mass : Fin 3 → ℕ} {β' : Module.Basis (Fin O'.planes) Zq E}
  {key' : Key S'} {openKey' : OpeningKey O'} {weight' : ℕ}
  (C : Compose P B Wire Coord Lead total) (vf : Verifier.VerifierData P)
  (T : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord E (Option (List E × E)) Table)
  (N : Tier (Fin S'.blocks) (fun _ => Rq O'.carrierDegree) (Claim O' E)
    (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
    (Weak O') Wire Coord E Result' Source')
  (commit : Wire → (g : Fin 3) → (B.shape g).Payload) (value : Wire → Fin 3 → E)
  (place : Wire → Fin 3 → List E) (next : Wire → S'.Payload) (sourceOf : Wire → Source')

/-- The verifier of the compression argument over the Akita opening (SPEC.md Algorithm 10). It
accepts with the conditions `Compose.Ran` of the stages before the opening, and with the
conditions of `rootAccepts` for the batching root at the stage `Compose.root`: the transcript has
this stage, the link `RootLinked` to the claim of `T` with the weights of the transcript, the
link `Tier.Linked` from `T` to `N`, and a true final claim of the checks of `N`. -/
def composeAccepts (point : Prefix Wire (Coord → E) total) : Prop :=
  C.Ran vf commit value place point ∧ C.root vf < point.rest.length ∧
    RootLinked T (C.root vf) commit value place
      ((claimsRoot (Coord := Coord) (Slot := E) (total := total) (C.root vf) C.read value
        place).weights point) point ∧
    T.Linked N next point ∧ N.Passed (sourceOf (point.wire (N.start + 1))) point

/-- The checkpoint structure of the last level `N` under the verifier `composeAccepts`. -/
noncomputable def composeLast :=
  N.over (composeAccepts C vf T N commit value place next sourceOf)
    (finish (transcript total (composeAccepts C vf T N commit value place next sourceOf)))
    (fun point => sourceOf (point.wire (N.start + 1))) (fun _ _ => ())

/-- The checkpoint structure of the fold `T` with opening groups over the last level `N`. -/
noncomputable def composeFold :=
  T.over (composeAccepts C vf T N commit value place next sourceOf)
    (composeLast C vf T N commit value place next sourceOf)
    (fun inner => tableOf ((composeLast C vf T N commit value place next sourceOf).witness inner))
    (fun inner other => Sum.inl (Tier.collide key' openKey'
      (2 * (2 * weight') * fold'.sourceBound S')
      ((composeLast C vf T N commit value place next sourceOf).witness inner)
      ((composeLast C vf T N commit value place next sourceOf).witness other)))

/-- The checkpoint structure of the Akita opening: the batching root at the stage `Compose.root`
over the fold `T` with opening groups and the last level `N`. -/
noncomputable def composeRoot :=
  (claimsRoot (Coord := Coord) (Slot := E) (total := total) (C.root vf) C.read value
      place).segment.over
    (composeAccepts C vf T N commit value place next sourceOf)
    (composeFold C vf T N commit value place next sourceOf)
    (fun inner =>
      Batch.tablesOf ((composeFold C vf T N commit value place next sourceOf).witness inner))
    (fun inner other => Sum.inl (Batch.collide key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      ((composeFold C vf T N commit value place next sourceOf).witness inner)
      ((composeFold C vf T N commit value place next sourceOf).witness other)))
    (OpensAt (C.root vf) commit value place key mass)

variable {C vf T N commit value place next sourceOf}

/-- The Akita opening is a suffix from the stage `Compose.root` (`batchRoot_suffix` for a batching
root at this stage). -/
theorem composeRoot_suffix {blocks : ℕ} (card : Fintype.card B.Block = blocks) (valid : T.Valid)
    (valid' : N.Valid) (fair : ∃ decode : (Coord → E) → Fin 3 → E, Fair decode)
    (packed : T.method = Batch.method β key openKey mass)
    (packed' : N.method = packing β' key' openKey' weight') (first : T.start = C.root vf + 1)
    (second : N.start = T.finish) (fits : N.finish ≤ total)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) → T.passes table (some (r2, v))) :
    Tier.Suffix (composeAccepts C vf T N commit value place next sourceOf) (C.root vf)
      (OpensAt (C.root vf) commit value place key mass)
      (composeRoot C vf T N commit value place next sourceOf)
      (1 / Fintype.card E + (∑ b, T.chance b + T.costs.sum + (∑ b, N.chance b + N.costs.sum)))
      (1 + (blocks + 1 + (S'.blocks + 1))) (2 * (2 * (blocks + 1) * (2 * (S'.blocks + 1)))) :=
  root_suffix (C.root vf) C.read value place fair packed first
    (by
      have ends := T.finish_eq
      have ends' := N.finish_eq
      omega)
    commit (fun point accepted => ⟨accepted.2.1, accepted.2.2.1⟩) Sum.inl
    (fun solution found => by
      show T.method.broken solution
      rw [packed]
      exact found)
    (Tier.suffix_push card valid packed' second next
      (by
        have ends' := N.finish_eq
        omega)
      uniform passes (fun _ accepted => accepted.2.2.2.1) Sum.inl
      (fun solution found => by
        show N.method.broken solution
        rw [packed']
        exact found)
      (Tier.suffix_last (Fintype.card_fin _) valid' sourceOf fits
        fun _ accepted => accepted.2.2.2.2))

/-- The compression argument over the Akita opening is a suffix from the stage of the lookup
challenges: `Compose.suffix` over `composeRoot_suffix`. -/
theorem compose_suffix [Nonempty Coord] {blocks : ℕ} (sized : C.Sized vf)
    (card : Fintype.card B.Block = blocks) (valid : T.Valid) (valid' : N.Valid)
    (fair : ∃ decode : (Coord → E) → Fin 3 → E, Fair decode)
    (packed : T.method = Batch.method β key openKey mass)
    (packed' : N.method = packing β' key' openKey' weight') (first : T.start = C.root vf + 1)
    (second : N.start = T.finish) (fits : N.finish ≤ total)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) → T.passes table (some (r2, v))) :
    Tier.Suffix (composeAccepts C vf T N commit value place next sourceOf) C.start
      (C.Extracted key mass)
      (C.chain vf (composeAccepts C vf T N commit value place next sourceOf)
        (composeRoot C vf T N commit value place next sourceOf) key openKey mass Sum.inl)
      (C.leadCosts.sum + Verifier.runCost vf +
        (1 / Fintype.card E + (∑ b, T.chance b + T.costs.sum + (∑ b, N.chance b + N.costs.sum))))
      (2 + (1 + (blocks + 1 + (S'.blocks + 1))))
      (4 * (2 * (2 * (blocks + 1) * (2 * (S'.blocks + 1))))) :=
  Compose.suffix sized (fun _ accepted => accepted.1) Sum.inl
    (fun solution found => by
      show T.method.broken solution
      rw [packed]
      exact found)
    (composeRoot_suffix card valid valid' fair packed packed' first second fits uniform passes)

end Root

section Instance

variable {B : Batch (Fin 3)} {S' : Shape} {O' : OpeningShape S'} {fold' : Descriptor} {E : Type}
  [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] {Wire Coord Lead : Type} [Fintype Coord]
  [DecidableEq Coord] [Nonempty Coord] {total : ℕ} {P : Verifier.Params Zq E}
  {C : Compose P B Wire Coord Lead total} {vf : Verifier.VerifierData P}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin 3) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells < 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin 3) → E → Option (Rq (B.carrierDegree g))} {mass : Fin 3 → ℕ} {base : ℕ}
  {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'} {openKey' : OpeningKey O'}
  {claim' : Wire → Claim O' E} {μ' : ℕ} {fits' : O'.cells fold' < 2 ^ μ'} {batch' : ℕ}
  {rowsFit' : O'.rowCount + 1 ≤ 2 ^ batch'} {block' : Fin S'.blocks → Coord}
  {decode' : E → Option (Rq O'.carrierDegree)} {weight' : ℕ}
  {commit : Wire → (g : Fin 3) → (B.shape g).Payload} {value : Wire → Fin 3 → E}
  {place : Wire → Fin 3 → List E} {next : Wire → S'.Payload} {witnessOf : Wire → Witness O' fold'}

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

variable {peak : Fin 3 → ℕ} {peak' : ℕ} (sized : C.Sized vf)
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

include sized injective units sparse response uniform small varsFit batchFit groupsFit
  injective' units' sparse' base' small' varsFit' batchFit'

/-- The compression argument over a batching root, a committed fold with three opening groups
and a last packing fold is a suffix from the stage of the lookup challenges. `sized`: the
conditions `Compose.Sized` of the compression argument. `inside`: the stages are stages of the
transcript. The other hypotheses are those of `batchRootChain_suffix`. -/
theorem composeChain_suffix (inside : Tier.finish second ≤ total) :
    Tier.Suffix accepts C.start (C.Extracted key mass) whole
      (C.leadCosts.sum + Verifier.runCost vf + (1 / Fintype.card E + (∑ b, Tier.chance first b +
        (Tier.costs first).sum + (∑ b, Tier.chance second b + (Tier.costs second).sum))))
      (2 + (1 + (∑ g, (B.shape g).blocks + 1 + (S'.blocks + 1))))
      (4 * (2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S'.blocks + 1))))) :=
  compose_suffix sized B.card_block
    (committedBatch_valid injective units sparse response uniform small varsFit batchFit)
    (clearFold_valid injective' units' sparse' base' small' varsFit' batchFit')
    (exists_fair_slots groupsFit) rfl rfl rfl rfl inside committedBatch_uniform
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)

variable (start : C.start = 0)
  (prover : OracleComp (Prefix Wire (Coord → E) total) (Coord → E) (ℕ → Wire))

include start

/-- An output of the extractor for the compression argument over the Akita opening (Akita
Theorem 10.17): the verifier accepts the transcript of the prover, and the output has certified
decommitments of the commitments of the key polynomial and of `W` of the first message, and
`Compose.good` holds for its table of `W` when its table of the key polynomial is the table of
`Params.key`, or the output is a solution of Module-SIS for a matrix of the first level or of the
last level. -/
theorem compose_extract_sound (inside : Tier.finish second ≤ total) (budget : ℕ) (table : Prefix Wire (Coord → E) total → Coord → E)
    (tape : Tape (Prefix Wire (Coord → E) total) Coord E budget (Structure.protocol whole).stages)
    (out : Opened B.Weak (Opened (Weak O') (Prefix Wire (Coord → E) total)) ⊕
      (B.LevelBreak key openKey
          (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g)) ⊕
        (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕ Unit)))
    (result : Checkpoint.extract whole prover budget table tape = some out) :
    accepts ((Structure.protocol whole).final prover table) ∧
      Sum.elim
        (fun certificate => C.Extracted key mass ((Structure.protocol whole).input prover table 0)
          certificate.opening)
        (Sum.elim (fun solution => solution.isSome)
          (Sum.elim (fun solution => solution.isSome) fun _ => False))
        out :=
  (composeChain_suffix (commit := commit) (value := value) (place := place) (next := next)
    (witnessOf := witnessOf) sized injective units sparse response uniform small varsFit
    batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit' inside).extract_sound
    start prover budget table tape out result

variable [Finite Wire]

/-- Akita Theorem 10.17 for the compression argument over the Akita opening. For each prover
with at most `Q` queries to the oracle, the probability that the verifier accepts and the
extractor returns no output is at most `(Q + 1)·ε + (Q + 1)·(3 + r + 1 + r' + 1)/budget`. `ε` is
the bounds of the draws of `Compose.lead`, `runCost` of the compression argument, `1/|E|` for
the weights of the batching root, and the stage errors of the two levels with `r = Σ_g B_g` and
`r'` fold coordinates. -/
theorem compose_extraction_le (inside : Tier.finish second ≤ total) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → E) total) Coord E budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
          Checkpoint.extract whole prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (C.leadCosts.sum + Verifier.runCost vf + (1 / Fintype.card E +
        (∑ b, Tier.chance first b + (Tier.costs first).sum +
          (∑ b, Tier.chance second b + (Tier.costs second).sum)))) +
        (queries + 1) * (2 + (1 + (∑ g, (B.shape g).blocks + 1 + (S'.blocks + 1))) : ℕ) / budget :=
  (composeChain_suffix (commit := commit) (value := value) (place := place) (next := next)
    (witnessOf := witnessOf) sized injective units sparse response uniform small varsFit
    batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit' inside).extraction_le
    start prover bounded positive

/-- The expected number of runs of the prover in the extraction is at most
`K̂ + (Q + 1)·(K̂ − 1)` with `K̂ = 4·2·2·(r + 1)·2·(r' + 1)` (Akita Theorem 10.17, the cost). -/
theorem compose_runs_le (inside : Tier.finish second ≤ total) {queries : ℕ}
    (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → E) total) Coord E budget
          (Structure.protocol whole).stages,
        (extractRuns whole prover budget table tape : ℝ) ≤
      (4 * (2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S'.blocks + 1)))) : ℕ) +
        (queries + 1) *
          ((4 * (2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S'.blocks + 1)))) : ℕ) - 1) :=
  (composeChain_suffix (commit := commit) (value := value) (place := place) (next := next)
    (witnessOf := witnessOf) sized injective units sparse response uniform small varsFit
    batchFit groupsFit injective' units' sparse' base' small' varsFit' batchFit' inside).runs_le start
    prover bounded budget

end Instance

end NightstreamFPrime.Spec.Compression.Akita.Extract
