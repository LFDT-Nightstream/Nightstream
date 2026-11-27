import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink

/-! The opening suffix of Akita for a packing level over a trace level over the terminal, for
`k = 2` (ePrint 2026/1983, revision of 2026-11-23, (127), Proposition 10.44, Theorem 10.17 and
Corollary 10.49 with (206)).

- `tripleAccepts` is the verifier of two levels and the terminal: the link `Tier.TraceLinked` from
  the first level to the second level, the link `Terminal.Linked` from the second level to the
  terminal, and the checks `Terminal.Passed` of the terminal.
- `triple` is the checkpoint structure of the two levels over the terminal, and `triple_suffix`
  states that it is a suffix.
- `traceTriple_extract_sound`, `traceTriple_extraction_le` and `traceTriple_runs_le` are the
  extraction theorem of a committed packing fold over a committed trace fold over the terminal,
  each with the tensor reduction of Akita Section 5.2 after its checks.

The quotient-lift reduction, more than one claim or group, the recomputation (131) of the last
block of the terminal, and the setup offloading are not in this file. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

variable {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E]
  [DecidableEq E] [Algebra Zq E] [Degree2 E]

section Triple

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {S'' : Shape} {Ring Stmt Brk Wire Coord Slot : Type} [Zero Ring]
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {key' : Key S'} {openKey' : OpeningKey O'} {weight' total m' m'' : ℕ}
  (A : Tier (Fin S.blocks) (fun _ => Ring) Stmt Brk (Witness O fold) (Weak O) Wire Coord Slot
    (Option (List E × E × E))
    Table)
  (B : Tier (Fin S'.blocks) (fun _ => Rq S'.innerDegree) (TraceClaim O' E)
    (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) (Witness O' fold')
    (Weak O') Wire Coord Slot
    (Option (List E × E × E)) Table)
  (L : Terminal S'' E Wire Coord Slot total)
  (quarter' : 4 * m' = S'.innerDegree) (quarter'' : 4 * m'' = S''.innerDegree)
  (commit : Wire → S'.Payload)
  (stateOf : Wire → Fin S''.blocks → Fin S''.innerRank → Rq S''.innerDegree)

/-- The verifier of two levels and the terminal: the level `A` with the link `Tier.TraceLinked`
to the claim of the trace level `B`, the level `B` with the link `Terminal.Linked` to the
statement of the terminal `L`, and the checks (128), (129) and (130) of the terminal with the
bound of its clear response. -/
def tripleAccepts (point : Prefix Wire (Option Coord → Slot) total) : Prop :=
  A.TraceLinked B quarter' commit point ∧ Terminal.Linked B L quarter'' stateOf point ∧
    L.Passed point

/-- The checkpoint structure of the trace level `B` over the terminal. -/
noncomputable def tripleLast :
    Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire)
      (Opened (Weak O') (Prefix Wire (Option Coord → Slot) total × Option (TerminalFork S'')))
      (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕
        Option (ShortKernel L.matrix (8 * L.weight * L.bound)))
      (Table ⊕ Unit) ((Σ _ : Fin S'.blocks, Rq S'.innerDegree) ⊕ Rq S''.innerDegree)
      (Option (Weak O')) :=
  B.over (tripleAccepts A B L quarter' quarter'' commit stateOf)
    (L.body (tripleAccepts A B L quarter' quarter'' commit stateOf))
    (fun inner => flatten (E := E) (unpackTable S''.innerDegree m'' (S''.decode (Terminal.blocksOf
      ((L.body (tripleAccepts A B L quarter' quarter'' commit stateOf)).witness inner)))))
    (fun inner other => L.collide
      ((L.body (tripleAccepts A B L quarter' quarter'' commit stateOf)).witness inner)
      ((L.body (tripleAccepts A B L quarter' quarter'' commit stateOf)).witness other))

/-- The checkpoint structure of two levels over the terminal (Akita Proposition 10.44). A witness
of `R_bind` is a solution of Module-SIS for `B`, `D` or a compression map of `A`, for the matrix
`A`, `B`, `D` or a compression map of `B`, or for `A_term`. -/
noncomputable def triple :
    Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire)
      (Opened (Weak O)
        (Opened (Weak O') (Prefix Wire (Option Coord → Slot) total × Option (TerminalFork S''))))
      (Brk ⊕ (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕
        Option (ShortKernel L.matrix (8 * L.weight * L.bound))))
      (Table ⊕ (Table ⊕ Unit))
      ((Σ _ : Fin S.blocks, Ring) ⊕
        ((Σ _ : Fin S'.blocks, Rq S'.innerDegree) ⊕ Rq S''.innerDegree))
      (Option (Weak O)) :=
  A.over (tripleAccepts A B L quarter' quarter'' commit stateOf)
    (tripleLast A B L quarter' quarter'' commit stateOf)
    (fun inner => flatten (E := E) (unpackTable S'.innerDegree m' (tableOf
      ((tripleLast A B L quarter' quarter'' commit stateOf).witness inner))))
    (fun inner other => Sum.inl (Tier.collide key' openKey'
      (2 * (2 * weight') * fold'.sourceBound S')
      ((tripleLast A B L quarter' quarter'' commit stateOf).witness inner)
      ((tripleLast A B L quarter' quarter'' commit stateOf).witness other)))

variable {A B L quarter' quarter'' commit stateOf}

omit [Fintype E] [DecidableEq E] in
/-- Two levels over the terminal are a suffix: the stage errors, the retry loops and the tree size
are those of `Tier.suffix_push_trace` over `Terminal.suffix_linked`. The hypotheses on the
terminal are: its fold challenge is a stage of the transcript after the stages of `B`, the blocks
have different coordinates, and the challenge family has units as differences and members of the
norm `κ_{1,term}`. -/
theorem triple_suffix (valid : A.Valid) (valid' : B.Valid)
    (traced : B.method = trace key' openKey' weight') (next : B.start = A.finish)
    (next' : L.start = B.finish) (fits : L.start < total) (injective : Function.Injective L.block)
    {peak : ℕ} (units : UnitDifferences L.family)
    (sparse : ∀ challenge ∈ L.family, Sparse L.weight peak challenge)
    (uniform : ∀ wire challenge table other,
      bare (A.check wire challenge table) = bare (A.check wire challenge other))
    (passes : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      A.passes table (some (ρ, θ, v)))
    (uniform' : ∀ wire challenge table other,
      bare (B.check wire challenge table) = bare (B.check wire challenge other))
    (passes' : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      B.passes table (some (ρ, θ, v))) :
    Tier.Suffix (tripleAccepts A B L quarter' quarter'' commit stateOf) A.start A.Opens
      (triple A B L quarter' quarter'' commit stateOf)
      (A.levelError +
        (B.levelError + L.termError))
      (S.blocks + 1 + (S'.blocks + 1 + S''.blocks))
      (2 * (S.blocks + 1) * (2 * (S'.blocks + 1) * (S''.blocks + 1))) :=
  Tier.suffix_push_trace valid traced next quarter' commit
    (by
      have ends := B.finish_eq
      omega)
    uniform passes (fun _ accepted => accepted.1) Sum.inl
    (fun solution found => by
      show B.method.broken solution
      rw [traced]
      exact found)
    (Terminal.suffix_linked valid' ⟨injective, fits, fun _ accepted => accepted.2.2⟩ next' units
      ⟨peak, sparse⟩ quarter'' stateOf uniform' passes' fun _ accepted => accepted.2.1)

end Triple

section Trace

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {S'' : Shape} {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]
  [Nonempty Coord]
  {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}
  {claim : Wire → Claim O E} {μ : ℕ} {fits : O.cells fold ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
  {decode : Slot → Option (Rq O.carrierDegree)} {read : Reader Wire} {weight ν : ℕ}
  {key' : Key S'} {openKey' : OpeningKey O'} {claim' : Wire → TraceClaim O' E} {μ' : ℕ}
  {fits' : O'.cells fold' ≤ 2 ^ μ'} {batch' : ℕ}
  {rowsFit' : O'.traceRowCount + 1 ≤ 2 ^ batch'} {block' : Fin S'.blocks → Coord}
  {decode' : Slot → Option (Rq S'.innerDegree)} {weight' ν' m' m'' : ℕ}
  {quarter' : 4 * m' = S'.innerDegree} {quarter'' : 4 * m'' = S''.innerDegree}
  {commit : Wire → S'.Payload}
  {stateOf : Wire → Fin S''.blocks → Fin S''.innerRank → Rq S''.innerDegree}

local notation "first" =>
  Tier.tensored
    (committedFold β key openKey claim μ fits batch rowsFit block decode read weight 0) ν

local notation "second" =>
  Tier.tensored
    (committedTrace key' openKey' claim' μ' fits' batch' rowsFit' block' decode' read weight'
      (Tier.finish first)) ν'

variable {total : ℕ} {L : Terminal S'' E Wire Coord Slot total}

local notation "chain" => triple first second L quarter' quarter'' commit stateOf

/-- A committed packing fold over a committed trace fold over the terminal is a suffix from the
stage 0. Each fold has the tensor reduction after its checks. `next` and `inside`: the fold
challenge of the terminal is the stage after the stages of the trace fold, and it is a stage of
the transcript. -/
theorem traceTriple_suffix {peak peak' peak'' : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (next : L.start = Tier.finish second) (inside : L.start < total)
    (injective'' : Function.Injective L.block)
    (units'' : UnitDifferences L.family)
    (sparse'' : ∀ challenge ∈ L.family, Sparse L.weight peak'' challenge) :
    Tier.Suffix (tripleAccepts first second L quarter' quarter'' commit stateOf) 0
      (Tier.Opens first) chain
      (Tier.levelError first +
        (Tier.levelError second +
          L.termError))
      (S.blocks + 1 + (S'.blocks + 1 + S''.blocks))
      (2 * (S.blocks + 1) * (2 * (S'.blocks + 1) * (S''.blocks + 1))) :=
  triple_suffix
    (Tier.tensored_valid (committedFold_valid injective units sparse base small varsFit batchFit)
      (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)
      fun _ _ table rest tail rejected ended =>
        foldCheck_stagedTo _ _ _ _ table varsFit batchFit rest tail rejected ended)
    (Tier.tensored_valid
      (committedTrace_valid injective' units' sparse' base' small' varsFit' batchFit')
      (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)
      fun _ _ table rest tail rejected ended =>
        foldCheck_stagedTo _ _ _ _ table varsFit' batchFit' rest tail rejected ended)
    rfl rfl next inside injective'' units'' sparse''
    (Tier.tensored_uniform committedFold_uniform) (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    (Tier.tensored_uniform committedTrace_uniform) (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)

/-- An output of the extractor for a committed packing fold over a committed trace fold over the
terminal: the verifier `tripleAccepts` accepts the transcript of the prover, and the output is a
root weak opening of the claim of the first level with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f` (Akita
Definition 10.22), or a solution of Module-SIS for `B`, `D` or a compression map of the first
level, for `A`, `B`, `D` or a compression map of the trace level, or for `A_term` with the radius
`8·κ_{1,term}·Z_{∞,term}` of Akita (181). By `Tier.opens_evaluation`, the value of the root claim is
the multilinear evaluation of the table of the root weak opening at the point of the claim. -/
theorem traceTriple_extract_sound {peak peak' peak'' : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (next : L.start = Tier.finish second) (inside : L.start < total)
    (injective'' : Function.Injective L.block)
    (units'' : UnitDifferences L.family)
    (sparse'' : ∀ challenge ∈ L.family, Sparse L.weight peak'' challenge)
    (prover : OracleComp (Address Wire total)
      (Option Coord → Slot) (ℕ → Wire))
    (budget : ℕ)
    (table : Address Wire total → Option Coord → Slot)
    (tape : Tape (Address Wire total) (Option Coord) Slot
      budget (Structure.protocol chain).stages)
    (out : Opened (Weak O) (Opened (Weak O') (Prefix Wire (Option Coord → Slot) total ×
        Option (TerminalFork S''))) ⊕
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S) ⊕
        (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') ⊕
          Option (ShortKernel L.matrix (8 * L.weight * L.bound)))))
    (result : Checkpoint.extract chain prover budget table tape = some out) :
    tripleAccepts first second L quarter' quarter'' commit stateOf
        ((Structure.protocol chain).final prover table) ∧
      Sum.elim
        (fun certificate => ∃ ω, certificate.opening = some ω ∧
          ω.Opens β (2 * weight) (fold.sourceBound S)
            ((claim (((Structure.protocol chain).input prover table 0).wire 0)).statement key
              openKey)
            (S.decode ω.source))
        (Sum.elim (fun solution => solution.isSome)
          (Sum.elim (fun solution => solution.isSome)
            fun solution : Option (ShortKernel L.matrix (8 * L.weight * L.bound)) =>
              solution.isSome))
        out :=
  (traceTriple_suffix injective units sparse base small varsFit batchFit injective' units'
    sparse' base' small' varsFit' batchFit' next inside injective'' units''
    sparse'').extract_sound rfl prover budget table tape out result

variable [Finite Wire]

/-- Akita Theorem 10.17 for a committed packing fold over a committed trace fold over the
terminal, which is Akita Corollary 10.49 with (206) for these stages. For each prover with at most
`Q` queries to the oracle, the probability that the verifier `tripleAccepts` accepts and the
extractor returns no output is at most
`(Q + 1)·(ε + ε' + 2^{−g_term}·B_term·p_term) + (Q + 1)·(r + 1 + r' + 1 + B_term)/budget`.
`ε` and `ε'` are the stage errors `Tier.levelError` of the two levels: the weights of their fold
coordinates and the bounds of the draws of their checks with the tensor reduction (Akita Theorem
5.1 for one claim), each times `2^{−g}` for its work bits (Remark 10.18). `r`, `r'` and `B_term`
are the numbers of blocks, `p_term` is the largest probability of one fold challenge of the
terminal, and `budget` is the limit of the attempts of a retry loop of the extractor. By
`traceTriple_extract_sound`, an output is a root weak opening or a solution of
Module-SIS. -/
theorem traceTriple_extraction_le {peak peak' peak'' : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (next : L.start = Tier.finish second) (inside : L.start < total)
    (injective'' : Function.Injective L.block)
    (units'' : UnitDifferences L.family)
    (sparse'' : ∀ challenge ∈ L.family, Sparse L.weight peak'' challenge)
    (prover : OracleComp (Address Wire total)
      (Option Coord → Slot) (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord)
          Slot budget (Structure.protocol chain).stages,
        (if tripleAccepts first second L quarter' quarter'' commit stateOf
            ((Structure.protocol chain).final prover table) ∧
          Checkpoint.extract chain prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (Tier.levelError first + (Tier.levelError second + L.termError)) +
        (queries + 1) * (S.blocks + 1 + (S'.blocks + 1 + S''.blocks) : ℕ) / budget := by
  exact (traceTriple_suffix injective units sparse base small varsFit batchFit injective' units'
    sparse' base' small' varsFit' batchFit' next inside injective'' units''
    sparse'').extraction_le rfl prover (bounded.spread _) positive

/-- The expected number of runs of the prover in the extraction of a committed packing fold over
a committed trace fold over the terminal is at most `K̂ + (Q + 1)·(K̂ − 1)` with
`K̂ = 2·(r + 1)·2·(r' + 1)·(1 + B_term)` for `r`, `r'` and `B_term` blocks (Akita Theorem 10.17,
the cost, with the fold factor `1 + B_term` of (197)). -/
theorem traceTriple_runs_le {peak peak' peak'' : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (injective' : Function.Injective block')
    (units' : UnitDifferences {challenge | ∃ slot, decode' slot = some challenge})
    (sparse' : ∀ challenge, (∃ slot, decode' slot = some challenge) →
      Sparse weight' peak' challenge)
    (base' : 2 ≤ fold'.responseBase) (small' : S'.certifiedBase < fieldModulus)
    (varsFit' : μ' ≤ Fintype.card Coord) (batchFit' : batch' ≤ Fintype.card Coord)
    (next : L.start = Tier.finish second) (inside : L.start < total)
    (injective'' : Function.Injective L.block)
    (units'' : UnitDifferences L.family)
    (sparse'' : ∀ challenge ∈ L.family, Sparse L.weight peak'' challenge)
    (prover : OracleComp (Address Wire total)
      (Option Coord → Slot) (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Address Wire total) (Option Coord)
          Slot budget (Structure.protocol chain).stages,
        (extractRuns chain prover budget table tape : ℝ) ≤
      (2 * (S.blocks + 1) * (2 * (S'.blocks + 1) * (S''.blocks + 1)) : ℕ) +
        (queries + 1) * ((2 * (S.blocks + 1) * (2 * (S'.blocks + 1) * (S''.blocks + 1)) : ℕ) - 1) :=
  (traceTriple_suffix injective units sparse base small varsFit batchFit injective' units'
    sparse' base' small' varsFit' batchFit' next inside injective'' units''
    sparse'').runs_le rfl prover (bounded.spread _) budget

end Trace

end NightstreamFPrime.Spec.Compression.Akita.Extract
