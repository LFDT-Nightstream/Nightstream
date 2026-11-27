import NightstreamFPrime.Spec.Compression.Akita.FoldCheck
import NightstreamFPrime.Spec.Compression.Akita.TraceFold

/-! The link from the checks of one fold of Akita to its exact branch relation
(ePrint 2026/1983, revision of 2026-11-23, Sections 5.5, 6 and 7, and the hypothesis of
Corollary 10.32).

- `Placed flat positions v` states that each coefficient of the digit vector `v` is a cell of the
  flat witness at a position of `positions`. `Placed.balanced` and `Placed.negativeBinary` give
  the alphabets of `v` from the digit range statement and the binary statement of the flat
  witness.
- `Layout` has the laws of the flat layout of Akita (75) and (76) that the link needs: the map
  from the digit vectors of one fold to the flat witness, the set `I_bin`, and the placement of
  each digit vector. It does not give the addresses of Akita Appendix L.2.
- `Witness.Alphabets` is the alphabet part of the branch relation, and `Layout.alphabets` gives
  it from the two statements of the flat witness.
- `Layout.foldCheck_avoids` is the link for a relation `Target` of the digit vectors: when a
  witness with the certified alphabets whose flat table satisfies the relation of the ring
  reduction satisfies `Target` (`encodes`), a path of `foldCheck` that avoids the bad sets and ends
  in a true claim gives `Target`. `foldCheck_satisfies` and `foldCheck_traceSatisfies` are the
  cases of `Witness.Satisfies` and `Witness.TraceSatisfies`, and `foldCheck_satisfies_error` and
  `foldCheck_traceSatisfies_error` are their bounds. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

section Cells

variable {d : ℕ} [NeZero d] {ι : Type} {flat : Table} {v : ι → Rq d}

/-- Each coefficient of the digit vector `v` is a cell of the flat witness `flat` at a position
of `positions` (Akita (76)). -/
def Placed (flat : Table) (positions : ℕ → Prop) (v : ι → Rq d) : Prop :=
  ∀ j, ∀ i < d, ∃ x, positions x ∧ flat x = coeff (v j) i

/-- A digit vector on the cube of a flat witness with the digit range statement is in the
alphabet `A_{b⋆}` (Akita Section 6.1). -/
theorem Placed.balanced {b μ : ℕ} (small : b < fieldModulus)
    (placed : Placed flat (· < 2 ^ μ) v) (range : RangeHolds b μ flat) : Balanced b v := by
  intro j i lt
  obtain ⟨x, inside, cell⟩ := placed j i lt
  rw [← cell]
  exact (rangeHolds_iff small μ flat).mp range x inside

/-- A digit vector on the positions `I_bin` of a flat witness with the binary statement is in the
alphabet `{−1, 0}` (Akita Remark 6.2). -/
theorem Placed.negativeBinary {support : ℕ → Prop} {μ : ℕ}
    (placed : Placed flat (fun x => x < 2 ^ μ ∧ support x) v) (binary : BinaryOn support μ flat) :
    NegativeBinary v := by
  refine (negativeBinary_iff v).mpr fun j i lt => ?_
  obtain ⟨x, ⟨inside, member⟩, cell⟩ := placed j i lt
  rw [← cell]
  exact binary x inside member

end Cells

/-- Each coefficient of the digit vectors of a compression path is a cell of the flat witness at
a position of `positions`. -/
def PathDigits.Placed (flat : Table) (positions : ℕ → Prop) :
    (chain : Option ChainShape) → PathDigits chain → Prop
  | none, _ => True
  | some _, digits =>
      Akita.Placed flat positions (ChainDigits.first digits) ∧
        Akita.Placed flat positions (ChainDigits.second digits)

theorem PathDigits.Placed.binary {flat : Table} {support : ℕ → Prop} {μ : ℕ}
    {chain : Option ChainShape} {digits : PathDigits chain}
    (placed : PathDigits.Placed flat (fun x => x < 2 ^ μ ∧ support x) chain digits)
    (binary : BinaryOn support μ flat) : PathDigits.Binary chain digits := by
  cases chain with
  | none => trivial
  | some c => exact ⟨placed.1.negativeBinary binary, placed.2.negativeBinary binary⟩

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

/-- The laws of the flat layout of the witness of one fold on the cube of `μ` variables
(Akita (75) and (76)). `flat` maps the digit vectors to the flat witness. `support` is the set
`I_bin` of the positions of the compression digits (Akita Remark 6.2), and `compressed` states
that the level has compression digits. Each coefficient of `ẑ`,
`t̂_i` and `ê` is a cell of the cube, and each coefficient of the digits of `F_1`, `F_2`, `H_1` and
`H_2` is a cell of the cube at a position of `I_bin`. -/
structure Layout (O : OpeningShape S) (fold : Descriptor) (μ : ℕ) where
  flat : Witness O fold → Table
  support : ℕ → Prop
  compressed : Bool
  raw : compressed = false → ∀ x, ¬ support x
  response : ∀ w, Placed (flat w) (· < 2 ^ μ) w.response
  inner : ∀ w i, Placed (flat w) (· < 2 ^ μ) (w.inner i)
  partials : ∀ w, Placed (flat w) (· < 2 ^ μ) w.partials
  chain : ∀ w, PathDigits.Placed (flat w) (fun x => x < 2 ^ μ ∧ support x) S.chain w.chain
  openChain : ∀ w,
    PathDigits.Placed (flat w) (fun x => x < 2 ^ μ ∧ support x) O.chain w.openChain

/-- The alphabets of the digit vectors of one fold (Akita Definition 10.22 (v) and Section 5.5):
`A_{b⋆}` for `ẑ`, `t̂_i` and `ê`, and `{−1, 0}` for the digits of the compression maps. -/
structure Witness.Alphabets (w : Witness O fold) : Prop where
  responseDigits : Balanced S.certifiedBase w.response
  innerDigits : ∀ i, Balanced S.certifiedBase (w.inner i)
  partialDigits : Balanced S.certifiedBase w.partials
  chainDigits : PathDigits.Binary S.chain w.chain
  openDigits : PathDigits.Binary O.chain w.openChain

/-- The digit range statement and the binary statement of the flat witness give the alphabets of
the digit vectors, for `b⋆ < q`. -/
theorem Layout.alphabets {μ : ℕ} (L : Layout O fold μ) (small : S.certifiedBase < fieldModulus)
    (w : Witness O fold) (range : RangeHolds S.certifiedBase μ (L.flat w))
    (binary : BinaryOn L.support μ (L.flat w)) : w.Alphabets where
  responseDigits := (L.response w).balanced small range
  innerDigits i := (L.inner w i).balanced small range
  partialDigits := (L.partials w).balanced small range
  chainDigits := (L.chain w).binary binary
  openDigits := (L.openChain w).binary binary

section Link

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

omit [Fintype E] [DecidableEq E] in
theorem Witness.Satisfies.alphabets {β : Module.Basis (Fin O.planes) Zq E} {w : Witness O fold}
    {st : Statement O E} {challenge : Fin S.blocks → Rq O.carrierDegree}
    (satisfies : w.Satisfies β st challenge) : w.Alphabets :=
  ⟨satisfies.responseDigits, satisfies.carried.innerDigits, satisfies.carried.partialDigits,
    satisfies.carried.chainDigits, satisfies.carried.openDigits⟩

omit [Fintype E] [DecidableEq E] in
theorem Witness.TraceSatisfies.alphabets {w : Witness O fold} {st : TraceStatement O E}
    {challenge : Fin S.blocks → Rq S.innerDegree} (satisfies : w.TraceSatisfies st challenge) :
    w.Alphabets :=
  ⟨satisfies.responseDigits, satisfies.carried.innerDigits, satisfies.carried.partialDigits,
    satisfies.carried.chainDigits, satisfies.carried.openDigits⟩

variable (R : Reduction E) (L : Layout O fold R.vars) [DecidablePred L.support]

/-- The link from the checks of one fold to a relation `Target` of its digit vectors. `encodes`
states that the ring reduction `R` reads the layout `L`: a witness with the certified alphabets
whose flat table satisfies the relation of `R` satisfies `Target`. Then on a path of `foldCheck`
that avoids the bad sets, an accepted claim `(r_2, v_W)` that is true for the flat witness gives
`Target`. -/
theorem Layout.foldCheck_avoids (small : S.certifiedBase < fieldModulus)
    (Target : Witness O fold → Prop)
    (encodes : ∀ w : Witness O fold, w.Alphabets → R.Holds (L.flat w) → Target w)
    (w : Witness O fold) :
    (foldCheck R S.certifiedBase L.support L.compressed (L.flat w)).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E (L.flat w)) → Target w :=
  (Akita.foldCheck_avoids R S.certifiedBase L.support L.compressed L.raw (L.flat w)).mono
    fun _ checked r2 v equal opening =>
      encodes w (L.alphabets small w (checked r2 v equal opening).2.1
        (checked r2 v equal opening).2.2) (checked r2 v equal opening).1

/-- The link for a fold with coefficient packing: on a path of `foldCheck` that avoids the bad
sets, an accepted claim that is true for the flat witness gives the exact branch relation
`Witness.Satisfies` for the fold challenges, which is the hypothesis of
`extract_opens_or_collision` (Akita Corollary 10.32) for one child. -/
theorem foldCheck_satisfies (small : S.certifiedBase < fieldModulus)
    (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree)
    (encodes : ∀ w : Witness O fold, w.Alphabets → R.Holds (L.flat w) →
      w.Satisfies β st challenge)
    (w : Witness O fold) :
    (foldCheck R S.certifiedBase L.support L.compressed (L.flat w)).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E (L.flat w)) →
        w.Satisfies β st challenge :=
  L.foldCheck_avoids R small _ encodes w

/-- The link for a fold with the evaluation-trace method: on a path of `foldCheck` that avoids the
bad sets, an accepted claim that is true for the flat witness gives the exact branch relation
`Witness.TraceSatisfies`, which is the hypothesis of `traceExtract_opens_or_collision` for one
child. -/
theorem foldCheck_traceSatisfies (small : S.certifiedBase < fieldModulus)
    (st : TraceStatement O E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (encodes : ∀ w : Witness O fold, w.Alphabets → R.Holds (L.flat w) →
      w.TraceSatisfies st challenge)
    (w : Witness O fold) :
    (foldCheck R S.certifiedBase L.support L.compressed (L.flat w)).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E (L.flat w)) →
        w.TraceSatisfies st challenge :=
  L.foldCheck_avoids R small _ encodes w

/-- For each prover, the probability that the checks of a packing fold end in a true claim for a
witness that does not satisfy the exact branch relation is at most `foldError`. -/
theorem foldCheck_satisfies_error (small : S.certifiedBase < fieldModulus)
    (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree)
    (encodes : ∀ w : Witness O fold, w.Alphabets → R.Holds (L.flat w) →
      w.Satisfies β st challenge)
    (w : Witness O fold) (s : Strategy (Option (List E × E))) :
    prob s (foldCheck R S.certifiedBase L.support L.compressed (L.flat w)) (fun result =>
      ¬ ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E (L.flat w)) →
        w.Satisfies β st challenge) ≤ foldError R S.certifiedBase L.compressed :=
  prob_not_le s _ _ _ (foldCheck_satisfies R L small β st challenge encodes w)
    (foldCheck_costLe R S.certifiedBase L.support L.compressed (L.flat w))

/-- For each prover, the probability that the checks of a trace fold end in a true claim for a
witness that does not satisfy the exact branch relation is at most `foldError`. -/
theorem foldCheck_traceSatisfies_error (small : S.certifiedBase < fieldModulus)
    (st : TraceStatement O E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (encodes : ∀ w : Witness O fold, w.Alphabets → R.Holds (L.flat w) →
      w.TraceSatisfies st challenge)
    (w : Witness O fold) (s : Strategy (Option (List E × E))) :
    prob s (foldCheck R S.certifiedBase L.support L.compressed (L.flat w)) (fun result =>
      ¬ ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E (L.flat w)) →
        w.TraceSatisfies st challenge) ≤ foldError R S.certifiedBase L.compressed :=
  prob_not_le s _ _ _ (foldCheck_traceSatisfies R L small st challenge encodes w)
    (foldCheck_costLe R S.certifiedBase L.support L.compressed (L.flat w))

end Link

end NightstreamFPrime.Spec.Compression.Akita
