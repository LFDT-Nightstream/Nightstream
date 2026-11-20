import NightstreamFPrime.Spec.Compression.Akita.Prover
import NightstreamFPrime.Spec.Compression.Akita.Extract.TierBound
import NightstreamFPrime.Spec.Compression.Akita.Extract.Uniform

/-! The recursive opening suffix of Akita: the composition of its levels (ePrint 2026/1983,
revision of 2026-10-15, Proposition 10.44 with its proof, and Corollary 10.49 with (206)).

- `tableCollision` is the reduction of Akita Lemma 10.23 and Corollary 10.25 for two certified
  decommitments of one payload to different tables: `Coll_ℓ` of the checkpoint of a level whose
  next level commits to its flat witness.
- `Tier.Suffix` is the invariant of the stages from one level: the conditions of
  Definition 10.16 from its first stage, a weak opening of its claim in each first certificate
  (Proposition 10.44 (i)), and the sums of the stage errors, of the retry loops and the tree size.
- `Tier.suffix_last` is the suffix of a last level, whose proof has the source in the clear.
- `Tier.Linked` is the link that the verifier makes from a level to the next level: the payload
  of the next claim is the commitment that the proof has before the checks, and the point and the
  value of the next claim are the final claim `(r_2, v_W)` of the checks. `Tier.suffix_push` is
  the suffix of such a level over the suffix of the next level.
- `Tier.Suffix.lawful`, `Tier.Suffix.extract_sound`, `Tier.Suffix.extraction_le` and
  `Tier.Suffix.runs_le` are Akita Theorem 10.17 for a suffix from the stage 0. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

section Collision

variable {S : Shape}

/-- The reduction of Akita Lemma 10.23 for two certified decommitments: the solution of Lemma 4.3
for the path of `B`, or else the solution `z_A` for `A`. -/
noncomputable def tableCollision (key : Key S) (innerBound : ℕ) (st st' : Certified S) :
    Option (ShortKernel key.inner innerBound ⊕ OuterBreak key) :=
  ((outerCollision key st.inner st'.inner st.chain st'.chain).map Sum.inr).or
    ((sourceCollision key innerBound st st').map Sum.inl)

/-- Akita Lemma 10.23 for the commitment only (Corollary 10.25 for two certified occurrences):
two certified decommitments of one payload to different tables give a solution of Module-SIS, for
`A` with each bound `η_A ≥ κ̄'_1·β̄_∞ + κ̄_1·β̄'_∞`, for `B` with the bound `b⋆ − 1`, or for a
compression map with the bound 1. -/
theorem tableCollision_isSome {key : Key S}
    {innerBound factorBound sourceBound factorBound' sourceBound' : ℕ} {st st' : Certified S}
    {payload : S.Payload} {table table' : Table}
    (opens : st.Opens factorBound sourceBound key payload table)
    (opens' : st'.Opens factorBound' sourceBound' key payload table')
    (covers : factorBound' * sourceBound + factorBound * sourceBound' ≤ innerBound)
    (different : table ≠ table') : (tableCollision key innerBound st st').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases digits : st.inner ≠ st'.inner ∨ st.chain ≠ st'.chain
  · exact Or.inl (outerCollision_isSome opens.source.rows opens'.source.rows
      opens.source.innerDigits opens'.source.innerDigits opens.source.chainDigits
      opens'.source.chainDigits digits)
  · refine Or.inr (sourceCollision_isSome opens opens' covers
      (not_not.mp fun differs => digits (Or.inl differs)) fun same => different ?_)
    rw [opens.source.decodes, opens'.source.decodes, same]

end Collision

namespace Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

attribute [local instance low] Classical.propDecidable

/-- The table of the successor source of a weak opening: the table that its source blocks decode
to (Akita (201)). -/
noncomputable def tableOf {S : Shape} {O : OpeningShape S} : Option (Weak O) → Table
  | some ω => S.decode ω.source
  | none => fun _ => 0

namespace Tier

section Suffix

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}
  {E Wire Coord Slot Result Source : Type} [CommRing E] [Module Zq E]
  (T : Tier O fold E Wire Coord Slot Result Source) {total : ℕ}
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type}

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem opens_take {level : ℕ} (above : T.start ≤ level)
    (point : Prefix Wire (Coord → Slot) total) (opening : Option (Weak O)) :
    T.Opens (point.take level) opening ↔ T.Opens point opening := by
  unfold Tier.Opens
  rw [Prefix.wire_take above]

/-- `Coll_ℓ` for two weak openings of the level: the solution of `tableCollision` for their
certified decommitments. -/
noncomputable def collide (opening other : Option (Weak O)) : T.Break :=
  match opening, other with
  | some ω, some ω' =>
      (tableCollision T.key (2 * (2 * T.weight) * fold.sourceBound S) ω.toCertified
        ω'.toCertified).map (Sum.map id Sum.inl)
  | _, _ => none

/-- The invariant of the stages from the level `T` (Akita Proposition 10.44 for a suffix of the
levels). `body` is the checkpoint structure of these stages, and `embed` maps a solution of
Module-SIS for a matrix of the level to a witness of `R_bind` of `body`. `opened`: a first
certificate has an accepting transcript and a weak opening of the claim of the level
(Proposition 10.44 (i)). `lawfulFrom`: the conditions of Definition 10.16 from the first stage of
the level. `related`: `Wit` returns this weak opening. `errorSum`, `loopCount` and `treeSize`:
the sum of the stage errors (206), the number of the retry loops and the tree size of these
stages. -/
structure Suffix (accept : Prefix Wire (Coord → Slot) total → Prop)
    (body : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
      KeyT (Option (Weak O)))
    (embed : T.Break → BreakT) (error : ℝ) (loops tree : ℕ) : Prop where
  opened : ∀ inner, body.holds T.start inner →
    accept (body.point inner) ∧ T.Opens (body.point inner) (body.witness inner)
  embedded : ∀ solution : T.Break, solution.isSome → body.breaks (embed solution)
  lawfulFrom : body.LawfulFrom (transcript total accept) T.start
  protocol : body.protocol = transcript total accept
  related : ∀ root opening, body.relation root opening ↔ T.Opens root opening
  errorSum : body.chargeFrom T.start = error
  loopCount : body.loopsFrom T.start = loops
  treeSize : body.treeFrom T.start = tree

variable {T} {accept : Prefix Wire (Coord → Slot) total → Prop}

/-- The level over later stages that satisfy its interface `Fed` is a suffix. -/
theorem suffix_over {WitT : Type}
    {tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
      KeyT WitT}
    {source : Inner → Source} {clash : Inner → Inner → BreakT} {peak : ℕ} (valid : T.Valid peak)
    (fed : T.Fed accept tail source clash) (stages : tail.protocol.stages = total) :
    T.Suffix accept (T.over accept tail source clash) Sum.inl
      (S.blocks * T.chance + T.costs.sum + tail.chargeFrom T.finish)
      (S.blocks + 1 + tail.loopsFrom T.finish) (2 * (S.blocks + 1) * tail.treeFrom T.finish) where
  opened := T.over_opened
  embedded _ found := found
  lawfulFrom := T.over_lawfulFrom valid fed
  protocol := rfl
  related _ _ := Iff.rfl
  errorSum := T.over_chargeFrom fed.fits stages
  loopCount := T.over_loopsFrom valid.draws fed.fits stages
  treeSize := T.over_treeFrom valid.draws fed.fits stages

/-- The interface `Fed` of a last level: the proof has the source after the fold challenge, and
the verifier accepts when the final claim of the checks is true for this source. No stage is
after the level. -/
theorem fed_finish (sourceOf : Wire → Source) (fits : T.finish ≤ total)
    (passed : ∀ point, accept point → T.Passed (sourceOf (point.wire (T.start + 1))) point) :
    T.Fed accept (Checkpoint.finish (transcript total accept))
      (fun point => sourceOf (point.wire (T.start + 1))) (fun _ _ => ()) where
  fits := fits
  decodes point accepted := (passed point accepted).2.1
  lawful := finish_lawfulFrom _ _
  accepted _ held := held
  passed point held := passed point held
  clash inner other anchor _ _ found foundOther different := by
    obtain ⟨_, rfl⟩ := transcript_cut found
    obtain ⟨_, same⟩ := transcript_cut foundOther
    have same : inner.take (T.start + 1) = other.take (T.start + 1) := same
    refine absurd ?_ different
    show sourceOf (inner.wire (T.start + 1)) = sourceOf (other.wire (T.start + 1))
    rw [← Prefix.wire_take le_rfl inner, same, Prefix.wire_take le_rfl]

/-- The suffix of a last level, whose proof has the source in the clear (the model of a terminal
level): `r·p + Σ_j ε_j` for the stage errors, `r + 1` retry loops and the tree size `2·(r + 1)`,
for `r` blocks. -/
theorem suffix_last {peak : ℕ} (valid : T.Valid peak) (sourceOf : Wire → Source)
    (fits : T.finish ≤ total)
    (passed : ∀ point, accept point → T.Passed (sourceOf (point.wire (T.start + 1))) point) :
    T.Suffix accept
      (T.over accept (Checkpoint.finish (transcript total accept))
        (fun point => sourceOf (point.wire (T.start + 1))) (fun _ _ => ()))
      Sum.inl (S.blocks * T.chance + T.costs.sum) (S.blocks + 1) (2 * (S.blocks + 1)) := by
  have suffix := suffix_over valid (fed_finish sourceOf fits passed) rfl
  rw [finish_chargeFrom, finish_loopsFrom, finish_treeFrom, add_zero, Nat.add_zero,
    Nat.mul_one] at suffix
  exact suffix

end Suffix

section Push

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {E Wire Coord Slot Result' Source' : Type} [Field E] [Algebra Zq E]
  (T : Tier O fold E Wire Coord Slot (Option (List E × E)) Table)
  (N : Tier O' fold' E Wire Coord Slot Result' Source') {total : ℕ}
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type}

omit [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
/-- The scalar of a weak opening of the claim of a level is the multilinear evaluation of the
table of its source blocks at the point of the claim (Akita (202), with
`scalar_contract_eq_mle`). The point has the segments `(r_pack, r_tail, r_pos, r_blk)`, and `B`,
`M`, `k·h_car` and `d_car` are the powers of two of their lengths. -/
theorem opens_evaluation {point : Prefix Wire (Coord → Slot) total} {ω : Weak O'}
    (opens : N.Opens point (some ω)) {block position pack tail : List E}
    (blocks : S'.blocks = 2 ^ block.length) (length : S'.blockLength = 2 ^ position.length)
    (stride : O'.stride = 2 ^ pack.length) (degree : O'.carrierDegree = 2 ^ tail.length)
    (place : (N.claim (point.wire N.start)).point = Point.ofSegments O' block position pack tail) :
    mle (pack ++ (tail ++ (position ++ block))) (lifted E (S'.decode ω.source)) =
      (N.claim (point.wire N.start)).value := by
  obtain ⟨other, same, opens⟩ := opens
  obtain rfl := Option.some.inj same
  have scalar : scalar N.basis (N.claim (point.wire N.start)).point
      (fun i t => contract N.basis (N.claim (point.wire N.start)).point t
        (gadget S'.sourceBase (ω.source i))) = (N.claim (point.wire N.start)).value :=
    opens.value_source
  rw [place, scalar_contract_eq_mle N.basis block position pack tail blocks length stride degree
    ω.source] at scalar
  exact scalar

/-- The link of the verifier from the level `T` to the next level `N` on a transcript
(Akita Figure 4, steps 3 and 5): the transcript has the stages of `T`, each fold challenge
decodes, and the checks of `T` end in a final claim `(r_2, v_W)`. The payload of the claim of `N`
is the commitment `commit` of the message after the fold challenge of `T`. The point of the claim
of `N` is `r_2`, with the segments `(r_pack, r_tail, r_pos, r_blk)` of the lengths of the shape
of `N`, and its value is `v_W`. -/
def Linked (commit : Wire → S'.Payload) (point : Prefix Wire (Coord → Slot) total) : Prop :=
  T.finish ≤ point.rest.length ∧ T.Decodes point ∧
    ∃ block position pack tail v,
      T.game (fun _ => 0) point = .pure (some (pack ++ (tail ++ (position ++ block)), v)) ∧
      S'.blocks = 2 ^ block.length ∧ S'.blockLength = 2 ^ position.length ∧
      O'.stride = 2 ^ pack.length ∧ O'.carrierDegree = 2 ^ tail.length ∧
      (N.claim (point.wire N.start)).payload = commit (point.wire (T.start + 1)) ∧
      (N.claim (point.wire N.start)).point = Point.ofSegments O' block position pack tail ∧
      (N.claim (point.wire N.start)).value = v

variable {T N} {accept : Prefix Wire (Coord → Slot) total → Prop}
  {tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT (Option (Weak O'))}
  {embed : N.Break → BreakT} {error : ℝ} {loops tree : ℕ}

/-- The interface `Fed` of a level whose flat witness the next level commits to (Akita
Proposition 10.44, "Checkpoints" and "Segment soundness"). The successor source of a first
certificate of the next level is the table of its weak opening. Its final claim is true by the
value equation of the weak opening, which is the multilinear evaluation of this table
(`scalar_contract_eq_mle`). Two sources after one stage input of the checkpoint are two certified
decommitments of one payload, and `tableCollision` gives a solution of Module-SIS for a matrix of
the next level. `uniform`: the verifier of the checks does not read the source. `passes`: the
final claim `(r_2, v_W)` is true for a table `w` when `v_W = w̃(r_2)`. `linked`: the verifier
accepts only with the link `Linked`. -/
theorem fed_push (next : N.start = T.finish) (commit : Wire → S'.Payload)
    (fits : T.finish ≤ total)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) → T.passes table (some (r2, v)))
    (linked : ∀ point, accept point → T.Linked N commit point)
    (later : N.Suffix accept tail embed error loops tree) :
    T.Fed accept tail (fun inner => tableOf (tail.witness inner))
      (fun inner other => embed (N.collide (tail.witness inner) (tail.witness other))) where
  fits := fits
  decodes point accepted := (linked point accepted).2.1
  lawful := by
    rw [← next]
    exact later.lawfulFrom
  accepted inner held := (later.opened inner (by rw [next]; exact held)).1
  passed inner held := by
    obtain ⟨accepted, opens⟩ := later.opened inner (by rw [next]; exact held)
    obtain ⟨ω, found, _⟩ := id opens
    obtain ⟨long, decodes, block, position, pack, tail', v, ended, blocks, length, stride,
      degree, _, place, value⟩ := linked _ accepted
    rw [found] at opens
    have evaluation := N.opens_evaluation opens blocks length stride degree place
    rw [value] at evaluation
    refine ⟨long, decodes, _, ended_of_bare T.read (uniform _ _ _ _) ended, passes _ _ _ ?_⟩
    rw [found]
    exact evaluation.symm
  clash inner other anchor held heldOther found foundOther different := by
    obtain ⟨accepted, ω, foundω, opens⟩ := later.opened inner (by rw [next]; exact held)
    obtain ⟨acceptedOther, ω', foundω', opens'⟩ :=
      later.opened other (by rw [next]; exact heldOther)
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, payload, _, _⟩ := linked _ accepted
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, payload', _, _⟩ := linked _ acceptedOther
    obtain ⟨_, rfl⟩ := transcript_cut found
    obtain ⟨_, same⟩ := transcript_cut foundOther
    have sameWire : (tail.point other).wire (T.start + 1) =
        (tail.point inner).wire (T.start + 1) := by
      rw [← Prefix.wire_take le_rfl (tail.point other), ← same, Prefix.wire_take le_rfl]
    have left : ω.toCertified.Opens (2 * N.weight) (fold'.sourceBound S') N.key
        (N.claim ((tail.point inner).wire N.start)).payload (S'.decode ω.source) :=
      opens.certified
    have right : ω'.toCertified.Opens (2 * N.weight) (fold'.sourceBound S') N.key
        (N.claim ((tail.point other).wire N.start)).payload (S'.decode ω'.source) :=
      opens'.certified
    rw [payload] at left
    rw [payload', sameWire] at right
    have differ : S'.decode ω.source ≠ S'.decode ω'.source := fun same => different (by
      show tableOf (tail.witness inner) = tableOf (tail.witness other)
      rw [foundω, foundω']
      exact same)
    refine later.embedded _ ?_
    rw [foundω, foundω']
    show ((tableCollision N.key (2 * (2 * N.weight) * fold'.sourceBound S') ω.toCertified
      ω'.toCertified).map (Sum.map id Sum.inl)).isSome
    rw [Option.isSome_map]
    exact tableCollision_isSome left right (le_of_eq (by ring)) differ

/-- The suffix of a level over the suffix of the next level, which commits to its flat witness
(Akita Proposition 10.44 for one more level): the stage errors add `r·p + Σ_j ε_j`, the retry
loops add `r + 1`, and the tree size is multiplied by `2·(r + 1)`, for `r` blocks. A witness of
`R_bind` is a solution of Module-SIS for a matrix of this level or a witness of `R_bind` of the
next level. -/
theorem suffix_push {peak : ℕ} (valid : T.Valid peak) (next : N.start = T.finish)
    (commit : Wire → S'.Payload) (fits : T.finish ≤ total)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) → T.passes table (some (r2, v)))
    (linked : ∀ point, accept point → T.Linked N commit point)
    (later : N.Suffix accept tail embed error loops tree) :
    T.Suffix accept
      (T.over accept tail (fun inner => tableOf (tail.witness inner))
        (fun inner other => embed (N.collide (tail.witness inner) (tail.witness other))))
      Sum.inl (S.blocks * T.chance + T.costs.sum + error) (S.blocks + 1 + loops)
      (2 * (S.blocks + 1) * tree) := by
  have stages : tail.protocol.stages = total := by rw [later.protocol]; rfl
  have suffix := suffix_over valid (fed_push next commit fits uniform passes linked later) stages
  rw [← next, later.errorSum, later.loopCount, later.treeSize] at suffix
  exact suffix

end Push

section Extraction

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}
  {E Wire Coord Slot Result Source : Type} [CommRing E] [Module Zq E]
  {T : Tier O fold E Wire Coord Slot Result Source} {total : ℕ}
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type} {accept : Prefix Wire (Coord → Slot) total → Prop}
  {body : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT (Option (Weak O))}
  {embed : T.Break → BreakT} {error : ℝ} {loops tree : ℕ}

/-- A suffix from the stage 0 satisfies the conditions of Akita Definition 10.16 (Akita
Proposition 10.44). -/
theorem Suffix.lawful (suffix : T.Suffix accept body embed error loops tree)
    (root : T.start = 0) : body.Lawful := by
  have lawful : body.LawfulFrom body.protocol 0 := by
    rw [suffix.protocol, ← root]
    exact suffix.lawfulFrom
  refine lawful.lawful fun certificate statement held found => ?_
  obtain ⟨_, opens⟩ := suffix.opened certificate (by rw [root]; exact held)
  rw [suffix.protocol] at found
  obtain ⟨_, rfl⟩ := transcript_cut found
  exact (suffix.related _ _).mpr ((T.opens_take (by omega) _ _).mpr opens)

/-- An output of the extractor for a suffix from the stage 0 is correct: the verifier accepts
the transcript of the prover, and the output is a root weak opening of the claim of the first
level (Akita Definition 10.22, with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`) or a witness of `R_bind`,
which is a solution of Module-SIS for a matrix of one level. -/
theorem Suffix.extract_sound (suffix : T.Suffix accept body embed error loops tree)
    (root : T.start = 0)
    (prover : OracleComp (Prefix Wire (Coord → Slot) total) (Coord → Slot) (ℕ → Wire))
    (budget : ℕ) (table : Prefix Wire (Coord → Slot) total → Coord → Slot)
    (tape : Tape (Prefix Wire (Coord → Slot) total) Coord Slot budget body.protocol.stages)
    (out : Inner ⊕ BreakT)
    (result : Checkpoint.extract body prover budget table tape = some out) :
    body.protocol.accept (body.protocol.final prover table) ∧
      Sum.elim
        (fun certificate => T.Opens (body.protocol.input prover table 0)
          (body.witness certificate))
        body.breaks out := by
  obtain ⟨accepted, sound⟩ := Checkpoint.extract_sound body prover (suffix.lawful root) budget
    table tape out result
  refine ⟨accepted, ?_⟩
  rcases out with certificate | broken
  · exact (suffix.related _ _).mp sound
  · exact sound

variable [Finite Wire]

/-- Akita Theorem 10.17, in the form (161), for a suffix from the stage 0 (Corollary 10.49 with
(206) for the opening suffix). For each prover with at most `Q` queries, the probability that the
verifier accepts and the extractor returns no output is at most
`(Q + 1)·error + (Q + 1)·loops/budget`: `error` is the sum of the stage errors of the levels,
`loops` is the number of the retry loops, and `budget` is the limit of the attempts of a retry
loop. The probability is over the table of the oracle and the tape of the extractor. -/
theorem Suffix.extraction_le (suffix : T.Suffix accept body embed error loops tree)
    (root : T.start = 0)
    (prover : OracleComp (Prefix Wire (Coord → Slot) total) (Coord → Slot) (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → Slot) total) Coord Slot budget
          body.protocol.stages,
        (if body.protocol.accept (body.protocol.final prover table) ∧
          Checkpoint.extract body prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * error + (queries + 1) * loops / budget := by
  have bound := checkpoint_extraction_le body prover (suffix.lawful root) bounded positive
  have charge := suffix.errorSum
  have count := suffix.loopCount
  rw [root] at charge count
  rw [body.checkError_eq_chargeFrom, charge, count] at bound
  exact bound

/-- The expected number of runs of the prover in the extraction of a suffix from the stage 0 is
at most `K̂ + (Q + 1)·(K̂ − 1)` for the tree size `K̂ = tree` (Akita Theorem 10.17, the cost). -/
theorem Suffix.runs_le (suffix : T.Suffix accept body embed error loops tree)
    (root : T.start = 0)
    (prover : OracleComp (Prefix Wire (Coord → Slot) total) (Coord → Slot) (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → Slot) total) Coord Slot budget
          body.protocol.stages,
        (extractRuns body prover budget table tape : ℝ) ≤
      tree + (queries + 1) * (tree - 1) := by
  have bound := extract_runs_le body prover bounded budget
  have size := suffix.treeSize
  rw [root] at size
  unfold Structure.treeSize at bound
  rw [size] at bound
  exact bound

end Extraction

end Tier

end Extract

end NightstreamFPrime.Spec.Compression.Akita
