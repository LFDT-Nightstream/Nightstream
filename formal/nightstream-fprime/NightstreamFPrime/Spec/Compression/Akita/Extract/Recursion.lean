import NightstreamFPrime.Spec.Compression.Akita.Prover
import NightstreamFPrime.Spec.Compression.Akita.Extract.TierBound
import NightstreamFPrime.Spec.Compression.Akita.Extract.Uniform

/-! The recursive opening suffix of Akita: the composition of its levels (ePrint 2026/1983,
revision of 2026-10-15, Proposition 10.44 with its proof, and Corollary 10.49 with (206)).

- `tableCollision` is the reduction of Akita Lemma 10.23 and Corollary 10.25 for two certified
  decommitments of one payload to different tables: `Coll_ℓ` of the checkpoint of a level whose
  next level commits to its flat witness.
- `Tier.Suffix` is the invariant of the stages from one stage `start`: the conditions of
  Definition 10.16 from this stage, the relation `Opens` of each first certificate
  (Proposition 10.44 (i)), and the sums of the stage errors, of the retry loops and the tree size.
  The stages can be those of a level `Tier` or of the terminal.
- `Tier.suffix_last` is the suffix of a last level, whose proof has the source in the clear.
- `Tier.suffix_before` is the suffix of a level over a suffix of the later stages: `source` maps a
  first certificate of the later stages to the successor source of the level, `passed` is Akita
  Lemma 10.42 (ii), and `clashes` is `Coll_ℓ`.
- `Tier.suffix_committed` is `Tier.suffix_before` for a next level that commits to the flat
  witness: the source is the table of its weak opening, and `Coll_ℓ` is `tableCollision`.
- `Tier.Linked` is the link that the verifier makes from a level to a next level with coefficient
  packing: the payload of the next claim is the commitment that the proof has before the checks,
  and the point and the value of the next claim are the final claim `(r_2, v_W)` of the checks.
  `Tier.suffix_push` is `Tier.suffix_committed` for such a next level.
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

section Invariant

variable {Wire Coord Slot : Type} {total : ℕ} [Fintype Slot] [Nonempty Slot] [Fintype Coord]
  [DecidableEq Coord] {Inner BreakT CandT KeyT Wit : Type}

/-- The invariant of the stages from the stage `start` (Akita Proposition 10.44 for a suffix of
the levels). `body` is the checkpoint structure of these stages, and `Opens` is the relation of
their first certificates. `opened`: a first certificate has an accepting transcript and a witness
of `Opens` (Proposition 10.44 (i)). `rooted`: `Opens` reads the transcript before the stage
`start` only. `lawfulFrom`: the conditions of Definition 10.16 from the stage `start`. `related`:
`Wit` returns this witness. `errorSum`, `loopCount` and `treeSize`: the sum of the stage errors
(206), the number of the retry loops and the tree size of these stages. -/
structure Suffix (accept : Prefix Wire (Coord → Slot) total → Prop) (start : ℕ)
    (Opens : Prefix Wire (Coord → Slot) total → Wit → Prop)
    (body : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
      KeyT Wit)
    (error : ℝ) (loops tree : ℕ) : Prop where
  opened : ∀ inner, body.holds start inner →
    accept (body.point inner) ∧ Opens (body.point inner) (body.witness inner)
  rooted : ∀ point witness, Opens point witness → Opens (point.take start) witness
  lawfulFrom : body.LawfulFrom (transcript total accept) start
  protocol : body.protocol = transcript total accept
  related : ∀ root witness, body.relation root witness ↔ Opens root witness
  errorSum : body.chargeFrom start = error
  loopCount : body.loopsFrom start = loops
  treeSize : body.treeFrom start = tree

end Invariant

section Level

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}
  {Ring Stmt Brk Wire Coord Slot Result Source : Type} [Zero Ring]
  (T : Tier O fold Ring Stmt Brk Wire Coord Slot Result Source) {total : ℕ}
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type}

omit [Zero Ring] [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
theorem opens_take {level : ℕ} (above : T.start ≤ level)
    (point : Prefix Wire (Coord → Slot) total) (opening : Option (Weak O)) :
    T.Opens (point.take level) opening ↔ T.Opens point opening := by
  unfold Tier.Opens
  rw [Prefix.wire_take above]

variable {T} {accept : Prefix Wire (Coord → Slot) total → Prop}

/-- The level over later stages that satisfy its interface `Fed` is a suffix. -/
theorem suffix_over {WitT : Type}
    {tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
      KeyT WitT}
    {source : Inner → Source} {clash : Inner → Inner → BreakT} (valid : T.Valid)
    (fed : T.Fed accept tail source clash) (stages : tail.protocol.stages = total) :
    Suffix accept T.start T.Opens (T.over accept tail source clash)
      (S.blocks * T.chance + T.costs.sum + tail.chargeFrom T.finish)
      (S.blocks + 1 + tail.loopsFrom T.finish) (2 * (S.blocks + 1) * tail.treeFrom T.finish) where
  opened := T.over_opened
  rooted point opening opens := (T.opens_take le_rfl point opening).mpr opens
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

/-- The suffix of a last level, whose proof has the source in the clear: `r·p + Σ_j ε_j` for the
stage errors, `r + 1` retry loops and the tree size `2·(r + 1)`, for `r` blocks. -/
theorem suffix_last (valid : T.Valid) (sourceOf : Wire → Source) (fits : T.finish ≤ total)
    (passed : ∀ point, accept point → T.Passed (sourceOf (point.wire (T.start + 1))) point) :
    Suffix accept T.start T.Opens
      (T.over accept (Checkpoint.finish (transcript total accept))
        (fun point => sourceOf (point.wire (T.start + 1))) (fun _ _ => ()))
      (S.blocks * T.chance + T.costs.sum) (S.blocks + 1) (2 * (S.blocks + 1)) := by
  have suffix := suffix_over valid (fed_finish sourceOf fits passed) rfl
  rw [finish_chargeFrom, finish_loopsFrom, finish_treeFrom, add_zero, Nat.add_zero,
    Nat.mul_one] at suffix
  exact suffix

variable {Wit' : Type} {Opens' : Prefix Wire (Coord → Slot) total → Wit' → Prop}
  {tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT Wit'}
  {error : ℝ} {loops tree : ℕ}

/-- The interface `Fed` of a level over a suffix of the later stages (Akita Proposition 10.44,
"Checkpoints" and "Segment soundness"). `source` maps the witness of a first certificate of the
later stages to the successor source of the level. `decodes`: the verifier rejects an answer of
the stage `start` without a fold challenge for each block. `passed`: the final claim of the checks
of the level is true for the successor source (Akita Lemma 10.42 (ii)). `clashes`: two witnesses
with different sources, for two accepting transcripts with one stage input of the checkpoint,
give a witness of `R_bind` (Definition 10.16 (vi)). -/
theorem fed_before (later : Suffix accept T.finish Opens' tail error loops tree)
    (fits : T.finish ≤ total) (source : Wit' → Source) (clash : Wit' → Wit' → BreakT)
    (decodes : ∀ point, accept point → T.Decodes point)
    (passed : ∀ point witness, accept point → Opens' point witness →
      T.Passed (source witness) point)
    (clashes : ∀ point other witness witness', accept point → accept other →
      Opens' point witness → Opens' other witness' →
      point.take (T.start + 1) = other.take (T.start + 1) → source witness ≠ source witness' →
      tail.breaks (clash witness witness')) :
    T.Fed accept tail (fun inner => source (tail.witness inner))
      (fun inner other => clash (tail.witness inner) (tail.witness other)) where
  fits := fits
  decodes := decodes
  lawful := later.lawfulFrom
  accepted inner held := (later.opened inner held).1
  passed inner held := passed _ _ (later.opened inner held).1 (later.opened inner held).2
  clash inner other anchor held heldOther found foundOther different := by
    obtain ⟨_, rfl⟩ := transcript_cut found
    obtain ⟨_, same⟩ := transcript_cut foundOther
    exact clashes _ _ _ _ (later.opened inner held).1 (later.opened other heldOther).1
      (later.opened inner held).2 (later.opened other heldOther).2 same different

/-- The suffix of a level over a suffix of the later stages (Akita Proposition 10.44 for one more
level): the stage errors add `r·p + Σ_j ε_j`, the retry loops add `r + 1`, and the tree size is
multiplied by `2·(r + 1)`, for `r` blocks. The hypotheses are those of `fed_before`. -/
theorem suffix_before (valid : T.Valid)
    (later : Suffix accept T.finish Opens' tail error loops tree) (fits : T.finish ≤ total)
    (source : Wit' → Source) (clash : Wit' → Wit' → BreakT)
    (decodes : ∀ point, accept point → T.Decodes point)
    (passed : ∀ point witness, accept point → Opens' point witness →
      T.Passed (source witness) point)
    (clashes : ∀ point other witness witness', accept point → accept other →
      Opens' point witness → Opens' other witness' →
      point.take (T.start + 1) = other.take (T.start + 1) → source witness ≠ source witness' →
      tail.breaks (clash witness witness')) :
    Suffix accept T.start T.Opens
      (T.over accept tail (fun inner => source (tail.witness inner))
        (fun inner other => clash (tail.witness inner) (tail.witness other)))
      (S.blocks * T.chance + T.costs.sum + error) (S.blocks + 1 + loops)
      (2 * (S.blocks + 1) * tree) := by
  have stages : tail.protocol.stages = total := by rw [later.protocol]; rfl
  have suffix := suffix_over valid
    (fed_before later fits source clash decodes passed clashes) stages
  rw [later.errorSum, later.loopCount, later.treeSize] at suffix
  exact suffix

end Level

section Committed

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {Ring Stmt Brk Ring' Stmt' Brk' Wire Coord Slot Result Result' Source' : Type}
  [Zero Ring] [Zero Ring']
  {T : Tier O fold Ring Stmt Brk Wire Coord Slot Result Table}
  {N : Tier O' fold' Ring' Stmt' Brk' Wire Coord Slot Result' Source'}
  {total : ℕ} [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type} {accept : Prefix Wire (Coord → Slot) total → Prop}
  {tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT (Option (Weak O'))}
  {error : ℝ} {loops tree : ℕ}

/-- `Coll_ℓ` for two weak openings of a level with the matrices `key`: the solution of
`tableCollision` for their certified decommitments. -/
noncomputable def collide {S : Shape} {O : OpeningShape S} (key : Key S) (openKey : OpeningKey O)
    (innerBound : ℕ) (opening other : Option (Weak O)) : LevelBreak key openKey innerBound :=
  match opening, other with
  | some ω, some ω' =>
      (tableCollision key innerBound ω.toCertified ω'.toCertified).map (Sum.map id Sum.inl)
  | _, _ => none

omit [Zero Ring'] in
/-- The suffix of a level over the suffix of a next level `N` that commits to its flat witness
(Akita Proposition 10.44 for one more level, "Checkpoints"). The successor source of a first
certificate of the next level is the table of its weak opening. `certified`: this weak opening
has a certified decommitment of the commitment `commit` that the proof has before the checks of
the level (Akita Definition 10.22 (i) and (ii)). Two sources after one stage input of the
checkpoint are two certified decommitments of one payload, and `tableCollision` gives a solution
of Module-SIS for a matrix of the next level (Akita Corollary 10.25). `passed`: the final claim
of the checks of the level is true for the table (Akita Lemma 10.42 (ii)). `embedded`: `embed`
maps a solution of Module-SIS for a matrix of `N` to a witness of `R_bind` of the later stages.
The stage errors add `r·p + Σ_j ε_j`, the retry loops add `r + 1`, and the tree size is multiplied
by `2·(r + 1)`, for `r` blocks. -/
theorem suffix_committed (valid : T.Valid) (next : N.start = T.finish)
    (commit : Wire → S'.Payload) (fits : T.finish ≤ total)
    (decodes : ∀ point, accept point → T.Decodes point) {factorBound sourceBound innerBound : ℕ}
    (key' : Key S') (openKey' : OpeningKey O')
    (covers : factorBound * sourceBound + factorBound * sourceBound ≤ innerBound)
    (certified : ∀ point ω, accept point → N.Opens point (some ω) →
      ω.toCertified.Opens factorBound sourceBound key' (commit (point.wire (T.start + 1)))
        (S'.decode ω.source))
    (passed : ∀ point ω, accept point → N.Opens point (some ω) →
      T.Passed (S'.decode ω.source) point)
    (embed : LevelBreak key' openKey' innerBound → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Suffix accept N.start N.Opens tail error loops tree) :
    Suffix accept T.start T.Opens
      (T.over accept tail (fun inner => tableOf (tail.witness inner))
        (fun inner other =>
          embed (collide key' openKey' innerBound (tail.witness inner) (tail.witness other))))
      (S.blocks * T.chance + T.costs.sum + error) (S.blocks + 1 + loops)
      (2 * (S.blocks + 1) * tree) := by
  refine suffix_before valid (next ▸ later) fits tableOf
    (fun opening other => embed (collide key' openKey' innerBound opening other)) decodes ?_ ?_
  · rintro point _ accepted ⟨ω, rfl, opens⟩
    exact passed point ω accepted ⟨ω, rfl, opens⟩
  · rintro point other _ _ accepted acceptedOther ⟨ω, rfl, opens⟩ ⟨ω', rfl, opens'⟩ same different
    have sameWire : other.wire (T.start + 1) = point.wire (T.start + 1) := by
      rw [← Prefix.wire_take le_rfl other, ← same, Prefix.wire_take le_rfl]
    have left := certified point ω accepted ⟨ω, rfl, opens⟩
    have right := certified other ω' acceptedOther ⟨ω', rfl, opens'⟩
    rw [sameWire] at right
    refine embedded _ ?_
    show ((tableCollision key' innerBound ω.toCertified ω'.toCertified).map
      (Sum.map id Sum.inl)).isSome
    rw [Option.isSome_map]
    exact tableCollision_isSome left right covers different

end Committed

section Push

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {E Ring Stmt Brk Wire Coord Slot Result' Source' : Type} [Zero Ring]
  [Field E] [Algebra Zq E] {β' : Module.Basis (Fin O'.planes) Zq E} {key' : Key S'}
  {openKey' : OpeningKey O'} {weight' : ℕ}
  (T : Tier O fold Ring Stmt Brk Wire Coord Slot (Option (List E × E)) Table)
  (N : Tier O' fold' (Rq O'.carrierDegree) (Claim O' E)
    (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) Wire Coord Slot Result'
    Source')
  {total : ℕ} [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type}

omit [Zero Ring] [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord] in
/-- The scalar of a weak opening with coefficient packing is the multilinear evaluation of the
table of its source blocks at the point of the statement (Akita (202), with
`scalar_contract_eq_mle`). The point has the segments `(r_pack, r_tail, r_pos, r_blk)`, and `B`,
`M`, `k·h_car` and `d_car` are the powers of two of their lengths. -/
theorem opens_evaluation {st : Statement O' E} {factorBound sourceBound : ℕ} {table : Table}
    {ω : Weak O'} (opens : ω.Opens β' factorBound sourceBound st table)
    {block position pack tail : List E}
    (blocks : S'.blocks = 2 ^ block.length) (length : S'.blockLength = 2 ^ position.length)
    (stride : O'.stride = 2 ^ pack.length) (degree : O'.carrierDegree = 2 ^ tail.length)
    (place : st.point = Point.ofSegments O' block position pack tail) :
    mle (pack ++ (tail ++ (position ++ block))) (lifted E (S'.decode ω.source)) = st.value := by
  have scalar := opens.value_source
  rw [place, scalar_contract_eq_mle β' block position pack tail blocks length stride degree
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
  {error : ℝ} {loops tree : ℕ}

/-- `suffix_committed` for a next level with coefficient packing (Akita Proposition 10.44 for one
more level). The final claim of the level is true by the value equation of the weak opening of
the next level, which is the multilinear evaluation of its table (`scalar_contract_eq_mle`).
`packed`: the method of `N` is `packing`. `uniform`: the verifier of the checks
does not read the source. `passes`: the final claim `(r_2, v_W)` is true for a table `w` when
`v_W = w̃(r_2)`. `linked`: the verifier accepts only with the link `Linked`. `embedded`: `embed`
maps a solution of Module-SIS for a matrix of `N` to a witness of `R_bind` of the later stages.
The stage errors add `r·p + Σ_j ε_j`, the retry loops add `r + 1`, and the tree size is multiplied
by `2·(r + 1)`, for `r` blocks. -/
theorem suffix_push (valid : T.Valid) (packed : N.method = packing β' key' openKey' weight')
    (next : N.start = T.finish) (commit : Wire → S'.Payload) (fits : T.finish ≤ total)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) r2 v, v = mle r2 (lifted E table) → T.passes table (some (r2, v)))
    (linked : ∀ point, accept point → T.Linked N commit point)
    (embed : LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Suffix accept N.start N.Opens tail error loops tree) :
    Suffix accept T.start T.Opens
      (T.over accept tail (fun inner => tableOf (tail.witness inner))
        (fun inner other => embed (collide key' openKey'
          (2 * (2 * weight') * fold'.sourceBound S') (tail.witness inner) (tail.witness other))))
      (S.blocks * T.chance + T.costs.sum + error) (S.blocks + 1 + loops)
      (2 * (S.blocks + 1) * tree) := by
  refine suffix_committed (factorBound := 2 * weight') (sourceBound := fold'.sourceBound S')
    valid next commit fits (fun point accepted => (linked point accepted).2.1) key' openKey'
    (le_of_eq (by ring)) ?_ ?_ embed embedded later
  · rintro point ω accepted ⟨_, same, opens⟩
    obtain rfl := Option.some.inj same
    rw [packed] at opens
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, payload, _, _⟩ := linked _ accepted
    have left : ω.toCertified.Opens (2 * weight') (fold'.sourceBound S') key'
        (N.claim (point.wire N.start)).payload (S'.decode ω.source) :=
      (show ω.Opens β' _ _ ((N.claim (point.wire N.start)).statement key' openKey') _ from
        opens).certified
    rw [payload] at left
    exact left
  · rintro point ω accepted ⟨_, same, opens⟩
    obtain rfl := Option.some.inj same
    rw [packed] at opens
    obtain ⟨long, decodes, block, position, pack, tail', v, ended, blocks, length, stride,
      degree, _, place, value⟩ := linked _ accepted
    have evaluation := opens_evaluation
      (st := (N.claim (point.wire N.start)).statement key' openKey') opens blocks length stride
      degree place
    exact ⟨long, decodes, _, ended_of_bare T.read (uniform _ _ _ _) ended,
      passes _ _ _ (evaluation.trans value).symm⟩

end Push

section Extraction

variable {Wire Coord Slot : Type} {total : ℕ} [Fintype Slot] [Nonempty Slot] [Fintype Coord]
  [DecidableEq Coord] {Inner BreakT CandT KeyT Wit : Type}
  {accept : Prefix Wire (Coord → Slot) total → Prop} {start : ℕ}
  {Opens : Prefix Wire (Coord → Slot) total → Wit → Prop}
  {body : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT Wit}
  {error : ℝ} {loops tree : ℕ}

/-- A suffix from the stage 0 satisfies the conditions of Akita Definition 10.16 (Akita
Proposition 10.44). -/
theorem Suffix.lawful (suffix : Suffix accept start Opens body error loops tree)
    (root : start = 0) : body.Lawful := by
  subst root
  have lawful : body.LawfulFrom body.protocol 0 := by
    rw [suffix.protocol]
    exact suffix.lawfulFrom
  refine lawful.lawful fun certificate statement held found => ?_
  obtain ⟨_, opens⟩ := suffix.opened certificate held
  rw [suffix.protocol] at found
  obtain ⟨_, rfl⟩ := transcript_cut found
  exact (suffix.related _ _).mpr (suffix.rooted _ _ opens)

/-- An output of the extractor for a suffix from the stage 0 is correct: the verifier accepts
the transcript of the prover, and the output is a witness of `Opens` for the stage input of the
stage 0, which for a level is a root weak opening of its claim (Akita Definition 10.22), or a
witness of `R_bind`, which is a solution of Module-SIS for a matrix of one level. -/
theorem Suffix.extract_sound (suffix : Suffix accept start Opens body error loops tree)
    (root : start = 0)
    (prover : OracleComp (Prefix Wire (Coord → Slot) total) (Coord → Slot) (ℕ → Wire))
    (budget : ℕ) (table : Prefix Wire (Coord → Slot) total → Coord → Slot)
    (tape : Tape (Prefix Wire (Coord → Slot) total) Coord Slot budget body.protocol.stages)
    (out : Inner ⊕ BreakT)
    (result : Checkpoint.extract body prover budget table tape = some out) :
    body.protocol.accept (body.protocol.final prover table) ∧
      Sum.elim
        (fun certificate => Opens (body.protocol.input prover table 0) (body.witness certificate))
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
theorem Suffix.extraction_le (suffix : Suffix accept start Opens body error loops tree)
    (root : start = 0)
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
theorem Suffix.runs_le (suffix : Suffix accept start Opens body error loops tree)
    (root : start = 0)
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
