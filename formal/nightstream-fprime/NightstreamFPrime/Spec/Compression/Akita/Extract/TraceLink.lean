import NightstreamFPrime.Spec.Compression.Akita.Tensor
import NightstreamFPrime.Spec.Compression.Akita.TracePack
import NightstreamFPrime.Spec.Compression.Akita.Extract.Chain
import NightstreamFPrime.Spec.Compression.Akita.Extract.TerminalSuffix
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLevel

/-! The link from a level of the opening suffix of Akita to a receiver with the evaluation-trace
opening method, for `k = 2` (ePrint 2026/1983, revision of 2026-10-15, Section 5.2 with
Theorem 5.1, Section 5.3 with (65) and (66), Section 7.1 with (95) to (101), Section 8.2 with
(130), Figure 4, step 0, and Lemma 10.42 (ii)).

The receiver commits to the ring elements `ψ(f̄)` of the packed polynomial `f̄` of the flat
witness `w` of the sender. The final claim `(r_2, v_W)` of the checks of the sender is a claim on
`w̃`, and the tensor reduction changes it to the claim `ṽ = θ·f̄(ρ)` that the scalar row (101) of
the receiver has.

- `tensorCosts` has the bounds of the draws of `tensorCheck`: `κ/|E|` for `η` and `2/|E|` for
  each round, with the sum `(κ + 2·m_max)/|E|` of Akita Theorem 5.1 for one claim and `κ = 1`.
- `Tier.tensored` is a level whose checks continue with the tensor reduction of their final
  claim. A final claim `(ρ, θ, ṽ)` is true for a table `w` when `ṽ = θ·f̄_w(ρ)`.
- `TraceClaim.ofPoint` and `TerminalStatement.ofPoint` are the claim of a trace level and the
  statement of the terminal for a point `(ρ_pack, ρ_pos, ρ_blk)`, a factor `θ` and a value `ṽ`.
  `traceOpens_evaluation` states that a weak opening of such a claim has
  `θ·f̄(ρ) = ṽ` for the packed polynomial `f̄` of its table (`traceScalar_eq_mle`).
- `Tier.TraceLinked` and `Terminal.Linked` are the links that the verifier makes from a level to
  a next trace level and to the terminal. `Tier.suffix_push_trace` and `Terminal.suffix_linked`
  are the suffix of the level over the receiver. The successor source of the level is
  `flatten (unpackTable …)` of the table of the receiver, and the condition of Akita
  Lemma 10.42 (ii) is a theorem. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Reduction

/-- The bounds of the draws of `tensorCheck` with `ν` tail variables: `η`, and the rounds of the
sum-check of degree 2 (Akita Theorem 5.1 for one claim and `κ = 1`). -/
noncomputable def tensorCosts (ν : ℕ) : List ℝ :=
  (((1 : ℕ) : ℝ) / Fintype.card Quadratic) ::
    List.replicate ν (((2 : ℕ) : ℝ) / Fintype.card Quadratic)

/-- The sum of the bounds of the draws is `(κ + 2·m_max)/|E|` of Akita Theorem 5.1, item 1, for
one claim and `κ = 1`. -/
theorem tensorCosts_sum (ν : ℕ) :
    (tensorCosts ν).sum = ((1 + 2 * ν : ℕ) : ℝ) / Fintype.card Quadratic := by
  simp only [tensorCosts, List.sum_cons, List.sum_replicate, nsmul_eq_mul]
  push_cast
  ring

theorem tensorCosts_nonnegative (ν : ℕ) : ∀ cost ∈ tensorCosts ν, 0 ≤ cost := by
  intro cost member
  simp only [tensorCosts, List.mem_cons, List.mem_replicate] at member
  rcases member with rfl | ⟨_, rfl⟩ <;> positivity

/-- The verifier of the tensor reduction is the same for each table. -/
theorem tensorCheck_bare (ν : ℕ) (w other : Table) (r : List Quadratic) (v : Quadratic) :
    bare (tensorCheck ν w r v) = bare (tensorCheck ν other r v) := by
  rcases r with _ | ⟨head, tail⟩
  · rfl
  change bare (ite (tail.length = ν) _ _) = bare (ite (tail.length = ν) _ _)
  by_cases length : tail.length = ν
  · rw [ite_eq_left length, ite_eq_left length]
    refine bare_recv_congr fun columns => ?_
    by_cases checked : v = (1 - head) * columns.1 + head * columns.2
    · rw [ite_eq_left checked, ite_eq_left checked]
      exact bare_draw_congr fun η => bare_bind_congr (sumcheck_bare 2 _ _ ν [] _) fun _ => rfl
    · rw [ite_eq_right checked, ite_eq_right checked]
  · rw [ite_eq_right length, ite_eq_right length]

/-- Each draw of `tensorCheck` has its bound of `tensorCosts`, when an answer of the oracle has a
slot. -/
theorem tensorCheck_staged {Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
    (ν : ℕ) (w : Table) (r : List Quadratic) (v : Quadratic) :
    Staged (Coord → Quadratic) (tensorCheck ν w r v) (tensorCosts ν) := by
  have field : ∃ decode : (Coord → Quadratic) → Quadratic, Fair decode := exists_fair_slot
  rcases r with _ | ⟨head, tail⟩
  · trivial
  change StagedTo _ (ite (tail.length = ν) _ _) _ _
  split
  swap
  · trivial
  intro columns
  dsimp only
  split
  swap
  · trivial
  refine stagedTo_draw field (density_le _ 1
    (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun η => ?_
  have rounds := sumcheck_staged (Reply := Coord → Quadratic) field 2 ν _
    (combRound_truth ν 2 (tensorTables w tail η) productComb productComb_degree).degree []
    (fun result rest => StagedTo (Coord → Quadratic) (match result with
      | none => Game.pure none
      | some (ρ, final) => Game.pure (some (ρ, mle ρ (tensorWeight tail η), final))) rest
        fun _ _ => True)
    (fun _ => trivial) (fun _ => trivial) ν []
    ((1 - η) * tensorRow columns false + η * tensorRow columns true) (by simp)
  rw [List.append_nil] at rounds
  exact (stagedTo_bind _ _ _ _).mpr rounds

end Reduction

section Tensored

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {Ring Stmt Brk Wire Coord : Type}

/-- The level `T` with the tensor reduction of the final claim `(r_2, v_W)` of its checks, for a
next level with evaluation trace (Akita Figure 4, step 0). The draws of the tensor reduction are
after the draws of the checks. A final claim `(ρ, θ, ṽ)` is true for a table `w` when
`ṽ = θ·f̄_w(ρ)`, for the packed polynomial `f̄_w` of `w`. `ν` is the number of the tail variables:
the point `r_2` has `ν + 1` coordinates. -/
noncomputable def Tier.tensored
    (T : Tier O fold Ring Stmt Brk Wire Coord Quadratic (Option (List Quadratic × Quadratic))
      Table) (ν : ℕ) :
    Tier O fold Ring Stmt Brk Wire Coord Quadratic
      (Option (List Quadratic × Quadratic × Quadratic)) Table where
  method := T.method
  claim := T.claim
  block := T.block
  decode := T.decode
  read := T.read
  digits := T.digits
  check wire challenge table := (T.check wire challenge table).bind fun
    | none => .pure none
    | some (r2, v) => tensorCheck ν table r2 v
  passes table result := ∃ ρ θ v, result = some (ρ, θ, v) ∧ v = θ * mle ρ (packed table)
  costs := T.costs ++ tensorCosts ν
  start := T.start

variable [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
  {T : Tier O fold Ring Stmt Brk Wire Coord Quadratic (Option (List Quadratic × Quadratic)) Table}
  {ν : ℕ}

/-- A level with the tensor reduction satisfies the conditions of `Tier.Valid`: the soundness of
its checks is that of the checks of the level with `tensorCheck_avoids`. `passes`: a final claim
`(r_2, v_W)` of the level is true for a table `w` when `v_W = w̃(r_2)`. `staged`: each draw of the
checks of the level has its bound, for a game that continues after an accepted end. -/
theorem Tier.tensored_valid (valid : T.Valid)
    (passes : ∀ (table : Table) r2 v, v = mle r2 (lifted Quadratic table) →
      T.passes table (some (r2, v)))
    (staged : ∀ wire challenge table (rest : List ℝ)
      (tail : Option (List Quadratic × Quadratic) → List ℝ → Prop),
      (∀ costs, tail none costs) → (∀ final, tail (some final) rest) →
      StagedTo (Coord → Quadratic) (T.check wire challenge table) (T.costs ++ rest) tail) :
    (T.tensored ν).Valid where
  injective := valid.injective
  merges := valid.merges
  sound wire challenge table :=
    (avoids_bind _ _ _).mpr ((valid.sound wire challenge table).mono fun result implies => by
      rcases result with _ | ⟨r2, v⟩
      · rintro ⟨_, _, _, ended, _⟩
        simp at ended
      · exact (tensorCheck_avoids ν table r2 v).mono fun _ opened ⟨ρ, θ, value, ended, opening⟩ =>
          implies (passes table r2 v (opened ρ θ value ended opening)))
  staged wire challenge table :=
    (stagedTo_bind _ _ _ _).mpr (staged wire challenge table (tensorCosts ν) _
      (fun _ => trivial) fun final => tensorCheck_staged ν table final.1 final.2)
  nonnegative cost member :=
    (List.mem_append.mp member).elim (valid.nonnegative cost) (tensorCosts_nonnegative ν cost)
  draws := List.append_ne_nil_of_left_ne_nil valid.draws _

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The verifier of the checks of a level with the tensor reduction does not read the table. -/
theorem Tier.tensored_uniform
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (wire : Wire) (challenge : Fin S.blocks → Ring) (table other : Table) :
    bare ((T.tensored ν).check wire challenge table) =
      bare ((T.tensored ν).check wire challenge other) :=
  bare_bind_congr (uniform wire challenge table other) fun result => by
    rcases result with _ | ⟨r2, v⟩
    · rfl
    · exact tensorCheck_bare ν table other r2 v

end Tensored

section Claims

variable {S : Shape} {O : OpeningShape S} {m : ℕ}

/-- The claim of a trace level for the packed point `ρ = (ρ_pack, ρ_pos, ρ_blk)` of Akita (65),
the factor `θ` and the value `ṽ` of the tensor reduction: the embedding of the fixed subfield of
Akita Section 3.1, the weights `χ_blk(i) = ẽq(ρ_blk, i)` of (95) and `a_x = ẽq(ρ_pos, x)` of (66),
and the functional `T_{ρ_pack}` of (97). -/
noncomputable def TraceClaim.ofPoint (quarter : 4 * m = S.innerDegree) (payload : S.Payload)
    (openPayload : O.Payload) (block position pack : List Quadratic) (θ v : Quadratic) :
    TraceClaim O Quadratic :=
  ⟨payload, openPayload, (Akita.subfield quarter).toRingHom, fun i => eqAt block i,
    fun x => eqAt position x, packFunctional quarter (eqAt pack), θ, v⟩

/-- The statement of the terminal for the packed point `(ρ_pack,term, ρ_pos,term, ρ_blk,term)` and
the pair `(θ_term, v_tensor)` of its tensor reduction (Akita Section 8.2). -/
noncomputable def TerminalStatement.ofPoint (quarter : 4 * m = S.innerDegree)
    (matrix : Matrix (Fin S.innerRank) (Fin S.blockLength × Fin S.sourceDigits)
      (Rq S.innerDegree))
    (state : Fin S.blocks → Fin S.innerRank → Rq S.innerDegree)
    (block position pack : List Quadratic) (θ v : Quadratic) : TerminalStatement S Quadratic :=
  ⟨matrix, state, (Akita.subfield quarter).toRingHom, fun i => eqAt block i,
    fun x => eqAt position x, packFunctional quarter (eqAt pack), θ, v⟩

/-- The scalar row (101) of a weak opening of a trace claim is the claim of the tensor reduction
for the table of its source blocks: `θ·f̄(ρ) = ṽ`, for the packed polynomial
`f̄ = unpackTable` of this table (Akita (98), with `traceScalar_eq_mle`). -/
theorem traceOpens_evaluation {quarter : 4 * m = S.innerDegree} {payload : S.Payload}
    {openPayload : O.Payload} {block position pack : List Quadratic} {θ v : Quadratic}
    {key : Key S} {openKey : OpeningKey O} {factorBound sourceBound : ℕ} {table : Table}
    {ω : Weak O}
    (opens : ω.TraceOpens factorBound sourceBound
      ((TraceClaim.ofPoint quarter payload openPayload block position pack θ v).statement key
        openKey) table)
    (blocks : S.blocks = 2 ^ block.length) (length : S.blockLength = 2 ^ position.length)
    (width : 2 * m = 2 ^ pack.length) :
    θ * mle (pack ++ (position ++ block)) (unpackTable S.innerDegree m (S.decode ω.source)) =
      v := by
  have partials : ∀ i, O.tracePartial ω.partials i =
      traceContract (Akita.subfield quarter).toRingHom
        (fun x : Fin S.blockLength => eqAt position x) (gadget S.sourceBase (ω.source i)) :=
    opens.partials
  rw [← traceScalar_eq_mle quarter block position pack blocks length width ω.source]
  simp only [← partials]
  exact opens.value

end Claims

section Receiver

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {Ring Stmt Brk Wire Coord Slot Result' Source' : Type} [Zero Ring]
  {key' : Key S'} {openKey' : OpeningKey O'} {weight' : ℕ}
  (T : Tier O fold Ring Stmt Brk Wire Coord Slot (Option (List Quadratic × Quadratic × Quadratic))
    Table)
  (N : Tier O' fold' (Rq S'.innerDegree) (TraceClaim O' Quadratic)
    (LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S')) Wire Coord Slot Result'
    Source')
  {total : ℕ} [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type} {m : ℕ}

/-- The link of the verifier from the level `T` to a next level `N` with evaluation trace
(Akita Figure 4, steps 0, 3 and 5): the transcript has the stages of `T`, each fold challenge
decodes, and the checks of `T` with the tensor reduction end in a claim `(ρ, θ, ṽ)`. The claim of
`N` is `TraceClaim.ofPoint` for the commitment `commit` of the message after the fold challenge
of `T`, the point `ρ` with the segments `(ρ_pack, ρ_pos, ρ_blk)` of the lengths of the shape of
`N`, the factor `θ` and the value `ṽ`. -/
def Tier.TraceLinked (quarter : 4 * m = S'.innerDegree) (commit : Wire → S'.Payload)
    (point : Prefix Wire (Coord → Slot) total) : Prop :=
  T.finish ≤ point.rest.length ∧ T.Decodes point ∧
    ∃ block position pack θ v openPayload,
      T.game (fun _ => 0) point = .pure (some (pack ++ (position ++ block), θ, v)) ∧
      S'.blocks = 2 ^ block.length ∧ S'.blockLength = 2 ^ position.length ∧
      2 * m = 2 ^ pack.length ∧
      N.claim (point.wire N.start) = TraceClaim.ofPoint quarter
        (commit (point.wire (T.start + 1))) openPayload block position pack θ v

variable {T N} {accept : Prefix Wire (Coord → Slot) total → Prop}
  {tail : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT (Option (Weak O'))}
  {error : ℝ} {loops tree : ℕ}

/-- `Tier.suffix_committed` for a next level `N` with evaluation trace (Akita Proposition 10.44
for one more level). The successor source of the level is the table of `Z_q` of the packed
polynomial `ψ^{-1}` of the table of a weak opening of `N`. The final claim of the level is true
for it by the scalar row (101) of this weak opening (`traceOpens_evaluation`), which is
Akita Lemma 10.42 (ii). `traced`: the method of `N` is `trace`. `uniform`: the verifier of the
checks does not read the source. `passes`: a final claim `(ρ, θ, ṽ)` is true for a table `w` when
`ṽ = θ·f̄_w(ρ)`. `linked`: the verifier accepts only with the link `Tier.TraceLinked`. -/
theorem Tier.suffix_push_trace (valid : T.Valid)
    (traced : N.method = trace key' openKey' weight') (next : N.start = T.finish)
    (quarter : 4 * m = S'.innerDegree) (commit : Wire → S'.Payload) (fits : T.finish ≤ total)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T.passes table (some (ρ, θ, v)))
    (linked : ∀ point, accept point → T.TraceLinked N quarter commit point)
    (embed : LevelBreak key' openKey' (2 * (2 * weight') * fold'.sourceBound S') → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Tier.Suffix accept N.start N.Opens tail error loops tree) :
    Tier.Suffix accept T.start T.Opens
      (T.over accept tail
        (fun inner => flatten (unpackTable S'.innerDegree m (tableOf (tail.witness inner))))
        (fun inner other => embed (Tier.collide key' openKey'
          (2 * (2 * weight') * fold'.sourceBound S') (tail.witness inner) (tail.witness other))))
      (S.blocks * T.chance + T.costs.sum + error) (S.blocks + 1 + loops)
      (2 * (S.blocks + 1) * tree) := by
  refine Tier.suffix_committed (factorBound := 2 * weight')
    (sourceBound := fold'.sourceBound S') valid next commit fits
    (fun point accepted => (linked point accepted).2.1) key' openKey' (le_of_eq (by ring))
    (fun table => flatten (unpackTable S'.innerDegree m table)) ?_ ?_ embed embedded later
  · rintro point ω accepted ⟨_, same, opens⟩
    obtain rfl := Option.some.inj same
    rw [traced] at opens
    obtain ⟨_, _, block, position, pack, θ, v, openPayload, _, _, _, _, claimed⟩ :=
      linked _ accepted
    have left : ω.TraceOpens (2 * weight') (fold'.sourceBound S')
        ((N.claim (point.wire N.start)).statement key' openKey') (S'.decode ω.source) := opens
    rw [claimed] at left
    exact left.certified
  · rintro point ω accepted ⟨_, same, opens⟩
    obtain rfl := Option.some.inj same
    rw [traced] at opens
    obtain ⟨long, decodes, block, position, pack, θ, v, openPayload, ended, blocks, length,
      width, claimed⟩ := linked _ accepted
    have left : ω.TraceOpens (2 * weight') (fold'.sourceBound S')
        ((N.claim (point.wire N.start)).statement key' openKey') (S'.decode ω.source) := opens
    rw [claimed] at left
    refine ⟨long, decodes, _, ended_of_bare T.read (uniform _ _ _ _) ended, passes _ _ _ _ ?_⟩
    rw [packed_flatten]
    exact (traceOpens_evaluation left blocks length width).symm

end Receiver

namespace Terminal

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape}
  {Ring Stmt Brk Wire Coord Slot : Type} [Zero Ring] {total : ℕ} [Fintype Slot] [Nonempty Slot]
  [Fintype Coord] [DecidableEq Coord]
  (T : Tier O fold Ring Stmt Brk Wire Coord Slot (Option (List Quadratic × Quadratic × Quadratic))
    Table)
  (L : Terminal S' Quadratic Wire Coord Slot total) {m : ℕ}

/-- The link of the verifier from the level `T` to the terminal (Akita Section 8.2): the
transcript has the stages of `T`, each fold challenge decodes, and the checks of `T` with the
tensor reduction end in a claim `(ρ, θ, ṽ)`. The statement of the terminal is
`TerminalStatement.ofPoint` for the state `(t_i)_i = stateOf` of the message after the fold
challenge of `T`, the point `ρ` with the segments `(ρ_pack, ρ_pos, ρ_blk)` of the lengths of the
shape of the terminal, and the pair `(θ_term, v_tensor) = (θ, ṽ)`. -/
def Linked (quarter : 4 * m = S'.innerDegree)
    (stateOf : Wire → Fin S'.blocks → Fin S'.innerRank → Rq S'.innerDegree)
    (point : Prefix Wire (Coord → Slot) total) : Prop :=
  T.finish ≤ point.rest.length ∧ T.Decodes point ∧
    ∃ block position pack θ v,
      T.game (fun _ => 0) point = .pure (some (pack ++ (position ++ block), θ, v)) ∧
      S'.blocks = 2 ^ block.length ∧ S'.blockLength = 2 ^ position.length ∧
      2 * m = 2 ^ pack.length ∧
      L.statement point = TerminalStatement.ofPoint quarter L.matrix
        (stateOf (point.wire (T.start + 1))) block position pack θ v

variable {T L} {accept : Prefix Wire (Coord → Slot) total → Prop}

/-- `Terminal.suffix_after` for a terminal with the link `Terminal.Linked` (Akita
Proposition 10.44 for the last nonterminal level). The successor source of the level is the table
of `Z_q` of the packed polynomial `ψ^{-1}` of the table of the source blocks of a certificate of
the terminal. The final claim of the level is true for it by the row (130) of the certificate
(`Terminal.Sourced.source` with `traceScalar_eq_mle`), which is Akita Lemma 10.42 (ii). Two
accepting transcripts with one stage input of the checkpoint have one state `(t_i)_i`, because the
message after the fold challenge of the level has this state. `uniform`, `passes` and `linked` are
as in `Tier.suffix_push_trace`. -/
theorem suffix_linked (valid : T.Valid) (terminal : L.Valid accept) (next : L.start = T.finish)
    {peak : ℕ} (units : UnitDifferences L.family)
    (sparse : ∀ challenge ∈ L.family, Sparse L.weight peak challenge)
    (quarter : 4 * m = S'.innerDegree)
    (stateOf : Wire → Fin S'.blocks → Fin S'.innerRank → Rq S'.innerDegree)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T.passes table (some (ρ, θ, v)))
    (linked : ∀ point, accept point → Linked T L quarter stateOf point) :
    Tier.Suffix accept T.start T.Opens
      (T.over accept (L.body accept)
        (fun inner => flatten (unpackTable S'.innerDegree m
          (S'.decode (blocksOf ((L.body accept).witness inner)))))
        (fun inner other =>
          L.collide ((L.body accept).witness inner) ((L.body accept).witness other)))
      (S.blocks * T.chance + T.costs.sum + S'.blocks * slotChance L.decode)
      (S.blocks + 1 + S'.blocks) (2 * (S.blocks + 1) * (S'.blocks + 1)) := by
  refine suffix_after valid terminal next units sparse
    (fun source => flatten (unpackTable S'.innerDegree m (S'.decode source))) ?_
    (fun point accepted => (linked point accepted).2.1) ?_
  · intro point other accepted accepted' same
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, claimed⟩ := linked point accepted
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, claimed'⟩ := linked other accepted'
    have state : (L.read (point.take L.start)).state = stateOf (point.wire (T.start + 1)) :=
      congrArg TerminalStatement.state claimed
    have state' : (L.read (other.take L.start)).state = stateOf (other.wire (T.start + 1)) :=
      congrArg TerminalStatement.state claimed'
    rw [state, state', ← Prefix.wire_take le_rfl other, ← same, Prefix.wire_take le_rfl]
  · intro point fork accepted sourced
    obtain ⟨long, decodes, block, position, pack, θ, v, ended, blocks, length, width, claimed⟩ :=
      linked point accepted
    have value := (sourced.source units).2.2
    rw [claimed] at value
    refine ⟨long, decodes, _, ended_of_bare T.read (uniform _ _ _ _) ended, passes _ _ _ _ ?_⟩
    rw [packed_flatten,
      ← traceScalar_eq_mle quarter block position pack blocks length width fork.source]
    exact value.symm

end Terminal

end NightstreamFPrime.Spec.Compression.Akita.Extract
