import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.Play
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceNorm

/-! The completeness of the checks of a trace level of Akita (ePrint 2026/1983, revision of
2026-10-15, Theorem 10.9 with Lemma 10.7, Theorem 3.3, Theorem 5.1, item 1, and Section 6.2).

- `Sends` states which messages of the checks satisfy the condition `sendable` of `WinsAt`: an
  element of `E`, a pair, a vector of at most `width` elements, and a natural number up to `cap`.
- `sumcheck_winsAt` is `Game.sumcheck_winsTo` for `WinsAt`.
- `normCheck_winsAt` is the completeness of the direct norm check: for a true claim
  `v_W = w̃(r_2)` and integer coordinates with `Σ_u (z^Z_u)² = E_resp ≤ S_max`, the prover sends
  `E_resp`, the true round polynomials, `z̃(r_u)` and `w̃(r_3)`.
- `energy_response` gives `Σ_u (z^Z_u)² = ‖G_{b_z,m_A}·ẑ‖_2²` in `Z_q` for the flat witness of
  digit vectors, so the cap of the honest prover is a bound on `‖z‖_2²`.
- `lockedTraceFold_winsTo` is the completeness of the checks of a committed trace fold.
- `Tier.tensored_winsAt` adds the tensor reduction, and `committedTrace_winsAt` and
  `committedTraceSq_winsAt` are the completeness of the checks of the levels `committedTrace` and
  `committedTraceSq`.
- `liftFold_winsAt` and `liftBatch_winsAt` are the completeness of the checks of the levels
  `liftFold` and `liftBatch` on the quotient-lift route, for the flat witness with the digits of
  the quotients (Akita Lemma 7.1, item 1). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

theorem magnitude_sq (a : Zq) : ((magnitude a : ℕ) : Zq) ^ 2 = a ^ 2 := by
  have square : ((a.valMinAbs.natAbs : ℤ)) ^ 2 = a.valMinAbs ^ 2 := Int.natAbs_sq _
  have cast := congrArg (Int.cast : ℤ → Zq) square
  rw [Int.cast_pow, Int.cast_pow, Int.cast_natCast, ZMod.coe_valMinAbs] at cast
  exact cast

section Energy

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

theorem vectorOf_flat (w : Witness O fold) :
    (vectorOf (S.responseCode fold) w.flat : S.Response fold) = w.response :=
  funext fun column => ofCoeffs_eq fun k small => by
    rw [w.flat_response (cell_lt _ small), cellsOf_apply _ _ _ small]

/-- The integer coordinates of the flat witness of digit vectors have the energy
`‖G_{b_z,m_A}·ẑ‖_2²` in `Z_q` (Akita (81) and (85)), when the cube of `μ` variables has the cells
of `ẑ` and the cube of `ν` variables has the `N_A` coordinates of `z`. -/
theorem energy_response (w : Witness O fold) {μ ν : ℕ} (cells : S.responseCells fold ≤ 2 ^ μ)
    (room : S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ ν) :
    ∑ u ∈ Finset.range (2 ^ ν), normTable μ (S.normKernel fold) w.flat u ^ 2 =
      ((vecNormSq (gadget fold.responseBase w.response) : ℕ) : Zq) := by
  rw [normTable_normKernel cells, sum_sq_cellsOf S.sourceCode _ room, vectorOf_flat, vecNormSq,
    Nat.cast_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [normSq, Nat.cast_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Nat.cast_pow, magnitude_sq]

end Energy

namespace Extract

open NightstreamFPrime.Spec.Checkpoint

/-- The messages of the checks that satisfy the condition `sendable` of `WinsAt`: each element of
`E`, each pair, each vector of at most `width` elements of `E`, and each natural number up to
`cap`. -/
structure Sends (sendable : (M : Type) → M → Prop) (E : Type) (width cap : ℕ) : Prop where
  scalar : ∀ message : E, sendable E message
  pair : ∀ message : E × E, sendable (E × E) message
  vector : ∀ (n : ℕ) (message : Fin n → E), n ≤ width → sendable (Fin n → E) message
  count : ∀ message : ℕ, message ≤ cap → sendable ℕ message

theorem normCosts_length (E : Type) [Fintype E] (ν μ : ℕ) :
    (normCosts E ν μ).length = 0 + μ + 1 + ν := by
  simp only [normCosts, List.length_append, List.length_cons, List.length_replicate]
  omega

section Checks

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  {sendable : (M : Type) → M → Prop} {slots : ℕ}

omit [Algebra Zq E] in
/-- `Game.sumcheck_winsTo` for `WinsAt`: the first round polynomial is message `index` after the
last draw, and the sum-check ends with at most `index` messages after its last draw. -/
theorem sumcheck_winsAt {d bound : ℕ} {truth : List E → E} {round : List E → E[X]}
    (true_rounds : RoundTruth d bound truth round)
    (message : ∀ coefficients : Fin (d + 1) → E, sendable (Fin (d + 1) → E) coefficients)
    (rest : ℕ) (tail : Option (List E × E) → ℕ → ℕ → Prop) :
    ∀ (rounds : ℕ) (point : List E) (claim : E) (index : ℕ), point.length + rounds ≤ bound →
      claim = truth point → index < slots →
      (∀ final : List E, final.length = point.length + rounds → ∀ place ≤ index,
        tail (some (final, truth final)) rest place) →
      WinsAt sendable slots (sumcheck d round rounds point claim) (rest + rounds) index tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim index _ true_claim _ ended
      rw [true_claim]
      exact ended point rfl index le_rfl
  | succ rounds step =>
      intro point claim index inside true_claim room ended
      have within : point.length < bound := by omega
      refine ⟨room, fun index => (round point).coeff index, message _, ?_⟩
      dsimp only
      rw [coeffPoly_coeff (round point) (true_rounds.degree point within),
        ite_eq_left (true_claim ▸ true_rounds.halves point within)]
      intro challenge
      show WinsAt sendable slots (sumcheck d round rounds (point ++ [challenge])
        ((round point).eval challenge)) (rest + rounds) 0 tail
      rw [true_rounds.binds point challenge within]
      exact step (point ++ [challenge]) _ 0 (by simp; omega) rfl (by omega) fun final length
        place early => ended final (by simp at length; omega) place (by omega)

/-- The completeness of the direct norm check (Akita Section 6.2 and Lemma 6.3): for a true
claim `v_W = w̃(r_2)` and a number `E_resp ≤ S_max` with `Σ_{u < 2^ν} (z^Z_u)² = E_resp` in `Z_q`,
the prover has a message for each draw sequence, and the check ends in the claim
`(r_3, w̃(r_3))` after `ν + 1 + μ` draws. The first messages are `E_resp` and the first round
polynomial, after `index` messages. -/
theorem normCheck_winsAt {width cap : ℕ} (sends : Sends sendable E width cap) (wide : 3 ≤ width)
    {ν μ limit claimed index : ℕ} (kernel : ℕ → ℕ → Zq) (w : Table) {r2 : List E}
    (length : r2.length = μ) (small : claimed ≤ limit) (capped : claimed ≤ cap)
    (energy : ∑ u ∈ Finset.range (2 ^ ν), normTable μ kernel w u ^ 2 = (claimed : Zq))
    (room : index + 2 < slots) :
    WinsAt sendable slots (normCheck ν μ limit kernel w r2 (mle r2 (lifted E w)))
      (0 + μ + 1 + ν) index fun result _ place =>
        place ≤ 1 ∧ ∃ r3 : List E, r3.length = μ ∧ result = some (r3, mle r3 (lifted E w)) := by
  unfold normCheck
  rw [ite_eq_left length]
  refine ⟨by omega, claimed, sends.count _ capped, ?_⟩
  dsimp only
  rw [ite_eq_left small]
  refine (winsAt_bind sendable slots _ _ _ _ _).mpr (sumcheck_winsAt
    (combRound_truth ν 2 (normTables μ kernel w) productComb productComb_degree)
    (fun coefficients => sends.vector _ coefficients wide) (0 + μ + 1) _ ν [] _ (index + 1)
    (by simp) ?_ (by omega) fun ru count place early => ?_)
  · rw [product_truth_nil, cubeSum_eq_sum_range, ← map_natCast (algebraMap Zq E), ← energy,
      map_sum]
    refine Finset.sum_congr rfl fun u _ => ?_
    simp only [normTables, Matrix.cons_val_zero, Matrix.cons_val_one, lifted, pow_two, map_mul]
  simp only [List.length_nil, zero_add] at count
  have last : combTruth ν (normTables μ kernel w) productComb ru =
      mle ru (lifted E (normTable μ kernel w)) * mle ru (lifted E (normTable μ kernel w)) := by
    have final := product_truth_final ru (normTables μ kernel w)
    rw [count] at final
    exact final
  refine ⟨by omega, mle ru (lifted E (normTable μ kernel w)), sends.scalar _, ?_⟩
  dsimp only
  rw [ite_eq_left last]
  intro η
  refine (winsAt_bind sendable slots _ _ _ _ _).mpr (sumcheck_winsAt
    (combRound_truth μ 2 (bindTables ν kernel w r2 ru η) productComb productComb_degree)
    (fun coefficients => sends.vector _ coefficients wide) 0 _ μ [] _ 0 (by simp) ?_ (by omega)
    fun r3 count3 place3 early3 => ?_)
  · rw [product_truth_nil, mle_eq_cubeSum r2, length, mle_normTable kernel w count,
      cubeSum_eq_sum_range, cubeSum_eq_sum_range, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [bindTables, Matrix.cons_val_zero, Matrix.cons_val_one]
    ring
  simp only [List.length_nil, zero_add] at count3
  have last3 := product_truth_final r3 (bindTables ν kernel w r2 ru η)
  rw [count3] at last3
  have checked : combTruth μ (bindTables ν kernel w r2 ru η) productComb r3 =
      mle r3 (lifted E w) * (mle r3 (eqAt r2) + η * mle r3 (normWeight ν kernel ru)) := by
    rw [last3]
    simp only [bindTables, Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [mle_add, mle_mul_left]
  refine ⟨by omega, mle r3 (lifted E w), sends.scalar _, ?_⟩
  dsimp only
  rw [ite_eq_left checked]
  exact ⟨by omega, r3, count3, rfl⟩

end Checks

section Trace

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E]
  [Algebra Zq E] {st : TraceStatement O E} {challenge : Fin S.blocks → Rq S.innerDegree} {μ : ℕ}
  {cells : O.cells fold < 2 ^ μ}

/-- A table with `traceRelation` and the cell `cells` equal to 0 satisfies
`lockedTraceRelation`. -/
theorem lockedTraceRelation_of_flat {table : Table} (zero : table (O.cells fold) = 0)
    (holds : (traceRelation O fold st challenge μ cells).Holds table) :
    (lockedTraceRelation O fold st challenge μ cells).Holds table := by
  refine ⟨(forall_append (RingRow.Holds · table) (traceRows O fold st challenge)
    (fun _ : Fin 1 => (zeroForm O.carrierDegree (O.cells fold)).row 0)).mpr
    ⟨holds.1, fun _ => ((zeroForm O.carrierDegree (O.cells fold)).row_holds 0 table).mpr ?_⟩,
    holds.2⟩
  show (∑ _c : Unit, (1 : Rq O.carrierDegree) * ofCoeffs fun _ => table (O.cells fold)) = 0
  rw [zero, ofCoeffs_eq (x := 0) fun j _ => (coeff_zero j).symm]
  simp

variable [Fintype E] [DecidableEq E]

/-- The completeness of the checks of a trace fold whose flat witness the next level commits to
(Akita Lemma 10.7 with Theorem 10.9): for digit vectors with the exact branch relation, the
prover has a message for each draw sequence, and the checks end in the claim `(r_2, w̃(r_2))` for
the flat witness. -/
theorem lockedTraceFold_winsTo (admits : Type → Prop) (scalar : admits E)
    (range : admits (Fin (S.certifiedBase / 2 + 1 + 1) → E)) (cubic : admits (Fin (3 + 1) → E))
    (two : 2 ≤ S.certifiedBase) (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 + 1 ≤ 2 ^ batch) {w : Witness O fold}
    (satisfies : w.TraceSatisfies st challenge) :
    WinsTo admits (foldCheck (lockedTraceReduction st challenge μ cells batch rowsFit)
      S.certifiedBase (O.binarySupport fold) w.flat) (0 + μ + 1 + 1 + μ + 1 + 1 + 1) true
      fun result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w.flat)) :=
  foldCheck_winsTo admits (lockedTraceReduction st challenge μ cells batch rowsFit)
    S.certifiedBase scalar range cubic _
    (lockedTraceRelation_of_flat (w.flat_of_le le_rfl) (satisfies.flat μ cells))
    (satisfies.alphabets.rangeHolds two small μ) (satisfies.alphabets.binaryOn two μ)

end Trace

section Levels

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} {sendable : (M : Type) → M → Prop}
  {slots width cap : ℕ}

/-- The completeness of the checks of a level with the tensor reduction (Akita Theorem 5.1,
item 1): when the checks of the level end in a true claim `(r_2, w̃(r_2))` of `ν + 1` coordinates
with at most `bound` messages after the last draw, the checks with the tensor reduction end in a
claim `(ρ, θ, θ·f̄_w(ρ))` with `ν` coordinates. -/
theorem Tier.tensored_winsAt [Degree2 E] {Ring Stmt Brk : Type}
    {T : Tier (Fin S.blocks) (fun _ => Ring) Stmt Brk (Witness O fold) (Weak O) Wire Coord E
      (Option (List E × E)) Table}
    {ν : ℕ} (sends : Sends sendable E width cap) (wide : 3 ≤ width) {wire : Wire}
    {challenge : Fin S.blocks → Ring} {table : Table} {draws bound : ℕ}
    (wins : WinsAt sendable slots (T.check wire challenge table) draws 0 fun result _ place =>
      place ≤ bound ∧ ∃ r2 : List E, r2.length = ν + 1 ∧
        result = some (r2, mle r2 (lifted E table)))
    (room : bound < slots) (count : T.costs.length = draws) :
    WinsAt sendable slots ((T.tensored ν).check wire challenge table) (T.tensored ν).costs.length
      0 fun result _ _ => ∃ (ρ : List E) (θ : E),
        ρ.length = ν ∧ result = some (ρ, θ, θ * mle ρ (packed table)) := by
  have length : (T.tensored ν).costs.length = draws + (0 + ν + 1) := by
    show (T.costs ++ tensorCosts E ν).length = _
    rw [List.length_append, count, tensorCosts, List.length_cons, List.length_replicate,
      Nat.zero_add]
  rw [length]
  refine (WinsAt.bind (last := bound + 1) wins fun result ⟨r2, long, ended⟩ => ?_).mono
    fun _ _ _ kept => kept.2
  subst ended
  obtain ⟨head, tail, rfl⟩ := List.exists_cons_of_length_eq_add_one long
  exact winsAt_of_fresh (admits := fun M => ∀ message : M, sendable M message)
    (fun _ message all => all message)
    (tensorCheck_winsTo _ sends.pair (fun message => sends.vector _ message wide) table head
      (by simpa using long)) room

variable {key : Key S} {openKey : OpeningKey O} {claim : Wire → TraceClaim O E} {μ : ℕ}
  {cells : O.cells fold < 2 ^ μ} {batch : ℕ} {rowsFit : O.traceRowCount + 1 + 1 ≤ 2 ^ batch}
  {block : Fin S.blocks → Coord} {decode : E → Option (Rq S.innerDegree)} {read : Reader Wire}
  {weight start : ℕ}

/-- The completeness of the checks of the level `committedTrace` (Akita Theorem 10.9 for the
checks of Figure 6): for digit vectors with the exact branch relation of the claim of a wire
message, the checks end in the claim `(r_2, w̃(r_2))` for the flat witness. -/
theorem committedTrace_winsAt (sends : Sends sendable E width cap)
    (wide : S.certifiedBase / 2 + 1 + 1 ≤ width) (four : 4 ≤ width) (two : 2 ≤ S.certifiedBase)
    (small : S.certifiedBase < fieldModulus) (room : 0 < slots) {wire : Wire}
    {challenge : Fin S.blocks → Rq S.innerDegree} {w : Witness O fold}
    (satisfies : w.TraceSatisfies ((claim wire).statement key openKey) challenge) :
    WinsAt sendable slots
      ((committedTrace (fold := fold) key openKey claim μ cells batch rowsFit block decode read
        weight start).check wire challenge w.flat) (0 + μ + 1 + 1 + μ + 1 + 1 + 1) 0
      fun result _ place => place ≤ 1 ∧
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w.flat)) :=
  winsAt_of_fresh (admits := fun M => ∀ message : M, sendable M message)
    (fun _ message all => all message)
    (lockedTraceFold_winsTo _ sends.scalar (fun message => sends.vector _ message wide)
      (fun message => sends.vector _ message four) two small batch rowsFit satisfies) room

/-- The completeness of the checks of the level `committedTraceSq` on the Euclidean route (Akita
Theorem 10.9 with Section 6.2): for digit vectors with the exact branch relation of the claim of
a wire message and a response with `‖G_{b_z,m_A}·ẑ‖_2² ≤ S_max`, the checks with the direct norm
check end in the claim `(r_3, w̃(r_3))` for the flat witness. `normRoom`: the cube of `ν_norm`
variables has the `N_A` coordinates of `z`. `capped`: a wire message can have the energy. -/
theorem committedTraceSq_winsAt (sends : Sends sendable E width cap)
    (wide : S.certifiedBase / 2 + 1 + 1 ≤ width) (four : 4 ≤ width) (two : 2 ≤ S.certifiedBase)
    (small : S.certifiedBase < fieldModulus) (room : 3 < slots) {ν operator limit : ℕ}
    (normRoom : S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ ν) {wire : Wire}
    {challenge : Fin S.blocks → Rq S.innerDegree} {w : Witness O fold}
    (satisfies : w.TraceSatisfies ((claim wire).statement key openKey) challenge)
    (short : vecNormSq (gadget fold.responseBase w.response) ≤ limit)
    (capped : limit ≤ cap) :
    WinsAt sendable slots
      ((committedTraceSq (fold := fold) key openKey claim μ cells batch rowsFit ν block decode
        read weight operator limit start).check wire challenge w.flat)
      (0 + μ + 1 + 1 + μ + 1 + 1 + 1 + (0 + μ + 1 + ν)) 0
      fun result _ place => place ≤ 1 ∧
        ∃ r3 : List E, r3.length = μ ∧ result = some (r3, mle r3 (lifted E w.flat)) := by
  have responses : S.responseCells fold ≤ 2 ^ μ :=
    Nat.le_of_lt (Nat.lt_of_le_of_lt (O.responseCells_le_cells fold) cells)
  change WinsAt sendable slots (Game.bind _ _) _ _ _
  refine WinsAt.bind (committedTrace_winsAt (claim := claim) (cells := cells) (block := block)
    (decode := decode) (read := read) (weight := weight) (start := start) sends wide four two
    small (by omega) satisfies) fun result ⟨r2, long, ended⟩ => ?_
  subst ended
  exact normCheck_winsAt sends (by omega) (S.normKernel fold) w.flat long short
    (short.trans capped) (energy_response w responses normRoom) (by omega)

theorem committedTraceSq_costs_length {ν operator limit : ℕ} :
    (committedTraceSq (fold := fold) key openKey claim μ cells batch rowsFit ν block decode read
      weight operator limit start).costs.length =
      0 + μ + 1 + 1 + μ + 1 + 1 + 1 + (0 + μ + 1 + ν) := by
  show (foldCosts E _ μ batch _ ++ normCosts E ν μ).length = _
  rw [List.length_append, foldCosts_length, normCosts_length]

end Levels

section Lift

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} {sendable : (M : Type) → M → Prop}
  {slots width cap : ℕ} {β : Module.Basis (Fin O.planes) Zq E} {key : Key S}
  {openKey : OpeningKey O} {claim : Wire → Claim O E} {μ : ℕ} {cells : O.cells fold < 2 ^ μ}
  {digits : ℕ} {room : O.quotientRoom fold digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
  {decode : E → Option (Rq O.carrierDegree)} {read : Reader Wire} {weight start : ℕ}

/-- The completeness of the checks of the level `liftFold` (Akita Lemma 7.1, item 1, with
Theorem 10.9): for digit vectors with the exact branch relation of the claim of a wire message,
the checks end in the claim `(r_2, w̃(r_2))` for the flat witness with the digits of the
quotients. `covers`: each element of `Z_q` has `digits` balanced digits of the base `b⋆`. -/
theorem liftFold_winsAt (sends : Sends sendable E width cap)
    (wide : S.certifiedBase / 2 + 1 + 1 ≤ width) (four : 4 ≤ width)
    (radix : Radix S.certifiedBase)
    (covers : ∀ x : Zq, -((S.certifiedBase / 2 : ℕ) : ℤ) *
        ∑ u ∈ Finset.range digits, (S.certifiedBase : ℤ) ^ u ≤ x.valMinAbs ∧
      x.valMinAbs ≤ (((S.certifiedBase / 2 : ℕ) : ℤ) - 1) *
        ∑ u ∈ Finset.range digits, (S.certifiedBase : ℤ) ^ u)
    (positive : 0 < slots) {wire : Wire} {challenge : Fin S.blocks → Rq O.carrierDegree}
    {w : Witness O fold}
    (satisfies : w.Satisfies β ((claim wire).statement key openKey) challenge) :
    WinsAt sendable slots
      ((liftFold β key openKey claim μ cells digits room batch rowsFit block decode read weight
        start).check wire challenge
        ((lockedRelation O fold β ((claim wire).statement key openKey) challenge μ
          cells).completion S.certifiedBase digits (O.cells fold + 1) w.flat))
      (0 + μ + 1 + 1 + μ + 1 + 1 + 1) 0 fun result _ place => place ≤ 1 ∧
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
          ((lockedRelation O fold β ((claim wire).statement key openKey) challenge μ
            cells).completion S.certifiedBase digits (O.cells fold + 1) w.flat))) :=
  winsAt_of_fresh (admits := fun M => ∀ message : M, sendable M message)
    (fun _ message all => all message)
    (liftedFold_winsTo _ sends.scalar (fun message => sends.vector _ message wide)
      (fun message => sends.vector _ message four) radix covers room batch rowsFit satisfies)
    positive

end Lift

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord : Type} {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {cells : B.cells < 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin n → ℕ} {start : ℕ}

/-- The completeness of the checks of the level `liftBatch` with opening groups (Akita
Lemma 7.1, item 1, with Theorem 10.9): for digit vectors with the exact branch relation
`Batch.Witness.Satisfies` of the claim of a wire message, the checks end in the claim
`(r_2, w̃(r_2))` for the flat witness with the digits of the quotients. -/
theorem liftBatch_winsAt (sends : Sends sendable E width cap) (wide : base / 2 + 1 + 1 ≤ width)
    (four : 4 ≤ width) (uniform : ∀ g, (B.shape g).certifiedBase = base) (radix : Radix base)
    (covers : ∀ x : Zq, -((base / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u ≤
        x.valMinAbs ∧
      x.valMinAbs ≤ (((base / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u)
    (positive : 0 < slots) {wire : Wire} {challenge : (b : B.Block) → Rq (B.carrierDegree b.1)}
    {w : B.Witness}
    (satisfies : w.Satisfies β ((claim wire).statement key openKey)
      fun g i => challenge ⟨g, i⟩) :
    WinsAt sendable slots
      ((liftBatch β key openKey claim μ cells base digits room batch rowsFit block decode read
        mass start).check wire challenge
        ((B.lockedRelation β ((claim wire).statement key openKey) (fun g i => challenge ⟨g, i⟩)
          μ cells).completion base digits (B.cells + 1) w.flat))
      (0 + μ + 1 + 1 + μ + 1 + 1 + 1) 0 fun result _ place => place ≤ 1 ∧
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
          ((B.lockedRelation β ((claim wire).statement key openKey)
            (fun g i => challenge ⟨g, i⟩) μ cells).completion base digits (B.cells + 1)
            w.flat))) :=
  winsAt_of_fresh (admits := fun M => ∀ message : M, sendable M message)
    (fun _ message all => all message)
    (Batch.liftedFold_winsTo _ sends.scalar (fun message => sends.vector _ message wide)
      (fun message => sends.vector _ message four) uniform radix covers room batch rowsFit
      satisfies)
    positive

end Groups

end Extract

end NightstreamFPrime.Spec.Compression.Akita
