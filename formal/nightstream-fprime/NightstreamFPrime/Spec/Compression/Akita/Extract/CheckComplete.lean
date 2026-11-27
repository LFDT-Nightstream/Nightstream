import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.Play
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceNorm

/-! The completeness of the checks of a trace level of Akita (ePrint 2026/1983, revision of
2026-11-23, Theorem 10.9 with Lemma 10.7, Theorem 3.3, Theorem 5.1, item 1, and Section 6.2).

- `Sends` states which messages of the checks satisfy the condition `sendable` of `WinsAt`: an
  element of `E`, a pair, a vector of at most `width` elements, and a natural number up to `cap`.
- `sumcheck_winsAt` is `Game.sumcheck_winsTo` for `WinsAt`, and `eqSumcheck_winsAt` is
  `eqSumcheck_winsTo` for `WinsAt`.
- `foldCheckSq_winsAt` is the completeness of the checks of one fold with the direct norm check
  (`rangeTree_winsAt`, `normLeafStage_winsAt`): for integer coordinates with
  `Σ_u (z^Z_u)² = E_resp ≤ S_max`, the prover sends `E_resp`, the true round polynomials,
  `z̃(r_virt)`, `s̃(r_virt)` and `w̃(r_2)`.
- `energy_response` gives `Σ_u (z^Z_u)² = ‖G_{b_z,m_A}·ẑ‖_2²` in `Z_q` for the flat witness of
  digit vectors, so the cap of the honest prover is a bound on `‖z‖_2²`.
- `traceFold_winsTo` is the completeness of the checks of a committed trace fold.
- `Tier.tensored_winsAt` adds the tensor reduction, and `committedTrace_winsAt` and
  `committedTraceSq_winsAt` are the completeness of the checks of the levels `committedTrace` and
  `committedTraceSq`.
- `liftFold_winsAt` and `liftBatch_winsAt` are the completeness of the checks of the levels
  `liftFold` and `liftBatch` on the quotient-lift route, for the witness of Akita (76) with the
  digits of the quotients (Akita Lemma 7.1, item 1). -/

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

/-- The number of the draws of `foldCheckSq`: the draws of `foldCheck`, `ζ_norm` and
`η_norm`. -/
def foldDrawsSq (μ b : ℕ) (bin : Bool) : ℕ :=
  0 + μ + 1 + binDraws bin + 1 + treeDraws μ (μ + 1) (rangeArities b) + 1 + 1 + 1

/-- A bound on the number of field elements in one message of `foldCheckSq` before its fused
sum-check. -/
def rangeWidthSq (b : ℕ) : ℕ :=
  max (max (rangeLeaf b + 1) 2 + 1) (treeWidth (rangeArities b) 1)

theorem foldCostsSq_length (E : Type) [Fintype E] (badBound vars batch base : ℕ)
    (bin : Bool) :
    (foldCostsSq E badBound vars batch base bin).length = foldDrawsSq vars base bin := by
  simp only [foldCostsSq, rangeCostsSq, foldDrawsSq, List.length_cons, List.length_append,
    List.length_replicate, binCosts_length, ← treeCosts_length E vars (vars + 1) (rangeArities
      base) 1]
  omega

section Checks

variable {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] [DecidableEq E] [Algebra Zq E]
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

omit [Algebra Zq E] in
/-- `sumcheck_winsAt` for the sum-check with a second bad set in each round. -/
theorem sumcheckWith_winsAt {d bound : ℕ} {truth : List E → E} {round : List E → E[X]}
    (true_rounds : RoundTruth d bound truth round)
    (message : ∀ coefficients : Fin (d + 1) → E, sendable (Fin (d + 1) → E) coefficients)
    (extra : List E → Finset E) (rest : ℕ) (tail : Option (List E × E) → ℕ → ℕ → Prop) :
    ∀ (rounds : ℕ) (point : List E) (claim : E) (index : ℕ), point.length + rounds ≤ bound →
      claim = truth point → index < slots →
      (∀ final : List E, final.length = point.length + rounds → ∀ place ≤ index,
        tail (some (final, truth final)) rest place) →
      WinsAt sendable slots (sumcheckWith d round extra rounds point claim) (rest + rounds) index
        tail := by
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
      show WinsAt sendable slots (sumcheckWith d round extra rounds (point ++ [challenge])
        ((round point).eval challenge)) (rest + rounds) 0 tail
      rw [true_rounds.binds point challenge within]
      exact step (point ++ [challenge]) _ 0 (by simp; omega) rfl (by omega) fun final length
        place early => ended final (by simp at length; omega) place (by omega)

omit [Algebra Zq E] in
/-- `sumcheck_winsAt` for the equality-factored sum-check: the prover sends the messages
`(q_1, …, q_D)` of the true normalized round polynomials. -/
theorem eqSumcheck_winsAt {D bound : ℕ} {τ : List E} {truth : List E → E}
    {normal : List E → E[X]} (true_rounds : EqRoundTruth D bound τ truth normal)
    (message : ∀ coefficients : Fin D → E, sendable (Fin D → E) coefficients)
    (round : List E → E[X]) (extra : List E → Finset E) (rest : ℕ)
    (tail : List E × E → ℕ → ℕ → Prop) :
    ∀ (rounds : ℕ) (point : List E) (claim : E) (index : ℕ), point.length + rounds ≤ bound →
      claim = truth point → index < slots →
      (∀ final : List E, final.length = point.length + rounds → ∀ place ≤ index,
        tail (final, truth final) rest place) →
      WinsAt sendable slots (eqSumcheck D round extra τ rounds point claim) (rest + rounds) index
        tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim index _ true_claim _ ended
      rw [true_claim]
      exact ended point rfl index le_rfl
  | succ rounds step =>
      intro point claim index inside true_claim room ended
      have within : point.length < bound := by omega
      refine ⟨room, fun i => (normal point).coeff (i + 1), message _, ?_⟩
      intro challenge
      show WinsAt sendable slots (eqSumcheck D round extra τ rounds (point ++ [challenge])
        ((innerPoly claim (τ.getD point.length 0) fun i => (normal point).coeff (i + 1)).eval
          challenge)) (rest + rounds) 0 tail
      rw [innerPoly_tail (normal point) (true_rounds.degree point within) claim _
        (true_claim ▸ true_rounds.halves point within), true_rounds.binds point challenge within]
      exact step (point ++ [challenge]) _ 0 (by simp; omega) rfl (by omega) fun final length
        place early => ended final (by simp at length; omega) place (by omega)

/-- `rangeTree_winsTo` for `WinsAt`: the completeness of the range tree for a last stage with
`WinsAt` after a draw. -/
theorem rangeTree_winsAt {β : Type} {width cap : ℕ} (sends : Sends sendable E width cap)
    (room : 0 < slots) (leaf μ : ℕ) (w : Table)
    (last : ℕ → (ℕ → E) → List E → E → Game (Option β)) (lastDraws rest : ℕ)
    (tail : Option β → ℕ → ℕ → Prop)
    (lastWins : ∀ nodes weights point claim, point.length = μ →
      claim = ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E leaf m w) →
      WinsAt sendable slots (last nodes weights point claim) (rest + lastDraws) 0 tail) :
    ∀ (arities : List ℕ) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E),
      point.length = μ → treeWidth arities nodes ≤ width →
      claim = ∑ m ∈ Finset.range nodes,
        weights m * mle point (nodeTable E (arities.prod * leaf) m w) →
      WinsAt sendable slots (rangeTree leaf μ w last arities nodes weights point claim)
        (rest + treeDraws μ lastDraws arities) 0 tail := by
  intro arities
  induction arities with
  | nil =>
      intro nodes weights point claim long _ true_claim
      rw [List.prod_nil, one_mul] at true_claim
      exact lastWins nodes weights point claim long true_claim
  | cons a others step =>
      intro nodes weights point claim long wide true_claim
      subst long
      have widths : a + 1 + 1 ≤ width ∧ nodes * a ≤ width ∧
          treeWidth others (nodes * a) ≤ width := by
        simpa [treeWidth] using wide
      rw [List.prod_cons, mul_assoc, ← stage_eqTruth_nil] at true_claim
      rw [treeDraws, ← Nat.add_assoc, ← Nat.add_assoc]
      refine (winsAt_bind sendable slots _ _ _ _ _).mpr (eqSumcheck_winsAt
        (stage_eqRoundTruth a (others.prod * leaf) nodes weights point w)
        (fun coefficients => sends.vector _ coefficients (by omega)) _ _ _ _ point.length []
        claim 0 (by simp) true_claim room fun final length place early => ?_)
      simp only [List.length_nil, zero_add] at length
      obtain rfl : place = 0 := by omega
      refine ⟨room, fun i => mle final (nodeTable E (others.prod * leaf) i w),
        sends.vector _ _ widths.2.1, ?_⟩
      have inside (m t : ℕ) (node : m ∈ Finset.range nodes) (place : t ∈ Finset.range a) :
          m * a + t < nodes * a :=
        Nat.lt_of_lt_of_le (Nat.add_lt_add_left (Finset.mem_range.mp place) _)
          (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right a (Finset.mem_range.mp node))
      have final_claim := stage_eqTruth_final a (others.prod * leaf) nodes weights point final w
        length
      dsimp only
      rw [ite_eq_left (final_claim.trans (Finset.sum_congr rfl fun m node =>
        congrArg _ (Finset.prod_congr rfl fun t place =>
          (entry_of (fun i => mle final (nodeTable E (others.prod * leaf) i w))
            (inside m t node place)).symm)))]
      intro γ
      refine step (nodes * a) (fun m => γ ^ m) final _ length widths.2.2 ?_
      rw [rowPoly_eval]
      exact Finset.sum_congr rfl fun m member => congrArg _
        (entry_of (fun i => mle final (nodeTable E (others.prod * leaf) i w))
          (Finset.mem_range.mp member))

/-- The completeness of the last stage of the range tree on the Euclidean route (Akita
Section 6.3 and Lemma 6.3): for a true claim and a number `E_resp ≤ S_max` with
`Σ_{u < 2^{μ'}} (z^Z_u)² = E_resp` in `Z_q`, the prover sends `E_resp`, the true round
polynomials, and the values `z̃(r_virt)` and `s̃(r_virt)`. -/
theorem normLeafStage_winsAt {width cap : ℕ} (sends : Sends sendable E width cap)
    (room : 0 < slots) {leaf μ limit claimed : ℕ} (wide : max (leaf + 1) 2 + 1 ≤ width)
    (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (small : claimed ≤ limit) (capped : claimed ≤ cap)
    (energy : ∑ u ∈ Finset.range (2 ^ μ), normTable μ kernel w u ^ 2 = (claimed : Zq))
    (rest : ℕ) (tail : Option (List E × E × E) → ℕ → ℕ → Prop)
    (ended : ∀ r : List E, r.length = μ → tail (some (r, mle r (derived E w),
      mle r (lifted E (normTable μ kernel w)))) rest 1)
    (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E) (long : point.length = μ)
    (true_claim : claim = ∑ m ∈ Finset.range nodes,
      weights m * mle point (nodeTable E leaf m w)) :
    WinsAt sendable slots (normLeafStage leaf μ limit kernel support w nodes weights point claim)
      (rest + μ + 1) 0 tail := by
  subst long
  refine ⟨room, claimed, sends.count _ capped, ?_⟩
  dsimp only
  rw [ite_eq_left small]
  intro ζ
  have merged : claim + ζ * claimed =
      combTruth point.length (normLeafTables point.length kernel point w)
        (normLeafComb E leaf nodes weights ζ) [] := by
    rw [normLeafTruth_nil, true_claim, ← map_natCast (algebraMap Zq E), ← energy, map_sum]
    exact congrArg _ (congrArg _ (Finset.sum_congr rfl fun u _ => by
      simp only [lifted, map_pow]))
  refine (winsAt_bind sendable slots _ _ _ _ _).mpr (sumcheckWith_winsAt
    (normLeaf_roundTruth leaf nodes point.length weights ζ kernel point w)
    (fun coefficients => sends.vector _ coefficients wide) _ rest _ point.length [] _ 0
    (by simp) merged room fun final length place early => ?_)
  simp only [List.length_nil, zero_add] at length
  obtain rfl : place = 0 := by omega
  have last := normLeafTruth_final leaf nodes weights ζ point.length kernel point final w
  rw [length] at last
  refine ⟨room, (mle final (lifted E (normTable point.length kernel w)),
    mle final (derived E w)), sends.pair _, ?_⟩
  dsimp only
  rw [ite_eq_left last]
  exact ended final length

omit [DecidableEq E] [Algebra Zq E] in
/-- The prover wins after `drawBin` when it wins for each coefficient `ζ_bin`. -/
theorem drawBin_winsAt {β : Type} (bin : Bool) {bad : Finset E} {next : E → Game β} {rest : ℕ}
    {tail : β → ℕ → ℕ → Prop} (each : ∀ ζ, WinsAt sendable slots (next ζ) rest 0 tail) :
    WinsAt sendable slots (drawBin bin bad next) (rest + binDraws bin) 0 tail := by
  cases bin
  · exact each 0
  · exact each

/-- The completeness of the checks of one fold on the Euclidean route (Akita Theorem 10.9 with
Sections 6.2 and 6.3): for a witness with the relation, the digit range statement, the binary
statement and `Σ_{u < 2^{μ'}} (z^Z_u)² = E_resp ≤ S_max` in `Z_q`, the prover has a message for
each draw sequence, and `foldCheckSq` ends in the claim `(r_2, w̃(r_2))`. -/
theorem foldCheckSq_winsAt {width cap : ℕ} (sends : Sends sendable E width cap)
    (room : 0 < slots) (R : Reduction E) (b : ℕ) (wide : rangeWidthSq b ≤ width)
    (four : 4 ≤ width) (support : ℕ → Prop) [DecidablePred support] {bin : Bool}
    {limit claimed : ℕ}
    (kernel : ℕ → ℕ → Zq) {w : Table} (holds : R.Holds w) (ranged : RangeHolds b R.vars w)
    (binary : BinaryOn support R.vars w) (small : claimed ≤ limit) (capped : claimed ≤ cap)
    (energy : ∑ u ∈ Finset.range (2 ^ R.vars), normTable R.vars kernel w u ^ 2 =
      (claimed : Zq)) :
    WinsAt sendable slots (foldCheckSq R b support bin limit kernel w)
      (foldDrawsSq R.vars b bin) 0
      fun result _ place => place ≤ 1 ∧
        ∃ r2 : List E, r2.length = R.vars ∧ result = some (r2, mle r2 (lifted E w)) := by
  intro α τ0 τ1
  refine (winsAt_bind sendable slots _ _ _ _ _).mpr (rangeTree_winsAt sends room (rangeLeaf b)
    R.vars w _ (R.vars + 1) (0 + R.vars + 1 + binDraws bin + 1) _
    (fun nodes weights point claim => normLeafStage_winsAt sends room
      ((le_max_left _ _).trans wide) kernel support w small capped energy _ _
      (fun final length => ?_) nodes weights point claim)
    (rangeArities b) 1 (fun _ => 1) (rangePoint R.low τ0) 0 (rangePoint_length _ _)
    ((le_max_right _ _).trans wide) ?_)
  swap
  · rw [root_sum b w (rangePoint R.low τ0)]
    exact (mle_rangeResidual_eq_zero ranged _ (rangePoint_length _ _)).symm
  intro η
  refine drawBin_winsAt bin fun ζ => ?_
  intro γ
  refine (winsAt_of_fresh (admits := fun M => ∀ message : M, sendable M message)
    (fun _ message all => all message) (fusedCheck_winsTo _ sends.scalar
      (fun message => sends.vector _ message four) R.vars support w _ _ final _ ζ γ 0 ?_)
    room).mono fun _ _ _ kept => kept
  have virt := fusedTruth_virt support w (fun x => (R.rows α).batchWeight (List.ofFn τ1) x +
    η * normWeight R.vars kernel final x) final γ ζ
  rw [length] at virt
  have batched : (R.rows α).Batched R.vars w (List.ofFn τ1) := R.batched_of_holds holds α _
  have zero : mle final (supportTable support w) = (0 : E) := by
    rw [mle_congr final (u := fun _ => 0) fun x small => by
      unfold supportTable
      split
      · rename_i member
        rw [derived, binary x (length ▸ small) member, map_zero]
      · rfl]
    exact mle_const final 0
  have scaled (x : ℕ) : algebraMap Zq E (w x) * (η * normWeight R.vars kernel final x) =
      η * (algebraMap Zq E (w x) * normWeight R.vars kernel final x) := by ring
  rw [virt, zero, mul_zero, add_zero, mle_normTable kernel w length, ← batched]
  simp only [lifted, mul_add, Finset.sum_add_distrib, scaled, ← Finset.mul_sum]

end Checks

section Trace

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E]
  [Algebra Zq E] {st : TraceStatement O E} {challenge : Fin S.blocks → Rq S.innerDegree} {μ : ℕ}
  {cells : O.cells fold ≤ 2 ^ μ}

variable [Fintype E] [DecidableEq E]

/-- The completeness of the checks of a trace fold whose flat witness the next level commits to
(Akita Lemma 10.7 with Theorem 10.9): for digit vectors with the exact branch relation, the
prover has a message for each draw sequence, and the checks end in the claim `(r_2, w̃(r_2))` for
the flat witness. -/
theorem traceFold_winsTo (admits : Type → Prop) (scalar : admits E)
    (range : ∀ n, n ≤ rangeWidth S.certifiedBase → admits (Fin n → E)) (cubic : admits (Fin (3 + 1) → E))
    (two : 2 ≤ S.certifiedBase) (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) {w : Witness O fold}
    (satisfies : w.TraceSatisfies st challenge) :
    WinsTo admits (foldCheck (traceReduction st challenge μ cells batch rowsFit)
      S.certifiedBase (O.binarySupport fold) (O.compressed fold) w.flat) (foldDraws μ
        S.certifiedBase (O.compressed fold)) true
      fun result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w.flat)) :=
  foldCheck_winsTo admits (traceReduction st challenge μ cells batch rowsFit)
    S.certifiedBase scalar range cubic _
    (satisfies.flat μ cells)
    (satisfies.alphabets.rangeHolds two small μ) (satisfies.alphabets.binaryOn two μ)

end Trace

section Levels

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} {sendable : (M : Type) → M → Prop}
  {slots width cap : ℕ}

omit [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] in
/-- The completeness of the checks of a level with the tensor reduction (Akita Theorem 5.1,
item 1): when the checks of the level end in a true claim `(r_2, w̃(r_2))` of `ν + 1` coordinates
with at most `bound` messages after the last draw, the checks with the tensor reduction end in a
claim `(ρ, θ, θ·f̄_w(ρ))` with `ν` coordinates. -/
theorem Tier.tensored_winsAt [Degree2 E] {Ring Stmt Brk : Type}
    {T : Tier (Fin S.blocks) (fun _ => Ring) Stmt Brk (Witness O fold) (Weak O) Wire Coord Slot
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
    (tensorCheck_winsTo _ sends.pair sends.scalar (fun message => sends.vector _ message wide) table
      head
      (by simpa using long)) room

variable {key : Key S} {openKey : OpeningKey O} {claim : Wire → TraceClaim O E} {μ : ℕ}
  {cells : O.cells fold ≤ 2 ^ μ} {batch : ℕ} {rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch}
  {block : Fin S.blocks → Coord} {decode : Slot → Option (Rq S.innerDegree)} {read : Reader Wire}
  {weight start : ℕ}

omit [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] in
/-- The completeness of the checks of the level `committedTrace` (Akita Theorem 10.9 for the
checks of Figure 6): for digit vectors with the exact branch relation of the claim of a wire
message, the checks end in the claim `(r_2, w̃(r_2))` for the flat witness. -/
theorem committedTrace_winsAt (sends : Sends sendable E width cap)
    (wide : rangeWidth S.certifiedBase ≤ width) (four : 4 ≤ width) (two : 2 ≤ S.certifiedBase)
    (small : S.certifiedBase < fieldModulus) (room : 0 < slots) {wire : Wire}
    {challenge : Fin S.blocks → Rq S.innerDegree} {w : Witness O fold}
    (satisfies : w.TraceSatisfies ((claim wire).statement key openKey) challenge) :
    WinsAt sendable slots
      ((committedTrace (fold := fold) key openKey claim μ cells batch rowsFit block decode read
        weight start).check wire challenge w.flat) (foldDraws μ S.certifiedBase (O.compressed
          fold)) 0
      fun result _ place => place ≤ 1 ∧
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w.flat)) :=
  winsAt_of_fresh (admits := fun M => ∀ message : M, sendable M message)
    (fun _ message all => all message)
    (traceFold_winsTo _ sends.scalar
      (fun _ small message => sends.vector _ message (small.trans wide))
      (fun message => sends.vector _ message four) two small batch rowsFit satisfies) room

omit [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] in
/-- The completeness of the checks of the level `committedTraceSq` on the Euclidean route (Akita
Theorem 10.9 with Sections 6.2 and 6.3): for digit vectors with the exact branch relation of the
claim of a wire message and a response with `‖G_{b_z,m_A}·ẑ‖_2² ≤ S_max`, the checks end in the
claim `(r_2, w̃(r_2))` for the flat witness. `normRoom`: the cube of the `μ'` variables has the
`N_A` coordinates of `z`. `capped`: a wire message can have the energy. -/
theorem committedTraceSq_winsAt (sends : Sends sendable E width cap)
    (wide : rangeWidthSq S.certifiedBase ≤ width) (four : 4 ≤ width) (two : 2 ≤ S.certifiedBase)
    (small : S.certifiedBase < fieldModulus) (room : 0 < slots) {operator limit : ℕ}
    (normRoom : S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ μ) {wire : Wire}
    {challenge : Fin S.blocks → Rq S.innerDegree} {w : Witness O fold}
    (satisfies : w.TraceSatisfies ((claim wire).statement key openKey) challenge)
    (short : vecNormSq (gadget fold.responseBase w.response) ≤ limit)
    (capped : limit ≤ cap) :
    WinsAt sendable slots
      ((committedTraceSq (fold := fold) key openKey claim μ cells batch rowsFit block decode
        read weight operator limit start).check wire challenge w.flat)
      (foldDrawsSq μ S.certifiedBase (O.compressed fold)) 0
      fun result _ place => place ≤ 1 ∧
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w.flat)) :=
  foldCheckSq_winsAt sends room
    (traceReduction ((claim wire).statement key openKey) challenge μ cells batch rowsFit)
    S.certifiedBase wide four _ _
    (satisfies.flat μ cells)
    (satisfies.alphabets.rangeHolds two small μ) (satisfies.alphabets.binaryOn two μ) short
    (short.trans capped)
    (energy_response w ((O.responseCells_le_cells fold).trans cells)
      normRoom)

omit [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] in
theorem committedTraceSq_costs_length {operator limit : ℕ} :
    (committedTraceSq (fold := fold) key openKey claim μ cells batch rowsFit block decode read
      weight operator limit start).costs.length = foldDrawsSq μ S.certifiedBase (O.compressed fold)
        :=
  foldCostsSq_length E _ _ _ _ _

end Levels

section Lift

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} {sendable : (M : Type) → M → Prop}
  {slots width cap : ℕ} {β : Module.Basis (Fin O.planes) Zq E} {key : Key S}
  {openKey : OpeningKey O} {claim : Wire → Claim O E} {μ : ℕ} {cells : O.cells fold ≤ 2 ^ μ}
  {digits : ℕ} {room : O.quotientRoom fold digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : O.rowCount + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
  {decode : Slot → Option (Rq O.carrierDegree)} {read : Reader Wire} {weight start : ℕ}

omit [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] in
/-- The completeness of the checks of the level `liftFold` (Akita Lemma 7.1, item 1, with
Theorem 10.9): for digit vectors with the exact branch relation of the claim of a wire message,
the checks end in the claim `(r_2, w̃(r_2))` for the flat witness with the digits of the
quotients. `covers`: each element of `Z_q` has `digits` balanced digits of the base `b⋆`. -/
theorem liftFold_winsAt (sends : Sends sendable E width cap)
    (wide : rangeWidth S.certifiedBase ≤ width) (four : 4 ≤ width)
    (radix : Radix S.certifiedBase)
    (covers : fieldModulus ≤ S.certifiedBase ^ digits)
    (positive : 0 < slots) {wire : Wire} {challenge : Fin S.blocks → Rq O.carrierDegree}
    {w : Witness O fold}
    (satisfies : w.Satisfies β ((claim wire).statement key openKey) challenge) :
    WinsAt sendable slots
      ((liftFold β key openKey claim μ cells digits room batch rowsFit block decode read weight
        start).check wire challenge
        ((flatRelation O fold β ((claim wire).statement key openKey) challenge μ
          cells).completion (O.gaps fold digits) S.certifiedBase digits w.flat))
      (foldDraws μ S.certifiedBase (O.compressed fold)) 0 fun result _ place => place ≤ 1 ∧
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
          ((flatRelation O fold β ((claim wire).statement key openKey) challenge μ
            cells).completion (O.gaps fold digits) S.certifiedBase digits w.flat))) :=
  winsAt_of_fresh (admits := fun M => ∀ message : M, sendable M message)
    (fun _ message all => all message)
    (liftedFold_winsTo _ sends.scalar
      (fun _ small message => sends.vector _ message (small.trans wide))
      (fun message => sends.vector _ message four) radix covers room batch rowsFit satisfies)
    positive

end Lift

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord : Type} {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {cells : B.cells ≤ 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin n) → Slot → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin n → ℕ} {start : ℕ}

omit [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] in
/-- The completeness of the checks of the level `liftBatch` with opening groups (Akita
Lemma 7.1, item 1, with Theorem 10.9): for digit vectors with the exact branch relation
`Batch.Witness.Satisfies` of the claim of a wire message, the checks end in the claim
`(r_2, w̃(r_2))` for the flat witness with the digits of the quotients. -/
theorem liftBatch_winsAt (sends : Sends sendable E width cap) (wide : rangeWidth base ≤ width)
    (four : 4 ≤ width) (uniform : ∀ g, (B.shape g).certifiedBase = base) (radix : Radix base)
    (covers : fieldModulus ≤ base ^ digits)
    (positive : 0 < slots) {wire : Wire} {challenge : (b : B.Block) → Rq (B.carrierDegree b.1)}
    {w : B.Witness}
    (satisfies : w.Satisfies β ((claim wire).statement key openKey)
      fun g i => challenge ⟨g, i⟩) :
    WinsAt sendable slots
      ((liftBatch β key openKey claim μ cells base digits room batch rowsFit block decode read
        mass start).check wire challenge
        ((B.flatRelation β ((claim wire).statement key openKey) (fun g i => challenge ⟨g, i⟩)
          μ cells).completion (B.gaps digits) base digits w.flat))
      (foldDraws μ base B.compressed) 0 fun result _ place => place ≤ 1 ∧
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E
          ((B.flatRelation β ((claim wire).statement key openKey)
            (fun g i => challenge ⟨g, i⟩) μ cells).completion (B.gaps digits) base digits
            w.flat))) :=
  winsAt_of_fresh (admits := fun M => ∀ message : M, sendable M message)
    (fun _ message all => all message)
    (Batch.liftedFold_winsTo _ sends.scalar
      (fun _ small message => sends.vector _ message (small.trans wide))
      (fun message => sends.vector _ message four) uniform radix covers room batch rowsFit
      satisfies)
    positive

end Groups

end Extract

end NightstreamFPrime.Spec.Compression.Akita
