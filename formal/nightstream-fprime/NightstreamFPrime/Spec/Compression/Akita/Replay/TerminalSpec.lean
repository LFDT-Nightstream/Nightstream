import NightstreamFPrime.Spec.Compression.Akita.Replay.Terminal

/-! The terminal of the specification for the values of the checks of `Replay/Terminal.lean`, and
the theorems that relate the two (Akita Section 8.2, (128) to (131); Lemma 10.47 for the digests).

- `zSource`, `matrixOf`, `stateOf`, `partialsOf` and `challengesOf` are the response `z`, the
  matrix `A_term` (the prefix view of the flat setup, Akita (35)), the state `(t_i)_i`, the
  partials `(e_i)_i` and the fold challenges `(c_i)_i` of the specification for the words of the
  code, with the last blocks of `t` and `e` as arguments. `statementOf` is the statement
  `TerminalStatement.ofPoint` of `Extract/TraceLink.lean` for the split point of the terminal.
- `ringDigest` and `rowsDigestOf` are the field digests of the canonical coefficients of a ring
  element and of the rows of the last block of `t`; `terminalDigest` is `H_dig` of Akita (213).
- `toRq_folded`, `toRq_reduced`, `toRq_aRow`, `toRq_tPartial` and `into_opening` give the values
  of the code in the specification. `terminalChecks_ok_iff` is the statement for acceptance and
  rejection. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Lib (eqAt)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Lemmas

theorem sumExt_into {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E] (f : ℕ → Ext) :
    ∀ n, ((sumExt f n).into : E) = ∑ i ∈ Finset.range n, ((f i).into : E)
  | 0 => by simp [sumExt, Ext.into_zero]
  | n + 1 => by rw [sumExt, Ext.into_add, sumExt_into f n, Finset.sum_range_succ]

/-- `Σ_{x<M} Σ_{u<δ} g(x·δ + u) = Σ_{c<M·δ} g c`. -/
theorem sum_range_mul {M : Type} [AddCommMonoid M] (g : ℕ → M) (δ : ℕ) :
    ∀ n, ∑ x ∈ Finset.range n, ∑ u ∈ Finset.range δ, g (x * δ + u) =
      ∑ c ∈ Finset.range (n * δ), g c
  | 0 => by simp
  | n + 1 => by
      rw [Finset.sum_range_succ, sum_range_mul g δ n, Nat.succ_mul, Finset.sum_range_add]

/-- A sum over the blocks is the sum over the sent blocks and the last block. -/
theorem sum_fin_last {M : Type} [AddCommMonoid M] {B : ℕ} (blocks : 1 ≤ B) (g : ℕ → M) :
    ∑ i : Fin B, g i = ∑ i ∈ Finset.range (B - 1), g i + g (B - 1) := by
  rw [Fin.sum_univ_eq_sum_range g B]
  conv_lhs => rw [show B = B - 1 + 1 by omega]
  rw [Finset.sum_range_succ]

theorem Fq.ofNat_mod (n : ℕ) : Fq.ofNat (n % fieldModulus) = Fq.ofNat n := by
  rw [Fq.ofNat, Fq.ofNat, Nat.mod_mod]

theorem half_toZq : half.toZq = (2 : Zq)⁻¹ := by
  refine (eq_inv_of_mul_eq_one_left ?_)
  rw [half, Fq.toZq_ofNat]
  have value : (((fieldModulus + 1) / 2 * 2 : ℕ) : Zq) = 1 := by
    rw [show (fieldModulus + 1) / 2 * 2 = fieldModulus + 1 by decide, Nat.cast_add,
      ZMod.natCast_self, zero_add, Nat.cast_one]
  exact_mod_cast value

end Lemmas

namespace Poly

variable {d : ℕ} [NeZero d]

theorem equal_iff (a b : Array Fq) : equal d a b = true ↔ toRq d a = toRq d b := by
  rw [equal, List.all_eq_true]
  constructor
  · intro each
    refine toRq_ext fun k small => ?_
    have same := each k (List.mem_range.mpr small)
    rw [beq_iff_eq] at same
    rw [Fq.toZq, Fq.toZq, ← ZMod.natCast_mod, same, ZMod.natCast_mod]
  · intro same k member
    have value := congrArg (fun x => coeff x k) same
    simp only [coeff_toRq _ (List.mem_range.mp member), Fq.toZq] at value
    rw [beq_iff_eq, ← ZMod.val_natCast, value, ZMod.val_natCast]

theorem toRq_ring_coeff (a : Array Fq) (i : ℕ) {k : ℕ} (small : k < d) :
    coeff (toRq d (ring d a i)) k = (get a (i * d + k)).toZq := by
  rw [coeff_toRq _ small, ring, get_ofFn _ small]

theorem canon_toRq (a : Array Fq) :
    ofFn d (fun k => Fq.ofNat (coeff (toRq d a) k).val) = canon d a := by
  unfold canon ofFn
  congr 1
  funext k
  beta_reduce
  rw [coeff_toRq _ k.isLt, Fq.toZq, ZMod.val_natCast, Fq.ofNat_mod]

end Poly

section Recover

variable (S : Shape)

theorem recover_some {c inv part target : Array Fq}
    (inverse : Poly.toRq S.innerDegree inv * Poly.toRq S.innerDegree c = 1) :
    recover S c inv part target = some (Poly.mul S.innerDegree inv
      (Poly.sub S.innerDegree target part)) := by
  unfold recover
  rw [ite_eq_left ((Poly.equal_iff _ _).mpr (by
    rw [Poly.toRq_add, Poly.toRq_mul, Poly.toRq_mul, Poly.toRq_sub, ← mul_assoc,
      mul_comm (Poly.toRq _ c), inverse, one_mul, add_sub_cancel]))]

theorem recover_eq {c inv part target x : Array Fq}
    (found : recover S c inv part target = some x) :
    x = Poly.mul S.innerDegree inv (Poly.sub S.innerDegree target part) := by
  unfold recover at found
  dsimp only at found
  split at found
  · exact (Option.some.inj found).symm
  · cases found

/-- With an inverse of `c`, the block `x` of `part + c·x = target` is the recovered block. -/
theorem solution_iff {c inv part target : Array Fq}
    (inverse : Poly.toRq S.innerDegree inv * Poly.toRq S.innerDegree c = 1) (y : Rq S.innerDegree) :
    Poly.toRq _ part + Poly.toRq _ c * y = Poly.toRq _ target ↔
      y = Poly.toRq _ (Poly.mul S.innerDegree inv (Poly.sub S.innerDegree target part)) := by
  rw [Poly.toRq_mul, Poly.toRq_sub]
  constructor
  · intro solved
    rw [← solved, add_sub_cancel_left, ← mul_assoc, inverse, one_mul]
  · intro equal
    rw [equal, ← mul_assoc, mul_comm (Poly.toRq _ c), inverse, one_mul, add_sub_cancel]

theorem mapM_some {α β : Type} (f : α → Option β) (g : α → β) :
    ∀ l : List α, (∀ a ∈ l, f a = some (g a)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | a :: l, each => by
      rw [List.mapM_cons, each a List.mem_cons_self,
        mapM_some f g l fun b member => each b (List.mem_cons_of_mem _ member)]
      rfl

theorem rowsDigest_map (x : ℕ → Array Fq) :
    rowsDigest S ((List.range S.innerRank).map x) = fieldDigest ((List.range S.innerRank).foldl
      (fun out r => out ++ Poly.ofFn S.innerDegree fun k =>
        Fq.ofNat (coeff (Poly.toRq S.innerDegree (x r)) k).val) #[]) := by
  rw [rowsDigest, List.foldl_map]
  congr 1
  refine List.foldl_ext _ _ _ fun out r _ => ?_
  rw [Poly.canon_toRq]

end Recover

section Values

variable (S : Shape) {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

/-- The response `z` of the specification: column `x·δ + u` of the decoded values. -/
noncomputable def zSource (z : Array ℤ) : S.Source :=
  fun xu => Poly.toRq S.innerDegree (zRing S z (xu.1 * S.sourceDigits + xu.2))

/-- `A_term`: entry `(r, (x, u))` is the ring element `r·M·δ + x·δ + u` of the flat setup, the
prefix view of Akita (35). -/
noncomputable def matrixOf (setup : Array Fq) :
    Matrix (Fin S.innerRank) (Fin S.blockLength × Fin S.sourceDigits) (Rq S.innerDegree) :=
  Matrix.of fun r xu => Poly.toRq S.innerDegree (Poly.ring S.innerDegree setup
    (r * (S.blockLength * S.sourceDigits) + (xu.1 * S.sourceDigits + xu.2)))

/-- The state `(t_i)_i`: the sent blocks, and the last block `tLast`. -/
noncomputable def stateOf (msgs : TerminalMessages) (tLast : ℕ → Rq S.innerDegree) :
    Fin S.blocks → Fin S.innerRank → Rq S.innerDegree := fun i r =>
  if i.val < S.blocks - 1 then
    Poly.toRq S.innerDegree (Poly.ring S.innerDegree msgs.t (i * S.innerRank + r))
  else tLast r

/-- The partials `(e_i)_i`: the sent blocks, and the last block `eLast`. -/
noncomputable def partialsOf (msgs : TerminalMessages) (eLast : Rq S.innerDegree) :
    Fin S.blocks → Rq S.innerDegree := fun i =>
  if i.val < S.blocks - 1 then Poly.toRq S.innerDegree (Poly.ring S.innerDegree msgs.e i)
  else eLast

/-- The fold challenges `(c_i)_i`. -/
noncomputable def challengesOf (ch : TerminalChallenges) : Fin S.blocks → Rq S.innerDegree :=
  fun i => Poly.toRq S.innerDegree (challenge S ch i)

/-- The statement of the terminal (`TerminalStatement.ofPoint` of `Extract/TraceLink.lean`) for
the split point of the terminal, the pair `(θ, v)`, `A_term = matrixOf setup` and the state with
the last block `tLast`. -/
noncomputable def statementOf (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (setup : Array Fq) (msgs : TerminalMessages) (ch : TerminalChallenges)
    (tLast : ℕ → Rq S.innerDegree) : TerminalStatement S E :=
  ⟨matrixOf S setup, stateOf S msgs tLast, (subfield (E := E) quarter).toRingHom,
    fun i => eqAt ((blockPoint S ch.point).map Ext.into) i,
    fun x => eqAt ((positionPoint S ch.point).map Ext.into) x,
    packFunctional quarter (eqAt ((packPoint S ch.point).map Ext.into)), ch.scale.into,
    ch.target.into⟩

/-- The field digest of the canonical coefficients of a ring element. -/
noncomputable def ringDigest {d : ℕ} [NeZero d] (y : Rq d) : ByteArray :=
  fieldDigest (Poly.ofFn d fun k => Fq.ofNat (coeff y k).val)

/-- The field digest of the rows `0, …, n_A − 1` of the last block of `t`. -/
noncomputable def rowsDigestOf (rows : ℕ → Rq S.innerDegree) : ByteArray :=
  fieldDigest ((List.range S.innerRank).foldl (fun out r => out ++ Poly.ofFn S.innerDegree
    fun k => Fq.ofNat (coeff (rows r) k).val) #[])

/-- `H_dig` of Akita Lemma 10.47 (213) for the concrete instance, a DEFINITION: the field digest
(SHA3-256 with the tag `akita-pcs/field-digest/v8`) of the last block of `e` (`Sum.inl`) and of
the rows of the last block of `t` (`Sum.inr`). `Digest.terminal_le` bounds the bad events of a
uniform digest table `hash`; with `hash := terminalDigest S` its premise is that this hash is such
a table, independent of `H_FS` (record (b), part of `[KECCAK]`). -/
noncomputable def terminalDigest : Rq S.innerDegree ⊕ (ℕ → Rq S.innerDegree) → ByteArray
  | .inl e => ringDigest e
  | .inr t => rowsDigestOf S t

theorem toRq_folded (msgs : TerminalMessages) (ch : TerminalChallenges) :
    Poly.toRq S.innerDegree (folded S msgs ch) = ∑ i ∈ Finset.range (S.blocks - 1),
      Poly.toRq S.innerDegree (challenge S ch i) *
        Poly.toRq S.innerDegree (Poly.ring S.innerDegree msgs.e i) := by
  rw [folded, Poly.toRq_sum]
  exact Finset.sum_congr rfl fun i _ => Poly.toRq_mul _ _

theorem toRq_tPartial (msgs : TerminalMessages) (ch : TerminalChallenges) (r : ℕ) :
    Poly.toRq S.innerDegree (tPartial S msgs ch r) = ∑ i ∈ Finset.range (S.blocks - 1),
      Poly.toRq S.innerDegree (challenge S ch i) *
        Poly.toRq S.innerDegree (Poly.ring S.innerDegree msgs.t (i * S.innerRank + r)) := by
  rw [tPartial, Poly.toRq_sum]
  exact Finset.sum_congr rfl fun i _ => Poly.toRq_mul _ _

/-- Row `r` of `A·z` is row `r` of `A_term·z` (Akita (128)). -/
theorem toRq_aRow (setup : Array Fq) (z : Array ℤ) (r : Fin S.innerRank) :
    Poly.toRq S.innerDegree (aRow S setup z r) = (matrixOf S setup).mulVec (zSource S z) r := by
  let g : ℕ → Rq S.innerDegree := fun c => Poly.toRq S.innerDegree (Poly.ring S.innerDegree
    setup (r * (S.blockLength * S.sourceDigits) + c)) * Poly.toRq S.innerDegree (zRing S z c)
  have inner : ∀ x : Fin S.blockLength, ∑ u : Fin S.sourceDigits,
      matrixOf S setup r (x, u) * zSource S z (x, u) =
        ∑ u ∈ Finset.range S.sourceDigits, g (x * S.sourceDigits + u) := fun x =>
    Fin.sum_univ_eq_sum_range (fun u => g (x * S.sourceDigits + u)) S.sourceDigits
  have rows : Poly.toRq S.innerDegree (aRow S setup z r) =
      ∑ c ∈ Finset.range (S.blockLength * S.sourceDigits), g c := by
    unfold aRow
    split
    · rename_i fits
      have small : ∀ c j, (z.getD (c * S.innerDegree + j) 0).natAbs ≤ 2 ^ 15 := fun c j => by
        rw [Array.getD_eq_getD_getElem?]
        cases found : z[c * S.innerDegree + j]? with
        | none => simp
        | some n =>
            have member := Array.mem_of_getElem? found
            have bounded := Array.all_eq_true_iff_forall_mem.mp fits.2 n member
            simpa using bounded
      rw [Poly.toRq_dotSmall fits.1 _ _ small]
      rfl
    · rw [Poly.toRq_sum]
      exact Finset.sum_congr rfl fun c _ => Poly.toRq_mul _ _
  rw [rows, Matrix.mulVec, dotProduct, Fintype.sum_prod_type,
    Finset.sum_congr rfl fun x _ => inner x,
    Fin.sum_univ_eq_sum_range (fun x => ∑ u ∈ Finset.range S.sourceDigits,
      g (x * S.sourceDigits + u)) S.blockLength, sum_range_mul g]

/-- The right side of (129): `⟨a, G_{b,M}·z⟩` with `a_x = ẽq(ρ_pos, x)` in the fixed
subfield. -/
theorem toRq_reduced (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (ch : TerminalChallenges) (z : Array ℤ) :
    Poly.toRq S.innerDegree (reduced S ch z) =
      traceContract (subfield (E := E) quarter).toRingHom
        (fun x : Fin S.blockLength => eqAt ((positionPoint S ch.point).map Ext.into) x)
        (gadget S.sourceBase (zSource S z)) := by
  let g : ℕ → Rq S.innerDegree := fun x =>
    subfield (E := E) quarter (eqAt ((positionPoint S ch.point).map Ext.into) x) *
      ∑ u ∈ Finset.range S.sourceDigits, (S.sourceBase : Rq S.innerDegree) ^ u *
        Poly.toRq S.innerDegree (zRing S z (x * S.sourceDigits + u))
  have term : ∀ x : Fin S.blockLength, (subfield (E := E) quarter).toRingHom
      (eqAt ((positionPoint S ch.point).map Ext.into) x) *
        gadget S.sourceBase (zSource S z) x = g x := fun x => by
    simp only [gadget, g, AlgHom.toRingHom_eq_coe, RingHom.coe_coe]
    congr 1
    exact Fin.sum_univ_eq_sum_range (fun u => (S.sourceBase : Rq S.innerDegree) ^ u *
      Poly.toRq S.innerDegree (zRing S z (x * S.sourceDigits + u))) S.sourceDigits
  rw [traceContract, Finset.sum_congr rfl fun x _ => term x, Fin.sum_univ_eq_sum_range g,
    reduced, Poly.toRq_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Poly.toRq_subMul (E := E) quarter, eqIndex_into, Poly.toRq_sum]
  congr 1
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [Poly.toRq_scale, Fq.toZq_ofNat, Algebra.smul_def, map_natCast, Nat.cast_pow]

theorem unpackAt_into {d m : ℕ} [NeZero d] (quarter : 4 * m = d) (two : ((2 * m : ℕ) : Zq) ≠ 0)
    (Z : Array Fq) {u : ℕ} (small : u < 2 * m) :
    ((unpackAt m Z u).into : E) = unpack d m u (Poly.toRq d Z) := by
  have nonzero : ((m : ℕ) : Zq) ≠ 0 := fun zero => two (by rw [Nat.cast_mul, zero, mul_zero])
  have key : ((2 * m : ℕ) : Zq)⁻¹ * (m : Zq) = (2 : Zq)⁻¹ := by
    rw [Nat.cast_mul, Nat.cast_ofNat, mul_inv, mul_assoc, inv_mul_cancel₀ nonzero, mul_one]
  have slotSmall : slot m u < d := by unfold slot; split <;> omega
  have fixedCoeff : coeff (fixed d m * Poly.toRq d Z) (slot m u) =
      (if slot m u < m then -(Poly.get Z (d + slot m u - m)).toZq
        else (Poly.get Z (slot m u - m)).toZq) -
      (if slot m u < 3 * m then -(Poly.get Z (d + slot m u - 3 * m)).toZq
        else (Poly.get Z (slot m u - 3 * m)).toZq) := by
    rw [fixed, sub_mul, coeff_sub, coeff_root_pow_mul (by omega) _ slotSmall,
      coeff_root_pow_mul (by omega) _ slotSmall]
    congr 1
    · split
      · rw [Poly.coeff_toRq _ (by omega)]
      · rw [Poly.coeff_toRq _ (by omega)]
    · split
      · rw [Poly.coeff_toRq _ (by omega)]
      · rw [Poly.coeff_toRq _ (by omega)]
  refine Degree2.ext ?_ ?_
  · simp only [unpack, LinearMap.smul_apply, LinearMap.comp_apply, LinearMap.mulRight_apply,
      traceForm, LinearMap.coe_mk, AddHom.coe_mk, Degree2.re_smul, Degree2.re_pair]
    rw [coeff_mul_conj_root_pow _ slotSmall, Poly.coeff_toRq _ slotSmall, ← mul_assoc,
      inv_mul_cancel₀ two, one_mul, re_into, unpackAt, slot]
    by_cases low : u < m
    · simp only [low, ite_true]
    · simp only [low, ite_false]
  · simp only [unpack, LinearMap.smul_apply, LinearMap.comp_apply, LinearMap.mulRight_apply,
      traceForm, LinearMap.coe_mk, AddHom.coe_mk, Degree2.im_smul, Degree2.im_pair]
    rw [← mul_assoc (fixed d m), coeff_mul_conj_root_pow _ slotSmall, fixedCoeff, ← mul_assoc,
      key, im_into, unpackAt]
    by_cases low : u < m
    · have shape : slot m u = u := by unfold slot; rw [ite_eq_left low]
      rw [shape, ite_eq_left low, ite_eq_left low, ite_eq_left (by omega), Fq.toZq_mul,
        half_toZq, Fq.toZq_sub, show d + u - m = u + 3 * m by omega,
        show d + u - 3 * m = u + m by omega]
      ring
    · have shape : slot m u = u + m := by unfold slot; rw [ite_eq_right low]
      rw [shape, ite_eq_right low, ite_eq_right (by omega), ite_eq_left (by omega), Fq.toZq_mul,
        half_toZq, Fq.toZq_add, show u + m - m = u by omega,
        show d + (u + m) - 3 * m = u + 2 * m by omega]
      ring

theorem functional_into {d m : ℕ} [NeZero d] (quarter : 4 * m = d)
    (two : ((2 * m : ℕ) : Zq) ≠ 0) (pack : List Ext) (Z : Array Fq) :
    ((functional m pack Z).into : E) =
      packFunctional quarter (eqAt (pack.map Ext.into)) (Poly.toRq d Z) := by
  rw [functional, sumExt_into, packFunctional_apply]
  refine Finset.sum_congr rfl fun u member => ?_
  rw [Ext.into_mul, eqIndex_into, unpackAt_into quarter two Z (Finset.mem_range.mp member)]

/-- The left side of (130) without `θ`: `Σ_i χ_blk(i)·T_{ρ_pack}(e_i)`. -/
theorem into_opening (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (ch : TerminalChallenges)
    (msgs : TerminalMessages) (eLast : Array Fq) :
    ((opening S ch msgs eLast).into : E) = ∑ i ∈ Finset.range S.blocks,
      eqAt ((blockPoint S ch.point).map Ext.into) i *
        packFunctional quarter (eqAt ((packPoint S ch.point).map Ext.into))
          (Poly.toRq S.innerDegree (eFull S msgs eLast i)) := by
  rw [opening, sumExt_into]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Ext.into_mul, eqIndex_into, functional_into quarter two]

variable {S}

theorem sum_partials (blocks : 1 ≤ S.blocks) (msgs : TerminalMessages)
    (ch : TerminalChallenges) (eLast : Rq S.innerDegree) :
    ∑ i, challengesOf S ch i * partialsOf S msgs eLast i =
      Poly.toRq S.innerDegree (folded S msgs ch) +
        Poly.toRq S.innerDegree (challenge S ch (S.blocks - 1)) * eLast := by
  let g : ℕ → Rq S.innerDegree := fun n => Poly.toRq S.innerDegree (challenge S ch n) *
    if n < S.blocks - 1 then Poly.toRq S.innerDegree (Poly.ring S.innerDegree msgs.e n) else eLast
  calc ∑ i, challengesOf S ch i * partialsOf S msgs eLast i = ∑ i : Fin S.blocks, g i := rfl
    _ = ∑ n ∈ Finset.range (S.blocks - 1), g n + g (S.blocks - 1) := sum_fin_last blocks g
    _ = _ := by
      rw [toRq_folded]
      congr 1
      · exact Finset.sum_congr rfl fun n member => by
          simp only [g, ite_eq_left (Finset.mem_range.mp member)]
      · simp only [g, ite_eq_right (Nat.lt_irrefl _)]

theorem sum_state (blocks : 1 ≤ S.blocks) (msgs : TerminalMessages) (ch : TerminalChallenges)
    (tLast : ℕ → Rq S.innerDegree) (r : Fin S.innerRank) :
    (∑ i, challengesOf S ch i • stateOf S msgs tLast i) r =
      Poly.toRq S.innerDegree (tPartial S msgs ch r) +
        Poly.toRq S.innerDegree (challenge S ch (S.blocks - 1)) * tLast r := by
  let g : ℕ → Rq S.innerDegree := fun n => Poly.toRq S.innerDegree (challenge S ch n) *
    if n < S.blocks - 1 then
      Poly.toRq S.innerDegree (Poly.ring S.innerDegree msgs.t (n * S.innerRank + r))
    else tLast r
  calc (∑ i, challengesOf S ch i • stateOf S msgs tLast i) r = ∑ i : Fin S.blocks, g i := by
        rw [Finset.sum_apply]
        rfl
    _ = ∑ n ∈ Finset.range (S.blocks - 1), g n + g (S.blocks - 1) := sum_fin_last blocks g
    _ = _ := by
      rw [toRq_tPartial]
      congr 1
      · exact Finset.sum_congr rfl fun n member => by
          simp only [g, ite_eq_left (Finset.mem_range.mp member)]
      · simp only [g, ite_eq_right (Nat.lt_irrefl _)]

theorem sum_value (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (msgs : TerminalMessages)
    (ch : TerminalChallenges) (eLast : Array Fq) :
    ∑ i : Fin S.blocks, eqAt ((blockPoint S ch.point).map Ext.into) i *
        packFunctional quarter (eqAt ((packPoint S ch.point).map Ext.into))
          (partialsOf S msgs (Poly.toRq S.innerDegree eLast) i) =
      ((opening S ch msgs eLast).into : E) := by
  rw [into_opening S quarter two, ← Fin.sum_univ_eq_sum_range]
  refine Finset.sum_congr rfl fun i _ => ?_
  congr 2
  unfold partialsOf eFull
  split <;> rfl

theorem inverse_mul' {d : ℕ} [NeZero d] {n : ℕ} (power : 2 ^ n = d) {a b : Array Fq}
    (found : Poly.inverse n a = some b) : Poly.toRq d b * Poly.toRq d a = 1 := by
  subst power
  exact Poly.inverse_mul n a b found

theorem inverse_isSome' {d : ℕ} [NeZero d] {n : ℕ} (power : 2 ^ n = d) {a : Array Fq}
    (unit : IsUnit (Poly.toRq d a)) : (Poly.inverse n a).isSome := by
  subst power
  exact Poly.inverse_isSome n a unit

/-- The last-block variant of Akita Section 8.2 for one inverse of the last challenge: the
blocks `eLast` and `tLast` with the digests of the proof and the checks (128), (129) and (130)
exist exactly when the blocks of (131) have these digests and pass (130). -/
theorem lastBlocks_iff (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (blocks : 1 ≤ S.blocks)
    {setup : Array Fq} {msgs : TerminalMessages} {ch : TerminalChallenges} {z : Array ℤ}
    {inv : Array Fq} (inverse : Poly.toRq S.innerDegree inv *
      Poly.toRq S.innerDegree (challenge S ch (S.blocks - 1)) = 1) :
    (∃ (eLast : Rq S.innerDegree) (tLast : ℕ → Rq S.innerDegree),
        ringDigest eLast = msgs.eDigest ∧ rowsDigestOf S tLast = msgs.tDigest ∧
        (statementOf (E := E) S quarter setup msgs ch tLast).Accepts (partialsOf S msgs eLast)
          (challengesOf S ch) (zSource S z)) ↔
      fieldDigest (Poly.canon S.innerDegree (Poly.mul S.innerDegree inv
          (Poly.sub S.innerDegree (reduced S ch z) (folded S msgs ch)))) = msgs.eDigest ∧
      rowsDigest S ((List.range S.innerRank).map fun r => Poly.mul S.innerDegree inv
          (Poly.sub S.innerDegree (aRow S setup z r) (tPartial S msgs ch r))) = msgs.tDigest ∧
      (ch.scale * opening S ch msgs (Poly.mul S.innerDegree inv
          (Poly.sub S.innerDegree (reduced S ch z) (folded S msgs ch)))).equals ch.target = true := by
  let xe := Poly.mul S.innerDegree inv (Poly.sub S.innerDegree (reduced S ch z) (folded S msgs ch))
  let xr : ℕ → Array Fq := fun r =>
    Poly.mul S.innerDegree inv (Poly.sub S.innerDegree (aRow S setup z r) (tPartial S msgs ch r))
  have rows : rowsDigest S ((List.range S.innerRank).map xr) =
      rowsDigestOf S fun r => Poly.toRq S.innerDegree (xr r) := rowsDigest_map S xr
  have value : ∀ eLast, eLast = Poly.toRq S.innerDegree xe →
      ((statementOf (E := E) S quarter setup msgs ch (fun r => 0)).scale *
        ∑ i, (statementOf (E := E) S quarter setup msgs ch (fun r => 0)).block i *
          (statementOf (E := E) S quarter setup msgs ch (fun r => 0)).functional
            (partialsOf S msgs eLast i) =
        (statementOf (E := E) S quarter setup msgs ch (fun r => 0)).value ↔
      (ch.scale * opening S ch msgs xe).equals ch.target = true) := by
    intro eLast same
    rw [Ext.equals_iff (E := E), Ext.into_mul, ← sum_value quarter two msgs ch xe, same]
    rfl
  constructor
  · rintro ⟨eLast, tLast, eDigest, tDigest, accepts⟩
    have evaluation := accepts.evaluation
    rw [sum_partials blocks] at evaluation
    have eSame : eLast = Poly.toRq S.innerDegree xe :=
      (solution_iff S inverse eLast).mp (evaluation.trans (toRq_reduced S quarter ch z).symm)
    have tSame : ∀ r < S.innerRank, tLast r = Poly.toRq S.innerDegree (xr r) := fun r small => by
      have row := congrFun accepts.inner ⟨r, small⟩
      dsimp only [statementOf] at row
      rw [sum_state blocks] at row
      exact (solution_iff S inverse (tLast r)).mp (row.symm.trans (toRq_aRow S setup z ⟨r, small⟩).symm)
    refine ⟨?_, ?_, (value eLast eSame).mp accepts.value⟩
    · rw [← eDigest, eSame, ringDigest, Poly.canon_toRq]
    · rw [rows, ← tDigest, rowsDigestOf, rowsDigestOf]
      congr 1
      refine List.foldl_ext _ _ _ fun out r member => ?_
      rw [tSame r (List.mem_range.mp member)]
  · rintro ⟨eDigest, tDigest, trace⟩
    refine ⟨Poly.toRq S.innerDegree xe, fun r => Poly.toRq S.innerDegree (xr r), ?_, ?_, ?_⟩
    · rw [ringDigest, Poly.canon_toRq]
      exact eDigest
    · rw [← rows]
      exact tDigest
    · refine ⟨?_, ?_, (value _ rfl).mpr trace⟩
      · funext r
        dsimp only [statementOf]
        rw [sum_state blocks, ← toRq_aRow S setup z r]
        exact ((solution_iff S inverse _).mpr rfl).symm
      · show ∑ i, challengesOf S ch i * partialsOf S msgs (Poly.toRq S.innerDegree xe) i = _
        dsimp only [statementOf]
        rw [sum_partials blocks, ← toRq_reduced S quarter ch z]
        exact (solution_iff S inverse _).mpr rfl

/-- The checks of the terminal on a decoded `z` accept exactly when `‖z‖_2² ≤ S_max`, the fold
challenges are valid, the last one is a unit, and some last blocks of `e` and `t` with the digests
of the proof make the statement of the terminal accept the full partials, the challenges and `z`
(Akita Section 8.2, (128) to (131)). -/
theorem terminalChecks_ok_iff (power : 2 ^ Nat.log 2 S.innerDegree = S.innerDegree)
    (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (blocks : 1 ≤ S.blocks) (cap : ℕ)
    (setup : Array Fq) (msgs : TerminalMessages) (ch : TerminalChallenges) (z : Array ℤ) :
    (∃ out, terminalChecks S cap setup msgs ch z = .ok out) ↔
      centeredNormSq z ≤ cap ∧
      (ch.fold.length = S.blocks ∧ ch.fold.all (sparseValid S.innerDegree) = true) ∧
      IsUnit (Poly.toRq S.innerDegree (challenge S ch (S.blocks - 1))) ∧
      ∃ (eLast : Rq S.innerDegree) (tLast : ℕ → Rq S.innerDegree),
        ringDigest eLast = msgs.eDigest ∧ rowsDigestOf S tLast = msgs.tDigest ∧
        (statementOf (E := E) S quarter setup msgs ch tLast).Accepts (partialsOf S msgs eLast)
          (challengesOf S ch) (zSource S z) := by
  unfold terminalChecks
  by_cases norm : centeredNormSq z ≤ cap
  swap
  · rw [ite_eq_right norm]
    simp [norm]
  rw [ite_eq_left norm]
  by_cases valid : ch.fold.length = S.blocks ∧ ch.fold.all (sparseValid S.innerDegree) = true
  swap
  · rw [ite_eq_right valid]
    simp only [reduceCtorEq, exists_false, false_iff, not_and]
    exact fun _ valid' => absurd valid' valid
  rw [ite_eq_left valid]
  rcases found : Poly.inverse (Nat.log 2 S.innerDegree) (challenge S ch (S.blocks - 1)) with
    _ | inv
  · simp only [reduceCtorEq, exists_false, false_iff, not_and]
    intro _ _ unit
    have some := inverse_isSome' power unit
    rw [found] at some
    cases some
  have inverse := inverse_mul' power found
  have unit : IsUnit (Poly.toRq S.innerDegree (challenge S ch (S.blocks - 1))) :=
    ⟨⟨_, _, by rw [mul_comm]; exact inverse, inverse⟩, rfl⟩
  have rowsSome : tLastOf S msgs ch inv ((List.range S.innerRank).map (aRow S setup z)) =
      some ((List.range S.innerRank).map fun r => Poly.mul S.innerDegree inv
        (Poly.sub S.innerDegree (aRow S setup z r) (tPartial S msgs ch r))) := by
    refine mapM_some _ _ _ fun r member => ?_
    rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_range (List.mem_range.mp member), Option.map_some, Option.getD_some]
    exact recover_some S inverse
  simp only [recover_some S inverse, rowsSome]
  rw [lastBlocks_iff quarter two blocks inverse]
  simp only [norm, valid, unit, true_and]
  by_cases eDigest : fieldDigest (Poly.canon S.innerDegree (Poly.mul S.innerDegree inv
      (Poly.sub S.innerDegree (reduced S ch z) (folded S msgs ch)))) = msgs.eDigest
  · by_cases tDigest : rowsDigest S ((List.range S.innerRank).map fun r =>
        Poly.mul S.innerDegree inv (Poly.sub S.innerDegree (aRow S setup z r)
          (tPartial S msgs ch r))) = msgs.tDigest
    · by_cases trace : (ch.scale * opening S ch msgs (Poly.mul S.innerDegree inv
          (Poly.sub S.innerDegree (reduced S ch z) (folded S msgs ch)))).equals ch.target = true
      · simp [eDigest, tDigest, trace]
      · simp [eDigest, tDigest, trace]
    · simp [eDigest, tDigest]
  · simp [eDigest]

/-- The terminal of the code on a payload that decodes: `terminalChecks_ok_iff` for the
decoded `z`. -/
theorem terminalRun_ok_iff (power : 2 ^ Nat.log 2 S.innerDegree = S.innerDegree)
    (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (blocks : 1 ≤ S.blocks) (cap : ℕ)
    (P : Golomb.Params) (setup : Array Fq) (msgs : TerminalMessages) (ch : TerminalChallenges)
    {z : Array ℤ} (decoded : Golomb.decode P msgs.payload = some z) :
    (∃ out, terminalRun S cap P setup msgs ch = .ok out) ↔
      centeredNormSq z ≤ cap ∧
      (ch.fold.length = S.blocks ∧ ch.fold.all (sparseValid S.innerDegree) = true) ∧
      IsUnit (Poly.toRq S.innerDegree (challenge S ch (S.blocks - 1))) ∧
      ∃ (eLast : Rq S.innerDegree) (tLast : ℕ → Rq S.innerDegree),
        ringDigest eLast = msgs.eDigest ∧ rowsDigestOf S tLast = msgs.tDigest ∧
        (statementOf (E := E) S quarter setup msgs ch tLast).Accepts (partialsOf S msgs eLast)
          (challengesOf S ch) (zSource S z) := by
  rw [terminalRun, decoded]
  exact terminalChecks_ok_iff power quarter two blocks cap setup msgs ch z

end Values

end NightstreamFPrime.Spec.Compression.Akita.Replay
