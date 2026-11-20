import NightstreamFPrime.Spec.FieldExtension
import NightstreamFPrime.Spec.Compression.Akita.FoldCheck
import NightstreamFPrime.Spec.Compression.Game.Complete
import NightstreamFPrime.Spec.Compression.Akita.Trace

/-! The tensor reduction of Akita for one claim and the extension degree `k = 2`, as a game
(ePrint 2026/1983, revision of 2026-10-15, Section 3.7, Section 5.2 with (54), Theorem 5.1 with
its proof in Appendix D.1, and Figure 4, step 0).

The reduction has `κ = 1` head variable, which is the first variable of the point, and the basis
`(β_0, β_1) = (1, u)` of a field `E` with `Degree2 E`, for example `Z_q[u]/(u² − 2)`.

- `packed w` is the packed polynomial `f̄(x) = w(0, x)·β_0 + w(1, x)·β_1` of a table `w` of
  `Z_q`, and `flatten` is its inverse.
- `tensorWeight` is the batched equality factor `A_η(x) = (1 − η)·A_0(x) + η·A_1(x)`, for the
  coordinates `ẽq(r_tail, x) = A_0(x)·β_0 + A_1(x)·β_1`.
- `tensorCheck ν w` is the verifier for a claim `w̃(r_head, r_tail) = v` with `ν` tail variables.
  It receives the column partials `S_y`, checks `v = (1 − r_head)·S_0 + r_head·S_1`, draws `η`,
  and runs the sum-check of degree 2 for `c_η = Σ_x f̄(x)·A_η(x)`. It ends in the point `ρ`, the
  factor `θ = A_η(ρ)` and the final product `ṽ` of the sum-check. The bad sets are the root of the
  row discrepancy in `η` and the roots of each round.
- `tensorCheck_avoids` is the soundness part of Akita Theorem 5.1 for one claim: on a path that
  avoids the bad sets, a true forwarded claim `ṽ = θ·f̄(ρ)` gives the input claim.
- `tensorCheck_winsTo` and `tensorCheck_complete` are the completeness part: for a true input
  claim, the honest prover reaches a true forwarded claim for each draw sequence. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- 2 is not a square of `Z_q`, so `Quadratic` is a field. -/
scoped instance twoNonsquare : Fact (¬ IsSquare (2 : Zq)) := by
  have norm : ∀ real imaginary : Zq, real * real - 2 * imaginary * imaginary = 0 →
      real = 0 ∧ imaginary = 0 := FieldExtension.twoProjectiveNonresidue
  exact ⟨fun ⟨root, square⟩ =>
    one_ne_zero (α := Zq) (norm root 1 (by rw [← square]; ring)).2⟩

noncomputable scoped instance quadraticFintype : Fintype Quadratic :=
  Fintype.ofEquiv _ (QuadraticAlgebra.equivProd (2 : Zq) 0).symm

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The packed polynomial `f̄` of Akita Section 3.7 for `κ = 1` and the basis `(1, u)`:
`f̄(x) = w(0, x) + w(1, x)·u`, with the head variable on the lowest bit of the cell. -/
def packed (w : Table) (x : ℕ) : E := Degree2.pair (w (2 * x)) (w (2 * x + 1))

/-- The table of `Z_q` of a packed polynomial: cell `2·x + y` is coordinate `y` of value `x`. -/
def flatten (g : ℕ → E) : Table :=
  fun cell => if cell % 2 = 0 then Degree2.re (g (cell / 2)) else Degree2.im (g (cell / 2))

omit [Fintype E] [DecidableEq E] in
theorem packed_flatten (g : ℕ → E) : packed (flatten g) = g := by
  funext x
  have even : 2 * x % 2 = 0 := by omega
  have odd : ¬ (2 * x + 1) % 2 = 0 := by omega
  simp only [packed, flatten, even, odd, ite_true, ite_false,
    show 2 * x / 2 = x by omega, show (2 * x + 1) / 2 = x by omega, Degree2.pair_re_im]

/-- The batched equality factor `A_η(x) = (1 − η)·A_0(x) + η·A_1(x)` of Akita Section 3.7, for
the coordinates `ẽq(r_tail, x) = A_0(x) + A_1(x)·u`. -/
noncomputable def tensorWeight (tail : List E) (η : E) (x : ℕ) : E :=
  (1 - η) * algebraMap Zq E (Degree2.re (eqAt tail x)) +
    η * algebraMap Zq E (Degree2.im (eqAt tail x))

/-- The tables of the sum (54) of Akita for one claim: `f̄` and `A_η`. -/
noncomputable def tensorTables (w : Table) (tail : List E) (η : E) :
    Fin 2 → ℕ → E :=
  ![packed w, tensorWeight tail η]

/-- The row partial `row_u = S_{u,0} + S_{u,1}·u` of the column partials `S_y = S_{0,y} + S_{1,y}·u`
(Akita Section 3.7). -/
def tensorRow (columns : E × E) (u : Bool) : E :=
  if u then Degree2.pair (Degree2.im columns.1) (Degree2.im columns.2)
  else Degree2.pair (Degree2.re columns.1) (Degree2.re columns.2)

/-- The row partial `Σ_x A_u(x)·f̄(x)` of an honest prover (Akita Section 3.7, (⋆)). -/
noncomputable def tensorSum (ν : ℕ) (w : Table) (tail : List E) (u : Bool) : E :=
  cubeSum ν fun x => packed w x *
    algebraMap Zq E (if u then Degree2.im (eqAt tail x) else Degree2.re (eqAt tail x))

/-- The verifier of the tensor reduction for the table `w` and a claim `w̃(r_head, r_tail) = v`
with `ν` tail variables (Akita Sections 3.7 and 5.2, one claim, `κ = 1`). It ends in `none` when
a check fails, and otherwise in the point `ρ`, the factor `θ = A_η(ρ)` and the final product
`ṽ`, which is the claim `ṽ = θ·f̄(ρ)`. -/
noncomputable def tensorCheck (ν : ℕ) (w : Table) :
    List E → E → Game (Option (List E × E × E))
  | [], _ => .pure none
  | head :: tail, v =>
      if tail.length = ν then
        .recv (E × E) fun columns =>
          if v = (1 - head) * columns.1 + head * columns.2 then
            .draw E (zeros (linePoly
              (tensorRow columns false - tensorSum ν w tail false)
              (tensorRow columns true - tensorSum ν w tail true -
                (tensorRow columns false - tensorSum ν w tail false)))) fun η =>
              (sumcheck 2 (combRound ν (tensorTables w tail η) productComb) ν []
                ((1 - η) * tensorRow columns false + η * tensorRow columns true)).bind fun
                | none => .pure none
                | some (ρ, final) => .pure (some (ρ, mle ρ (tensorWeight tail η), final))
          else .pure none
      else .pure none

private def reHom : E →+ Zq := ⟨⟨Degree2.re, Degree2.re_zero⟩, Degree2.re_add⟩

private def imHom : E →+ Zq := ⟨⟨Degree2.im, Degree2.im_zero⟩, Degree2.im_add⟩

omit [Fintype E] [DecidableEq E] in
private theorem tensorRow_false (columns : E × E) :
    tensorRow columns false =
      Degree2.pair (Degree2.re columns.1) (Degree2.re columns.2) :=
  rfl

omit [Fintype E] [DecidableEq E] in
private theorem tensorRow_true (columns : E × E) :
    tensorRow columns true = Degree2.pair (Degree2.im columns.1) (Degree2.im columns.2) :=
  rfl

omit [Fintype E] [DecidableEq E] in
/-- A column partial of an honest prover from the row partials: when the two row partials are
`Σ_x A_u(x)·f̄(x)`, the column partial `S_y` is `w̃(y, r_tail)`. -/
theorem column_of_rows {ν : ℕ} {w : Table} {tail : List E}
    {columns : E × E} (length : tail.length = ν)
    (first : tensorRow columns false = tensorSum ν w tail false)
    (second : tensorRow columns true = tensorSum ν w tail true) :
    columns.1 = mle tail (fun x => lifted E w (2 * x)) ∧
      columns.2 = mle tail (fun x => lifted E w (2 * x + 1)) := by
  have firstRe := congrArg reHom first
  have firstIm := congrArg imHom first
  have secondRe := congrArg reHom second
  have secondIm := congrArg imHom second
  rw [tensorRow_false] at firstRe firstIm
  rw [tensorRow_true] at secondRe secondIm
  replace firstRe := (Degree2.re_pair _ _).symm.trans firstRe
  replace firstIm := (Degree2.im_pair _ _).symm.trans firstIm
  replace secondRe := (Degree2.re_pair _ _).symm.trans secondRe
  replace secondIm := (Degree2.im_pair _ _).symm.trans secondIm
  simp only [tensorSum, map_cubeSum] at firstRe firstIm secondRe secondIm
  rw [mle_eq_cubeSum, mle_eq_cubeSum, length]
  constructor
  · refine Degree2.ext ?_ ?_
    · refine (firstRe.trans ?_).trans (map_cubeSum reHom _ _).symm
      congr 1
      funext x
      simp [reHom, packed, lifted, Degree2.re_mul, Degree2.re_algebraMap, Degree2.im_algebraMap]
      ring
    · refine (secondRe.trans ?_).trans (map_cubeSum imHom _ _).symm
      congr 1
      funext x
      simp [reHom, imHom, packed, lifted, Degree2.re_mul, Degree2.im_mul, Degree2.re_algebraMap,
        Degree2.im_algebraMap]
      ring
  · refine Degree2.ext ?_ ?_
    · refine (firstIm.trans ?_).trans (map_cubeSum reHom _ _).symm
      congr 1
      funext x
      simp [reHom, imHom, packed, lifted, Degree2.re_mul, Degree2.im_mul, Degree2.re_algebraMap,
        Degree2.im_algebraMap]
      ring
    · refine (secondIm.trans ?_).trans (map_cubeSum imHom _ _).symm
      congr 1
      funext x
      simp [imHom, packed, lifted, Degree2.im_mul, Degree2.re_algebraMap, Degree2.im_algebraMap]
      ring

omit [Fintype E] [DecidableEq E] in
/-- The claim of the sum-check (54) for one claim: `Σ_x f̄(x)·A_η(x) = (1 − η)·row_0 + η·row_1`
for the row partials of an honest prover. -/
theorem tensorTruth_nil (ν : ℕ) (w : Table) (tail : List E) (η : E) :
    combTruth ν (tensorTables w tail η) productComb [] =
      (1 - η) * tensorSum ν w tail false + η * tensorSum ν w tail true := by
  rw [product_truth_nil, tensorSum, tensorSum, ← cubeSum_mul_left, ← cubeSum_mul_left,
    ← cubeSum_add]
  congr 1
  funext x
  simp only [tensorTables, tensorWeight, Matrix.cons_val_zero, Matrix.cons_val_one,
    Bool.false_eq_true, ite_false, ite_true]
  ring

/-- The soundness part of Akita Theorem 5.1 for one claim and `κ = 1`: on a path of the tensor
reduction that avoids the bad sets, a forwarded claim that is true for the packed polynomial of
the table, `ṽ = θ·f̄(ρ)`, gives the input claim `v = w̃(r)`. -/
theorem tensorCheck_avoids (ν : ℕ) (w : Table) (r : List E) (v : E) :
    (tensorCheck ν w r v).Avoids fun result =>
      ∀ ρ θ value, result = some (ρ, θ, value) → value = θ * mle ρ (packed w) →
        v = mle r (lifted E w) := by
  rcases r with _ | ⟨head, tail⟩
  · intro ρ θ value equal
    simp at equal
  change Game.Avoids (ite (tail.length = ν) _ _) _
  split
  swap
  · intro ρ θ value equal
    simp at equal
  rename_i length
  intro columns
  dsimp only
  split
  swap
  · intro ρ θ value equal
    simp at equal
  rename_i checked
  intro η outside
  refine (avoids_bind _ _ _).mpr ?_
  refine ((sumcheck_avoids 2 ν _ _ (combRound_truth ν 2 (tensorTables w tail η) productComb
    productComb_degree) ν [] _ (by simp)).and_ends (sumcheck_ends 2 _ ν [] _)).mono ?_
  rintro (_ | ⟨ρ, final⟩) ⟨sound, ends⟩
  · intro ρ θ value equal
    simp at equal
  intro ρ' θ' value' equal opening
  simp only [Option.some.injEq, Prod.mk.injEq] at equal
  obtain ⟨rfl, rfl, rfl⟩ := equal
  have count : ρ.length = ν := by
    obtain ⟨cs, prefixed, count⟩ := ends ρ final rfl
    rw [prefixed]
    simpa using count
  have claim := not_imp_not.mp (sound ρ final rfl) (by
    have last := product_truth_final ρ (tensorTables w tail η)
    rw [count] at last
    rw [last, opening, mul_comm]
    rfl)
  rw [tensorTruth_nil] at claim
  have root : tensorRow columns false - tensorSum ν w tail false +
      η * (tensorRow columns true - tensorSum ν w tail true -
        (tensorRow columns false - tensorSum ν w tail false)) = 0 := by
    linear_combination claim
  obtain ⟨first, second⟩ := eq_zero_of_not_mem_zeros_linePoly outside root
  obtain ⟨left, right⟩ := column_of_rows length (sub_eq_zero.mp first)
    (sub_eq_zero.mp (by rw [first, sub_zero] at second; exact second))
  rw [checked, left, right, mle_cons, ← mle_mul_left, ← mle_mul_left, ← mle_add]
  rfl

omit [Fintype E] [DecidableEq E] in
/-- The row partials of the column partials `S_y = w̃(y, r_tail)` of an honest prover are
`Σ_x A_u(x)·f̄(x)` (Akita Section 3.7, (⋆)): the converse of `column_of_rows`. -/
theorem rows_of_columns {ν : ℕ} (w : Table) {tail : List E} (length : tail.length = ν)
    (u : Bool) :
    tensorRow (mle tail (fun x => lifted E w (2 * x)),
      mle tail (fun x => lifted E w (2 * x + 1))) u = tensorSum ν w tail u := by
  rw [mle_eq_cubeSum, mle_eq_cubeSum, length]
  cases u
  · refine Degree2.ext ?_ ?_
    · rw [tensorRow_false, Degree2.re_pair]
      show reHom (cubeSum ν _) = reHom (cubeSum ν _)
      rw [map_cubeSum, map_cubeSum]
      congr 1
      funext x
      simp [reHom, packed, lifted, Degree2.re_mul, Degree2.re_algebraMap, Degree2.im_algebraMap]
      ring
    · rw [tensorRow_false, Degree2.im_pair]
      show reHom (cubeSum ν _) = imHom (cubeSum ν _)
      rw [map_cubeSum, map_cubeSum]
      congr 1
      funext x
      simp [reHom, imHom, packed, lifted, Degree2.re_mul, Degree2.im_mul, Degree2.re_algebraMap,
        Degree2.im_algebraMap]
      ring
  · refine Degree2.ext ?_ ?_
    · rw [tensorRow_true, Degree2.re_pair]
      show imHom (cubeSum ν _) = reHom (cubeSum ν _)
      rw [map_cubeSum, map_cubeSum]
      congr 1
      funext x
      simp [reHom, imHom, packed, lifted, Degree2.re_mul, Degree2.im_mul, Degree2.re_algebraMap,
        Degree2.im_algebraMap]
      ring
    · rw [tensorRow_true, Degree2.im_pair]
      show imHom (cubeSum ν _) = imHom (cubeSum ν _)
      rw [map_cubeSum, map_cubeSum]
      congr 1
      funext x
      simp [imHom, packed, lifted, Degree2.im_mul, Degree2.re_algebraMap, Degree2.im_algebraMap]
      ring

/-- The completeness part of Akita Theorem 5.1 for one claim and `κ = 1`: for a true input claim
`v = w̃(r_head, r_tail)`, the prover sends the column partials `S_y = w̃(y, r_tail)` and the true
round polynomials, and the tensor reduction ends in a point `ρ` of `ν` coordinates, a factor `θ`
and the forwarded claim `θ·f̄(ρ)`, after `ν + 1` draws. -/
theorem tensorCheck_winsTo (fits : Type → Prop) (columns : fits (E × E))
    (quadratic : fits (Fin (2 + 1) → E)) {ν : ℕ} (w : Table) (head : E)
    {tail : List E} (length : tail.length = ν) :
    WinsTo fits (tensorCheck ν w (head :: tail) (mle (head :: tail) (lifted E w)))
      (0 + ν + 1) true fun result _ _ =>
        ∃ (ρ : List E) (θ : E),
          ρ.length = ν ∧ result = some (ρ, θ, θ * mle ρ (packed w)) := by
  change WinsTo fits (ite (tail.length = ν) _ _) _ _ _
  rw [ite_eq_left length]
  have split : mle (head :: tail) (lifted E w) =
      (1 - head) * mle tail (fun x => lifted E w (2 * x)) +
        head * mle tail (fun x => lifted E w (2 * x + 1)) := by
    rw [← mle_mul_left, ← mle_mul_left, ← mle_add]
    rfl
  refine ⟨rfl, columns, (mle tail (fun x => lifted E w (2 * x)),
    mle tail (fun x => lifted E w (2 * x + 1))), ?_⟩
  dsimp only
  rw [ite_eq_left split]
  intro η
  refine (winsTo_bind fits _ _ _ _ _).mpr (sumcheck_winsTo fits
    (combRound_truth ν 2 (tensorTables w tail η) productComb productComb_degree) quadratic 0 _ ν
    [] _ (by simp) ?_ fun final count => ?_)
  · rw [tensorTruth_nil, rows_of_columns w length, rows_of_columns w length]
  · simp only [List.length_nil, zero_add] at count
    have last := product_truth_final final (tensorTables w tail η)
    rw [count] at last
    refine ⟨final, mle final (tensorWeight tail η), count, ?_⟩
    rw [last, mul_comm]
    rfl

/-- Perfect completeness of the tensor reduction for one claim: for a true input claim, a
strategy of the prover reaches a forwarded claim `ṽ = θ·f̄(ρ)` that is true for the packed
polynomial with probability 1. -/
theorem tensorCheck_complete {ν : ℕ} (w : Table) (head : E) {tail : List E}
    (length : tail.length = ν) :
    ∃ strategy : Strategy (Option (List E × E × E)),
      prob strategy (tensorCheck ν w (head :: tail) (mle (head :: tail) (lifted E w)))
        (fun result => ∃ (ρ : List E) (θ : E),
          ρ.length = ν ∧ result = some (ρ, θ, θ * mle ρ (packed w))) = 1 :=
  ⟨_, prob_honest (fun _ => True)
    (tensorCheck_winsTo (fun _ => True) trivial trivial w head length) fun _ _ _ ended => ended⟩

end NightstreamFPrime.Spec.Compression.Akita
