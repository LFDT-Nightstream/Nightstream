import NightstreamFPrime.Spec.FieldExtension
import NightstreamFPrime.Spec.Compression.Akita.FoldCheck
import NightstreamFPrime.Spec.Compression.Akita.Trace

/-! The tensor reduction of Akita for one claim and the extension degree `k = 2`, as a game
(ePrint 2026/1983, revision of 2026-10-15, Section 3.7, Section 5.2 with (54), Theorem 5.1 with
its proof in Appendix D.1, and Figure 4, step 0).

The reduction has `κ = 1` head variable, which is the first variable of the point, and the basis
`(β_0, β_1) = (1, u)` of `E = Z_q[u]/(u² − 2)`.

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
  avoids the bad sets, a true forwarded claim `ṽ = θ·f̄(ρ)` gives the input claim. -/

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

/-- The packed polynomial `f̄` of Akita Section 3.7 for `κ = 1` and the basis `(1, u)`:
`f̄(x) = w(0, x) + w(1, x)·u`, with the head variable on the lowest bit of the cell. -/
def packed (w : Table) (x : ℕ) : Quadratic := ⟨w (2 * x), w (2 * x + 1)⟩

/-- The table of `Z_q` of a packed polynomial: cell `2·x + y` is coordinate `y` of value `x`. -/
def flatten (g : ℕ → Quadratic) : Table :=
  fun cell => if cell % 2 = 0 then (g (cell / 2)).re else (g (cell / 2)).im

theorem packed_flatten (g : ℕ → Quadratic) : packed (flatten g) = g := by
  funext x
  have even : 2 * x % 2 = 0 := by omega
  have odd : ¬ (2 * x + 1) % 2 = 0 := by omega
  simp only [packed, flatten, even, odd, ite_true, ite_false,
    show 2 * x / 2 = x by omega, show (2 * x + 1) / 2 = x by omega]

/-- The batched equality factor `A_η(x) = (1 − η)·A_0(x) + η·A_1(x)` of Akita Section 3.7, for
the coordinates `ẽq(r_tail, x) = A_0(x) + A_1(x)·u`. -/
noncomputable def tensorWeight (tail : List Quadratic) (η : Quadratic) (x : ℕ) : Quadratic :=
  (1 - η) * algebraMap Zq Quadratic (eqAt tail x).re +
    η * algebraMap Zq Quadratic (eqAt tail x).im

/-- The tables of the sum (54) of Akita for one claim: `f̄` and `A_η`. -/
noncomputable def tensorTables (w : Table) (tail : List Quadratic) (η : Quadratic) :
    Fin 2 → ℕ → Quadratic :=
  ![packed w, tensorWeight tail η]

/-- The row partial `row_u = S_{u,0} + S_{u,1}·u` of the column partials `S_y = S_{0,y} + S_{1,y}·u`
(Akita Section 3.7). -/
def tensorRow (columns : Quadratic × Quadratic) (u : Bool) : Quadratic :=
  if u then ⟨columns.1.im, columns.2.im⟩ else ⟨columns.1.re, columns.2.re⟩

/-- The row partial `Σ_x A_u(x)·f̄(x)` of an honest prover (Akita Section 3.7, (⋆)). -/
noncomputable def tensorSum (ν : ℕ) (w : Table) (tail : List Quadratic) (u : Bool) : Quadratic :=
  cubeSum ν fun x => packed w x *
    algebraMap Zq Quadratic (if u then (eqAt tail x).im else (eqAt tail x).re)

/-- The verifier of the tensor reduction for the table `w` and a claim `w̃(r_head, r_tail) = v`
with `ν` tail variables (Akita Sections 3.7 and 5.2, one claim, `κ = 1`). It ends in `none` when
a check fails, and otherwise in the point `ρ`, the factor `θ = A_η(ρ)` and the final product
`ṽ`, which is the claim `ṽ = θ·f̄(ρ)`. -/
noncomputable def tensorCheck (ν : ℕ) (w : Table) :
    List Quadratic → Quadratic → Game (Option (List Quadratic × Quadratic × Quadratic))
  | [], _ => .pure none
  | head :: tail, v =>
      if tail.length = ν then
        .recv (Quadratic × Quadratic) fun columns =>
          if v = (1 - head) * columns.1 + head * columns.2 then
            .draw Quadratic (zeros (linePoly
              (tensorRow columns false - tensorSum ν w tail false)
              (tensorRow columns true - tensorSum ν w tail true -
                (tensorRow columns false - tensorSum ν w tail false)))) fun η =>
              (sumcheck 2 (combRound ν (tensorTables w tail η) productComb) ν []
                ((1 - η) * tensorRow columns false + η * tensorRow columns true)).bind fun
                | none => .pure none
                | some (ρ, final) => .pure (some (ρ, mle ρ (tensorWeight tail η), final))
          else .pure none
      else .pure none

private def reHom : Quadratic →+ Zq := ⟨⟨fun z => z.re, rfl⟩, fun _ _ => rfl⟩

private def imHom : Quadratic →+ Zq := ⟨⟨fun z => z.im, rfl⟩, fun _ _ => rfl⟩

/-- A column partial of an honest prover from the row partials: when the two row partials are
`Σ_x A_u(x)·f̄(x)`, the column partial `S_y` is `w̃(y, r_tail)`. -/
theorem column_of_rows {ν : ℕ} {w : Table} {tail : List Quadratic}
    {columns : Quadratic × Quadratic} (length : tail.length = ν)
    (first : tensorRow columns false = tensorSum ν w tail false)
    (second : tensorRow columns true = tensorSum ν w tail true) :
    columns.1 = mle tail (fun x => lifted Quadratic w (2 * x)) ∧
      columns.2 = mle tail (fun x => lifted Quadratic w (2 * x + 1)) := by
  have firstRe := congrArg reHom first
  have firstIm := congrArg imHom first
  have secondRe := congrArg reHom second
  have secondIm := congrArg imHom second
  simp only [tensorSum, map_cubeSum] at firstRe firstIm secondRe secondIm
  rw [mle_eq_cubeSum, mle_eq_cubeSum, length]
  constructor
  · refine QuadraticAlgebra.ext ?_ ?_
    · refine (firstRe.trans ?_).trans (map_cubeSum reHom _ _).symm
      congr 1
      funext x
      simp [reHom, packed, lifted]
      ring
    · refine (secondRe.trans ?_).trans (map_cubeSum imHom _ _).symm
      congr 1
      funext x
      simp [reHom, imHom, packed, lifted]
      ring
  · refine QuadraticAlgebra.ext ?_ ?_
    · refine (firstIm.trans ?_).trans (map_cubeSum reHom _ _).symm
      congr 1
      funext x
      simp [reHom, imHom, packed, lifted]
      ring
    · refine (secondIm.trans ?_).trans (map_cubeSum imHom _ _).symm
      congr 1
      funext x
      simp [imHom, packed, lifted]
      ring

/-- The claim of the sum-check (54) for one claim: `Σ_x f̄(x)·A_η(x) = (1 − η)·row_0 + η·row_1`
for the row partials of an honest prover. -/
theorem tensorTruth_nil (ν : ℕ) (w : Table) (tail : List Quadratic) (η : Quadratic) :
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
theorem tensorCheck_avoids (ν : ℕ) (w : Table) (r : List Quadratic) (v : Quadratic) :
    (tensorCheck ν w r v).Avoids fun result =>
      ∀ ρ θ value, result = some (ρ, θ, value) → value = θ * mle ρ (packed w) →
        v = mle r (lifted Quadratic w) := by
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

end NightstreamFPrime.Spec.Compression.Akita
