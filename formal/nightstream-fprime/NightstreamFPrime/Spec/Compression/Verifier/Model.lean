import NightstreamFPrime.Spec.Compression.Verifier.Slice
import NightstreamFPrime.Spec.Compression.Algebra.Batch
import NightstreamFPrime.Spec.Compression.Algebra.Key
import NightstreamFPrime.Spec.Compression.Lib.LogUp
import Mathlib.Algebra.MvPolynomial.CommRing

/-! The argument of Rust `ns-compress` as data (SPEC.md section 6).

`Params` holds what the verifier knows before a proof: the shapes and the layout of the claims,
the honest Ajtai key, and the matrix weights `matrixForm` of each claim, which this unit keeps
abstract. The prover names the values of the claims (`Statement`), commits the table `W` of the
witnesses and the counts, and later the table `V` of the shared inverses. Both tables are base-field
tables. One draw of the transcript gives the lookup challenges `τ` (`Fin L → F`), and one gives
`(μ, ω, δ, γ)` and `(ψ, ξ, λ, β)` (`Challenges`). Rust draws one `ξ` and one `λ` for each claim with
a table. This model draws one for each claim and reads only those of the claims with a table,
which does not change the law of the values that it reads.

`Relation` is what soundness gives for each claim: its slice of `W` satisfies each ring equation
(the `κ` Ajtai rows of its commitment, `Eval_K`, and each `Eval_A` relative to `matrixForm`), its
public input, and, with a table, the bound `[−bound, bound]` on each cell of its slice. -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra

/-- The public shape of one claim (Rust `layout::Cells` with the claim's sizes). With `lookup`,
the argument proves the bound `[−bound, bound]` with the counts in `counts`. -/
structure Shape where
  blocks : ℕ
  witness : Slice
  publicCells : ℕ
  lookup : Bool
  bound : ℕ
  counts : Slice

/-- What the verifier knows before a proof. -/
structure Params (F K : Type) where
  claims : ℕ
  shape : Fin claims → Shape
  /-- The variables of `W` (Rust `Layout::witness_vars`). -/
  witnessVars : ℕ
  /-- The witnesses with a table are below `2^spanVars` (Rust `Layout::table_witness_vars`). -/
  spanVars : ℕ
  /-- `L`, the number of lookup challenges (Rust `LOOKUPS`). -/
  lookups : ℕ
  /-- `D = 2^laneVars` lanes of a ring block (Rust `LANE_VARS`). -/
  laneVars : ℕ
  /-- `κ`, the rows of the Ajtai key, below `2^rowVars` (Rust `KAPPA`, `ROW_VARS`). -/
  rows : ℕ
  rowVars : ℕ
  /-- The key polynomial has `2^keyBlockVars` blocks of each row. -/
  keyBlockVars : ℕ
  /-- Lane `h` of key element `(i, j)`. -/
  key : ℕ → ℕ → ℕ → F
  /-- `t`, the matrices of each claim. -/
  matrices : ℕ
  /-- Lane `h` of the key element of block `j` in `Eval_A` of matrix `m`, for the point `r`. -/
  matrixForm : Fin claims → List K → ℕ → ℕ → ℕ → K

/-- The values of one claim that the prover names (Rust `CeClaim`): the commitment (row, lane),
the public input, the point `r`, `Eval_K` and each `Eval_A` (lanes). -/
structure ClaimValues (F K : Type) where
  commitment : ℕ → ℕ → F
  publicInput : ℕ → F
  point : List K
  evalK : ℕ → K
  evalA : ℕ → ℕ → K

instance {F K : Type} [Zero F] [Zero K] : Nonempty (ClaimValues F K) :=
  ⟨⟨fun _ _ => 0, fun _ => 0, [], fun _ => 0, fun _ _ => 0⟩⟩

variable {F K : Type} [Field F] [Field K] [Algebra F K]

/-- The values of every claim. -/
abbrev Statement (P : Params F K) := Fin P.claims → ClaimValues F K

/-- `(μ, ω, δ, γ)`: the batches of the claims, of the equations, the point of the ring equations
and the batch of the public cells. -/
abbrev RingDraw (K : Type) := K × K × K × K

/-- `(ψ, (ξ, λ), β)`: the batch of the lookups, the weights of the inverse checks and of the sums,
and the point of the inverse checks. -/
abbrev LookupDraw (K : Type) (L s span : ℕ) := (Fin L → K) × ((Fin s → K) × (Fin s → K)) × (Fin span → K)

/-- The draw after the commitment of `V` (Rust `challenges`). -/
abbrev Challenges (P : Params F K) := RingDraw K × LookupDraw K P.lookups P.claims P.spanVars

namespace Params

variable (P : Params F K)

/-- `D`, the lanes of a ring block. -/
def lanes : ℕ := 2 ^ P.laneVars

/-- `E = κ + 1 + t`, the ring equations of a claim. -/
def equations : ℕ := P.rows + 1 + P.matrices

/-- The key element of block `j` in ring equation `e` of claim `c` at the point `r`: the Ajtai row
`e`, then `Eval_K` (`eq(r_hi, j) · conjugate(eq(r_lo, ·))`), then `Eval_A` of matrix `e − κ − 1`. -/
def form (c : Fin P.claims) (r : List K) (e j : ℕ) : ℕ → K :=
  if e < P.rows then fun h => algebraMap F K (P.key e j h)
  else if e = P.rows then
    fun h => eqAt (r.drop P.laneVars) j * conjugate P.lanes (eqAt (r.take P.laneVars)) h
  else P.matrixForm c r (e - P.rows - 1) j

/-- The right side of ring equation `e`: commitment row `e`, then `Eval_K`, then `Eval_A`. -/
def right (v : ClaimValues F K) (e : ℕ) : ℕ → K :=
  if e < P.rows then fun l => algebraMap F K (v.commitment e l)
  else if e = P.rows then v.evalK else v.evalA (e - P.rows - 1)

end Params

/-- The cells of a slice of `W` in `K`. -/
def sliceOf (W : ℕ → F) (s : Slice) : ℕ → K := fun y => algebraMap F K (W (s.offset + y))

/-- Value `v` of the table `[−bound, bound]`: `v − bound` (Rust `lookup::code`). -/
def tableValue (bound v : ℕ) : F := (v : F) - bound

/-- What soundness gives for each claim. -/
def Relation (P : Params F K) (st : Statement P) (W : ℕ → F) : Prop :=
  ∀ c : Fin P.claims,
    (∀ e < P.equations, ∀ l < P.lanes, ringLeft P.lanes (P.shape c).blocks (P.form c (st c).point)
      (sliceOf W (P.shape c).witness) e l = P.right (st c) e l) ∧
    (∀ y < (P.shape c).publicCells, W ((P.shape c).witness.offset + y) = (st c).publicInput y) ∧
    ((P.shape c).lookup = true → ∀ y < 2 ^ (P.shape c).witness.vars,
      ∃ v < 2 * (P.shape c).bound + 1, W ((P.shape c).witness.offset + y) = tableValue (P.shape c).bound v)

section Draws

variable {P : Params F K}

def Challenges.mu (ch : Challenges P) : K := ch.1.1
def Challenges.omega (ch : Challenges P) : K := ch.1.2.1
def Challenges.delta (ch : Challenges P) : K := ch.1.2.2.1
def Challenges.gamma (ch : Challenges P) : K := ch.1.2.2.2
def Challenges.psi (ch : Challenges P) : Fin P.lookups → K := ch.2.1
def Challenges.xi (ch : Challenges P) : Fin P.claims → K := ch.2.2.1.1
def Challenges.lam (ch : Challenges P) : Fin P.claims → K := ch.2.2.1.2
def Challenges.beta (ch : Challenges P) : List K := List.ofFn ch.2.2.2

end Draws

section Tables

variable (P : Params F K) (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
  (ch : Challenges P)

/-- `1` for a claim with a table and `0` for another. -/
def lookupWeight (c : Fin P.claims) : K := if (P.shape c).lookup then 1 else 0

/-- The weights of claim `c` on its witness slice (Rust `weights::table`). -/
def claimWeights (c : Fin P.claims) : ℕ → K := fun y =>
  if y < P.lanes * (P.shape c).blocks then
    ch.mu ^ (c : ℕ) * claimWeight P.lanes P.equations (P.shape c).publicCells
      (P.form c (st c).point) ch.omega ch.delta ch.gamma y
  else 0

/-- The weights of the counts of claim `c` (Rust `count_weights`):
`−λ_c · Σ_k ψ_k / (τ_k − value)`. -/
def countWeights (c : Fin P.claims) : ℕ → K := fun v =>
  if v < 2 * (P.shape c).bound + 1 then
    -(ch.lam c * ∑ k, ch.psi k * algebraMap F K (1 / (τ k - tableValue (P.shape c).bound v)))
  else 0

/-- `G`, the public weights of `W`. -/
def weightTable : ℕ → K := fun x =>
  ∑ c, ((P.shape c).witness.place (claimWeights P st ch c) x +
    lookupWeight P c * (P.shape c).counts.place (countWeights P τ ch c) x)

/-- `E`, the weight of the inverse checks: `ξ_c · eq(β, ·)` on each witness with a table. -/
def checkTable : ℕ → K := fun x =>
  ∑ c, lookupWeight P c * (P.shape c).witness.place
    (fun y => ch.xi c * eqAt (ch.beta.take (P.shape c).witness.vars) y) x

/-- `S`, the weight of the lookup sums: `λ_c` on each witness with a table. -/
def sumTable : ℕ → K := fun x =>
  ∑ c, lookupWeight P c * (P.shape c).witness.place (fun _ => ch.lam c) x

/-- SPEC.md section 6. -/
def inverseTable : ℕ → K := fun x =>
  if x < 2 ^ P.spanVars then algebraMap F K (V x) else 0

def denominator {R : Type*} [CommRing R] {n : ℕ} (poles : Fin n → R) (w : R) : R :=
  ∏ k, (poles k - w)

def cofactor {R : Type*} [CommRing R] {n : ℕ} (poles : Fin n → R) (k : Fin n) (w : R) : R :=
  ∏ j ∈ univ.erase k, (poles j - w)

theorem factor_mul_cofactor {R : Type*} [CommRing R] {n : ℕ} (poles : Fin n → R)
    (k : Fin n) (w : R) : (poles k - w) * cofactor poles k w = denominator poles w :=
  Finset.mul_prod_erase univ (fun j => poles j - w) (mem_univ k)

def mainSummand {R : Type*} [CommRing R] {n : ℕ} (poles weights : Fin n → R)
    (w g e u s : R) : R :=
  w * g + e * (u * denominator poles w) + s * (u * ∑ k, weights k * cofactor poles k w)

def mainTables : Fin 5 → ℕ → K :=
  ![fun x => algebraMap F K (W x), weightTable P st τ ch, checkTable P ch, inverseTable P V,
    sumTable P ch]

noncomputable def mainComb : MvPolynomial (Fin 5) K :=
  mainSummand (fun k => MvPolynomial.C (algebraMap F K (τ k))) (fun k => MvPolynomial.C (ch.psi k))
    (.X 0) (.X 1) (.X 2) (.X 3) (.X 4)

theorem eval_mainComb (v : Fin 5 → K) :
    MvPolynomial.eval v (mainComb P τ ch) =
      mainSummand (fun k => algebraMap F K (τ k)) ch.psi (v 0) (v 1) (v 2) (v 3) (v 4) := by
  simp [mainComb, mainSummand, denominator, cofactor]

theorem mainComb_degree : (mainComb P τ ch).totalDegree ≤ P.lookups + 2 := by
  let factor := fun k => MvPolynomial.C (algebraMap F K (τ k)) - (MvPolynomial.X 0 : MvPolynomial (Fin 5) K)
  have degree (s : Finset (Fin P.lookups)) : (∏ i ∈ s, factor i).totalDegree ≤ s.card := by
    refine (MvPolynomial.totalDegree_finsetProd s factor).trans ?_
    calc ∑ i ∈ s, (factor i).totalDegree ≤ ∑ _i ∈ s, 1 := sum_le_sum fun i _ =>
          (MvPolynomial.totalDegree_sub _ _).trans (by simp [MvPolynomial.totalDegree_X])
         _ = _ := by simp
  have entire := degree univ
  rw [card_univ, Fintype.card_fin] at entire
  have num : (∑ k, MvPolynomial.C (ch.psi k) * ∏ i ∈ univ.erase k, factor i).totalDegree ≤ P.lookups := by
    apply MvPolynomial.totalDegree_finsetSum_le
    intro k _
    refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    simpa only [MvPolynomial.totalDegree_C, zero_add] using
      (degree (univ.erase k)).trans (by simp)
  unfold mainComb mainSummand denominator cofactor
  apply (MvPolynomial.totalDegree_add _ _).trans
  apply max_le
  · apply (MvPolynomial.totalDegree_add _ _).trans
    apply max_le
    · exact (MvPolynomial.totalDegree_mul _ _).trans (by simp [MvPolynomial.totalDegree_X])
    · apply (MvPolynomial.totalDegree_mul _ _).trans
      apply le_trans (Nat.add_le_add_left (MvPolynomial.totalDegree_mul _ _) _)
      simp only [MvPolynomial.totalDegree_X]
      change 1 + (1 + (∏ i, factor i).totalDegree) ≤ _
      omega
  · apply (MvPolynomial.totalDegree_mul _ _).trans
    apply le_trans (Nat.add_le_add_left (MvPolynomial.totalDegree_mul _ _) _)
    simp only [MvPolynomial.totalDegree_X]
    change 1 + (1 + (∑ k, MvPolynomial.C (ch.psi k) * ∏ i ∈ univ.erase k, factor i).totalDegree) ≤ _
    omega

/-- The sum that the first sum-check proves (Rust `claimed_sum`). -/
def claimedSum : K :=
  ∑ c : Fin P.claims, ch.mu ^ (c : ℕ) * claimTarget P.lanes P.equations (P.shape c).publicCells
      (P.right (st c)) (fun y => algebraMap F K ((st c).publicInput y)) ch.omega ch.delta ch.gamma +
    ∑ c, lookupWeight P c * ch.xi c

end Tables

end NightstreamFPrime.Spec.Compression.Verifier
