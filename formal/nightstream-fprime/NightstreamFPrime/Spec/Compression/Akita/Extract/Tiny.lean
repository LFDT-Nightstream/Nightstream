import NightstreamFPrime.Spec.Compression.Akita.Extract.Wire

/-! A small instance of two packing folds of Akita (ePrint 2026/1983, revision of 2026-11-23,
Theorem 10.9 and Theorem 10.17) with each parameter a numeral. It shows that the hypotheses of
`packingPair_extraction_le` and of `packingPair_complete` hold together for one verifier.

- `Sparse.one`, `Sparse.neg` and `isUnit_two` give the challenge family `{1, −1}`: its members
  have `‖c‖_1 = ‖c‖_∞ = 1`, and the difference of its two members is a unit (Akita
  Definition 9.2, E1).
- `Tiny.shape`, `Tiny.opening` and `Tiny.fold` have the ring degree 1, the extension degree 1
  and no compression map. The first level has one block of one element and a cube of `2^9`
  cells, and the second level has three blocks of 128 elements and a cube of `2^14` cells. The
  count of the blocks of the second level is not a power of two: its point has two coordinates
  for the blocks (Akita Section 4.5, (40) to (42)).
- `Tiny.encodable` states that each response is admissible for `b_z = 4` and `δ_f = 33`.
- `Tiny.complete` is `packingPair_complete` and `Tiny.extraction_le` is
  `packingPair_extraction_le` for this instance, with no hypothesis on the parameters. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Family

variable {d : ℕ} [NeZero d]

theorem normInf_one : normInf (1 : Rq d) = 1 := by
  have unit : magnitude (1 : Zq) = 1 := by
    have value : (1 : Zq).valMinAbs = 1 :=
      (ZMod.valMinAbs_spec _ _).mpr ⟨by simp, by constructor <;> decide⟩
    rw [magnitude, value]
    rfl
  refine le_antisymm ((normInf_le_iff _ _).mpr fun i _ => ?_) ?_
  · rw [coeff_one]
    split
    · exact unit.le
    · rw [magnitude_zero]
      exact Nat.zero_le _
  · have first := Finset.le_sup (f := fun i => magnitude (coeff (1 : Rq d) i))
      (Finset.mem_range.mpr (NeZero.pos d))
    rwa [coeff_one, ite_eq_left rfl, unit] at first

theorem Sparse.one : Sparse 1 1 (1 : Rq d) := ⟨normOne_one, normInf_one⟩

theorem Sparse.neg {mass peak : ℕ} {c : Rq d} (sparse : Sparse mass peak c) :
    Sparse mass peak (-c) := by
  have coefficient : ∀ i, magnitude (coeff (-c) i) = magnitude (coeff c i) := fun i => by
    rw [← zero_sub, coeff_sub, coeff_zero, zero_sub, magnitude_neg]
  exact ⟨by rw [← sparse.oneNorm, normOne, normOne, Finset.sum_congr rfl fun i _ => coefficient i],
    by rw [← sparse.infNorm, normInf, normInf, Finset.sup_congr rfl fun i _ => coefficient i]⟩

omit [NeZero d] in
theorem isUnit_two : IsUnit (2 : Rq d) := by
  have unit : IsUnit (2 : Zq) := isUnit_iff_ne_zero.mpr (by decide)
  have image := unit.map (algebraMap Zq (Rq d))
  rwa [map_ofNat] at image

end Family

namespace Extract.Tiny

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

attribute [local instance low] Classical.propDecidable

/-- A commitment shape with the ring degree 1, the source base `2^32` with 2 digits, the opening
base and the certified base 4 with 32 digits, one slice and no compression map. -/
def shape (blocks blockLength : ℕ) : Shape where
  innerDegree := 1
  outerDegree := 1
  divides := dvd_rfl
  blocks := blocks
  blockLength := blockLength
  sourceBase := 2 ^ 32
  sourceDigits := 2
  innerRank := 1
  openingBase := 4
  openingDigits := 32
  certifiedBase := 4
  outerRank := 1
  slices := 1
  chain := none

/-- The opening shape of `shape` with the carrier degree 1 and the extension degree 1. -/
def opening (blocks blockLength : ℕ) : OpeningShape (shape blocks blockLength) where
  carrierDegree := 1
  planes := 1
  packing := 1
  tower := rfl
  openDegree := 1
  openDivides := dvd_rfl
  openRank := 1
  chain := none

/-- The opening descriptor with `‖c‖_1 = 1`, `b_z = 4` and `δ_f = 33`. -/
def fold : Descriptor := ⟨1, 4, 33⟩

theorem shape_honest {blocks blockLength : ℕ} (positive : 0 < blockLength) :
    (shape blocks blockLength).Honest where
  source := (⟨by decide, by decide, by decide, by decide⟩ : Base (2 ^ 32) 2)
  opening := (⟨by decide, by decide, by decide, by decide⟩ : Base 4 32)
  certified := le_rfl
  block := positive
  path := trivial

theorem radix : Radix fold.responseBase := ⟨by decide, by decide, by decide⟩

/-- Each vector is admissible for `b_z = 4` and `δ_f = 33` (Akita Definition 10.5): the interval
of 33 digits has each representative in `(−q/2, q/2]`. -/
theorem encodable {d : ℕ} [NeZero d] {ι : Type} (v : ι → Rq d) :
    Encodable fold.responseBase fold.responseDigits v := by
  intro j i _
  have range := ZMod.valMinAbs_mem_Ioc (coeff (v j) i)
  have large : (fieldModulus : ℤ) ≤ 2 * ∑ u ∈ Finset.range 33, (4 : ℤ) ^ u := by
    simp only [Finset.sum_range_succ, Finset.sum_range_zero]
    decide
  have positive : (0 : ℤ) < fieldModulus := by decide
  obtain ⟨low, high⟩ := range
  show -(2 : ℤ) * ∑ u ∈ Finset.range 33, (4 : ℤ) ^ u ≤ _ ∧
    _ ≤ ((2 : ℤ) - 1) * ∑ u ∈ Finset.range 33, (4 : ℤ) ^ u
  generalize ∑ u ∈ Finset.range 33, (4 : ℤ) ^ u = total at large ⊢
  constructor <;> omega

/-- The fold challenge of a slot: 1 for the slot 0, and −1 for each other slot. -/
noncomputable def decode (slot : Zq) : Option (Rq 1) := some (if slot = 0 then 1 else -1)

theorem decode_member {c : Rq 1} (member : ∃ slot, decode slot = some c) : c = 1 ∨ c = -1 := by
  obtain ⟨slot, found⟩ := member
  have value := Option.some.inj found
  by_cases zero : slot = 0
  · exact Or.inl (by rw [← value, ite_eq_left zero])
  · exact Or.inr (by rw [← value, ite_eq_right zero])

theorem decode_units : UnitDifferences {c | ∃ slot, decode slot = some c} := by
  intro c member c' member' different
  rcases decode_member member with rfl | rfl <;> rcases decode_member member' with rfl | rfl
  · exact absurd rfl different
  · rw [sub_neg_eq_add, one_add_one_eq_two]
    exact isUnit_two
  · rw [← neg_add', one_add_one_eq_two]
    exact isUnit_two.neg
  · exact absurd rfl different

theorem decode_sparse (c : Rq 1) (member : ∃ slot, decode slot = some c) : Sparse 1 1 c := by
  rcases decode_member member with rfl | rfl
  · exact Sparse.one
  · exact Sparse.one.neg

/-- The coordinates of an answer of the oracle: 14 slots. -/
abbrev Coord : Type := Fin 14

/-- The wire messages: `Letter` with 4 elements of `Z_q` for one message of the checks. -/
abbrev Wire : Type := Letter (opening 1 1) (opening 3 128) fold Zq 4

/-- The basis `(1)` of `Z_q` over `Z_q`. -/
noncomputable def basis (blocks blockLength : ℕ) :
    Module.Basis (Fin (opening blocks blockLength).planes) Zq Zq :=
  Module.Basis.singleton (Fin 1) Zq

theorem cells_first : (opening 1 1).cells fold ≤ 2 ^ 9 := by decide

theorem cells_second : (opening 3 128).cells fold ≤ 2 ^ 14 := by decide

theorem rows_first : (opening 1 1).rowCount + 1 ≤ 2 ^ 3 := by decide

theorem rows_second : (opening 3 128).rowCount + 1 ≤ 2 ^ 3 := by decide

/-- The block of the second level with the index `i` has the slot `i`. -/
def place : Fin (shape 3 128).blocks → Coord := Fin.castLE (by decide)

variable {key : Key (shape 1 1)} {openKey : OpeningKey (opening 1 1)} {key' : Key (shape 3 128)}
  {openKey' : OpeningKey (opening 3 128)}

local notation "first" =>
  committedFold (basis 1 1) key openKey Letter.root 9 cells_first 3 rows_first
    (fun _ => (0 : Coord)) decode (Letter.read : Reader Wire) 1 0

local notation "second" =>
  clearFold (basis 3 128) key' openKey' Letter.next 14 cells_second 3 rows_second place decode
    (Letter.read : Reader Wire) 1 (Tier.finish first)

local notation "chain" => pair first second Letter.payload Letter.digits

/-- A vector of 4 elements gives each vector of at most 4 elements. -/
theorem surjective_castLE {n : ℕ} (small : n ≤ 4) :
    ∃ map : (Fin 4 → Zq) → Fin n → Zq, Function.Surjective map :=
  ⟨fun vector index => vector (Fin.castLE small index), fun short =>
    ⟨fun index => if inside : (index : ℕ) < n then short ⟨index, inside⟩ else 0, by
      funext index
      simp⟩⟩

/-- `packingPair_complete` for the small instance, with no hypothesis on the parameters: for each
matrices, each table, each opening point and each table of the oracle whose work slots have 57
zero work bits, the verifier accepts the transcript of a proof with the claim of the honest
prover. -/
theorem complete (source : Table) (point : Point (opening 1 1) Zq)
    (table : Address Wire (Tier.finish second) → Option Coord → Zq)
    (worked : ∀ input, table input none ∈ workZeros Zq 57) :
    ∃ proof : ℕ → Wire,
      (proof 0).root = honestClaim (basis 1 1) key openKey source point ∧
        pairAccepts first second Letter.payload Letter.digits
          ((Structure.protocol chain).final (.done proof) table) :=
  packingPair_complete (fun M => ∃ map : (Fin 4 → Zq) → M, Function.Surjective map) Letter.full
    ⟨fun vector => vector 0, fun value => ⟨fun _ => value, rfl⟩⟩ ⟨id, Function.surjective_id⟩
    (fun _ small => surjective_castLE
      (small.trans (show rangeWidth (shape 1 1).certifiedBase ≤ 4 by decide)))
    (fun _ small => surjective_castLE
      (small.trans (show rangeWidth (shape 3 128).certifiedBase ≤ 4 by decide)))
    (shape_honest (by decide)) radix
    le_rfl (show PathFits 1 (1 * 1) none from trivial) (by decide) (by decide)
    (shape_honest (by decide)) radix le_rfl (show PathFits 1 (1 * 1) none from trivial)
    (by decide) (by decide) (p := 0) (t := 0) (s := 7) (k := 2) rfl rfl rfl (by decide) rfl
    (by decide) source point
    (fun _ _ => encodable _) (fun _ _ _ _ => encodable _) table (fun _ _ => rfl) (fun _ _ => rfl)
    worked

/-- `packingPair_extraction_le` for the small instance, with no hypothesis on the parameters
(Akita Theorem 10.17 for two levels): the bound holds for each prover, and by `complete` its
event has the transcripts of the honest prover. -/
theorem extraction_le
    (prover : OracleComp (Address Wire (Tier.finish second)) (Option Coord → Zq)
      (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Address Wire (Tier.finish second)) (Option Coord) Zq
          budget
          (Structure.protocol chain).stages,
        (if pairAccepts first second Letter.payload Letter.digits
            ((Structure.protocol chain).final prover table) ∧
          Checkpoint.extract chain prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (Tier.levelError first + Tier.levelError second) +
        (queries + 1) * ((shape 1 1).blocks + 1 + ((shape 3 128).blocks + 1) : ℕ) / budget :=
  packingPair_extraction_le (S := shape 1 1) (O := opening 1 1) (fold := fold)
    (S' := shape 3 128) (O' := opening 3 128) (fold' := fold) (E := Zq) (Wire := Wire)
    (Coord := Coord) (β := basis 1 1) (key := key) (openKey := openKey) (claim := Letter.root)
    (μ := 9) (fits := cells_first) (batch := 3) (rowsFit := rows_first) (block := fun _ => 0)
    (decode := decode) (read := Letter.read) (weight := 1) (β' := basis 3 128) (key' := key')
    (openKey' := openKey') (claim' := Letter.next) (μ' := 14) (fits' := cells_second)
    (batch' := 3) (rowsFit' := rows_second) (block' := place) (decode' := decode) (weight' := 1)
    (commit := Letter.payload) (witnessOf := Letter.digits) (peak := 1) (peak' := 1)
    (fun left right _ => Subsingleton.elim (α := Fin 1) left right) decode_units decode_sparse
    (by decide) (by decide) (by simp) (by simp) (Fin.castLE_injective _) decode_units
    decode_sparse (by decide) (by decide) (by simp) (by simp) prover bounded positive

end Extract.Tiny

end NightstreamFPrime.Spec.Compression.Akita
