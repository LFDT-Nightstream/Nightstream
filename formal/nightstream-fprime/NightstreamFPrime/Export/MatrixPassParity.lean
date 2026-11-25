import Mathlib.FieldTheory.Finite.Basic
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Spec.Compression.Production
import NightstreamFPrime.Spec.Compression.Rust.Layout

/-! The output of the Lean model of Rust `weights::matrix_part` (`Spec.Compression.Verifier.matrixPass`)
on sample relations, for the comparison with Rust (`crates/ns-compress/tests/unit/weights.rs`).

`matrixPass` is a definition over any two fields. The production instance of `K` is a field by
`Fintype.fieldOfDomain`, which compiled code cannot run. `executableField` is `K` with the
inverse `a^(q² − 2)`, and `pass_matrixSpec` is `matrixPass_spec` for it: the emitted values are
values of a function that has `MatrixSpec`.

The value is `[schema, cases]`. A case is `[relations, claims, [μ, ω, δ], ρ, output]`. A relation
is `[rows, columns, ratio, starts, widths, entries]`, where `entries` holds, for each row and
each matrix, the pairs `[slot, coefficient]`. A claim is `[relation, point, norm]`, where `norm` is
the table bound plus 1, or 0 for no table. The layout is the layout of Rust `Layout::new`
(`Spec.Compression.Rust.layoutNew`) under the key of 512 blocks, and the first claim has a table,
as that layout requires. A value of `K` is two words. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.MatrixPassParity

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Compression NightstreamFPrime.Spec.Compression.Verifier
open NightstreamFPrime.Export.Codec

theorem pow_card_sub_one {value : K} (nonzero : value ≠ 0) :
    value ^ (fieldModulus ^ 2 - 1) = 1 := by
  have unit := @FiniteField.pow_card_sub_one_eq_one K (Fintype.groupWithZeroOfCancel K) _ value
    nonzero
  rw [card_K] at unit
  exact unit

/-- `a^(q² − 2)`, the inverse of a nonzero `a`. -/
@[irreducible] def inverse (value : K) : K := value ^ (fieldModulus ^ 2 - 2)

theorem mul_inverse {value : K} (nonzero : value ≠ 0) : value * inverse value = 1 := by
  unfold inverse
  rw [← pow_succ', show fieldModulus ^ 2 - 2 + 1 = fieldModulus ^ 2 - 1 by decide]
  exact pow_card_sub_one nonzero

theorem inverse_zero : inverse (0 : K) = 0 := by
  unfold inverse
  exact zero_pow (by decide)

/-- `K` as a field whose inverse compiled code can run. -/
@[reducible] def executableField : Field K :=
  { (carrierEquiv.commRing : CommRing K) with
    inv := inverse
    exists_pair_ne := ⟨0, 1, zero_ne_one⟩
    mul_inv_cancel := fun _ nonzero => mul_inverse nonzero
    inv_zero := inverse_zero
    nnqsmul := _
    qsmul := _ }

section Pass

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

local instance : Field F := Production.fieldF

local instance : Field K := executableField

/-- `F → K`, `x ↦ x + 0·u` (`Production.embedHom`), for the executable field. -/
def embedHom : F →+* K where
  toFun := K.embed
  map_one' := rfl
  map_mul' left right := by
    rw [mul_eq]
    simp only [K.embed, K.mul, mul_zero, zero_mul, add_zero]
  map_zero' := rfl
  map_add' left right := by
    rw [add_eq]
    simp only [K.embed, K.add, add_zero]

local instance : Algebra F K := embedHom.toAlgebra

/-- `Params` with the production sizes for `claims` claims without a table: claim `c` has the
blocks of its relation `rel (which c)` and the witness slice `witness c`. `matrixPass` reads the
shapes, `D`, `κ` and `t`, and it does not read `form`. -/
def passParams {claims : ℕ} (rel : ℕ → Factored F) (which : Fin claims → ℕ)
    (witness : Fin claims → Slice) (witnessVars : ℕ)
    (form : Fin claims → List K → ℕ → ℕ → ℕ → K) : Params F K where
  claims := claims
  shape c :=
    { blocks := (rel (which c)).columns / ringDegree
      witness := witness c
      publicCells := ringDegree * 5
      lookup := false
      bound := 0
      counts := ⟨0, 0⟩ }
  witnessVars := witnessVars
  spanVars := 0
  lookups := 4
  laneVars := 6
  rows := productionProfile.commitmentWidth
  rowVars := 5
  keyBlockVars := 0
  key := fun _ _ _ => 0
  matrices := productionProfile.ccsMatrices
  matrixForm := form

/-- The matrix weights of the factored rows of each claim's relation. -/
noncomputable def relationForm {claims : ℕ} (rel : ℕ → Factored F) (which : Fin claims → ℕ) :
    Fin claims → List K → ℕ → ℕ → ℕ → K :=
  fun c => (rel (which c)).matrices.form ringDegree

/-- The emitter runs `matrixPass` with the matrix weights 0. -/
def runParams {claims : ℕ} (rel : ℕ → Factored F) (which : Fin claims → ℕ)
    (witness : Fin claims → Slice) (witnessVars : ℕ) : Params F K :=
  passParams rel which witness witnessVars fun _ _ _ _ _ => 0

/-- `matrixPass` does not read the matrix weights of `Params`. -/
theorem matrixPass_runParams {claims : ℕ} (rel : ℕ → Factored F) (which : Fin claims → ℕ)
    (witness : Fin claims → Slice) (witnessVars : ℕ) :
    matrixPass (runParams rel which witness witnessVars) rel which =
      matrixPass (passParams rel which witness witnessVars (relationForm rel which)) rel which :=
  rfl

/-- `matrixPass` with the executable field computes the matrix part of the honest weights of
the factored rows. -/
theorem pass_matrixSpec {claims : ℕ} (rel : ℕ → Factored F) (which : Fin claims → ℕ)
    (witness : Fin claims → Slice) (witnessVars zeros : ℕ) :
    MatrixSpec (⟨matrixPass (passParams rel which witness witnessVars (relationForm rel which))
      rel which, zeros⟩ :
        VerifierData (passParams rel which witness witnessVars (relationForm rel which))) := by
  have form : ∀ c, (passParams rel which witness witnessVars (relationForm rel which)).matrixForm c =
      (rel (which c)).matrices.form
        (passParams rel which witness witnessVars (relationForm rel which)).lanes := fun _ => rfl
  have blocks : ∀ c c', which c = which c' →
      ((passParams rel which witness witnessVars (relationForm rel which)).shape c).blocks =
        ((passParams rel which witness witnessVars (relationForm rel which)).shape c').blocks :=
    fun c c' same => by
      change (rel (which c)).columns / ringDegree = (rel (which c')).columns / ringDegree
      rw [same]
  exact matrixPass_spec (passParams rel which witness witnessVars (relationForm rel which)) rel
    which form blocks zeros

/-- Sample word `index` of list `seed`, spread over the field. -/
def word (seed index : ℕ) : F :=
  Fin.ofNat fieldModulus ((seed + 1) * 11400714819323198485 + index * 15072723961408105743)

def extension (seed index : ℕ) : K := ⟨word seed (2 * index), word seed (2 * index + 1)⟩

/-- A relation as lists: 12 slots, some across one or two block borders, and up to 3 entries
for each row and matrix. -/
structure Relation where
  rows : ℕ
  blocks : ℕ
  ratio : F
  starts : List ℕ
  widths : List ℕ
  entries : List (List (List (ℕ × F)))

def sampleRelation (seed rows blocks : ℕ) : Relation where
  rows := rows
  blocks := blocks
  ratio := word seed 999
  starts := (List.range 12).map fun slot => (slot * 53 + 7 * seed) % (blocks * ringDegree - 131)
  widths := [1, 3, 9, 64, 70, 2, 5, 130, 1, 8, 33, 17]
  entries := (List.range rows).map fun row =>
    (List.range productionProfile.ccsMatrices).map fun matrix =>
      (List.range ((row + 2 * matrix + seed) % 4)).map fun entry =>
        ((row * 5 + matrix * 3 + entry * 7 + seed) % 12,
          word (seed + 3) (row * 64 + matrix * 8 + entry))

def Relation.factored (relation : Relation) : Factored F where
  rows := relation.rows
  columns := relation.blocks * ringDegree
  entries matrix row := (relation.entries.getD row []).getD matrix []
  start slot := relation.starts.getD slot 0
  width slot := relation.widths.getD slot 0
  ratio := relation.ratio

/-- A case: the relations, the relation of each claim, and a seed for the points and the
challenges. -/
structure Case where
  relations : List Relation
  which : List ℕ
  seed : ℕ

def cases : List Case :=
  [⟨[sampleRelation 1 11 3, sampleRelation 2 7 5], [0, 0, 1], 10⟩,
    ⟨[sampleRelation 3 16 4], [0], 20⟩,
    ⟨[sampleRelation 4 5 3, sampleRelation 5 9 3], [1, 0, 1, 0], 30⟩]

def pointVars : ℕ := 9

def scalar (value : K) : Value := .array [.atom value.c0.val, .atom value.c1.val]

def scalars (values : List K) : Value := .array (values.map scalar)

def relationValue (relation : Relation) : Value :=
  .array [.atom relation.rows, .atom (relation.blocks * ringDegree), .atom relation.ratio.val,
    .array (relation.starts.map .atom), .array (relation.widths.map .atom),
    .array (relation.entries.map fun row => .array (row.map fun matrix =>
      .array (matrix.map fun entry => .array [.atom entry.1, .atom entry.2.val])))]

/-- The case as a value, with the output of `matrixPass`, or the name of the layout error. -/
def caseValue (case : Case) : Value :=
  let rel (index : ℕ) : Factored F := (case.relations.getD index ⟨0, 0, 0, [], [], []⟩).factored
  let claims := case.which.length
  let which (c : Fin claims) : ℕ := case.which.getD c.val 0
  let point (c : ℕ) : List K := (List.range pointVars).map (extension (case.seed + c))
  let table (c : ℕ) : Option ℕ := if c = 0 then some 2 else none
  match Rust.layoutNew (case.which.zipIdx.map fun claim =>
      ⟨(rel claim.1).columns / ringDegree, table claim.2⟩) 512 with
  | .error error => .array [.atom 0, .array (error.name.toList.map fun char => .atom char.toNat)]
  | .ok layout =>
    let witness (c : Fin claims) : Slice := (layout.cells.getD c.val ⟨⟨0, 0⟩, 0, none⟩).witness
    let P := runParams rel which witness layout.witnessVars
    let mu := extension (case.seed + 7) 0
    let omega := extension (case.seed + 7) 1
    let delta := extension (case.seed + 7) 2
    let ρ := (List.range layout.witnessVars).map (extension (case.seed + 8))
    let statement : Statement P := fun c => ⟨fun _ _ => 0, fun _ => 0, point c.val, fun _ => 0,
      fun _ _ => 0⟩
    let challenges : Challenges P :=
      ((mu, omega, delta, 0), (fun _ => 0, (fun _ => 0, fun _ => 0), fun _ => 0))
    .array [.array (case.relations.map relationValue),
      .array (case.which.zipIdx.map fun claim => .array [.atom claim.1, scalars (point claim.2),
        .atom (match table claim.2 with | some bound => bound + 1 | none => 0)]),
      scalars [mu, omega, delta], scalars ρ,
      scalar (matrixPass P rel which statement challenges ρ)]

def schema : ℕ := 1

def parityValue : Value := .array [.atom schema, .array (cases.map caseValue)]

end Pass

end NightstreamFPrime.Export.MatrixPassParity
