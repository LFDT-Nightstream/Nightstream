import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tiny

/-! A small instance of the batching root of Akita over a fold with three opening groups over a
last packing fold (ePrint 2026/1983, revision of 2026-11-23, Theorem 10.46, "Batching root" and
"Root claims", and Theorem 10.17) with each parameter a numeral. It shows that the hypotheses of
`batchRoot_extraction_le` hold together, for three groups with three shapes.

- `Tiny.groups` has three groups with the shapes `Tiny.shape` of 1 block of 1 element, 2 blocks
  of 1 element and 1 block of 2 elements, and one shared matrix `D` with no compression map. Its
  flat witness has 520 cells on a cube of `2^10` cells.
- The last level has `Tiny.shape` with 4 blocks of 256 elements, which is the cube of the flat
  witness of the groups, and a cube of `2^15` cells.
- `Tiny.root_extract_sound`, `Tiny.root_extraction_le` and `Tiny.root_runs_le` are
  `batchRoot_extract_sound`, `batchRoot_extraction_le` and `batchRoot_runs_le` for this instance,
  with no hypothesis on the parameters. The messages, the readers of the verifier and the
  matrices are free.
- `Tiny.lift_winsTo` is `Batch.liftedFold_winsTo` for the three groups, with 33 quotient digits
  of the base 4: the quotient-lift relation of the groups has a witness for each digit vectors
  with the exact branch relation. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract.Tiny

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The numbers of blocks of the three groups. -/
def groupBlocks : Fin 3 → ℕ := ![1, 2, 1]

/-- The block lengths of the three groups. -/
def groupLengths : Fin 3 → ℕ := ![1, 1, 2]

/-- Three opening groups with the shapes `shape` of `groupBlocks` and `groupLengths`, the carrier
degree 1, the extension degree 1, and one shared matrix `D` of the rank 1 with no compression
map. -/
def groups : Batch (Fin 3) where
  shape g := shape (groupBlocks g) (groupLengths g)
  carrierDegree _ := 1
  carrierPositive _ := ⟨one_ne_zero⟩
  planes _ := 1
  packing _ := 1
  tower _ := rfl
  fold _ := fold
  openDegree := 1
  openDivides _ := dvd_rfl
  openRank := 1
  chain := none

theorem groups_cells : groups.cells ≤ 2 ^ 10 := by decide

theorem groups_rows : groups.rowCount + 1 ≤ 2 ^ 4 := by decide

theorem cells_last : (opening 4 256).cells fold ≤ 2 ^ 15 := by decide

theorem rows_last : (opening 4 256).rowCount + 1 ≤ 2 ^ 3 := by decide

/-- The coordinates of an answer of the oracle: 15 slots. -/
abbrev Slots : Type := Fin 15

/-- The fold coordinate `(g, i)` has the slot of its index in the order of the groups. -/
def groupPlace (b : groups.Block) : Slots :=
  Fin.castLE (by decide) (finSigmaFinEquiv (n := groupBlocks) b)

theorem groupPlace_injective : Function.Injective groupPlace :=
  (Fin.castLE_injective _).comp finSigmaFinEquiv.injective

/-- The block of the last level with the index `i` has the slot `i`. -/
def lastPlace : Fin (shape 4 256).blocks → Slots := Fin.castLE (by decide)

/-- The basis `(1)` of `Z_q` over `Z_q` for each group. -/
noncomputable def groupBasis (g : Fin 3) : Module.Basis (Fin (groups.planes g)) Zq Zq :=
  Module.Basis.singleton (Fin 1) Zq

theorem groups_room : groups.quotientRoom 33 ≤ 2 ^ 10 := by decide

/-- `q ≤ 4^33`: each element of `Z_q` has 33 balanced digits of the base 4. -/
theorem covers : fieldModulus ≤ 4 ^ 33 := by decide

/-- `Batch.liftedFold_winsTo` for the three groups, with no hypothesis on the parameters: for
digit vectors with the exact branch relation, the checks on the quotient-lift route end in a true
claim for the flat witness with the digits of the quotients. -/
theorem lift_winsTo (admits : Type → Prop) (scalar : admits Zq)
    (range : ∀ n, n ≤ rangeWidth 4 → admits (Fin n → Zq)) (cubic : admits (Fin (3 + 1) → Zq))
    {st : groups.Statement Zq} {challenge : groups.Challenge} {w : groups.Witness}
    (satisfies : w.Satisfies groupBasis st challenge) :
    WinsTo admits (foldCheck (Batch.liftedReduction groupBasis st challenge 10 groups_cells 4 33
      groups_room 4 groups_rows) 4 (groups.gaps 33).support groups.compressed
      ((groups.flatRelation groupBasis st challenge 10 groups_cells).completion (groups.gaps 33) 4
        33 w.flat)) (foldDraws 10 4 groups.compressed) true fun result _ _ =>
        ∃ r2 : List Zq, r2.length = 10 ∧ result = some (r2, mle r2 (lifted Zq
          ((groups.flatRelation groupBasis st challenge 10 groups_cells).completion
            (groups.gaps 33) 4 33 w.flat))) :=
  Batch.liftedFold_winsTo admits scalar range cubic (fun _ => rfl)
    ⟨by decide, by decide, by decide⟩ covers groups_room 4 groups_rows satisfies

variable {Wire : Type} {key : (g : Fin 3) → Key (groups.shape g)} {openKey : groups.OpenKey}
  {claim : Wire → groups.Claim Zq} {read : Reader Wire} {key' : Key (shape 4 256)}
  {openKey' : OpeningKey (opening 4 256)} {claim' : Wire → Claim (opening 4 256) Zq}
  {commit : Wire → (g : Fin 3) → (groups.shape g).Payload} {value : Wire → Fin 3 → Zq}
  {place : Wire → Fin 3 → List Zq} {next : Wire → (shape 4 256).Payload}
  {witnessOf : Wire → Witness (opening 4 256) fold}

local notation "first" =>
  committedBatch groupBasis key openKey claim 10 groups_cells 4 groups_rows groupPlace
    (fun _ => decode) read (fun _ => 1) 4 1

local notation "second" =>
  clearFold (basis 4 256) key' openKey' claim' 15 cells_last 3 rows_last lastPlace decode read 1
    (Tier.finish first)

local notation "accepts" => rootAccepts first second commit value place read next witnessOf

local notation "chain" => batchRoot first second commit value place read next witnessOf

/-- `batchRoot_extract_sound` for the small instance, with no hypothesis on the parameters: an
output of the extractor has, for each of the three groups, a certified decommitment of the
commitment of the group to a table with the claimed value at the point of the group, or it is a
solution of Module-SIS for a matrix of a group or of a level. -/
theorem root_extract_sound
    (prover : OracleComp (Address Wire (Tier.finish second)) (Option Slots → Zq)
      (ℕ → Wire))
    (budget : ℕ) (table : Address Wire (Tier.finish second) → Option Slots → Zq)
    (tape : Tape (Address Wire (Tier.finish second)) (Option Slots) Zq budget
      (Structure.protocol chain).stages)
    (out : Opened groups.Weak
        (Opened (Weak (opening 4 256)) (Prefix Wire (Option Slots → Zq) (Tier.finish second))) ⊕
      (groups.LevelBreak key openKey
          (fun g => 2 * (2 * 1) * (groups.fold g).sourceBound (groups.shape g)) ⊕
        (LevelBreak key' openKey' (2 * (2 * 1) * fold.sourceBound (shape 4 256)) ⊕ Unit)))
    (result : Checkpoint.extract chain prover budget table tape = some out) :
    accepts ((Structure.protocol chain).final prover table) ∧
      Sum.elim
        (fun certificate => ∃ ω, certificate.opening = some ω ∧ ∀ g,
          (ω.certified g).Opens (2 * 1) ((groups.fold g).sourceBound (groups.shape g)) (key g)
              (commit (((Structure.protocol chain).input prover table 0).wire 0) g)
              ((groups.shape g).decode (ω.certified g).source) ∧
            mle (place (((Structure.protocol chain).input prover table 0).wire 0) g)
                (lifted Zq ((groups.shape g).decode (ω.certified g).source)) =
              value (((Structure.protocol chain).input prover table 0).wire 0) g)
        (Sum.elim (fun solution => solution.isSome)
          (Sum.elim (fun solution => solution.isSome) fun _ => False))
        out :=
  batchRoot_extract_sound (B := groups) (S' := shape 4 256) (O' := opening 4 256)
    (fold' := fold) (E := Zq) (Wire := Wire) (Coord := Slots) (β := groupBasis) (key := key)
    (openKey := openKey) (claim := claim) (μ := 10) (fits := groups_cells) (batch := 4)
    (rowsFit := groups_rows)
    (block := groupPlace) (decode := fun _ => decode) (read := read) (mass := fun _ => 1)
    (base := 4) (β' := basis 4 256) (key' := key') (openKey' := openKey') (claim' := claim')
    (μ' := 15) (fits' := cells_last) (batch' := 3) (rowsFit' := rows_last) (block' := lastPlace)
    (decode' := decode) (weight' := 1) (commit := commit) (value := value) (place := place)
    (next := next) (witnessOf := witnessOf) (peak := fun _ => 1) (peak' := 1)
    groupPlace_injective
    (fun _ => decode_units) (fun _ => decode_sparse)
    (fun _ => (by decide : 2 ≤ fold.responseBase)) (fun _ => rfl)
    (by decide) (by simp) (by simp) (by simp) (Fin.castLE_injective _) decode_units decode_sparse
    (by decide) (by decide) (by simp) (by simp) prover budget table tape out result

variable [Finite Wire]

/-- `batchRoot_extraction_le` for the small instance, with no hypothesis on the parameters (Akita
Theorem 10.17 for the batching root, a fold with three opening groups and a last packing fold):
the bound holds for each prover. -/
theorem root_extraction_le
    (prover : OracleComp (Address Wire (Tier.finish second)) (Option Slots → Zq)
      (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Address Wire (Tier.finish second)) (Option Slots) Zq
      budget
          (Structure.protocol chain).stages,
        (if accepts ((Structure.protocol chain).final prover table) ∧
          Checkpoint.extract chain prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (1 / Fintype.card Zq / 2 ^ Root.rootBits 3 + (Tier.levelError first +
          Tier.levelError second)) +
        (queries + 1) *
          (1 + (∑ g, (groups.shape g).blocks + 1 + ((shape 4 256).blocks + 1)) : ℕ) / budget :=
  batchRoot_extraction_le (B := groups) (S' := shape 4 256) (O' := opening 4 256)
    (fold' := fold) (E := Zq) (Wire := Wire) (Coord := Slots) (β := groupBasis) (key := key)
    (openKey := openKey) (claim := claim) (μ := 10) (fits := groups_cells) (batch := 4)
    (rowsFit := groups_rows)
    (block := groupPlace) (decode := fun _ => decode) (read := read) (mass := fun _ => 1)
    (base := 4) (β' := basis 4 256) (key' := key') (openKey' := openKey') (claim' := claim')
    (μ' := 15) (fits' := cells_last) (batch' := 3) (rowsFit' := rows_last) (block' := lastPlace)
    (decode' := decode) (weight' := 1) (commit := commit) (value := value) (place := place)
    (next := next) (witnessOf := witnessOf) (peak := fun _ => 1) (peak' := 1)
    groupPlace_injective
    (fun _ => decode_units) (fun _ => decode_sparse)
    (fun _ => (by decide : 2 ≤ fold.responseBase)) (fun _ => rfl)
    (by decide) (by simp) (by simp) (by simp) (Fin.castLE_injective _) decode_units decode_sparse
    (by decide) (by decide) (by simp) (by simp) prover bounded positive

/-- `batchRoot_runs_le` for the small instance, with no hypothesis on the parameters. -/
theorem root_runs_le
    (prover : OracleComp (Address Wire (Tier.finish second)) (Option Slots → Zq)
      (ℕ → Wire))
    {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape (Address Wire (Tier.finish second)) (Option Slots) Zq
      budget
          (Structure.protocol chain).stages,
        (extractRuns chain prover budget table tape : ℝ) ≤
      (2 * (2 * (∑ g, (groups.shape g).blocks + 1) * (2 * ((shape 4 256).blocks + 1))) : ℕ) +
        (queries + 1) *
          ((2 * (2 * (∑ g, (groups.shape g).blocks + 1) * (2 * ((shape 4 256).blocks + 1))) :
            ℕ) - 1) :=
  batchRoot_runs_le (B := groups) (S' := shape 4 256) (O' := opening 4 256)
    (fold' := fold) (E := Zq) (Wire := Wire) (Coord := Slots) (β := groupBasis) (key := key)
    (openKey := openKey) (claim := claim) (μ := 10) (fits := groups_cells) (batch := 4)
    (rowsFit := groups_rows)
    (block := groupPlace) (decode := fun _ => decode) (read := read) (mass := fun _ => 1)
    (base := 4) (β' := basis 4 256) (key' := key') (openKey' := openKey') (claim' := claim')
    (μ' := 15) (fits' := cells_last) (batch' := 3) (rowsFit' := rows_last) (block' := lastPlace)
    (decode' := decode) (weight' := 1) (commit := commit) (value := value) (place := place)
    (next := next) (witnessOf := witnessOf) (peak := fun _ => 1) (peak' := 1)
    groupPlace_injective
    (fun _ => decode_units) (fun _ => decode_sparse)
    (fun _ => (by decide : 2 ≤ fold.responseBase)) (fun _ => rfl)
    (by decide) (by simp) (by simp) (by simp) (Fin.castLE_injective _) decode_units decode_sparse
    (by decide) (by decide) (by simp) (by simp) prover bounded budget

end NightstreamFPrime.Spec.Compression.Akita.Extract.Tiny
