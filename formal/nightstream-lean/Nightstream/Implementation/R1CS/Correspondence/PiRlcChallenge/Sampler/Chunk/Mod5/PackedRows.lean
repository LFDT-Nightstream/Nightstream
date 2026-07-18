import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk

/-!
Model-level semantics for the eight packed Mod-5 equations of one `Pi_RLC`
sampler chunk.

Owns: the fifteen-cell reduced witness, its linearly derived high quotient bit,
the sixteen direct Goldilocks residuals, the exact seven-bit-pair plus
one-residue-pair packing schedule, packing soundness/completeness, and the
small GateField algebra support used by this Mod-5 subtree.

Does not own: Rust trace decoding, CCS selector placement, inactive rows,
production column ownership, the Goldilocks nonresidue certificate, or row
removal authorization.

Emits constraints: no.

Authority boundary: the direct equations are independent mathematical
semantics. The packed equations become equivalent only under the explicit
`SevenNonresidue` premise. Neither predicate is evidence that production emits
this schedule until a separate artifact/Rust refinement theorem is proved.

| Stage path | Direct obligations | Packed equations | Mathematical property |
|---|---:|---:|---|
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.low_bit_pairs` | 12 | 6 | low quotient coordinates 0 through 11 are Boolean |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.high_bit_pair` | 2 | 1 | low coordinate 12 and the linearly derived high coordinate are Boolean |
| `nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.residue_pair` | 2 | 1 | the left centered cubic and centered-pair residual vanish |

| Result | Guarantee | Assumptions | Assurance tier | Permits row removal? |
|---|---|---|---|---|
| `quadraticZeroPair_iff` | one packed equation iff both residuals vanish | `SevenNonresidue` | model-level | no |
| `packedRows_iff_directRows` | exact 8-to-16 obligation equivalence | `SevenNonresidue` | model-level | no |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5

open Nightstream.Implementation.R1CS

private instance : NeZero goldilocksP := ⟨by decide⟩

/-- Canonical Goldilocks carrier used by the active Nat-based R1CS model. -/
abbrev GateField := Fin goldilocksP

/-- Canonical embedding into the active Goldilocks carrier. -/
def fieldResidue (value : Nat) : GateField :=
  ⟨value % goldilocksP, Nat.mod_lt _ (by decide)⟩

theorem fieldResidue_add_hom (left right : Nat) :
    fieldResidue (left + right) = fieldResidue left + fieldResidue right := by
  apply Fin.ext
  simp [fieldResidue, Fin.val_add, Nat.add_mod]

theorem fieldResidue_mul_hom (left right : Nat) :
    fieldResidue (left * right) = fieldResidue left * fieldResidue right := by
  apply Fin.ext
  simp [fieldResidue, Fin.val_mul, Nat.mul_mod]

theorem fieldResidue_val (value : GateField) :
    fieldResidue value.val = value := by
  apply Fin.ext
  simp [fieldResidue, Nat.mod_eq_of_lt value.isLt]

/-- Field subtraction written using the production canonical coefficient
`p - 1`; this avoids importing a second field implementation. -/
def fieldSub (left right : GateField) : GateField :=
  left + fieldResidue (goldilocksP - 1) * right

theorem gateField_add_comm (left right : GateField) :
    left + right = right + left := by
  apply Fin.ext
  simp only [Fin.val_add, Nat.add_comm]

theorem gateField_add_zero (value : GateField) : value + 0 = value := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_zero, Nat.add_zero,
    Nat.mod_eq_of_lt value.isLt]

theorem gateField_zero_add (value : GateField) : 0 + value = value := by
  rw [gateField_add_comm, gateField_add_zero]

theorem gateField_add_assoc (left middle right : GateField) :
    (left + middle) + right = left + (middle + right) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

theorem gateField_add_outer_comm (left middle right : GateField) :
    left + (middle + right) = right + (middle + left) := by
  apply Fin.ext
  simp only [Fin.val_add]
  simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_comm, Nat.add_left_comm,
    Nat.add_assoc]

theorem fieldResidue_one : fieldResidue 1 = 1 := by
  native_decide

theorem gateField_one_mul (value : GateField) : 1 * value = value := by
  apply Fin.ext
  change (1 % goldilocksP * value.val) % goldilocksP = value.val
  have oneMod : 1 % goldilocksP = 1 := by native_decide
  rw [oneMod, Nat.one_mul, Nat.mod_eq_of_lt value.isLt]

theorem gateField_mul_one (value : GateField) : value * 1 = value := by
  apply Fin.ext
  change (value.val * (1 % goldilocksP)) % goldilocksP = value.val
  have oneMod : 1 % goldilocksP = 1 := by native_decide
  rw [oneMod, Nat.mul_one, Nat.mod_eq_of_lt value.isLt]

theorem gateField_mul_zero (value : GateField) : value * 0 = 0 := by
  apply Fin.ext
  simp only [Fin.val_mul, Fin.val_zero, Nat.mul_zero, Nat.zero_mod]

theorem fieldResidue_eq_iff_mod (left right : Nat) :
    fieldResidue left = fieldResidue right ↔
      left % goldilocksP = right % goldilocksP := by
  constructor
  · intro equal
    exact congrArg Fin.val equal
  · intro equal
    apply Fin.ext
    exact equal

theorem gateField_mul_assoc (left middle right : GateField) :
    (left * middle) * right = left * (middle * right) := by
  apply Fin.ext
  simp only [Fin.val_mul]
  rw [Nat.mod_mul_mod, Nat.mul_mod_mod, Nat.mul_assoc]

theorem fieldResidue_mul_residue_mul
    (left middle : Nat) (right : GateField) :
    fieldResidue left * (fieldResidue middle * right) =
      fieldResidue (left * middle) * right := by
  rw [← gateField_mul_assoc, ← fieldResidue_mul_hom]

theorem gateField_mul_comm (left right : GateField) :
    left * right = right * left := by
  apply Fin.ext
  simp only [Fin.val_mul, Nat.mul_comm]

theorem gateField_mul_add (left middle right : GateField) :
    left * (middle + right) = left * middle + left * right := by
  apply Fin.ext
  simp only [Fin.val_mul, Fin.val_add]
  simp [Nat.mod_eq_of_lt middle.isLt, Nat.mod_eq_of_lt right.isLt,
    Nat.mul_mod_mod, Nat.mul_add, Nat.add_mod]

theorem gateField_add_mul (left middle right : GateField) :
    (left + middle) * right = left * right + middle * right := by
  rw [gateField_mul_comm, gateField_mul_add]
  congr 1 <;> rw [gateField_mul_comm]

theorem gateField_add_reassociate_four (a b c d : GateField) :
    (a + b) + (c + d) = (a + (b + c)) + d := by
  apply Fin.ext
  simp only [Fin.val_add]
  simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

theorem negOne_mul_add_self (value : GateField) :
    fieldResidue (goldilocksP - 1) * value + value = 0 := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul, fieldResidue]
  have raw : (goldilocksP - 1) * value.val + value.val =
      goldilocksP * value.val := by
    unfold goldilocksP
    omega
  have modular :
      ((goldilocksP - 1) * value.val + value.val) % goldilocksP = 0 := by
    rw [raw, Nat.mul_mod_right]
  simpa only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod,
    Nat.mod_eq_of_lt value.isLt] using modular

theorem fieldSub_eq_zero_iff (left right : GateField) :
    fieldSub left right = 0 ↔ left = right := by
  constructor
  · intro zero
    have shifted := congrArg (fun value : GateField => value + right) zero
    unfold fieldSub at shifted
    change (left + fieldResidue (goldilocksP - 1) * right) + right =
      0 + right at shifted
    rw [gateField_add_assoc, negOne_mul_add_self, gateField_add_zero,
      gateField_zero_add] at shifted
    exact shifted
  · rintro rfl
    unfold fieldSub
    rw [gateField_add_comm, negOne_mul_add_self]

theorem negOne_add_negOne :
    fieldResidue (goldilocksP - 1) + fieldResidue (goldilocksP - 1) =
      fieldResidue (goldilocksP - 2) := by
  native_decide

theorem negOne_mul_negOne :
    fieldResidue (goldilocksP - 1) * fieldResidue (goldilocksP - 1) = 1 := by
  native_decide

theorem negOne_mul_negOne_mul (value : GateField) :
    fieldResidue (goldilocksP - 1) *
        (fieldResidue (goldilocksP - 1) * value) = value := by
  rw [← gateField_mul_assoc, negOne_mul_negOne, gateField_one_mul]

theorem negSeven_mul_negTwo :
    fieldResidue (goldilocksP - 7) * fieldResidue (goldilocksP - 2) =
      fieldResidue 14 := by
  native_decide

/-- Residual of one Boolean-root equation. -/
def bitResidual (value : GateField) : GateField :=
  value * fieldSub value 1

/-- Residual of the centered cubic with roots `-1`, `0`, and `1`. -/
def centeredResidual (value : GateField) : GateField :=
  value * fieldSub value 1 * (value + 1)

/-- One equation packing two zero residuals with the fixed coefficient seven. -/
def QuadraticZeroPair (left right : GateField) : Prop :=
  left * left +
    fieldResidue (goldilocksP - 7) * (right * right) = 0

/-- Expanded production polynomial shared by every quadratic Boolean pair.
This definition is independent of any generated matrix or row schedule. -/
def booleanPairGatePolynomial (left right : GateField) : GateField :=
  ((((left * left * left * left +
          fieldResidue (goldilocksP - 2) * (left * left * left)) +
        left * left) +
      fieldResidue (goldilocksP - 7) *
        (right * right * right * right)) +
    fieldResidue 14 * (right * right * right)) +
  fieldResidue (goldilocksP - 7) * (right * right)

private theorem bitResidual_linear (value : GateField) :
    bitResidual value =
      value * value + fieldResidue (goldilocksP - 1) * value := by
  unfold bitResidual fieldSub
  rw [gateField_mul_add, gateField_mul_one,
    gateField_mul_comm value (fieldResidue (goldilocksP - 1))]

private theorem bitResidual_square_expansion (value : GateField) :
    bitResidual value * bitResidual value =
      (value * value * value * value +
          fieldResidue (goldilocksP - 2) * (value * value * value)) +
        value * value := by
  rw [bitResidual_linear, gateField_add_mul,
    gateField_mul_add, gateField_mul_add]
  calc
    _ = value * value * value * value +
          fieldResidue (goldilocksP - 1) * (value * value * value) +
          (fieldResidue (goldilocksP - 1) * (value * value * value) +
            (fieldResidue (goldilocksP - 1) *
                fieldResidue (goldilocksP - 1)) * (value * value)) := by
        ac_rfl
    _ = (value * value * value * value +
          (fieldResidue (goldilocksP - 1) * (value * value * value) +
            fieldResidue (goldilocksP - 1) * (value * value * value))) +
          value * value := by
      rw [negOne_mul_negOne, gateField_one_mul]
      apply gateField_add_reassociate_four
    _ = _ := by
      have combine (term : GateField) :
          fieldResidue (goldilocksP - 1) * term +
              fieldResidue (goldilocksP - 1) * term =
            fieldResidue (goldilocksP - 2) * term := by
        rw [← gateField_add_mul, negOne_add_negOne]
      rw [combine]

/-- The expanded production Boolean-pair polynomial is exactly the independent
quadratic packing of the two Boolean residuals. -/
theorem booleanPairGatePolynomial_eq_quadratic
    (left right : GateField) :
    booleanPairGatePolynomial left right =
      bitResidual left * bitResidual left +
        fieldResidue (goldilocksP - 7) *
          (bitResidual right * bitResidual right) := by
  rw [bitResidual_square_expansion, bitResidual_square_expansion,
    gateField_mul_add, gateField_mul_add]
  have middle :
      fieldResidue (goldilocksP - 7) *
          (fieldResidue (goldilocksP - 2) *
            (right * right * right)) =
        fieldResidue 14 * (right * right * right) := by
    rw [← gateField_mul_assoc, negSeven_mul_negTwo]
  rw [middle]
  apply Fin.ext
  simp only [booleanPairGatePolynomial, Fin.val_add]
  simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

/-- Narrow algebraic boundary still missing from the active no-Mathlib
project: seven is projectively nonresidual in the Goldilocks field. -/
def SevenNonresidue : Prop :=
  ∀ left right : GateField,
    QuadraticZeroPair left right → left = 0 ∧ right = 0

/-- The optimized block commits thirteen low quotient coordinates and two
centered residue coordinates. The fourteenth quotient coordinate is derived
linearly and therefore consumes no witness cell. -/
structure Witness where
  quotientLow : Fin 13 → GateField
  residueLeft : GateField
  residueRight : GateField

/-- The exact source column corresponding to one committed low quotient bit.
This is naming/ownership metadata only; it does not claim a trace bridge. -/
def lowBitSourceColumn (chunk : Nat) (index : Fin 13) : Nat :=
  ChunkRows.quotientBitCol chunk index.val

/-- Little-endian field value of the thirteen committed quotient bits. -/
def Witness.quotientLowValue (witness : Witness) : GateField :=
  witness.quotientLow 0 +
    fieldResidue 2 * witness.quotientLow 1 +
    fieldResidue 4 * witness.quotientLow 2 +
    fieldResidue 8 * witness.quotientLow 3 +
    fieldResidue 16 * witness.quotientLow 4 +
    fieldResidue 32 * witness.quotientLow 5 +
    fieldResidue 64 * witness.quotientLow 6 +
    fieldResidue 128 * witness.quotientLow 7 +
    fieldResidue 256 * witness.quotientLow 8 +
    fieldResidue 512 * witness.quotientLow 9 +
    fieldResidue 1024 * witness.quotientLow 10 +
    fieldResidue 2048 * witness.quotientLow 11 +
    fieldResidue 4096 * witness.quotientLow 12

/-- Unsigned residue index represented by the two centered cells. -/
def Witness.residueIndex (witness : Witness) : GateField :=
  witness.residueLeft + witness.residueRight + 2

/-- Centered-pair consistency residual used alongside the centered cubic. -/
def Witness.residuePairResidual (witness : Witness) : GateField :=
  witness.residueRight * fieldSub witness.residueLeft witness.residueRight

/-- Inverse of `5 * 2^13 = 40960` in the Goldilocks field. -/
def highDenominatorInverse : GateField :=
  fieldResidue 18446293709451952129

theorem highDenominatorInverse_correct :
    fieldResidue 40960 * highDenominatorInverse = 1 := by
  native_decide

/-- Linearly derived fourteenth quotient bit. This is the same rearrangement
used by the optimized Rust decoder:

`high = (chunk - 5 * low - residueIndex) / (5 * 2^13)`.

The source candidate is independently interpreted by `Chunk.chunkValue`; the
packed witness cannot choose it. -/
def derivedQuotientHigh
    (assignment : Nat → Nat) (chunk : Nat) (witness : Witness) : GateField :=
  highDenominatorInverse *
    fieldSub
      (fieldSub
        (fieldResidue (Chunk.chunkValue assignment chunk))
        (fieldResidue 5 * witness.quotientLowValue))
      witness.residueIndex

/-- The sixteen scalar obligations before quadratic-nonresidue packing:
thirteen low bit roots, one derived-high bit root, one left centered cubic,
and one centered residue-pair equation. -/
def DirectRows
    (assignment : Nat → Nat) (chunk : Nat) (witness : Witness) : Prop :=
  bitResidual (witness.quotientLow 0) = 0 ∧
    bitResidual (witness.quotientLow 1) = 0 ∧
    bitResidual (witness.quotientLow 2) = 0 ∧
    bitResidual (witness.quotientLow 3) = 0 ∧
    bitResidual (witness.quotientLow 4) = 0 ∧
    bitResidual (witness.quotientLow 5) = 0 ∧
    bitResidual (witness.quotientLow 6) = 0 ∧
    bitResidual (witness.quotientLow 7) = 0 ∧
    bitResidual (witness.quotientLow 8) = 0 ∧
    bitResidual (witness.quotientLow 9) = 0 ∧
    bitResidual (witness.quotientLow 10) = 0 ∧
    bitResidual (witness.quotientLow 11) = 0 ∧
    bitResidual (witness.quotientLow 12) = 0 ∧
    bitResidual (derivedQuotientHigh assignment chunk witness) = 0 ∧
    centeredResidual witness.residueLeft = 0 ∧
    witness.residuePairResidual = 0

/-- Exact production packing schedule: six ordinary low-bit pairs, one pair
containing low bit 12 and the derived high bit, then the centered residue
pair. -/
def PackedRows
    (assignment : Nat → Nat) (chunk : Nat) (witness : Witness) : Prop :=
  QuadraticZeroPair
      (bitResidual (witness.quotientLow 0))
      (bitResidual (witness.quotientLow 1)) ∧
    QuadraticZeroPair
      (bitResidual (witness.quotientLow 2))
      (bitResidual (witness.quotientLow 3)) ∧
    QuadraticZeroPair
      (bitResidual (witness.quotientLow 4))
      (bitResidual (witness.quotientLow 5)) ∧
    QuadraticZeroPair
      (bitResidual (witness.quotientLow 6))
      (bitResidual (witness.quotientLow 7)) ∧
    QuadraticZeroPair
      (bitResidual (witness.quotientLow 8))
      (bitResidual (witness.quotientLow 9)) ∧
    QuadraticZeroPair
      (bitResidual (witness.quotientLow 10))
      (bitResidual (witness.quotientLow 11)) ∧
    QuadraticZeroPair
      (bitResidual (witness.quotientLow 12))
      (bitResidual (derivedQuotientHigh assignment chunk witness)) ∧
    QuadraticZeroPair
      (centeredResidual witness.residueLeft)
      witness.residuePairResidual

/-- The explicit nonresidue premise is sufficient and necessary for this
module's generic pair-decoding step. -/
theorem quadraticZeroPair_iff
    (nonresidue : SevenNonresidue) {left right : GateField} :
    QuadraticZeroPair left right ↔ left = 0 ∧ right = 0 := by
  constructor
  · exact nonresidue left right
  · rintro ⟨rfl, rfl⟩
    unfold QuadraticZeroPair
    apply Fin.ext
    native_decide

/-- Model-level soundness and completeness of the exact 8-to-16 packing. -/
theorem packedRows_iff_directRows
    (nonresidue : SevenNonresidue)
    (assignment : Nat → Nat) (chunk : Nat) (witness : Witness) :
    PackedRows assignment chunk witness ↔
      DirectRows assignment chunk witness := by
  constructor
  · rintro ⟨h01, h23, h45, h67, h89, h1011, h12High, hResidue⟩
    have h01' := (quadraticZeroPair_iff nonresidue).mp h01
    have h23' := (quadraticZeroPair_iff nonresidue).mp h23
    have h45' := (quadraticZeroPair_iff nonresidue).mp h45
    have h67' := (quadraticZeroPair_iff nonresidue).mp h67
    have h89' := (quadraticZeroPair_iff nonresidue).mp h89
    have h1011' := (quadraticZeroPair_iff nonresidue).mp h1011
    have h12High' := (quadraticZeroPair_iff nonresidue).mp h12High
    have hResidue' := (quadraticZeroPair_iff nonresidue).mp hResidue
    exact ⟨h01'.1, h01'.2, h23'.1, h23'.2, h45'.1, h45'.2,
      h67'.1, h67'.2, h89'.1, h89'.2, h1011'.1, h1011'.2,
      h12High'.1, h12High'.2, hResidue'.1, hResidue'.2⟩
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11,
      h12, hHigh, hLeft, hPair⟩
    exact
      ⟨(quadraticZeroPair_iff nonresidue).mpr ⟨h0, h1⟩,
        (quadraticZeroPair_iff nonresidue).mpr ⟨h2, h3⟩,
        (quadraticZeroPair_iff nonresidue).mpr ⟨h4, h5⟩,
        (quadraticZeroPair_iff nonresidue).mpr ⟨h6, h7⟩,
        (quadraticZeroPair_iff nonresidue).mpr ⟨h8, h9⟩,
        (quadraticZeroPair_iff nonresidue).mpr ⟨h10, h11⟩,
        (quadraticZeroPair_iff nonresidue).mpr ⟨h12, hHigh⟩,
        (quadraticZeroPair_iff nonresidue).mpr ⟨hLeft, hPair⟩⟩

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5
