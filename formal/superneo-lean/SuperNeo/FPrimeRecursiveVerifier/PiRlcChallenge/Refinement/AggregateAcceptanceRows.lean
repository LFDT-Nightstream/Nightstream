import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.CanonicalAcceptanceSource

/-!
Owns: a Lean-only nine-row candidate for one sixteen-bit chunk acceptance
check: a balanced fourteen-edge product tree, seven packed output-bitness rows,
one balanced-radix-three aggregate row, and one final acceptance row.

Does not own: Rust row emission, selector wiring, the global enough-accepts
check, transcript sampling, or an exact generated trace artifact.

Emits constraints: no. This file specifies and proves a candidate relation.

Authority boundary: all sixteen source coordinates remain authoritative and
must satisfy their existing bit roots. The fourteen tree outputs and acceptance
cell are derived witnesses. The aggregate is sound only because every edge
residual is first bounded to `{-1,0,1}` by source/output bitness, its exact
balanced-radix-three lift is below the Goldilocks modulus, and bounded radix
representations are unique. A prover-supplied aggregate is never authority.

| Family | Mathematical obligation | Multiplicity | Existing role | Degree before selector | Permits Rust row removal? |
|---|---|---:|---|---:|---|
| `ProductTreeOutputBitRows` | Every `o_i` is Boolean | 7 | `QuadraticZeroPair` / Mod5 | 4 | No |
| `ProductTreeAggregateRow` | `sum_{i=0}^{13} 3^i * (o_i-l_i*r_i) = 0` | 1 | ProductSum | 2 | No |
| `FinalAcceptanceRow` | `o_6 * o_13 = 1-a` | 1 | ProductSum | 2 | No |

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits Rust row removal? |
|---|---|---|---|---|
| `productTreeAggregateRow_iff` | proposed `challenge.sampler.chunk.accept` | One aggregate equals all fourteen tree equations | Boolean inputs/outputs; `3^14 < q` | No |
| `aggregateAcceptanceRows_iff_treeAndFinal` | proposed chunk acceptance lowering | Nine rows equal tree plus final equation | Source bit roots | No |
| `aggregateAcceptanceRows_iff_sourceMeaning` | source acceptance bridge | Nine rows derive exactly the source accept bit | Source bit roots | No |
| `aggregateAcceptanceRows_extension_exact` | inactive extension | Every Boolean source has one unique extension | Fixed tree | No |

No Rust row may be removed from these theorems alone. Row removal requires an
exact generated artifact proving that production coordinates, coefficients,
matrix roles, selectors, and inactive materialization instantiate this model.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-! ## Explicit fourteen-edge product tree -/

/-- The fourteen internal outputs of the balanced product tree. -/
abbrev ProductTreeOutputs := Fin 14 → F

/-- Left operand of each edge, in topological order. -/
def productTreeLeft
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs)
    (index : Fin 14) : F :=
  match index.val with
  | 0 => bits 0
  | 1 => bits 2
  | 2 => bits 4
  | 3 => bits 6
  | 4 => outputs 0
  | 5 => outputs 2
  | 6 => outputs 4
  | 7 => bits 8
  | 8 => bits 10
  | 9 => bits 12
  | 10 => bits 14
  | 11 => outputs 7
  | 12 => outputs 9
  | _ => outputs 11

/-- Right operand of each edge, in topological order. -/
def productTreeRight
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs)
    (index : Fin 14) : F :=
  match index.val with
  | 0 => bits 1
  | 1 => bits 3
  | 2 => bits 5
  | 3 => bits 7
  | 4 => outputs 1
  | 5 => outputs 3
  | 6 => outputs 5
  | 7 => bits 9
  | 8 => bits 11
  | 9 => bits 13
  | 10 => bits 15
  | 11 => outputs 8
  | 12 => outputs 10
  | _ => outputs 12

/-- One edge residual `output - left * right`. -/
def productTreeResidual
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs)
    (index : Fin 14) : F :=
  outputs index -
    productTreeLeft bits outputs index * productTreeRight bits outputs index

/-- Readable source semantics of the entire topological product tree. -/
def ProductTreeMeaning
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) : Prop :=
  ∀ index, productTreeResidual bits outputs index = 0

/-- The fourteen equations written without an indexing abstraction. -/
def ProductTreeEquations
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) : Prop :=
  outputs 0 = bits 0 * bits 1 ∧
    outputs 1 = bits 2 * bits 3 ∧
    outputs 2 = bits 4 * bits 5 ∧
    outputs 3 = bits 6 * bits 7 ∧
    outputs 4 = outputs 0 * outputs 1 ∧
    outputs 5 = outputs 2 * outputs 3 ∧
    outputs 6 = outputs 4 * outputs 5 ∧
    outputs 7 = bits 8 * bits 9 ∧
    outputs 8 = bits 10 * bits 11 ∧
    outputs 9 = bits 12 * bits 13 ∧
    outputs 10 = bits 14 * bits 15 ∧
    outputs 11 = outputs 7 * outputs 8 ∧
    outputs 12 = outputs 9 * outputs 10 ∧
    outputs 13 = outputs 11 * outputs 12

/-- The indexed and explicit forms of the fourteen tree equations coincide. -/
theorem productTreeMeaning_iff_equations
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) :
    ProductTreeMeaning bits outputs ↔ ProductTreeEquations bits outputs := by
  constructor
  · intro hMeaning
    have h0 := sub_eq_zero.mp (hMeaning (0 : Fin 14))
    have h1 := sub_eq_zero.mp (hMeaning (1 : Fin 14))
    have h2 := sub_eq_zero.mp (hMeaning (2 : Fin 14))
    have h3 := sub_eq_zero.mp (hMeaning (3 : Fin 14))
    have h4 := sub_eq_zero.mp (hMeaning (4 : Fin 14))
    have h5 := sub_eq_zero.mp (hMeaning (5 : Fin 14))
    have h6 := sub_eq_zero.mp (hMeaning (6 : Fin 14))
    have h7 := sub_eq_zero.mp (hMeaning (7 : Fin 14))
    have h8 := sub_eq_zero.mp (hMeaning (8 : Fin 14))
    have h9 := sub_eq_zero.mp (hMeaning (9 : Fin 14))
    have h10 := sub_eq_zero.mp (hMeaning (10 : Fin 14))
    have h11 := sub_eq_zero.mp (hMeaning (11 : Fin 14))
    have h12 := sub_eq_zero.mp (hMeaning (12 : Fin 14))
    have h13 := sub_eq_zero.mp (hMeaning (13 : Fin 14))
    unfold ProductTreeEquations
    simpa [productTreeResidual, productTreeLeft, productTreeRight] using
      (show
        outputs 0 = bits 0 * bits 1 ∧
          outputs 1 = bits 2 * bits 3 ∧
          outputs 2 = bits 4 * bits 5 ∧
          outputs 3 = bits 6 * bits 7 ∧
          outputs 4 = outputs 0 * outputs 1 ∧
          outputs 5 = outputs 2 * outputs 3 ∧
          outputs 6 = outputs 4 * outputs 5 ∧
          outputs 7 = bits 8 * bits 9 ∧
          outputs 8 = bits 10 * bits 11 ∧
          outputs 9 = bits 12 * bits 13 ∧
          outputs 10 = bits 14 * bits 15 ∧
          outputs 11 = outputs 7 * outputs 8 ∧
          outputs 12 = outputs 9 * outputs 10 ∧
          outputs 13 = outputs 11 * outputs 12 from
        ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩)
  · intro hEquations index
    rcases hEquations with
      ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩
    fin_cases index <;>
      simp [productTreeResidual, productTreeLeft, productTreeRight,
        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13]

private theorem aggregateFieldBitRoot_cases
    {value : F} (hRoot : FieldBitRoot value) :
    value = 0 ∨ value = 1 := by
  rcases mul_eq_zero.mp hRoot with hZero | hOne
  · exact Or.inl hZero
  · exact Or.inr (sub_eq_zero.mp hOne)

private theorem aggregateFieldBitRoot_mul
    {left right : F}
    (hLeft : FieldBitRoot left) (hRight : FieldBitRoot right) :
    FieldBitRoot (left * right) := by
  rcases aggregateFieldBitRoot_cases hLeft with rfl | rfl <;>
    rcases aggregateFieldBitRoot_cases hRight with rfl | rfl <;>
    simp [FieldBitRoot, fieldBitResidual]

/-- Tree equations propagate source bitness to every internal output. -/
theorem productTreeMeaning_outputs_boolean
    {bits : Fin 16 → F} {outputs : ProductTreeOutputs}
    (hBits : ChunkBitsAreBoolean bits)
    (hMeaning : ProductTreeMeaning bits outputs) :
    ∀ index, FieldBitRoot (outputs index) := by
  rcases (productTreeMeaning_iff_equations bits outputs).mp hMeaning with
    ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩
  have b0 : FieldBitRoot (outputs 0) := by
    rw [h0]
    exact aggregateFieldBitRoot_mul (hBits 0) (hBits 1)
  have b1 : FieldBitRoot (outputs 1) := by
    rw [h1]
    exact aggregateFieldBitRoot_mul (hBits 2) (hBits 3)
  have b2 : FieldBitRoot (outputs 2) := by
    rw [h2]
    exact aggregateFieldBitRoot_mul (hBits 4) (hBits 5)
  have b3 : FieldBitRoot (outputs 3) := by
    rw [h3]
    exact aggregateFieldBitRoot_mul (hBits 6) (hBits 7)
  have b4 : FieldBitRoot (outputs 4) := by
    rw [h4]
    exact aggregateFieldBitRoot_mul b0 b1
  have b5 : FieldBitRoot (outputs 5) := by
    rw [h5]
    exact aggregateFieldBitRoot_mul b2 b3
  have b6 : FieldBitRoot (outputs 6) := by
    rw [h6]
    exact aggregateFieldBitRoot_mul b4 b5
  have b7 : FieldBitRoot (outputs 7) := by
    rw [h7]
    exact aggregateFieldBitRoot_mul (hBits 8) (hBits 9)
  have b8 : FieldBitRoot (outputs 8) := by
    rw [h8]
    exact aggregateFieldBitRoot_mul (hBits 10) (hBits 11)
  have b9 : FieldBitRoot (outputs 9) := by
    rw [h9]
    exact aggregateFieldBitRoot_mul (hBits 12) (hBits 13)
  have b10 : FieldBitRoot (outputs 10) := by
    rw [h10]
    exact aggregateFieldBitRoot_mul (hBits 14) (hBits 15)
  have b11 : FieldBitRoot (outputs 11) := by
    rw [h11]
    exact aggregateFieldBitRoot_mul b7 b8
  have b12 : FieldBitRoot (outputs 12) := by
    rw [h12]
    exact aggregateFieldBitRoot_mul b9 b10
  have b13 : FieldBitRoot (outputs 13) := by
    rw [h13]
    exact aggregateFieldBitRoot_mul b11 b12
  intro index
  fin_cases index <;> assumption

/-! ## Seven existing nonresidue-packed output bit rows -/

/-- Seven quadratic-nonresidue rows enforce all fourteen output bit roots. -/
def ProductTreeOutputBitRows (outputs : ProductTreeOutputs) : Prop :=
  QuadraticZeroPair
      (fieldBitResidual (outputs 0)) (fieldBitResidual (outputs 1)) ∧
    QuadraticZeroPair
      (fieldBitResidual (outputs 2)) (fieldBitResidual (outputs 3)) ∧
    QuadraticZeroPair
      (fieldBitResidual (outputs 4)) (fieldBitResidual (outputs 5)) ∧
    QuadraticZeroPair
      (fieldBitResidual (outputs 6)) (fieldBitResidual (outputs 7)) ∧
    QuadraticZeroPair
      (fieldBitResidual (outputs 8)) (fieldBitResidual (outputs 9)) ∧
    QuadraticZeroPair
      (fieldBitResidual (outputs 10)) (fieldBitResidual (outputs 11)) ∧
    QuadraticZeroPair
      (fieldBitResidual (outputs 12)) (fieldBitResidual (outputs 13))

/-- The seven packed rows are exactly the fourteen output bit obligations. -/
theorem productTreeOutputBitRows_iff (outputs : ProductTreeOutputs) :
    ProductTreeOutputBitRows outputs ↔
      ∀ index, FieldBitRoot (outputs index) := by
  constructor
  · rintro ⟨h01, h23, h45, h67, h89, h1011, h1213⟩ index
    have h01' := quadraticZeroPair_iff.mp h01
    have h23' := quadraticZeroPair_iff.mp h23
    have h45' := quadraticZeroPair_iff.mp h45
    have h67' := quadraticZeroPair_iff.mp h67
    have h89' := quadraticZeroPair_iff.mp h89
    have h1011' := quadraticZeroPair_iff.mp h1011
    have h1213' := quadraticZeroPair_iff.mp h1213
    fin_cases index
    · exact h01'.1
    · exact h01'.2
    · exact h23'.1
    · exact h23'.2
    · exact h45'.1
    · exact h45'.2
    · exact h67'.1
    · exact h67'.2
    · exact h89'.1
    · exact h89'.2
    · exact h1011'.1
    · exact h1011'.2
    · exact h1213'.1
    · exact h1213'.2
  · intro hBits
    exact ⟨quadraticZeroPair_iff.mpr ⟨hBits 0, hBits 1⟩,
      quadraticZeroPair_iff.mpr ⟨hBits 2, hBits 3⟩,
      quadraticZeroPair_iff.mpr ⟨hBits 4, hBits 5⟩,
      quadraticZeroPair_iff.mpr ⟨hBits 6, hBits 7⟩,
      quadraticZeroPair_iff.mpr ⟨hBits 8, hBits 9⟩,
      quadraticZeroPair_iff.mpr ⟨hBits 10, hBits 11⟩,
      quadraticZeroPair_iff.mpr ⟨hBits 12, hBits 13⟩⟩

/-! ## Reusable balanced-radix-three no-cancellation proof -/

/-- Least-significant-digit-first base-three value over a fixed `Fin` width. -/
def radix3Nat : {count : Nat} → (Fin count → Nat) → Nat
  | 0, _ => 0
  | _count + 1, digits =>
      digits 0 + 3 * radix3Nat (fun index => digits index.succ)

/-- The same fixed-width linear combination inside Goldilocks. -/
def radix3Field : {count : Nat} → (Fin count → F) → F
  | 0, _ => 0
  | _count + 1, values =>
      values 0 + 3 * radix3Field (fun index => values index.succ)

/-- Bounded base-three digits always encode below the width's radix. -/
theorem radix3Nat_lt
    {count : Nat} {digits : Fin count → Nat}
    (hDigits : ∀ index, digits index < 3) :
    radix3Nat digits < 3 ^ count := by
  induction count with
  | zero => simp [radix3Nat]
  | succ count ih =>
      have hHead := hDigits (0 : Fin (count + 1))
      have hTail :
          radix3Nat (fun index : Fin count => digits index.succ) < 3 ^ count :=
        ih (fun index => hDigits index.succ)
      simp only [radix3Nat, pow_succ]
      omega

/-- Fixed-width base-three representation is unique for digits below three. -/
theorem radix3Nat_injective
    {count : Nat} {left right : Fin count → Nat}
    (hLeft : ∀ index, left index < 3)
    (hRight : ∀ index, right index < 3)
    (hEqual : radix3Nat left = radix3Nat right) :
    left = right := by
  induction count with
  | zero =>
      funext index
      exact Fin.elim0 index
  | succ count ih =>
      have hExpanded := hEqual
      simp only [radix3Nat] at hExpanded
      have hLeftHead := hLeft (0 : Fin (count + 1))
      have hRightHead := hRight (0 : Fin (count + 1))
      have hHead : left 0 = right 0 := by omega
      have hTail :
          radix3Nat (fun index : Fin count => left index.succ) =
            radix3Nat (fun index : Fin count => right index.succ) := by
        omega
      have hTailFunctions :
          (fun index : Fin count => left index.succ) =
            fun index : Fin count => right index.succ :=
        ih (fun index => hLeft index.succ)
          (fun index => hRight index.succ) hTail
      funext index
      refine Fin.cases hHead ?_ index
      intro tail
      exact congrFun hTailFunctions tail

private theorem aggregateFieldOfNat_add (left right : Nat) :
    F.ofNat (left + right) = F.ofNat left + F.ofNat right := by
  apply Fin.ext
  simp [F.ofNat, Nat.add_mod]

private theorem aggregateFieldOfNat_mul (left right : Nat) :
    F.ofNat (left * right) = F.ofNat left * F.ofNat right := by
  apply Fin.ext
  simp [F.ofNat, Nat.mul_mod]

/-- Natural radix values embed as the identical Goldilocks linear combination. -/
theorem fieldOfNat_radix3Nat
    {count : Nat} (digits : Fin count → Nat) :
    F.ofNat (radix3Nat digits) =
      radix3Field (fun index => F.ofNat (digits index)) := by
  induction count with
  | zero => simp [radix3Nat, radix3Field]
  | succ count ih =>
      simp only [radix3Nat, radix3Field, aggregateFieldOfNat_add,
        aggregateFieldOfNat_mul]
      rw [ih]
      rfl

/-- Radix-three evaluation is additive coordinate by coordinate. -/
theorem radix3Field_add
    {count : Nat} (left right : Fin count → F) :
    radix3Field (fun index => left index + right index) =
      radix3Field left + radix3Field right := by
  induction count with
  | zero => simp [radix3Field]
  | succ count ih =>
      simp only [radix3Field]
      rw [ih]
      ring

/-- An all-zero vector has zero radix-three value. -/
@[simp] theorem radix3Field_zero (count : Nat) :
    radix3Field (fun _ : Fin count => (0 : F)) = 0 := by
  induction count with
  | zero => simp [radix3Field]
  | succ count ih => simp [radix3Field, ih]

/-- At width fourteen, the recursion has exactly coefficients `3^0..3^13`. -/
theorem radix3Field_fourteen_weights (values : Fin 14 → F) :
    radix3Field values =
      3 ^ 0 * values 0 + 3 ^ 1 * values 1 +
      3 ^ 2 * values 2 + 3 ^ 3 * values 3 +
      3 ^ 4 * values 4 + 3 ^ 5 * values 5 +
      3 ^ 6 * values 6 + 3 ^ 7 * values 7 +
      3 ^ 8 * values 8 + 3 ^ 9 * values 9 +
      3 ^ 10 * values 10 + 3 ^ 11 * values 11 +
      3 ^ 12 * values 12 + 3 ^ 13 * values 13 := by
  simp [radix3Field]
  ring

/-! ## One exact aggregate row -/

/-- Shift a centered edge residual from `{-1,0,1}` into an ordinary trit. -/
def shiftedProductDigit (output left right : F) : Nat :=
  if left = 1 ∧ right = 1 then
    if output = 1 then 1 else 0
  else if output = 1 then 2 else 1

private theorem shiftedProductDigit_lt_three
    {output left right : F}
    (hOutput : FieldBitRoot output)
    (hLeft : FieldBitRoot left)
    (hRight : FieldBitRoot right) :
    shiftedProductDigit output left right < 3 := by
  rcases aggregateFieldBitRoot_cases hOutput with rfl | rfl <;>
    rcases aggregateFieldBitRoot_cases hLeft with rfl | rfl <;>
    rcases aggregateFieldBitRoot_cases hRight with rfl | rfl <;>
    decide

private theorem shiftedProductDigit_field
    {output left right : F}
    (hOutput : FieldBitRoot output)
    (hLeft : FieldBitRoot left)
    (hRight : FieldBitRoot right) :
    F.ofNat (shiftedProductDigit output left right) =
      (output - left * right) + 1 := by
  rcases aggregateFieldBitRoot_cases hOutput with rfl | rfl <;>
    rcases aggregateFieldBitRoot_cases hLeft with rfl | rfl <;>
    rcases aggregateFieldBitRoot_cases hRight with rfl | rfl <;>
    native_decide

private theorem shiftedProductDigit_eq_one_iff
    {output left right : F}
    (hOutput : FieldBitRoot output)
    (hLeft : FieldBitRoot left)
    (hRight : FieldBitRoot right) :
    shiftedProductDigit output left right = 1 ↔ output = left * right := by
  rcases aggregateFieldBitRoot_cases hOutput with rfl | rfl <;>
    rcases aggregateFieldBitRoot_cases hLeft with rfl | rfl <;>
    rcases aggregateFieldBitRoot_cases hRight with rfl | rfl <;>
    decide

/-- Ordinary trit corresponding to one bounded tree residual. -/
def productTreeDigit
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs)
    (index : Fin 14) : Nat :=
  shiftedProductDigit (outputs index)
    (productTreeLeft bits outputs index) (productTreeRight bits outputs index)

private theorem productTreeOperands_boolean
    {bits : Fin 16 → F} {outputs : ProductTreeOutputs}
    (hBits : ChunkBitsAreBoolean bits)
    (hOutputs : ∀ index, FieldBitRoot (outputs index)) :
    ∀ index,
      FieldBitRoot (productTreeLeft bits outputs index) ∧
        FieldBitRoot (productTreeRight bits outputs index) := by
  intro index
  fin_cases index <;>
    simp only [productTreeLeft, productTreeRight] <;>
    constructor <;> first | apply hBits | apply hOutputs

/-- One ProductSum-role row for all fourteen product-tree equations. -/
def ProductTreeAggregateRow
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) : Prop :=
  radix3Field (fun index => productTreeResidual bits outputs index) = 0

/-- The aggregate's maximum shifted value is strictly below Goldilocks. -/
theorem fourteenTrits_no_goldilocks_wrap
    {digits : Fin 14 → Nat} (hDigits : ∀ index, digits index < 3) :
    radix3Nat digits < Goldilocks.q :=
  lt_trans (radix3Nat_lt hDigits) (by native_decide)

/-- The balanced aggregate row is sound and complete for all fourteen edges. -/
theorem productTreeAggregateRow_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs)
    (hBits : ChunkBitsAreBoolean bits)
    (hOutputs : ∀ index, FieldBitRoot (outputs index)) :
    ProductTreeAggregateRow bits outputs ↔ ProductTreeMeaning bits outputs := by
  have hOperands := productTreeOperands_boolean hBits hOutputs
  have hDigitLt :
      ∀ index, productTreeDigit bits outputs index < 3 := by
    intro index
    exact shiftedProductDigit_lt_three (hOutputs index)
      (hOperands index).1 (hOperands index).2
  constructor
  · intro hAggregate
    have hDigitField :
        (fun index => F.ofNat (productTreeDigit bits outputs index)) =
          fun index => productTreeResidual bits outputs index + 1 := by
      funext index
      exact shiftedProductDigit_field (hOutputs index)
        (hOperands index).1 (hOperands index).2
    have hFieldEqual :
        F.ofNat (radix3Nat (productTreeDigit bits outputs)) =
          F.ofNat (radix3Nat (fun _ : Fin 14 => 1)) := by
      calc
        F.ofNat (radix3Nat (productTreeDigit bits outputs)) =
            radix3Field
              (fun index => F.ofNat (productTreeDigit bits outputs index)) :=
          fieldOfNat_radix3Nat _
        _ = radix3Field
              (fun index => productTreeResidual bits outputs index + 1) := by
          rw [hDigitField]
        _ = radix3Field
              (fun index => productTreeResidual bits outputs index) +
            radix3Field (fun _ : Fin 14 => 1) :=
          radix3Field_add _ _
        _ = radix3Field (fun _ : Fin 14 => 1) := by
          rw [hAggregate]
          simp
        _ = F.ofNat (radix3Nat (fun _ : Fin 14 => 1)) :=
          (fieldOfNat_radix3Nat _).symm
    have hDigitsCanonical :
        radix3Nat (productTreeDigit bits outputs) < Goldilocks.q :=
      fourteenTrits_no_goldilocks_wrap hDigitLt
    have hOnesCanonical :
        radix3Nat (fun _ : Fin 14 => 1) < Goldilocks.q :=
      fourteenTrits_no_goldilocks_wrap (by intro; omega)
    have hNatEqual :
        radix3Nat (productTreeDigit bits outputs) =
          radix3Nat (fun _ : Fin 14 => 1) := by
      have hValues := congrArg Fin.val hFieldEqual
      simpa [F.ofNat_val_eq_of_canonical hDigitsCanonical,
        F.ofNat_val_eq_of_canonical hOnesCanonical] using hValues
    have hDigitsEqual :
        productTreeDigit bits outputs = fun _ : Fin 14 => 1 :=
      radix3Nat_injective hDigitLt (by intro; omega) hNatEqual
    intro index
    apply sub_eq_zero.mpr
    exact (shiftedProductDigit_eq_one_iff (hOutputs index)
      (hOperands index).1 (hOperands index).2).mp
        (congrFun hDigitsEqual index)
  · intro hMeaning
    have hResiduals :
        (fun index => productTreeResidual bits outputs index) =
          fun _ : Fin 14 => 0 := by
      funext index
      exact hMeaning index
    rw [ProductTreeAggregateRow, hResiduals, radix3Field_zero]

/-! ## Complete nine-row acceptance relation -/

/-- Final ProductSum-role row deriving the acceptance cell from both roots. -/
def FinalAcceptanceRow (outputs : ProductTreeOutputs) (accept : F) : Prop :=
  outputs 6 * outputs 13 = 1 - accept

/-- Seven bit rows, one aggregate row, and one final acceptance row. -/
def AggregateAcceptanceRows
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) : Prop :=
  ProductTreeOutputBitRows outputs ∧
    ProductTreeAggregateRow bits outputs ∧
    FinalAcceptanceRow outputs accept

/-- Under source bitness, the nine rows are exactly the tree plus final row. -/
theorem aggregateAcceptanceRows_iff_treeAndFinal
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    AggregateAcceptanceRows bits outputs accept ↔
      ProductTreeMeaning bits outputs ∧ FinalAcceptanceRow outputs accept := by
  constructor
  · rintro ⟨hOutputRows, hAggregate, hFinal⟩
    have hOutputs := (productTreeOutputBitRows_iff outputs).mp hOutputRows
    exact ⟨(productTreeAggregateRow_iff bits outputs hBits hOutputs).mp hAggregate,
      hFinal⟩
  · rintro ⟨hMeaning, hFinal⟩
    have hOutputs := productTreeMeaning_outputs_boolean hBits hMeaning
    exact ⟨(productTreeOutputBitRows_iff outputs).mpr hOutputs,
      (productTreeAggregateRow_iff bits outputs hBits hOutputs).mpr hMeaning,
      hFinal⟩

/-! ## Source semantics and canonical extension -/

private theorem lowHalfProduct_explicit (bits : Fin 16 → F) :
    lowHalfProduct bits =
      ((bits 0 * bits 1) * (bits 2 * bits 3)) *
        ((bits 4 * bits 5) * (bits 6 * bits 7)) := by
  simp [lowHalfProduct, lowHalfIndex, Fin.prod_univ_succ]
  ring

private theorem highHalfProduct_explicit (bits : Fin 16 → F) :
    highHalfProduct bits =
      ((bits 8 * bits 9) * (bits 10 * bits 11)) *
        ((bits 12 * bits 13) * (bits 14 * bits 15)) := by
  simp [highHalfProduct, highHalfIndex, Fin.prod_univ_succ]
  ring

/-- The two tree roots are exactly the two readable eight-coordinate products. -/
theorem productTreeMeaning_roots
    {bits : Fin 16 → F} {outputs : ProductTreeOutputs}
    (hMeaning : ProductTreeMeaning bits outputs) :
    outputs 6 = lowHalfProduct bits ∧ outputs 13 = highHalfProduct bits := by
  rcases (productTreeMeaning_iff_equations bits outputs).mp hMeaning with
    ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩
  constructor
  · calc
      outputs 6 = outputs 4 * outputs 5 := h6
      _ = (outputs 0 * outputs 1) * (outputs 2 * outputs 3) := by
        rw [h4, h5]
      _ = ((bits 0 * bits 1) * (bits 2 * bits 3)) *
          ((bits 4 * bits 5) * (bits 6 * bits 7)) := by
        rw [h0, h1, h2, h3]
      _ = lowHalfProduct bits := (lowHalfProduct_explicit bits).symm
  · calc
      outputs 13 = outputs 11 * outputs 12 := h13
      _ = (outputs 7 * outputs 8) * (outputs 9 * outputs 10) := by
        rw [h11, h12]
      _ = ((bits 8 * bits 9) * (bits 10 * bits 11)) *
          ((bits 12 * bits 13) * (bits 14 * bits 15)) := by
        rw [h7, h8, h9, h10]
      _ = highHalfProduct bits := (highHalfProduct_explicit bits).symm

/-- Deterministic topological materialization of the fourteen tree outputs. -/
def canonicalProductTreeOutputs (bits : Fin 16 → F) : ProductTreeOutputs :=
  fun index =>
    match index.val with
    | 0 => bits 0 * bits 1
    | 1 => bits 2 * bits 3
    | 2 => bits 4 * bits 5
    | 3 => bits 6 * bits 7
    | 4 => (bits 0 * bits 1) * (bits 2 * bits 3)
    | 5 => (bits 4 * bits 5) * (bits 6 * bits 7)
    | 6 => ((bits 0 * bits 1) * (bits 2 * bits 3)) *
        ((bits 4 * bits 5) * (bits 6 * bits 7))
    | 7 => bits 8 * bits 9
    | 8 => bits 10 * bits 11
    | 9 => bits 12 * bits 13
    | 10 => bits 14 * bits 15
    | 11 => (bits 8 * bits 9) * (bits 10 * bits 11)
    | 12 => (bits 12 * bits 13) * (bits 14 * bits 15)
    | _ => ((bits 8 * bits 9) * (bits 10 * bits 11)) *
        ((bits 12 * bits 13) * (bits 14 * bits 15))

/-- The deterministic materializer satisfies every edge equation. -/
theorem canonicalProductTreeOutputs_meaning (bits : Fin 16 → F) :
    ProductTreeMeaning bits (canonicalProductTreeOutputs bits) := by
  apply (productTreeMeaning_iff_equations bits _).mpr
  simp [ProductTreeEquations, canonicalProductTreeOutputs]

/-- The topological tree witness is pointwise unique. -/
theorem productTreeMeaning_unique
    {bits : Fin 16 → F} {left right : ProductTreeOutputs}
    (hLeft : ProductTreeMeaning bits left)
    (hRight : ProductTreeMeaning bits right) :
    left = right := by
  rcases (productTreeMeaning_iff_equations bits left).mp hLeft with
    ⟨hl0, hl1, hl2, hl3, hl4, hl5, hl6,
      hl7, hl8, hl9, hl10, hl11, hl12, hl13⟩
  rcases (productTreeMeaning_iff_equations bits right).mp hRight with
    ⟨hr0, hr1, hr2, hr3, hr4, hr5, hr6,
      hr7, hr8, hr9, hr10, hr11, hr12, hr13⟩
  have h0 : left 0 = right 0 := hl0.trans hr0.symm
  have h1 : left 1 = right 1 := hl1.trans hr1.symm
  have h2 : left 2 = right 2 := hl2.trans hr2.symm
  have h3 : left 3 = right 3 := hl3.trans hr3.symm
  have h4 : left 4 = right 4 := by rw [hl4, hr4, h0, h1]
  have h5 : left 5 = right 5 := by rw [hl5, hr5, h2, h3]
  have h6 : left 6 = right 6 := by rw [hl6, hr6, h4, h5]
  have h7 : left 7 = right 7 := hl7.trans hr7.symm
  have h8 : left 8 = right 8 := hl8.trans hr8.symm
  have h9 : left 9 = right 9 := hl9.trans hr9.symm
  have h10 : left 10 = right 10 := hl10.trans hr10.symm
  have h11 : left 11 = right 11 := by rw [hl11, hr11, h7, h8]
  have h12 : left 12 = right 12 := by rw [hl12, hr12, h9, h10]
  have h13 : left 13 = right 13 := by rw [hl13, hr13, h11, h12]
  funext index
  fin_cases index <;> assumption

/-- The nine rows are exactly tree semantics plus the canonical source accept bit. -/
theorem aggregateAcceptanceRows_iff_sourceMeaning
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    AggregateAcceptanceRows bits outputs accept ↔
      ProductTreeMeaning bits outputs ∧ AcceptanceSourceMeaning bits accept := by
  constructor
  · intro hRows
    rcases (aggregateAcceptanceRows_iff_treeAndFinal
      bits outputs accept hBits).mp hRows with ⟨hMeaning, hFinal⟩
    have hRoots := productTreeMeaning_roots hMeaning
    have hAccept : accept =
        1 - lowHalfProduct bits * highHalfProduct bits := by
      rw [FinalAcceptanceRow, hRoots.1, hRoots.2] at hFinal
      rw [hFinal]
      ring
    have hPacked :
        PackedAcceptanceMeaning bits (lowHalfProduct bits)
          (highHalfProduct bits) accept := by
      refine ⟨rfl, rfl, ?_, ?_, hAccept⟩
      · rw [← hRoots.1]
        exact productTreeMeaning_outputs_boolean hBits hMeaning 6
      · rw [← hRoots.2]
        exact productTreeMeaning_outputs_boolean hBits hMeaning 13
    exact ⟨hMeaning,
      (packedAcceptanceMeaning_iff_sourceMeaning bits accept hBits).mp hPacked⟩
  · rintro ⟨hMeaning, hSource⟩
    have hPacked :=
      (packedAcceptanceMeaning_iff_sourceMeaning bits accept hBits).mpr hSource
    have hRoots := productTreeMeaning_roots hMeaning
    apply (aggregateAcceptanceRows_iff_treeAndFinal
      bits outputs accept hBits).mpr
    refine ⟨hMeaning, ?_⟩
    rw [FinalAcceptanceRow, hRoots.1, hRoots.2, hPacked.2.2.2.2]
    ring

/-- The aggregate relation derives, rather than separately constrains, `accept` bitness. -/
theorem aggregateAcceptanceRows_implies_accept_bit
    {bits : Fin 16 → F} {outputs : ProductTreeOutputs} {accept : F}
    (hBits : ChunkBitsAreBoolean bits)
    (hRows : AggregateAcceptanceRows bits outputs accept) :
    FieldBitRoot accept :=
  ((aggregateAcceptanceRows_iff_sourceMeaning
    bits outputs accept hBits).mp hRows).2.1

/-- Canonical source rows and existential aggregate outputs accept identically. -/
theorem canonicalAcceptanceSourceRows_iff_exists_aggregateAcceptanceRows
    (bits : Fin 16 → F) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    (∃ inverse, CanonicalAcceptanceSourceRows bits accept inverse) ↔
      ∃ outputs, AggregateAcceptanceRows bits outputs accept := by
  rw [canonicalAcceptanceSourceRows_exists_iff bits accept hBits]
  constructor
  · intro hSource
    refine ⟨canonicalProductTreeOutputs bits, ?_⟩
    exact (aggregateAcceptanceRows_iff_sourceMeaning
      bits _ accept hBits).mpr
        ⟨canonicalProductTreeOutputs_meaning bits, hSource⟩
  · rintro ⟨outputs, hRows⟩
    exact ((aggregateAcceptanceRows_iff_sourceMeaning
      bits outputs accept hBits).mp hRows).2

/-- Complete witness carried by an inactive/unconditional acceptance block. -/
@[ext] structure AggregateAcceptanceWitness where
  outputs : ProductTreeOutputs
  accept : F

/-- Every Boolean source chunk has one and only one aggregate-row extension. -/
theorem aggregateAcceptanceRows_extension_exact
    (bits : Fin 16 → F) (hBits : ChunkBitsAreBoolean bits) :
    ∃! witness : AggregateAcceptanceWitness,
      AggregateAcceptanceRows bits witness.outputs witness.accept := by
  let sourceWitness := canonicalAcceptanceMaterializer bits
  let witness : AggregateAcceptanceWitness :=
    { outputs := canonicalProductTreeOutputs bits
      accept := sourceWitness.accept }
  have hSourceRows := canonicalAcceptanceMaterializer_holds bits
  have hSourceMeaning : AcceptanceSourceMeaning bits sourceWitness.accept :=
    (canonicalAcceptanceSourceRows_exists_iff bits sourceWitness.accept hBits).mp
      ⟨sourceWitness.inverse, hSourceRows⟩
  have hWitnessRows :
      AggregateAcceptanceRows bits witness.outputs witness.accept :=
    (aggregateAcceptanceRows_iff_sourceMeaning
      bits witness.outputs witness.accept hBits).mpr
        ⟨canonicalProductTreeOutputs_meaning bits, hSourceMeaning⟩
  refine ⟨witness, hWitnessRows, ?_⟩
  intro other hOtherRows
  have hWitnessMeaning := (aggregateAcceptanceRows_iff_treeAndFinal
    bits witness.outputs witness.accept hBits).mp hWitnessRows
  have hOtherMeaning := (aggregateAcceptanceRows_iff_treeAndFinal
    bits other.outputs other.accept hBits).mp hOtherRows
  have hOutputs : witness.outputs = other.outputs :=
    productTreeMeaning_unique hWitnessMeaning.1 hOtherMeaning.1
  have hAccept : witness.accept = other.accept := by
    have hWitnessFinal := hWitnessMeaning.2
    have hOtherFinal := hOtherMeaning.2
    rw [hOutputs] at hWitnessFinal
    unfold FinalAcceptanceRow at hWitnessFinal hOtherFinal
    have hEqual : 1 - witness.accept = 1 - other.accept :=
      hWitnessFinal.symm.trans hOtherFinal
    calc
      witness.accept = 1 - (1 - witness.accept) := by ring
      _ = 1 - (1 - other.accept) := by rw [hEqual]
      _ = other.accept := by ring
  apply AggregateAcceptanceWitness.ext
  · exact hOutputs.symm
  · exact hAccept.symm

/-! ## Per-family necessity witnesses -/

/-- Cancellation witness available when output bit rows are omitted. -/
private def outputBitRowsCounterexample : ProductTreeOutputs :=
  fun index =>
    match index.val with
    | 0 => 3
    | 1 => -1
    | 4 => -3
    | _ => 0

/-- Without output bitness, residuals `3` and `-1` cancel at weights `1,3`. -/
theorem productTreeOutputBitRows_are_necessary :
    ∃ bits : Fin 16 → F, ∃ outputs : ProductTreeOutputs, ∃ accept : F,
      ChunkBitsAreBoolean bits ∧
        ProductTreeAggregateRow bits outputs ∧
        FinalAcceptanceRow outputs accept ∧
        ¬ ProductTreeOutputBitRows outputs ∧
        ¬ ProductTreeMeaning bits outputs := by
  refine ⟨fun _ => 0, outputBitRowsCounterexample, 1, ?_, ?_, ?_, ?_, ?_⟩
  · intro index
    simp [FieldBitRoot, fieldBitResidual]
  · norm_num [ProductTreeAggregateRow, radix3Field, productTreeResidual,
      productTreeLeft, productTreeRight, outputBitRowsCounterexample]
  · norm_num [FinalAcceptanceRow, outputBitRowsCounterexample]
  · intro hRows
    have hOutputZero :=
      (productTreeOutputBitRows_iff outputBitRowsCounterexample).mp hRows 0
    rcases aggregateFieldBitRoot_cases hOutputZero with hZero | hOne
    · have hValues := congrArg Fin.val hZero
      norm_num [outputBitRowsCounterexample, F.ofNat, F.val_zero,
        Goldilocks.q] at hValues
    · have hValues := congrArg Fin.val hOne
      norm_num [outputBitRowsCounterexample, F.ofNat, Goldilocks.q] at hValues
  · intro hMeaning
    have hFirst := hMeaning (0 : Fin 14)
    have hValues := congrArg Fin.val hFirst
    norm_num [productTreeResidual, productTreeLeft, productTreeRight,
      outputBitRowsCounterexample, F.ofNat, F.val_zero,
      Goldilocks.q] at hValues

/-- Boolean-output witness available when the aggregate row is omitted. -/
private def aggregateRowCounterexample : ProductTreeOutputs :=
  fun index =>
    match index.val with
    | 0 => 1
    | _ => 0

/-- Without the aggregate row, the other families allow a false first edge. -/
theorem productTreeAggregateRow_is_necessary :
    ∃ bits : Fin 16 → F, ∃ outputs : ProductTreeOutputs, ∃ accept : F,
      ChunkBitsAreBoolean bits ∧
        ProductTreeOutputBitRows outputs ∧
        FinalAcceptanceRow outputs accept ∧
        ¬ ProductTreeMeaning bits outputs := by
  refine ⟨fun _ => 0, aggregateRowCounterexample, 1, ?_, ?_, ?_, ?_⟩
  · intro index
    simp [FieldBitRoot, fieldBitResidual]
  · apply (productTreeOutputBitRows_iff aggregateRowCounterexample).mpr
    intro index
    fin_cases index <;>
      simp [aggregateRowCounterexample, FieldBitRoot, fieldBitResidual]
  · norm_num [FinalAcceptanceRow, aggregateRowCounterexample]
  · intro hMeaning
    have hFirst := hMeaning (0 : Fin 14)
    norm_num [productTreeResidual, productTreeLeft, productTreeRight,
      aggregateRowCounterexample, F.ofNat] at hFirst

/-- Without the final row, a correct all-zero tree admits the wrong accept bit. -/
theorem finalAcceptanceRow_is_necessary :
    ∃ bits : Fin 16 → F, ∃ outputs : ProductTreeOutputs, ∃ accept : F,
      ChunkBitsAreBoolean bits ∧
        ProductTreeOutputBitRows outputs ∧
        ProductTreeAggregateRow bits outputs ∧
        ProductTreeMeaning bits outputs ∧
        ¬ AcceptanceSourceMeaning bits accept := by
  refine ⟨fun _ => 0, fun _ => 0, 0, ?_, ?_, ?_, ?_, ?_⟩
  · intro index
    simp [FieldBitRoot, fieldBitResidual]
  · apply (productTreeOutputBitRows_iff (fun _ => 0)).mpr
    intro index
    simp [FieldBitRoot, fieldBitResidual]
  · rw [ProductTreeAggregateRow]
    have hResiduals :
        (fun index => productTreeResidual (fun _ => 0) (fun _ => 0) index) =
          fun _ : Fin 14 => 0 := by
      funext index
      fin_cases index <;>
        simp [productTreeResidual, productTreeLeft, productTreeRight]
    rw [hResiduals, radix3Field_zero]
  · intro index
    fin_cases index <;>
      simp [productTreeResidual, productTreeLeft, productTreeRight]
  · intro hSource
    have hNotAll : ¬ AllChunkBitsOne (fun _ : Fin 16 => (0 : F)) := by
      intro hAll
      have hZeroOne := hAll (0 : Fin 16)
      exact zero_ne_one hZeroOne
    have hZeroOne := hSource.2.mpr hNotAll
    exact zero_ne_one hZeroOne

/-! ## Exact model shape, role, and degree contract -/

/-- Fourteen internal outputs plus the derived accept cell. -/
def aggregateAcceptanceCommittedCoordinates : Nat := 14 + 1

/-- Seven packed bit rows, one tree aggregate, and one final product row. -/
def aggregateAcceptanceRowsPerChunk : Nat := 7 + 1 + 1

/-- Number of multiplication identities represented in the tree ProductSum. -/
def aggregateAcceptanceTreeProducts : Nat := 14

/-- Existing ProductSum trace capacity used by the production role model. -/
def aggregateAcceptanceProductSumCapacity : Nat := 18

/-- Candidate model introduces no matrix role beyond Mod5 and ProductSum. -/
def aggregateAcceptanceAdditionalMatrixRoles : Nat := 0

/-- Candidate model introduces no polynomial term family beyond existing roles. -/
def aggregateAcceptanceAdditionalPolynomialFamilies : Nat := 0

/-- Raw degree of a packed pair of two quadratic bit residuals. -/
def aggregateAcceptanceOutputBitRowDegree : Nat := 4

/-- Raw degree of the weighted tree ProductSum equation. -/
def aggregateAcceptanceTreeRowDegree : Nat := 2

/-- Raw degree of `o_6 * o_13 = 1-a`. -/
def aggregateAcceptanceFinalRowDegree : Nat := 2

/-- A linear selector raises one row's degree by exactly one. -/
def aggregateAcceptanceSelectorDegree (degree : Nat) : Nat := degree + 1

/-- Exact coordinate/row count and existing-role capacity contract. -/
theorem aggregateAcceptance_existing_role_contract :
    aggregateAcceptanceCommittedCoordinates = 15 ∧
      aggregateAcceptanceRowsPerChunk = 9 ∧
      aggregateAcceptanceTreeProducts = 14 ∧
      aggregateAcceptanceTreeProducts ≤ aggregateAcceptanceProductSumCapacity ∧
      aggregateAcceptanceAdditionalMatrixRoles = 0 ∧
      aggregateAcceptanceAdditionalPolynomialFamilies = 0 := by
  norm_num [aggregateAcceptanceCommittedCoordinates,
    aggregateAcceptanceRowsPerChunk, aggregateAcceptanceTreeProducts,
    aggregateAcceptanceProductSumCapacity,
    aggregateAcceptanceAdditionalMatrixRoles,
    aggregateAcceptanceAdditionalPolynomialFamilies]

/-- Every candidate family remains within the fixed degree-eight ceiling. -/
theorem aggregateAcceptance_degree_contract :
    aggregateAcceptanceOutputBitRowDegree = 4 ∧
      aggregateAcceptanceTreeRowDegree = 2 ∧
      aggregateAcceptanceFinalRowDegree = 2 ∧
      aggregateAcceptanceSelectorDegree
          aggregateAcceptanceOutputBitRowDegree = 5 ∧
      aggregateAcceptanceSelectorDegree
          aggregateAcceptanceTreeRowDegree = 3 ∧
      aggregateAcceptanceSelectorDegree
          aggregateAcceptanceFinalRowDegree = 3 ∧
      aggregateAcceptanceSelectorDegree
          aggregateAcceptanceOutputBitRowDegree ≤ 8 ∧
      aggregateAcceptanceSelectorDegree
          aggregateAcceptanceTreeRowDegree ≤ 8 ∧
      aggregateAcceptanceSelectorDegree
          aggregateAcceptanceFinalRowDegree ≤ 8 := by
  norm_num [aggregateAcceptanceOutputBitRowDegree,
    aggregateAcceptanceTreeRowDegree, aggregateAcceptanceFinalRowDegree,
    aggregateAcceptanceSelectorDegree]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
