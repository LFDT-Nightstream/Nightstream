import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.Semantics

/-!
Exactness of the model-level aggregate acceptance relation for one chunk.

Owns: decoding paired output-bit equations, the collision-free radix-three
aggregate, the root/accept equation, soundness and completeness against the
independent tree/source semantics, and uniqueness of the canonical extension.

Does not own: generated rows, Rust emission, production column placement,
fixed selectors, inactive placement, the 960-chunk outer image, cost totals,
or row-removal authority.

Emits constraints: no. These predicates describe a candidate relation.

| Exact Rust stage path | Mathematical obligation | Equations | Principal result |
|---|---|---:|---|
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.tree_bit_pairs` | pair fourteen output bit residuals | 7 | `productTreeOutputBitRows_iff` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.product_aggregate` | compare two width-14 radix-three images | 1 | `productTreeAggregateRow_iff` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.root_binding` | derive accept from both tree roots | 1 | `aggregateAcceptanceRows_iff_sourceMeaning` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.root_binding` | prove canonical extension uniqueness | 0 | `aggregateAcceptanceRows_extension_exact` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Mod5

private theorem gateField_mul_eq_zero_cases
    (prime : EuclidPrime goldilocksP) {left right : GateField}
    (zero : left * right = 0) : left = 0 ∨ right = 0 := by
  have values := congrArg Fin.val zero
  simp only [Fin.val_mul, Fin.val_zero] at values
  rcases prime left.val right.val values with leftZero | rightZero
  · left
    apply Fin.ext
    simpa [Nat.mod_eq_of_lt left.isLt] using leftZero
  · right
    apply Fin.ext
    simpa [Nat.mod_eq_of_lt right.isLt] using rightZero

theorem bitResidual_zero_iff
    (prime : EuclidPrime goldilocksP) (value : GateField) :
    bitResidual value = 0 ↔ FieldBit value := by
  constructor
  · intro zero
    unfold bitResidual at zero
    rcases gateField_mul_eq_zero_cases prime zero with valueZero | subZero
    · exact Or.inl valueZero
    · exact Or.inr ((fieldSub_eq_zero_iff value 1).mp subZero)
  · rintro (rfl | rfl) <;>
      apply Fin.ext <;> native_decide

/-! ## Seven paired output-bit equations -/

def ProductTreeOutputBitRows (outputs : ProductTreeOutputs) : Prop :=
  QuadraticZeroPair (bitResidual (outputs 0)) (bitResidual (outputs 1)) ∧
    QuadraticZeroPair (bitResidual (outputs 2)) (bitResidual (outputs 3)) ∧
    QuadraticZeroPair (bitResidual (outputs 4)) (bitResidual (outputs 5)) ∧
    QuadraticZeroPair (bitResidual (outputs 6)) (bitResidual (outputs 7)) ∧
    QuadraticZeroPair (bitResidual (outputs 8)) (bitResidual (outputs 9)) ∧
    QuadraticZeroPair (bitResidual (outputs 10)) (bitResidual (outputs 11)) ∧
    QuadraticZeroPair (bitResidual (outputs 12)) (bitResidual (outputs 13))

theorem productTreeOutputBitRows_iff
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    (outputs : ProductTreeOutputs) :
    ProductTreeOutputBitRows outputs ↔
      ∀ index, FieldBit (outputs index) := by
  constructor
  · rintro ⟨h01, h23, h45, h67, h89, h1011, h1213⟩
    have h01' := (quadraticZeroPair_iff nonresidue).mp h01
    have h23' := (quadraticZeroPair_iff nonresidue).mp h23
    have h45' := (quadraticZeroPair_iff nonresidue).mp h45
    have h67' := (quadraticZeroPair_iff nonresidue).mp h67
    have h89' := (quadraticZeroPair_iff nonresidue).mp h89
    have h1011' := (quadraticZeroPair_iff nonresidue).mp h1011
    have h1213' := (quadraticZeroPair_iff nonresidue).mp h1213
    exact fin14_all
      ((bitResidual_zero_iff prime _).mp h01'.1)
      ((bitResidual_zero_iff prime _).mp h01'.2)
      ((bitResidual_zero_iff prime _).mp h23'.1)
      ((bitResidual_zero_iff prime _).mp h23'.2)
      ((bitResidual_zero_iff prime _).mp h45'.1)
      ((bitResidual_zero_iff prime _).mp h45'.2)
      ((bitResidual_zero_iff prime _).mp h67'.1)
      ((bitResidual_zero_iff prime _).mp h67'.2)
      ((bitResidual_zero_iff prime _).mp h89'.1)
      ((bitResidual_zero_iff prime _).mp h89'.2)
      ((bitResidual_zero_iff prime _).mp h1011'.1)
      ((bitResidual_zero_iff prime _).mp h1011'.2)
      ((bitResidual_zero_iff prime _).mp h1213'.1)
      ((bitResidual_zero_iff prime _).mp h1213'.2)
  · intro outputBits
    exact
      ⟨(quadraticZeroPair_iff nonresidue).mpr
          ⟨(bitResidual_zero_iff prime _).mpr (outputBits 0),
            (bitResidual_zero_iff prime _).mpr (outputBits 1)⟩,
        (quadraticZeroPair_iff nonresidue).mpr
          ⟨(bitResidual_zero_iff prime _).mpr (outputBits 2),
            (bitResidual_zero_iff prime _).mpr (outputBits 3)⟩,
        (quadraticZeroPair_iff nonresidue).mpr
          ⟨(bitResidual_zero_iff prime _).mpr (outputBits 4),
            (bitResidual_zero_iff prime _).mpr (outputBits 5)⟩,
        (quadraticZeroPair_iff nonresidue).mpr
          ⟨(bitResidual_zero_iff prime _).mpr (outputBits 6),
            (bitResidual_zero_iff prime _).mpr (outputBits 7)⟩,
        (quadraticZeroPair_iff nonresidue).mpr
          ⟨(bitResidual_zero_iff prime _).mpr (outputBits 8),
            (bitResidual_zero_iff prime _).mpr (outputBits 9)⟩,
        (quadraticZeroPair_iff nonresidue).mpr
          ⟨(bitResidual_zero_iff prime _).mpr (outputBits 10),
            (bitResidual_zero_iff prime _).mpr (outputBits 11)⟩,
        (quadraticZeroPair_iff nonresidue).mpr
          ⟨(bitResidual_zero_iff prime _).mpr (outputBits 12),
            (bitResidual_zero_iff prime _).mpr (outputBits 13)⟩⟩

/-! ## Collision-free weighted product aggregate -/

def radix3Nat : {count : Nat} → (Fin count → Nat) → Nat
  | 0, _ => 0
  | _count + 1, digits =>
      digits 0 + 3 * radix3Nat (fun index => digits index.succ)

/-- Radix-three interpretation directly in the active Goldilocks carrier. -/
def radix3Field : {count : Nat} → (Fin count → GateField) → GateField
  | 0, _ => 0
  | _count + 1, digits =>
      digits 0 + fieldResidue 3 *
        radix3Field (fun index => digits index.succ)

theorem radix3Field_eq_residue
    {count : Nat} (digits : Fin count → GateField) :
    radix3Field digits =
      fieldResidue (radix3Nat (fun index => (digits index).val)) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [radix3Field, radix3Nat]
      rw [fieldResidue_add_hom, fieldResidue_mul_hom,
        fieldResidue_val, inductionHypothesis]

theorem radix3Nat_lt
    {count : Nat} {digits : Fin count → Nat}
    (digitBound : ∀ index, digits index < 3) :
    radix3Nat digits < 3 ^ count := by
  induction count with
  | zero => simp [radix3Nat]
  | succ count inductionHypothesis =>
      have headBound := digitBound (0 : Fin (count + 1))
      have tailBound :
          radix3Nat (fun index : Fin count => digits index.succ) < 3 ^ count :=
        inductionHypothesis (fun index => digitBound index.succ)
      simp only [radix3Nat, Nat.pow_succ]
      omega

theorem radix3Nat_injective
    {count : Nat} {left right : Fin count → Nat}
    (leftBound : ∀ index, left index < 3)
    (rightBound : ∀ index, right index < 3)
    (equal : radix3Nat left = radix3Nat right) :
    left = right := by
  induction count with
  | zero =>
      funext index
      exact Fin.elim0 index
  | succ count inductionHypothesis =>
      have expanded := equal
      simp only [radix3Nat] at expanded
      have head : left 0 = right 0 := by
        have leftHead := leftBound (0 : Fin (count + 1))
        have rightHead := rightBound (0 : Fin (count + 1))
        omega
      have tail :
          radix3Nat (fun index : Fin count => left index.succ) =
            radix3Nat (fun index : Fin count => right index.succ) := by
        omega
      have tailFunctions := inductionHypothesis
        (fun index => leftBound index.succ)
        (fun index => rightBound index.succ) tail
      funext index
      refine Fin.cases head ?_ index
      intro tailIndex
      exact congrFun tailFunctions tailIndex

def productTreeProducts
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs) :
    Fin 14 → GateField :=
  fun index =>
    productTreeLeft bits outputs index * productTreeRight bits outputs index

/-- One Goldilocks equality with weights `3^0` through `3^13`. -/
def ProductTreeAggregateRow
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs) : Prop :=
  radix3Nat (fun index => (outputs index).val) % goldilocksP =
    radix3Nat (fun index => (productTreeProducts bits outputs index).val) %
      goldilocksP

theorem fourteen_radix3_digits_no_wrap
    {digits : Fin 14 → Nat} (digitBound : ∀ index, digits index < 3) :
    radix3Nat digits < goldilocksP :=
  Nat.lt_trans (radix3Nat_lt digitBound) (by native_decide)

private theorem fieldBit_val_lt_three
    {value : GateField} (bit : FieldBit value) : value.val < 3 := by
  rcases bit with rfl | rfl <;> decide

theorem productTreeAggregateRow_iff
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (sourceBoolean : ∀ index, FieldBit (bits index))
    (outputsBoolean : ∀ index, FieldBit (outputs index)) :
    ProductTreeAggregateRow bits outputs ↔
      ProductTreeMeaning bits outputs := by
  have operandsBoolean :=
    productTreeOperands_boolean sourceBoolean outputsBoolean
  have productsBoolean :
      ∀ index, FieldBit (productTreeProducts bits outputs index) := by
    intro index
    exact fieldBit_mul (operandsBoolean index).1 (operandsBoolean index).2
  have outputsBound : ∀ index, (outputs index).val < 3 :=
    fun index => fieldBit_val_lt_three (outputsBoolean index)
  have productsBound :
      ∀ index, (productTreeProducts bits outputs index).val < 3 :=
    fun index => fieldBit_val_lt_three (productsBoolean index)
  constructor
  · intro aggregate
    have outputsNoWrap := fourteen_radix3_digits_no_wrap outputsBound
    have productsNoWrap := fourteen_radix3_digits_no_wrap productsBound
    unfold ProductTreeAggregateRow at aggregate
    rw [Nat.mod_eq_of_lt outputsNoWrap,
      Nat.mod_eq_of_lt productsNoWrap] at aggregate
    have valuesEqual :=
      radix3Nat_injective outputsBound productsBound aggregate
    intro index
    apply Fin.ext
    exact congrFun valuesEqual index
  · intro meaning
    have pointwise :
        (fun index => (outputs index).val) =
          fun index => (productTreeProducts bits outputs index).val := by
      funext index
      exact congrArg Fin.val (meaning index)
    unfold ProductTreeAggregateRow
    rw [pointwise]

/-- The final equation derives the acceptance bit from both tree roots. -/
def FinalAcceptanceRow
    (outputs : ProductTreeOutputs) (accept : GateField) : Prop :=
  accept = fieldSub 1 (outputs 6 * outputs 13)

/-- Seven paired output-bit equations, one collision-free aggregate, and one
root/accept equation. This is a model-level candidate relation, not a claim
about production row placement. -/
def AggregateAcceptanceRows
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) : Prop :=
  ProductTreeOutputBitRows outputs ∧
    ProductTreeAggregateRow bits outputs ∧
    FinalAcceptanceRow outputs accept

theorem aggregateAcceptanceRows_iff_treeAndFinal
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) (sourceBoolean : ∀ index, FieldBit (bits index)) :
    AggregateAcceptanceRows bits outputs accept ↔
      ProductTreeMeaning bits outputs ∧
        FinalAcceptanceRow outputs accept := by
  constructor
  · rintro ⟨outputRows, aggregate, final⟩
    have outputBoolean :=
      (productTreeOutputBitRows_iff prime nonresidue outputs).mp outputRows
    exact ⟨(productTreeAggregateRow_iff bits outputs
      sourceBoolean outputBoolean).mp aggregate, final⟩
  · rintro ⟨meaning, final⟩
    have outputBoolean :=
      productTreeMeaning_outputs_boolean sourceBoolean meaning
    exact
      ⟨(productTreeOutputBitRows_iff prime nonresidue outputs).mpr outputBoolean,
        (productTreeAggregateRow_iff bits outputs
          sourceBoolean outputBoolean).mpr meaning,
        final⟩

theorem aggregateAcceptanceRows_iff_sourceMeaning
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs)
    (accept : GateField) (sourceBoolean : ∀ index, FieldBit (bits index)) :
    AggregateAcceptanceRows bits outputs accept ↔
      ProductTreeMeaning bits outputs ∧
        SourceAcceptanceMeaning bits accept := by
  constructor
  · intro rows
    rcases (aggregateAcceptanceRows_iff_treeAndFinal prime nonresidue
      bits outputs accept sourceBoolean).mp rows with ⟨meaning, final⟩
    have roots := productTreeMeaning_roots meaning
    unfold FinalAcceptanceRow at final
    unfold SourceAcceptanceMeaning
    rw [roots.1, roots.2] at final
    exact ⟨meaning, final⟩
  · rintro ⟨meaning, sourceMeaning⟩
    apply (aggregateAcceptanceRows_iff_treeAndFinal prime nonresidue
      bits outputs accept sourceBoolean).mpr
    have roots := productTreeMeaning_roots meaning
    refine ⟨meaning, ?_⟩
    unfold FinalAcceptanceRow
    unfold SourceAcceptanceMeaning at sourceMeaning
    rw [roots.1, roots.2]
    exact sourceMeaning

theorem aggregateAcceptanceRows_iff_verifierMeaning
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : BitsBoolean assignment chunk) (outputs : ProductTreeOutputs)
    (accept : GateField) :
    AggregateAcceptanceRows (sourceBits assignment chunk) outputs accept ↔
      ProductTreeMeaning (sourceBits assignment chunk) outputs ∧
        VerifierAcceptanceMeaning assignment chunk bits accept := by
  rw [aggregateAcceptanceRows_iff_sourceMeaning prime nonresidue
    (sourceBits assignment chunk) outputs accept (sourceBits_are_boolean bits),
    sourceAcceptanceMeaning_iff_verifier bits accept]

/-! ## Canonical extension -/

@[ext] structure AggregateAcceptanceWitness where
  outputs : ProductTreeOutputs
  accept : GateField

def canonicalAggregateAcceptanceWitness
    (bits : Fin 16 → GateField) : AggregateAcceptanceWitness where
  outputs := canonicalProductTreeOutputs bits
  accept := fieldSub 1 (lowHalfProduct bits * highHalfProduct bits)

theorem aggregateAcceptanceRows_extension_exact
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    (bits : Fin 16 → GateField)
    (sourceBoolean : ∀ index, FieldBit (bits index)) :
    ∃ witness : AggregateAcceptanceWitness,
      AggregateAcceptanceRows bits witness.outputs witness.accept ∧
        ∀ other : AggregateAcceptanceWitness,
          AggregateAcceptanceRows bits other.outputs other.accept →
            other = witness := by
  let witness := canonicalAggregateAcceptanceWitness bits
  have witnessMeaning : ProductTreeMeaning bits witness.outputs := by
    exact canonicalProductTreeOutputs_meaning bits
  have witnessSource : SourceAcceptanceMeaning bits witness.accept := by
    rfl
  have witnessRows :
      AggregateAcceptanceRows bits witness.outputs witness.accept :=
    (aggregateAcceptanceRows_iff_sourceMeaning prime nonresidue
      bits witness.outputs witness.accept sourceBoolean).mpr
        ⟨witnessMeaning, witnessSource⟩
  refine ⟨witness, witnessRows, ?_⟩
  intro other otherRows
  have otherSemantic :=
    (aggregateAcceptanceRows_iff_sourceMeaning prime nonresidue
      bits other.outputs other.accept sourceBoolean).mp otherRows
  have outputEqual : other.outputs = witness.outputs :=
    productTreeMeaning_unique otherSemantic.1 witnessMeaning
  have acceptEqual : other.accept = witness.accept := by
    unfold SourceAcceptanceMeaning at otherSemantic witnessSource
    exact otherSemantic.2.trans witnessSource.symm
  apply AggregateAcceptanceWitness.ext
  · exact outputEqual
  · exact acceptEqual

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance
