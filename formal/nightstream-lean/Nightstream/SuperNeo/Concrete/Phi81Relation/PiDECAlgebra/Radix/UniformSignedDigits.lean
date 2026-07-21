import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix

/-!
Model-level uniqueness of the production signed-binary `PiDEC` split.

Protocol: SuperNeo `Pi_DEC` at production `b = 2`, `k = 14`.
Phase: verifier-owned public-input digit decomposition.
Constraint family: abstract common-sign digit acceptance; this file emits no
rows.

Owns: a row-encoding-independent predicate saying that one sign is in
`{-1, 0, 1}`, every digit is either zero or that common sign, and the digits
recompose to the parent; honest completeness for `Radix.splitScalar`; and
uniqueness of the accepted digits for every strictly `B`-bounded parent.

Does not own: a polynomial or R1CS encoding of the predicate; canonical field
representative checks at an implementation boundary; Rust refinement; row
counts; generated artifacts; or authorization to remove any constraint.

Authority boundary: the soundness theorem starts from the independent
strict-`B` parent bound and the semantic radix recomposition. In particular,
fourteen independent centered-unit facts are not mistaken for canonical digit
selection: all digits must share the one accepted sign.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.paper.public_split.sign` | one common sign lies in `{-1,0,1}` | checked | `SignAllowed` |
| `nifs.pi_dec.paper.public_split.digit` | every child digit is zero or the common sign | checked | `ConstraintPredicate` |
| `nifs.pi_dec.paper.public_split.recompose` | signed digits recompose to the bounded parent | checked | `Accepted.recomposition` |
| `nifs.pi_dec.paper.public_split.exact` | accepted digits equal `Radix.splitScalar` | derived | `accepted_digits_eq_splitScalar` |
| `nifs.pi_dec.paper.public_split.complete` | canonical digits satisfy the predicate | derived | `honest_complete` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm

/-! ## Row-independent accepted roots -/

/-- The shared signed-binary direction. This is an accepted-root predicate,
not a claim about any particular polynomial encoding. -/
def SignAllowed (sign : F) : Prop :=
  sign = 0 ∨ sign = 1 ∨ sign = -1

/-- The semantic content of one common-sign family: after selecting one
allowed sign, each digit is either inactive or equal to that sign. -/
def ConstraintPredicate (sign : F) (digits : ChildIndex → F) : Prop :=
  SignAllowed sign ∧ ∀ index, digits index = 0 ∨ digits index = sign

/-- The row-count-neutral candidate accepted by the proposed family. -/
structure Accepted (parent sign : F) (digits : ChildIndex → F) : Prop where
  constraint : ConstraintPredicate sign digits
  recomposition : recomposeScalar digits = parent

/-- The honest common sign is derived from the same centered branch used by
the canonical production split. Zero uses the positive branch. -/
def honestSign (parent : F) : F :=
  if isNonnegative parent then 1 else -1

/-! ## Finite binary uniqueness -/

/-- Head-first natural sum over a fixed finite index set. -/
private def sumNats : {count : Nat} → (Fin count → Nat) → Nat
  | 0, _ => 0
  | _ + 1, values => values 0 + sumNats (fun index => values index.succ)

private theorem sumNats_mul (factor : Nat) {count : Nat}
    (values : Fin count → Nat) :
    sumNats (fun index => factor * values index) =
      factor * sumNats values := by
  induction count with
  | zero => simp [sumNats]
  | succ count inductionHypothesis =>
      simp only [sumNats]
      rw [inductionHypothesis (fun index => values index.succ)]
      exact (Nat.mul_add factor _ _).symm

/-- Little-endian natural value of a fixed binary word. -/
private def binaryValue {count : Nat} (bits : Fin count → Nat) : Nat :=
  sumNats fun index => 2 ^ index.val * bits index

private theorem binaryValue_succ {count : Nat}
    (bits : Fin (count + 1) → Nat) :
    binaryValue bits =
      bits 0 + 2 * binaryValue (fun index => bits index.succ) := by
  unfold binaryValue
  simp only [sumNats, Fin.val_zero, Nat.pow_zero, Nat.one_mul]
  congr 1
  calc
    sumNats (fun index : Fin count =>
        2 ^ index.succ.val * bits index.succ) =
        sumNats (fun index : Fin count =>
          2 * (2 ^ index.val * bits index.succ)) := by
      congr 1
      funext index
      simp [Nat.pow_succ', Nat.mul_assoc]
    _ = 2 * binaryValue (fun index => bits index.succ) := by
      rw [sumNats_mul]
      rfl

private theorem binaryValue_lt_pow {count : Nat}
    (bits : Fin count → Nat)
    (binary : ∀ index, bits index < 2) :
    binaryValue bits < 2 ^ count := by
  induction count with
  | zero => simp [binaryValue, sumNats]
  | succ count inductionHypothesis =>
      rw [binaryValue_succ, Nat.pow_succ']
      have headBound := binary 0
      have tailBound := inductionHypothesis
        (fun index => bits index.succ)
        (fun index => binary index.succ)
      omega

private theorem natBit_succ (value index : Nat) :
    natBit value (index + 1) = natBit (value / 2) index := by
  simp [natBit, Nat.div_div_eq_div_mul, Nat.pow_succ']

/-- Fixed-width binary expansion is injective. This proof is structural in the
width; it does not enumerate the `2^14` possible production words. -/
private theorem bits_eq_natBit_binaryValue {count : Nat}
    (bits : Fin count → Nat)
    (binary : ∀ index, bits index < 2) :
    bits = fun index => natBit (binaryValue bits) index.val := by
  induction count with
  | zero =>
      funext index
      exact Fin.elim0 index
  | succ count inductionHypothesis =>
      funext index
      refine Fin.cases ?_ (fun tail => ?_) index
      · rw [binaryValue_succ]
        unfold natBit
        simp only [Fin.val_zero, Nat.pow_zero, Nat.div_one]
        rw [Nat.add_mul_mod_self_left]
        exact (Nat.mod_eq_of_lt (binary 0)).symm
      · let tailBits : Fin count → Nat := fun item => bits item.succ
        have tailBinary : ∀ item, tailBits item < 2 := by
          intro item
          exact binary item.succ
        have tailExact := congrFun
          (inductionHypothesis tailBits tailBinary) tail
        have quotient : binaryValue bits / 2 = binaryValue tailBits := by
          rw [binaryValue_succ]
          rw [Nat.add_mul_div_left _ _ (by decide : 0 < 2)]
          rw [Nat.div_eq_of_lt (binary 0), Nat.zero_add]
        calc
          bits tail.succ = natBit (binaryValue tailBits) tail.val := tailExact
          _ = natBit (binaryValue bits) tail.succ.val := by
            simpa [Fin.val_succ, quotient] using
              (natBit_succ (binaryValue bits) tail.val).symm

/-! ## Field recomposition of binary words -/

/-- Local scalar fold definitionally equal to the fold hidden behind
`Radix.recomposeScalar`. It exists only to prove a generic arithmetic bridge. -/
private def scalarFold : {count : Nat} →
    (Fin count → F) → (Fin count → F) → F
  | 0, _, _ => 0
  | _ + 1, weights, values =>
      weights 0 * values 0 +
        scalarFold
          (fun index => weights index.succ)
          (fun index => values index.succ)

private theorem recomposeScalar_eq_scalarFold (values : ChildIndex → F) :
    recomposeScalar values =
      scalarFold
        Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        values := by
  rfl

private theorem scalarFold_fieldOfNat {count : Nat}
    (weights digits : Fin count → Nat) :
    scalarFold
        (fun index => fieldOfNat (weights index))
        (fun index => fieldOfNat (digits index)) =
      fieldOfNat (sumNats fun index => weights index * digits index) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [scalarFold, sumNats]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => digits index.succ)]
      rw [← fieldOfNat_mul, ← fieldOfNat_add]

private theorem scalarFold_neg {count : Nat}
    (weights values : Fin count → F) :
    scalarFold weights (fun index => -(values index)) =
      -(scalarFold weights values) := by
  induction count with
  | zero => exact Lean.Grind.AddCommGroup.neg_zero.symm
  | succ count inductionHypothesis =>
      simp only [scalarFold]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => values index.succ)]
      have mulNeg : weights 0 * -values 0 = -(weights 0 * values 0) := by
        calc
          weights 0 * -values 0 = -values 0 * weights 0 := Fin.mul_comm _ _
          _ = -(values 0 * weights 0) := Lean.Grind.Fin.neg_mul _ _
          _ = -(weights 0 * values 0) := by
            rw [Fin.mul_comm (values 0) (weights 0)]
      rw [mulNeg]
      exact (Lean.Grind.AddCommGroup.neg_add _ _).symm

private theorem radixWeight_eq_fieldOfNat (index : ChildIndex) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight index =
      fieldOfNat (2 ^ index.val) := by
  rfl

private theorem recompose_fieldBits (bits : ChildIndex → Nat) :
    recomposeScalar (fun index => fieldOfNat (bits index)) =
      fieldOfNat (binaryValue bits) := by
  rw [recomposeScalar_eq_scalarFold]
  rw [show
      Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight =
        (fun index : ChildIndex => fieldOfNat (2 ^ index.val)) by
    funext index
    exact radixWeight_eq_fieldOfNat index]
  exact scalarFold_fieldOfNat
    (fun index : ChildIndex => 2 ^ index.val) bits

private theorem recompose_negFieldBits (bits : ChildIndex → Nat) :
    recomposeScalar (fun index => -(fieldOfNat (bits index))) =
      -(fieldOfNat (binaryValue bits)) := by
  rw [recomposeScalar_eq_scalarFold, scalarFold_neg]
  rw [← recomposeScalar_eq_scalarFold]
  exact congrArg Neg.neg (recompose_fieldBits bits)

/-! ## Selector-bit interpretation -/

/-- Natural selector of the nonzero common-sign branch. -/
private def selectorBit (digit : F) : Nat :=
  if digit = 0 then 0 else 1

private theorem selectorBit_lt_two (digit : F) : selectorBit digit < 2 := by
  by_cases zero : digit = 0 <;> simp [selectorBit, zero]

private theorem positiveDigit_eq_fieldBit {digit : F}
    (allowed : digit = 0 ∨ digit = 1) :
    digit = fieldOfNat (selectorBit digit) := by
  rcases allowed with zero | one
  · subst digit
    rw [selectorBit, if_pos rfl, fieldOfNat_zero]
  · subst digit
    rw [selectorBit, if_neg (by decide : (1 : F) ≠ 0), fieldOfNat_one]

private theorem negativeDigit_eq_negFieldBit {digit : F}
    (allowed : digit = 0 ∨ digit = -1) :
    digit = -(fieldOfNat (selectorBit digit)) := by
  rcases allowed with zero | negative
  · subst digit
    rw [selectorBit, if_pos rfl, fieldOfNat_zero]
    exact Lean.Grind.AddCommGroup.neg_zero.symm
  · subst digit
    rw [selectorBit, if_neg (by decide : (-1 : F) ≠ 0), fieldOfNat_one]

private theorem magnitudeFieldDigit_cases (parent : F) (index : ChildIndex) :
    fieldOfNat (magnitudeDigit parent index) = 0 ∨
      fieldOfNat (magnitudeDigit parent index) = 1 := by
  have digitBound := magnitudeDigit_lt_two parent index
  have cases : magnitudeDigit parent index = 0 ∨
      magnitudeDigit parent index = 1 := by omega
  rcases cases with zero | one
  · left
    rw [zero, fieldOfNat_zero]
  · right
    rw [one, fieldOfNat_one]

/-! ## Honest completeness -/

/-- The canonical signed-binary split satisfies the common-sign predicate and
the existing exact recomposition equation. This is model-level completeness;
it does not compile the predicate to rows. -/
theorem honest_complete (parent : F)
    (bounded : centeredMagnitude parent < combinedBound) :
    Accepted parent (honestSign parent) (splitScalar parent) := by
  constructor
  · constructor
    · by_cases nonnegative : isNonnegative parent
      · exact Or.inr (Or.inl (by simp [honestSign, nonnegative]))
      · exact Or.inr (Or.inr (by simp [honestSign, nonnegative]))
    · intro index
      rw [splitScalar, if_pos bounded]
      by_cases nonnegative : isNonnegative parent
      · rw [show honestSign parent = 1 by simp [honestSign, nonnegative]]
        rw [show boundedDigit parent index =
            fieldOfNat (magnitudeDigit parent index) by
          simp [boundedDigit, nonnegative]]
        exact magnitudeFieldDigit_cases parent index
      · rw [show honestSign parent = -1 by simp [honestSign, nonnegative]]
        rw [show boundedDigit parent index =
            -(fieldOfNat (magnitudeDigit parent index)) by
          simp [boundedDigit, nonnegative]]
        rcases magnitudeFieldDigit_cases parent index with zero | one
        · left
          rw [zero]
          exact Lean.Grind.AddCommGroup.neg_zero
        · right
          rw [one]
  · exact splitScalar_recompose parent

/-! ## Exact soundness -/

private theorem centeredMagnitude_one : centeredMagnitude (1 : F) = 1 := by
  decide

/-- Every accepted common-sign digit is strictly inside the production fresh
bound. This follows from the accepted roots themselves; it is not an
additional parent- or prover-supplied premise. -/
theorem Accepted.digitBounded {parent sign : F}
    {digits : ChildIndex → F} (accepted : Accepted parent sign digits)
    (index : ChildIndex) :
    centeredMagnitude (digits index) < productionGlobalParams.b := by
  rcases accepted.constraint.1 with signZero | signOne | signNegative
  · rcases accepted.constraint.2 index with digitZero | digitSign
    · simp [digitZero, productionGlobalParams]
    · simp [digitSign, signZero, productionGlobalParams]
  · rcases accepted.constraint.2 index with digitZero | digitSign
    · simp [digitZero, productionGlobalParams]
    · simp [digitSign, signOne, productionGlobalParams,
        centeredMagnitude_one]
  · rcases accepted.constraint.2 index with digitZero | digitSign
    · simp [digitZero, productionGlobalParams]
    · simp [digitSign, signNegative, productionGlobalParams,
        Centered.centeredMagnitude_neg, centeredMagnitude_one]

/-- Recomposition of fourteen accepted common-sign digits is automatically
strictly `B`-bounded. Consequently the exactness theorem below does not need a
separate parent-bound check at the R1CS boundary. -/
theorem Accepted.parentBounded {parent sign : F}
    {digits : ChildIndex → F} (accepted : Accepted parent sign digits) :
    centeredMagnitude parent < combinedBound := by
  rw [← accepted.recomposition]
  simpa [recomposeScalar, combinedBound] using
    recomposeScalar_norm digits accepted.digitBounded

private theorem splitScalar_zero : splitScalar (0 : F) = fun _ => 0 := by
  funext index
  simp [splitScalar, combinedBound, productionGlobalParams, GlobalParams.bigB,
    boundedDigit, isNonnegative, magnitudeDigit, natBit, centeredMagnitude]

private theorem sound_positive {parent : F} {digits : ChildIndex → F}
    (bounded : centeredMagnitude parent < combinedBound)
    (allowed : ∀ index, digits index = 0 ∨ digits index = 1)
    (recomposition : recomposeScalar digits = parent) :
    digits = splitScalar parent := by
  let bits : ChildIndex → Nat := fun index => selectorBit (digits index)
  have bitsBinary : ∀ index, bits index < 2 := by
    intro index
    exact selectorBit_lt_two (digits index)
  have digitsAsBits : digits = fun index => fieldOfNat (bits index) := by
    funext index
    exact positiveDigit_eq_fieldBit (allowed index)
  let magnitude : Nat := binaryValue bits
  have magnitudeBound : magnitude < 2 ^ productionGlobalParams.k :=
    binaryValue_lt_pow bits bitsBinary
  have magnitudeLtModulus : magnitude < goldilocksModulus := by
    have productionBound : 2 ^ productionGlobalParams.k < goldilocksModulus := by
      decide
    exact Nat.lt_trans magnitudeBound productionBound
  have encoded : fieldOfNat magnitude = parent := by
    calc
      fieldOfNat magnitude = recomposeScalar (fun index => fieldOfNat (bits index)) :=
        (recompose_fieldBits bits).symm
      _ = recomposeScalar digits := by rw [← digitsAsBits]
      _ = parent := recomposition
  have magnitudeEqVal : magnitude = parent.val := by
    have values := congrArg Fin.val encoded
    simpa [fieldOfNat, Nat.mod_eq_of_lt magnitudeLtModulus] using values
  have parentNonnegative : isNonnegative parent := by
    unfold isNonnegative
    rw [← magnitudeEqVal]
    unfold PiRLCAlgebra.Norm.Centered.halfModulus goldilocksModulus
    have concreteBound : magnitude < 16384 := by
      simpa [productionGlobalParams] using magnitudeBound
    omega
  have magnitudeExact : centeredMagnitude parent = magnitude := by
    rw [PiRLCAlgebra.Norm.Centered.centeredMagnitude_eq_distance]
    unfold PiRLCAlgebra.Norm.Centered.distance
    rw [if_pos (by simpa [isNonnegative] using parentNonnegative)]
    exact magnitudeEqVal.symm
  have bitsExact := bits_eq_natBit_binaryValue bits bitsBinary
  funext index
  rw [splitScalar, if_pos bounded, boundedDigit, if_pos parentNonnegative,
    magnitudeDigit, magnitudeExact, ← congrFun bitsExact index]
  exact positiveDigit_eq_fieldBit (allowed index)

private theorem sound_negative {parent : F} {digits : ChildIndex → F}
    (bounded : centeredMagnitude parent < combinedBound)
    (allowed : ∀ index, digits index = 0 ∨ digits index = -1)
    (recomposition : recomposeScalar digits = parent) :
    digits = splitScalar parent := by
  let bits : ChildIndex → Nat := fun index => selectorBit (digits index)
  have bitsBinary : ∀ index, bits index < 2 := by
    intro index
    exact selectorBit_lt_two (digits index)
  have digitsAsBits : digits = fun index => -(fieldOfNat (bits index)) := by
    funext index
    exact negativeDigit_eq_negFieldBit (allowed index)
  let magnitude : Nat := binaryValue bits
  have magnitudeBound : magnitude < 2 ^ productionGlobalParams.k :=
    binaryValue_lt_pow bits bitsBinary
  have magnitudeLtModulus : magnitude < goldilocksModulus := by
    have productionBound : 2 ^ productionGlobalParams.k < goldilocksModulus := by
      decide
    exact Nat.lt_trans magnitudeBound productionBound
  have encoded : -(fieldOfNat magnitude) = parent := by
    calc
      -(fieldOfNat magnitude) =
          recomposeScalar (fun index => -(fieldOfNat (bits index))) :=
        (recompose_negFieldBits bits).symm
      _ = recomposeScalar digits := by rw [← digitsAsBits]
      _ = parent := recomposition
  have bitsExact := bits_eq_natBit_binaryValue bits bitsBinary
  by_cases magnitudeZero : magnitude = 0
  · have parentZero : parent = 0 := by
      rw [← encoded, magnitudeZero, fieldOfNat_zero]
      exact Lean.Grind.AddCommGroup.neg_zero
    have digitsZero : digits = fun _ => 0 := by
      funext index
      rw [negativeDigit_eq_negFieldBit (allowed index)]
      have selectorZero : selectorBit (digits index) = 0 := by
        change bits index = 0
        have magnitudeZero' : binaryValue bits = 0 := by
          simpa [magnitude] using magnitudeZero
        rw [congrFun bitsExact index, magnitudeZero']
        simp [natBit]
      rw [selectorZero, fieldOfNat_zero]
      exact Lean.Grind.AddCommGroup.neg_zero
    rw [parentZero, digitsZero, splitScalar_zero]
  · have embeddedNonzero : fieldOfNat magnitude ≠ 0 := by
      intro equal
      have values := congrArg Fin.val equal
      simp [fieldOfNat, Nat.mod_eq_of_lt magnitudeLtModulus] at values
      exact magnitudeZero values
    have negatedValue : (-(fieldOfNat magnitude)).val =
        goldilocksModulus - magnitude := by
      rw [Fin.val_neg, if_neg embeddedNonzero]
      simp [fieldOfNat, Nat.mod_eq_of_lt magnitudeLtModulus]
    have parentValue : parent.val = goldilocksModulus - magnitude := by
      have values := congrArg Fin.val encoded
      rw [negatedValue] at values
      exact values.symm
    have parentNegative : ¬ isNonnegative parent := by
      unfold isNonnegative
      rw [parentValue]
      unfold PiRLCAlgebra.Norm.Centered.halfModulus goldilocksModulus
      have concreteBound : magnitude < 16384 := by
        simpa [productionGlobalParams] using magnitudeBound
      omega
    have magnitudeExact : centeredMagnitude parent = magnitude := by
      rw [PiRLCAlgebra.Norm.Centered.centeredMagnitude_eq_distance]
      unfold PiRLCAlgebra.Norm.Centered.distance
      rw [if_neg (by simpa [isNonnegative] using parentNegative), parentValue]
      omega
    funext index
    rw [splitScalar, if_pos bounded, boundedDigit, if_neg parentNegative,
      magnitudeDigit, magnitudeExact, ← congrFun bitsExact index]
    exact negativeDigit_eq_negFieldBit (allowed index)

/-- Under the independent strict combined bound, the common-sign predicate
and radix recomposition force every child digit to be exactly the
verifier-computed production split. -/
theorem accepted_digits_eq_splitScalar {parent sign : F}
    {digits : ChildIndex → F}
    (bounded : centeredMagnitude parent < combinedBound)
    (accepted : Accepted parent sign digits) :
    digits = splitScalar parent := by
  rcases accepted.constraint.1 with signZero | signPositive | signNegative
  · have digitsZero : digits = fun _ => 0 := by
      funext index
      rcases accepted.constraint.2 index with zero | signed
      · exact zero
      · simpa [signZero] using signed
    have parentZero : parent = 0 := by
      rw [← accepted.recomposition, digitsZero, ← splitScalar_zero]
      exact splitScalar_recompose 0
    rw [parentZero, digitsZero, splitScalar_zero]
  · subst sign
    exact sound_positive bounded accepted.constraint.2 accepted.recomposition
  · subst sign
    exact sound_negative bounded accepted.constraint.2 accepted.recomposition

/-- Accepted common-sign roots and recomposition alone determine the exact
verifier-computed split. The strict parent bound is derived internally by
`Accepted.parentBounded`. -/
theorem Accepted.digits_eq_splitScalar {parent sign : F}
    {digits : ChildIndex → F} (accepted : Accepted parent sign digits) :
    digits = splitScalar parent :=
  accepted_digits_eq_splitScalar accepted.parentBounded accepted

/-- Soundness and honest completeness packaged as the exact accepted-language
characterization. The witness sign is deliberately existential, while the
accepted digit vector is deterministic. -/
theorem exists_accepted_iff_exact {parent : F} {digits : ChildIndex → F}
    (bounded : centeredMagnitude parent < combinedBound) :
    (∃ sign, Accepted parent sign digits) ↔ digits = splitScalar parent := by
  constructor
  · rintro ⟨sign, accepted⟩
    exact accepted_digits_eq_splitScalar bounded accepted
  · intro exact
    subst digits
    exact ⟨honestSign parent, honest_complete parent bounded⟩

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits
