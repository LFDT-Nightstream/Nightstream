import NightstreamFPrime.Layout.Range.CanonicalU64
import NightstreamFPrime.Layout.PiRlcWideSampler.Rows

/-! Structural R1CS cost of the unchanged checked range gadget. This is the
witness-construction view; its compact CCS plan has a separate row ledger. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.RangePhysical

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling.WideReduction

private theorem linear_affine (terms : List (Nat × Expr))
    (atoms : ∀ term ∈ terms, R1CS.IsAffine term.2) : R1CS.IsAffine (linearExpr terms) := by
  induction terms with
  | nil => exact R1CS.isAffine_const _
  | cons head rest ih =>
      exact R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ (atoms head (by simp)))
        (ih (fun term member => atoms term (by simp [member])))

private theorem reduced_affine (modulus : Nat) (terms : List (Nat × Expr))
    (atoms : ∀ term ∈ terms, R1CS.IsAffine term.2) : R1CS.IsAffine (linearExpr (reduceTerms modulus terms)) := by
  apply linear_affine
  intro term member
  obtain ⟨original, inside, rfl⟩ := List.mem_map.mp member
  exact atoms original inside

theorem check_affine (offset : Nat) (check : Fin checkCount) : R1CS.IsAffine (checkRow offset check) := by
  apply R1CS.IsAffine.add
  · apply R1CS.IsAffine.add
    · apply reduced_affine
      simp only [drawTerms, List.forall_mem_append, fieldTerms, List.forall_mem_map, fieldBit, Gadgets.Range.CanonicalU64.bitExpr]
      simp
    · exact R1CS.isAffine_const _
  · apply R1CS.IsAffine.const_mul
    apply R1CS.IsAffine.add
    · apply reduced_affine
      simp only [resultTerms, List.forall_mem_append, quotientTerms, List.forall_mem_map, List.forall_mem_flatMap, digitTerms, quotientBit, digitBit]
      simp
    · apply linear_affine
      simp only [checkTerms, List.forall_mem_map, checkBit]
      simp

theorem boolean_fresh (column : Nat) : R1CS.constraintFreshCount (booleanRow (.var column)) = 2 := rfl

theorem digit_fresh (offset position : Nat) : R1CS.constraintFreshCount (digitRangeRow offset position) = 1 := rfl

private theorem newBits_vars (offset : Nat) (atom : Expr) (member : atom ∈ newBits offset) :
    ∃ column, atom = .var column := by
  have all : ∀ atom ∈ newBits offset, ∃ column, atom = Expr.var column := by
    simp only [newBits, List.forall_mem_append, List.forall_mem_map, List.forall_mem_flatMap,
      quotientBit, digitBit, checkBit]
    simp
  exact all atom member

private theorem boolean_total (bits : List Expr) (atomsAreVars : ∀ atom ∈ bits, ∃ column, atom = .var column) :
    R1CS.totalFreshCount (bits.map booleanRow) = bits.length * 2 := by
  induction bits with
  | nil => rfl
  | cons atom rest ih =>
      obtain ⟨column, rfl⟩ := atomsAreVars atom (by simp)
      simp only [List.map_cons, R1CS.totalFreshCount, List.sum_cons, List.length_cons]
      change R1CS.constraintFreshCount (booleanRow (.var column)) + R1CS.totalFreshCount (rest.map booleanRow) = _
      rw [boolean_fresh, ih (fun atom member => atomsAreVars atom (by simp [member]))]
      omega

private theorem flatMap_single {α β : Type*} (items : List α) (f : α → β) :
    items.flatMap (fun item => [f item]) = items.map f := by
  induction items with
  | nil => rfl
  | cons item rest ih => simp only [List.flatMap_cons, List.map_cons, List.singleton_append, ih]

private theorem rows_fresh (offset : Nat) : R1CS.totalFreshCount (flatConstraints (rowOps offset)) = 760 := by
  have constraints : flatConstraints (rowOps offset) =
      (newBits offset).map booleanRow ++ (List.range digitCount).map (digitRangeRow offset) ++
        (List.finRange checkCount).map (checkRow offset) := by
    simp [rowOps, flatConstraints, List.flatMap_map, Op.flatConstraints, flatMap_single]
  rw [constraints, R1CS.totalFreshCount_append, R1CS.totalFreshCount_append,
    boolean_total _ (newBits_vars offset), newBits_length, newBitCount_eq]
  have digits : R1CS.totalFreshCount ((List.range digitCount).map (digitRangeRow offset)) = digitCount := by
    simp only [R1CS.totalFreshCount, List.map_map, Function.comp_def, digit_fresh,
      List.map_const', List.length_range, List.sum_replicate, smul_eq_mul, Nat.mul_one]
  have checks : R1CS.totalFreshCount ((List.finRange checkCount).map (checkRow offset)) = 0 := by
    apply R1CS.totalFreshCount_eq_zero_of_noFresh
    intro expression member
    obtain ⟨check, _, rfl⟩ := List.mem_map.mp member
    exact R1CS.constraintFreshCount_eq_zero_of_affine _ (check_affine offset check)
  rw [digits, checks]
  rfl

private theorem children_fresh (interface : Interface) (offset : Nat)
    (inputs : ∀ lane, R1CS.IsAffine (interface.source lane offset)) :
    R1CS.totalFreshCount (flatConstraints (childOps interface offset)) = 788 := by
  have all : ∀ children : List (Fin fieldCount),
      R1CS.totalFreshCount (flatConstraints (children.map (fun lane =>
        Sequence.childOp (childName lane) (Gadgets.Range.CanonicalU64.circuit (childInterface interface offset lane))
          (childOffset offset lane)))) = children.length * 197 := by
    intro children
    induction children with
    | nil => rfl
    | cons lane rest ih =>
        simp only [List.map_cons, flatConstraints, List.flatMap_cons, R1CS.totalFreshCount_append, List.length_cons]
        have head := Range.CanonicalU64.totalFreshCount_eq (childInterface interface offset lane)
          (childOffset offset lane) ⟨inputs lane⟩
        change R1CS.totalFreshCount (Range.CanonicalU64.logicalConstraints _ _) +
          R1CS.totalFreshCount (flatConstraints _) = _
        rw [head, ih]
        omega
  simpa only [childOps, List.length_finRange] using! all (List.finRange fieldCount)

theorem counts (interface : Interface) (hints : Nat → List Hint) (offset : Nat)
    (inputs : ∀ lane, R1CS.IsAffine (interface.source lane offset)) :
    R1CS.totalFreshCount (flatConstraints (operations interface hints offset)) = 1548 ∧
      R1CS.totalRowCount (flatConstraints (operations interface hints offset)) = 2229 := by
  have fresh : R1CS.totalFreshCount (flatConstraints (operations interface hints offset)) = 1548 := by
    simp only [operations, flatConstraints_append, R1CS.totalFreshCount_append]
    rw [children_fresh interface offset inputs, rows_fresh]
    rfl
  refine ⟨fresh, ?_⟩
  rw [R1CS.totalRowCount_eq_fresh_add_length, fresh, NightstreamFPrime.Gadgets.Sampling.WideReduction.rowCount_eq]
  rfl

end NightstreamFPrime.Layout.PiRlcWideSampler.RangePhysical
