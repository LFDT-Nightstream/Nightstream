import NightstreamFPrime.Gadgets.Sampling.WideReduction

/-! Structural read exclusion for temporary values before the checked
wide-reduction gadget. Hint sources are not constraints. Every retained row
reads only the declared inputs and the checked gadget's own variables. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range

private theorem linearExpr_supported (allowed : Nat → Prop) (terms : List (Nat × Expr))
    (atoms : ∀ term ∈ terms, term.2.VarsSatisfy allowed) :
    (linearExpr terms).VarsSatisfy allowed := by
  induction terms with
  | nil => trivial
  | cons term rest ih =>
      exact ⟨⟨trivial, atoms term (by simp)⟩, ih (fun x hx => atoms x (by simp [hx]))⟩

private theorem reduceTerms_supported (allowed : Nat → Prop) (terms : List (Nat × Expr))
    (modulus : Nat) (atoms : ∀ term ∈ terms, term.2.VarsSatisfy allowed) :
    (linearExpr (reduceTerms modulus terms)).VarsSatisfy allowed := by
  apply linearExpr_supported
  intro term member
  obtain ⟨original, inside, rfl⟩ := List.mem_map.mp member
  exact atoms original inside

theorem flatConstraints_varsSatisfy (interface : Interface) (hints : Nat → List Hint)
    (offset : Nat) (allowed : Nat → Prop)
    (sources : ∀ lane, (interface.source lane offset).VarsSatisfy allowed)
    (locals : ∀ index, offset ≤ index → index < offset + privateCount → allowed index) :
    ∀ expression ∈ flatConstraints (operations interface hints offset),
      expression.VarsSatisfy allowed := by
  have field (lane bit : Nat) (laneBound : lane < fieldCount) (bitBound : bit < CanonicalU64.bitCount) :
      (fieldBit offset lane bit).VarsSatisfy allowed := by
    apply locals <;>
      simp only [childOffset, childWidth, CanonicalU64.auxiliaryCount, CanonicalU64.bitCount,
        privateCount, newBitCount, fieldCount, quotientBitCount, digitCount, digitBitCount,
        checkBitCount, checkCount] at * <;> omega
  have quotient (bit : Nat) (bound : bit < quotientBitCount) :
      (quotientBit offset bit).VarsSatisfy allowed := by
    apply locals <;>
      simp only [quotientStart, privateCount, newBitCount, quotientBitCount, digitCount,
        digitBitCount, checkCount, checkBitCount] at * <;> omega
  have digit (position bit : Nat) (positionBound : position < digitCount)
      (bitBound : bit < digitBitCount) : (digitBit offset position bit).VarsSatisfy allowed := by
    apply locals <;>
      simp only [digitStart, quotientStart, privateCount, newBitCount, quotientBitCount, digitCount,
        digitBitCount, checkCount, checkBitCount] at * <;> omega
  have check (position bit : Nat) (positionBound : position < checkCount)
      (bitBound : bit < checkBitCount) : (checkBit offset position bit).VarsSatisfy allowed := by
    apply locals <;>
      simp only [checkStart, digitStart, quotientStart, privateCount, newBitCount, quotientBitCount,
        digitCount, digitBitCount, checkCount, checkBitCount] at * <;> omega
  have draw : ∀ term ∈ drawTerms offset, term.2.VarsSatisfy allowed := by
    intro term member
    simp only [drawTerms, fieldTerms, List.mem_append, List.mem_map, List.mem_range] at member
    rcases member with ((⟨bit, below, rfl⟩ | ⟨bit, below, rfl⟩) | ⟨bit, below, rfl⟩) | ⟨bit, below, rfl⟩ <;>
      exact field _ bit (by decide) below
  have result : ∀ term ∈ resultTerms offset, term.2.VarsSatisfy allowed := by
    intro term member
    simp only [resultTerms, quotientTerms, digitTerms, List.mem_append, List.mem_map,
      List.mem_flatMap, List.mem_range] at member
    rcases member with ⟨bit, below, rfl⟩ | ⟨position, positionBelow, bit, bitBelow, rfl⟩
    · exact quotient bit below
    · exact digit position bit positionBelow bitBelow
  intro expression member
  simp only [operations, flatConstraints, List.flatMap_append, List.mem_append] at member
  rcases member with (childMember | witnessMember) | rowMember
  · obtain ⟨child, childMember, member⟩ := List.mem_flatMap.mp childMember
    obtain ⟨lane, _, rfl⟩ := List.mem_map.mp childMember
    change expression ∈ flatConstraints
      (CanonicalU64.operations (childInterface interface offset lane) (childOffset offset lane)) at member
    apply CanonicalU64.flatConstraints_varsSatisfy _ _ allowed (sources lane) _ expression member
    intro index below
    have laneBound := lane.isLt
    apply locals <;>
      simp only [childOffset, childWidth, CanonicalU64.auxiliaryCount, privateCount,
        newBitCount, fieldCount, quotientBitCount, digitCount, digitBitCount,
        checkBitCount, checkCount] at * <;> omega
  · simp [Op.flatConstraints, recipeConstraints, WitnessBatch.hinted] at witnessMember
  · obtain ⟨operation, inside, member⟩ := List.mem_flatMap.mp rowMember
    simp only [rowOps, List.mem_append, List.mem_map, List.mem_range] at inside
    rcases inside with (⟨atom, atomMember, rfl⟩ | ⟨position, positionBelow, rfl⟩) | ⟨index, _, rfl⟩ <;>
      (have same := List.mem_singleton.mp member; subst same)
    · have supported : atom.VarsSatisfy allowed := by
        simp only [newBits, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_range] at atomMember
        rcases atomMember with (⟨bit, below, rfl⟩ | ⟨position, positionBelow, bit, bitBelow, rfl⟩) |
          ⟨index, _, bit, bitBelow, rfl⟩
        · exact quotient bit below
        · exact digit position bit positionBelow bitBelow
        · exact check index.val bit index.isLt bitBelow
      exact Expr.VarsSatisfy.mul _ _ allowed supported
        (Expr.VarsSatisfy.sub _ _ allowed supported trivial)
    · exact ⟨digit position 2 positionBelow (by decide),
        ⟨digit position 0 positionBelow (by decide), digit position 1 positionBelow (by decide)⟩⟩
    · apply Expr.VarsSatisfy.sub
      · exact ⟨reduceTerms_supported allowed _ _ draw, trivial⟩
      · refine ⟨reduceTerms_supported allowed _ _ result, ?_⟩
        apply linearExpr_supported
        intro term member
        obtain ⟨bit, bitMember, rfl⟩ := List.mem_map.mp member
        exact check index.val bit index.isLt (List.mem_range.mp bitMember)


theorem flatConstraints_varsBelow (interface : Interface) (hints : Nat → List Hint)
    (offset : Nat) (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints (operations interface hints offset),
      expression.VarsBelow (offset + privateCount) := by
  have supported := flatConstraints_varsSatisfy interface hints offset
    (fun index => index < offset + privateCount)
    (fun lane => (Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr
      (Expr.VarsBelow.mono _ (inputs lane) (by omega)))
    (fun _ _ upper => upper)
  intro expression member
  exact (Expr.varsSatisfy_lt_iff_varsBelow _ _).mp (supported expression member)

def replaceTemporary (env : Env) (start count : Nat) (temporary : Nat → F) : Env :=
  fun index => if start ≤ index ∧ index < start + count then temporary (index - start) else env index

theorem temporary_read_exclusion (interface : Interface) (hints : Nat → List Hint)
    (start count : Nat)
    (inputs : ∀ lane, (interface.source lane (start + count)).VarsBelow start)
    (env : Env) (temporary : Nat → F) :
    holdsFlat (replaceTemporary env start count temporary)
        (operations interface hints (start + count)) ↔
      holdsFlat env (operations interface hints (start + count)) := by
  have support := flatConstraints_varsSatisfy interface hints (start + count)
    (fun index => index < start ∨ start + count ≤ index)
    (fun lane => (Expr.varsSatisfy_lt_iff_varsBelow _ start).mpr (inputs lane) |>.mono _
      (fun _ below => Or.inl below))
    (fun _ above _ => Or.inr above)
  have same : ∀ expression ∈ flatConstraints (operations interface hints (start + count)),
      expression.eval (replaceTemporary env start count temporary) = expression.eval env := by
    intro expression member
    apply Expr.eval_eq_of_agree_satisfy expression _ _ _ (support expression member)
    intro index outside
    unfold replaceTemporary
    rw [if_neg (by omega)]
  constructor <;> intro rows expression member
  · rw [← same expression member]
    exact rows expression member
  · rw [same expression member]
    exact rows expression member

end NightstreamFPrime.Gadgets.Sampling.WideReduction
