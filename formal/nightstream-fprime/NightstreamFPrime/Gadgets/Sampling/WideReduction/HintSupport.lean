import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintProgram

/-! Each helper instruction reads only variables that execution has
already filled. These bounds concern hint sources, not constraint rows. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HintSupport

open NightstreamFPrime.Circuit

theorem linearExpr_below (terms : List (Nat × Expr)) (bound : Nat)
    (atoms : ∀ term ∈ terms, term.2.VarsBelow bound) :
    (linearExpr terms).VarsBelow bound := by
  induction terms with
  | nil => trivial
  | cons term rest ih =>
      exact ⟨⟨trivial, atoms term (by simp)⟩, ih (fun x hx => atoms x (by simp [hx]))⟩

theorem limbTerms_below (offset position : Nat) :
    (linearExpr (HintProgram.limbTerms offset position)).VarsBelow
      (HintProgram.limbStart offset position) := by
  apply linearExpr_below
  intro term member
  unfold HintProgram.limbTerms at member
  rcases List.mem_append.mp member with carryMember | inputMember
  · split_ifs at carryMember with zero
    · simp at carryMember
    · obtain ⟨bit, bitMember, rfl⟩ := List.mem_map.mp carryMember
      have bitBound := List.mem_range.mp bitMember
      change _ < _
      simp only [HintProgram.limbStart, HintProgram.limbStride, HintProgram.accumulatorBits,
        HintProgram.limbBits] at *
      omega
  · obtain ⟨lane, laneMember, blockMember⟩ := List.mem_flatMap.mp inputMember
    obtain ⟨word, wordMember, member⟩ := List.mem_flatMap.mp blockMember
    have laneBound := List.mem_range.mp laneMember
    have wordBound := List.mem_range.mp wordMember
    split at member
    · obtain ⟨bit, bitMember, rfl⟩ := List.mem_map.mp member
      have bitBound := List.mem_range.mp bitMember
      change _ < _
      simp only [HintProgram.limbStart, HintProgram.sourceBitCount, HintProgram.limbBits,
        fieldCount] at *
      omega
    · simp at member

theorem integerBit_below (offset bit : Nat) (bound : bit < 256) :
    (HintProgram.integerBit offset bit).VarsBelow (HintProgram.divisionStart offset) := by
  change _ < _
  simp only [HintProgram.limbStart, HintProgram.divisionStart,
    HintProgram.limbCount, HintProgram.limbBits, HintProgram.limbStride,
    HintProgram.accumulatorBits]
  omega

theorem initialLimb_below (offset position : Nat) :
    (HintProgram.initialDivisionLimb offset position).VarsBelow (HintProgram.divisionStart offset) := by
  apply linearExpr_below
  intro term member
  obtain ⟨bit, _, selected⟩ := List.mem_filterMap.mp member
  dsimp only at selected
  split_ifs at selected with included
  · cases selected
    exact integerBit_below offset _ included

theorem divisionInput_below (offset round position : Nat) :
    (HintProgram.divisionInput offset round position).VarsBelow
      (HintProgram.divisionColumn offset round position) := by
  unfold HintProgram.divisionInput
  refine ⟨⟨trivial, ?_⟩, ?_⟩
  · split
    · trivial
    · change _ < _
      unfold HintProgram.divisionColumn
      omega
  · split
    · apply Expr.VarsBelow.mono _ (initialLimb_below offset _)
      unfold HintProgram.divisionColumn
      omega
    · change _ < _
      unfold HintProgram.divisionColumn HintProgram.divisionStride HintProgram.divisionLimbs
      omega

theorem checkQuotient_below (offset : Nat) (check : Fin checkCount) :
    (HintProgram.checkQuotient offset check).VarsBelow (checkStart offset) := by
  have draw : (linearExpr (reduceTerms (modulus check) (drawTerms offset))).VarsBelow
      (checkStart offset) := by
    apply linearExpr_below
    intro term member
    obtain ⟨original, inside, rfl⟩ := List.mem_map.mp member
    have field (lane bit : Nat) (laneBound : lane < 4) (bitBound : bit < 64) :
        (fieldBit offset lane bit).VarsBelow (checkStart offset) := by
      change _ < _
      simp only [childOffset, childWidth, Range.CanonicalU64.auxiliaryCount,
        checkStart, digitStart, quotientStart, fieldCount, quotientBitCount, digitCount, digitBitCount]
      omega
    simp only [drawTerms, fieldTerms, List.mem_append, List.mem_map, List.mem_range] at inside
    rcases inside with ((⟨bit, below, rfl⟩ | ⟨bit, below, rfl⟩) |
      ⟨bit, below, rfl⟩) | ⟨bit, below, rfl⟩ <;> exact field _ bit (by decide) below
  have result : (linearExpr (reduceTerms (modulus check) (resultTerms offset))).VarsBelow
      (checkStart offset) := by
    apply linearExpr_below
    intro term member
    obtain ⟨original, inside, rfl⟩ := List.mem_map.mp member
    simp only [resultTerms, quotientTerms, digitTerms, List.mem_append, List.mem_map,
      List.mem_range, List.mem_flatMap] at inside
    rcases inside with ⟨bit, below, rfl⟩ | ⟨digit, digitBelow, bit, bitBelow, rfl⟩
    · change _ < _
      unfold checkStart digitStart
      omega
    · change _ < _
      simp only [checkStart, digitStart, digitCount, digitBitCount, quotientBitCount] at *
      omega
  exact ⟨⟨⟨draw, trivial⟩, ⟨trivial, result⟩⟩, trivial⟩

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HintSupport
