import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.ChunkRows
import Nightstream.Implementation.R1CS.Core.Program
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

/-!
Semantic refinement for one 16-bit `Pi_RLC` sampler candidate.

Owns: the independent bitwise-complement interpretation of the 16 raw source
bits and the proof that the four acceptance rows force the verifier-owned
production predicate. Mod-5 decoding and cumulative-prefix refinement are
added as separate theorem families below this acceptance boundary.

Does not own: transcript generation, source-bit decomposition, first-accepted
selection, whole-lane composition, production column placement, Rust
conformance, aggregate row replacements, or constraint totals.

Emits constraints: no.

Authority boundary: `ProductionAlphabet.verifier.accepts` is the semantic
authority. The R1CS accept wire is merely an implementation witness and is
accepted only after the source bits and all four inverse rows refine that
predicate.

| Protocol | Phase | Constraint family | Lean result | Guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/chunk | source value | `chunkValue_lt_bound` | Boolean bits determine a unique value below `2^16` |
| `Pi_RLC` | sampler/chunk | acceptance | `acceptanceRows_sound` | accept wire is one exactly when the candidate is not 65535 |
| `Pi_RLC` | sampler/chunk | production bridge | `acceptanceRows_refine_verifier` | accept wire equals the verifier-owned Boolean decision |
| `Pi_RLC` | sampler/chunk | residue range | `residueRangeRows_sound` | the decoded residue is one of `0,1,2,3,4` |
| `Pi_RLC` | sampler/chunk | quotient bits | `quotientBits_sound` | all 14 radix-2 digits are Boolean |
| `Pi_RLC` | sampler/chunk | quotient recomposition | `quotientRecompositionRow_sound` | the quotient wire is exactly the 14-bit integer |
| `Pi_RLC` | sampler/chunk | Euclidean decomposition | `decompositionRow_sound` | `chunk = 5 * quotient + residue` over the integers |
| `Pi_RLC` | sampler/chunk | decoder bridge | `residue_refines_verifier_symbol` | residue equals the verifier-owned mod-5 symbol |
| `Pi_RLC` | sampler/chunk | centered encoding | `symbolRow_refines_verifier` | symbol wire is the Goldilocks encoding of the centered verifier symbol |
| `Pi_RLC` | sampler/chunk | accepted prefix | `cumulativeRow_refines_verifier` | cumulative count advances by the verifier-owned accept decision |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

private theorem range16 :
    List.range 16 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] := by
  decide

private theorem range14 :
    List.range 14 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] := by
  decide

/-- Independent little-endian interpretation of the complemented raw word. -/
def chunkValue (assignment : Nat → Nat) (chunk : Nat) : Nat :=
  (List.range 16).foldl
    (fun value offset =>
      value + 2 ^ offset *
        (1 - assignment (ChunkRows.sourceBitCol chunk offset))) 0

def BitsBoolean (assignment : Nat → Nat) (chunk : Nat) : Prop :=
  ∀ offset, offset < 16 →
    assignment (ChunkRows.sourceBitCol chunk offset) ≤ 1

/-- Independent little-endian integer interpretation of the 14 quotient
digits. -/
def quotientValue (assignment : Nat → Nat) (chunk : Nat) : Nat :=
  (List.range 14).foldl
    (fun value offset =>
      value + 2 ^ offset * assignment (ChunkRows.quotientBitCol chunk offset)) 0

def QuotientBitsBoolean (assignment : Nat → Nat) (chunk : Nat) : Prop :=
  ∀ offset, offset < 14 →
    assignment (ChunkRows.quotientBitCol chunk offset) ≤ 1

theorem chunkValue_lt_bound
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : BitsBoolean assignment chunk) :
    chunkValue assignment chunk < ProductionAlphabet.chunkModulus := by
  have b0 := bits 0 (by decide)
  have b1 := bits 1 (by decide)
  have b2 := bits 2 (by decide)
  have b3 := bits 3 (by decide)
  have b4 := bits 4 (by decide)
  have b5 := bits 5 (by decide)
  have b6 := bits 6 (by decide)
  have b7 := bits 7 (by decide)
  have b8 := bits 8 (by decide)
  have b9 := bits 9 (by decide)
  have b10 := bits 10 (by decide)
  have b11 := bits 11 (by decide)
  have b12 := bits 12 (by decide)
  have b13 := bits 13 (by decide)
  have b14 := bits 14 (by decide)
  have b15 := bits 15 (by decide)
  simp [chunkValue, range16, ProductionAlphabet.chunkModulus]
  omega

theorem quotientValue_lt_bound
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : QuotientBitsBoolean assignment chunk) :
    quotientValue assignment chunk < 16384 := by
  have b0 := bits 0 (by decide)
  have b1 := bits 1 (by decide)
  have b2 := bits 2 (by decide)
  have b3 := bits 3 (by decide)
  have b4 := bits 4 (by decide)
  have b5 := bits 5 (by decide)
  have b6 := bits 6 (by decide)
  have b7 := bits 7 (by decide)
  have b8 := bits 8 (by decide)
  have b9 := bits 9 (by decide)
  have b10 := bits 10 (by decide)
  have b11 := bits 11 (by decide)
  have b12 := bits 12 (by decide)
  have b13 := bits 13 (by decide)
  simp [quotientValue, range14]
  omega

/-- Candidate determined by the checked source bits. -/
def candidate
    (assignment : Nat → Nat) (chunk : Nat)
    (bits : BitsBoolean assignment chunk) : ProductionAlphabet.Chunk :=
  ⟨chunkValue assignment chunk, chunkValue_lt_bound bits⟩

/-- The emitted chunk linear combination is the independent complemented
candidate value when the source cells are Boolean and column zero is one. -/
theorem chunkTerms_value
    {assignment : Nat → Nat} {chunk : Nat}
    (one : assignment 0 = 1) (bits : BitsBoolean assignment chunk) :
    lcEval assignment (ChunkRows.chunkTerms chunk) =
      chunkValue assignment chunk := by
  have b0 := bits 0 (by decide)
  have b1 := bits 1 (by decide)
  have b2 := bits 2 (by decide)
  have b3 := bits 3 (by decide)
  have b4 := bits 4 (by decide)
  have b5 := bits 5 (by decide)
  have b6 := bits 6 (by decide)
  have b7 := bits 7 (by decide)
  have b8 := bits 8 (by decide)
  have b9 := bits 9 (by decide)
  have b10 := bits 10 (by decide)
  have b11 := bits 11 (by decide)
  have b12 := bits 12 (by decide)
  have b13 := bits 13 (by decide)
  have b14 := bits 14 (by decide)
  have b15 := bits 15 (by decide)
  simp [ChunkRows.sourceBitCol] at b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
  simp [lcEval, ChunkRows.chunkTerms, chunkValue, range16,
    ChunkRows.sourceBitCol, one, goldilocksP]
  omega

private theorem lcEval_quotientTerms
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : QuotientBitsBoolean assignment chunk) :
    lcEval assignment (ChunkRows.quotientTerms chunk) =
      quotientValue assignment chunk := by
  have valueLt := quotientValue_lt_bound bits
  have valueGoldilocks : quotientValue assignment chunk < goldilocksP := by
    have bound : 16384 < goldilocksP := by decide
    exact Nat.lt_trans valueLt bound
  simpa [lcEval, ChunkRows.quotientTerms, quotientValue, range14] using
    (Nat.mod_eq_of_lt valueGoldilocks)

private theorem rawLcEval_append
    (assignment : Nat → Nat) (left right : List (Nat × Nat)) :
    Program.rawLcEval assignment (left ++ right) =
      Program.rawLcEval assignment left + Program.rawLcEval assignment right := by
  induction left with
  | nil => simp [Program.rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [Program.rawLcEval, inductionHypothesis, Nat.add_assoc]

private theorem lcEval_append
    (assignment : Nat → Nat) (left right : List (Nat × Nat)) :
    lcEval assignment (left ++ right) =
      (lcEval assignment left + lcEval assignment right) % goldilocksP := by
  rw [Program.lcEval_eq_raw_mod, rawLcEval_append, Nat.add_mod,
    ← Program.lcEval_eq_raw_mod, ← Program.lcEval_eq_raw_mod]

/-- Generic equality decoder for the exact subtraction-form row used by the
sampler's multi-term integer identities. -/
private theorem linearEquality_sound
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (left right : List (Nat × Nat))
    (rightCanonical : Program.CanonicalTerms right)
    (holds : RowHolds assignment
      ⟨left ++ Program.negateTerms right, [(0, 1)], []⟩) :
    lcEval assignment left = lcEval assignment right := by
  have combinedZero :
      lcEval assignment (left ++ Program.negateTerms right) = 0 := by
    simpa [RowHolds, lcEval, one] using holds
  have rightCancel := Program.lcEval_append_negateTerms_eq_zero
    assignment right rightCanonical
  rw [lcEval_append] at combinedZero rightCancel
  have modulusPositive : 0 < goldilocksP := by decide
  have leftLt : lcEval assignment left < goldilocksP := by
    rw [Program.lcEval_eq_raw_mod]
    exact Nat.mod_lt _ modulusPositive
  have rightLt : lcEval assignment right < goldilocksP := by
    rw [Program.lcEval_eq_raw_mod]
    exact Nat.mod_lt _ modulusPositive
  have complementLt :
      lcEval assignment (Program.negateTerms right) < goldilocksP := by
    rw [Program.lcEval_eq_raw_mod]
    exact Nat.mod_lt _ modulusPositive
  simp only [goldilocksP] at combinedZero rightCancel leftLt rightLt complementLt ⊢
  omega

private theorem lcEval_differenceTerms
    {assignment : Nat → Nat} {chunk : Nat}
    (one : assignment 0 = 1) (bits : BitsBoolean assignment chunk) :
    lcEval assignment (ChunkRows.differenceTerms chunk) =
      (chunkValue assignment chunk + goldilocksP - 65535) %
        goldilocksP := by
  rw [ChunkRows.differenceTerms, lcEval_append,
    chunkTerms_value one bits]
  simp [lcEval, one, goldilocksP]

private theorem lcEval_oneMinusAccept
    {assignment : Nat → Nat} {chunk : Nat}
    (one : assignment 0 = 1)
    (acceptLe : assignment (ChunkRows.acceptCol chunk) ≤ 1) :
    lcEval assignment (ChunkRows.oneMinusAcceptTerms chunk) =
      1 - assignment (ChunkRows.acceptCol chunk) := by
  have acceptCases : assignment (ChunkRows.acceptCol chunk) = 0 ∨
      assignment (ChunkRows.acceptCol chunk) = 1 := by omega
  rcases acceptCases with acceptZero | acceptOne
  · simp [lcEval, ChunkRows.oneMinusAcceptTerms, one, acceptZero,
      goldilocksP]
  · simp [lcEval, ChunkRows.oneMinusAcceptTerms, one, acceptOne,
      goldilocksP]

private theorem difference_mod_eq_zero_iff
    {value : Nat}
    (valueLt : value < ProductionAlphabet.chunkModulus) :
    (value + goldilocksP - 65535) % goldilocksP = 0 ↔
      value = 65535 := by
  change value < 65536 at valueLt
  change (value + 18446744069414584321 - 65535) %
      18446744069414584321 = 0 ↔ value = 65535
  constructor
  · intro zero
    by_cases equal : value = 65535
    · exact equal
    · have valueSmall : value < 65535 := by omega
      have sumLt : value + 18446744069414584321 - 65535 <
          18446744069414584321 := by
        omega
      rw [Nat.mod_eq_of_lt sumLt] at zero
      omega
  · intro equal
    subst value
    simp

private theorem satisfies_acceptanceRows
    {assignment : Nat → Nat} {chunk : Nat}
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    Satisfies (ChunkRows.acceptanceRows chunk) assignment := by
  intro row member
  apply satisfies row
  simp [ChunkRows.chunkRows, member]

private theorem fieldSub_eq_zero_iff
    {value amount : Nat}
    (valueLt : value < goldilocksP)
    (amountPositive : 0 < amount)
    (amountLt : amount < goldilocksP) :
    (value + (goldilocksP - amount)) % goldilocksP = 0 ↔
      value = amount := by
  have shiftedForm :
      value + (goldilocksP - amount) =
        value + goldilocksP - amount := by
    omega
  rw [shiftedForm]
  constructor
  · intro zero
    by_cases valueSmall : value < amount
    · have shiftedLt :
          value + goldilocksP - amount < goldilocksP := by
        omega
      rw [Nat.mod_eq_of_lt shiftedLt] at zero
      omega
    · have amountLe : amount ≤ value := by omega
      have rearranged :
          value + goldilocksP - amount =
            (value - amount) + goldilocksP := by
        omega
      rw [rearranged, Nat.add_mod] at zero
      simp only [Nat.mod_self, Nat.add_zero, Nat.mod_mod] at zero
      rw [Nat.mod_eq_of_lt (by omega : value - amount < goldilocksP)] at zero
      omega
  · intro equal
    subst value
    have rearranged :
        amount + goldilocksP - amount = goldilocksP := by
      omega
    rw [rearranged, Nat.mod_self]

private theorem satisfies_residueRangeRows
    {assignment : Nat → Nat} {chunk : Nat}
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    Satisfies (ChunkRows.residueRangeRows chunk) assignment := by
  intro row member
  apply satisfies row
  simp [ChunkRows.chunkRows, member]

private theorem satisfies_quotientRangeRows
    {assignment : Nat → Nat} {chunk : Nat}
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    Satisfies (ChunkRows.quotientRangeRows chunk) assignment := by
  intro row member
  apply satisfies row
  simp [ChunkRows.chunkRows, member]

/-- The 14 exact range rows force every quotient digit to be Boolean. -/
theorem quotientBits_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    QuotientBitsBoolean assignment chunk := by
  intro offset offsetLt
  apply bitRow_le_one prime (canonical _) one
  apply satisfies_quotientRangeRows satisfies
  rw [ChunkRows.quotientRangeRows]
  exact List.mem_map.mpr ⟨offset, List.mem_range.mpr offsetLt, rfl⟩

private theorem quotientPower_canonical
    {offset : Nat} (offsetLt : offset < 14) :
    0 < 2 ^ offset ∧ 2 ^ offset < goldilocksP := by
  constructor
  · exact Nat.two_pow_pos offset
  · have offsetLe : offset ≤ 13 := by omega
    have powerLe : 2 ^ offset ≤ 2 ^ 13 :=
      Nat.pow_le_pow_right (by decide) offsetLe
    have bound : 2 ^ 13 < goldilocksP := by decide
    exact Nat.lt_of_le_of_lt powerLe bound

private theorem quotientTerms_canonical (chunk : Nat) :
    Program.CanonicalTerms (ChunkRows.quotientTerms chunk) := by
  intro term member
  rw [ChunkRows.quotientTerms] at member
  rcases List.mem_map.mp member with ⟨offset, offsetMember, rfl⟩
  exact quotientPower_canonical (List.mem_range.mp offsetMember)

private theorem quotientRecompositionRow_holds
    {assignment : Nat → Nat} {chunk : Nat}
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    RowHolds assignment (ChunkRows.quotientRecompositionRow chunk) := by
  apply satisfies
  simp [ChunkRows.chunkRows]

/-- The range rows and exact subtraction-form recomposition row force the
quotient wire to equal the independently interpreted 14-bit integer. -/
theorem quotientRecompositionRow_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    assignment (ChunkRows.quotientCol chunk) =
      quotientValue assignment chunk := by
  have bits := quotientBits_sound prime canonical one satisfies
  have holds := quotientRecompositionRow_holds satisfies
  have builderHolds : RowHolds assignment
      (Program.builderLinearRow (ChunkRows.quotientCol chunk)
        (ChunkRows.quotientTerms chunk)) := by
    simpa [ChunkRows.quotientRecompositionRow, ChunkRows.zeroEqualityRow,
      Program.builderLinearRow, Program.negateTerms, Program.negCoeff] using holds
  have decoded := Program.builderLinearRow_sound canonical one
    (ChunkRows.quotientCol chunk) (ChunkRows.quotientTerms chunk)
    (quotientTerms_canonical chunk) builderHolds
  rw [lcEval_quotientTerms bits] at decoded
  exact decoded

/-- The exact four-row product chain forces the residue witness into the
five-element decoder alphabet. This is a semantic range theorem, not merely a
row-inclusion certificate. -/
theorem residueRangeRows_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    assignment (ChunkRows.residueCol chunk) <
      ProductionAlphabet.alphabetSize := by
  have residueRows := satisfies_residueRangeRows satisfies
  have first := residueRows
    ⟨[(ChunkRows.residueCol chunk, 1)],
      [(ChunkRows.residueCol chunk, 1), (0, goldilocksP - 1)],
      [(ChunkRows.residueProductCol chunk 0, 1)]⟩
    (by simp [ChunkRows.residueRangeRows])
  have second := residueRows
    ⟨[(ChunkRows.residueProductCol chunk 0, 1)],
      [(ChunkRows.residueCol chunk, 1), (0, goldilocksP - 2)],
      [(ChunkRows.residueProductCol chunk 1, 1)]⟩
    (by simp [ChunkRows.residueRangeRows])
  have third := residueRows
    ⟨[(ChunkRows.residueProductCol chunk 1, 1)],
      [(ChunkRows.residueCol chunk, 1), (0, goldilocksP - 3)],
      [(ChunkRows.residueProductCol chunk 2, 1)]⟩
    (by simp [ChunkRows.residueRangeRows])
  have fourth := residueRows
    ⟨[(ChunkRows.residueProductCol chunk 2, 1)],
      [(ChunkRows.residueCol chunk, 1), (0, goldilocksP - 4)], []⟩
    (by simp [ChunkRows.residueRangeRows])
  simp only [RowHolds, lcEval, List.foldl, one, Nat.one_mul,
    Nat.mul_one, Nat.zero_add, Nat.zero_mod,
    Nat.mod_eq_of_lt (canonical _)] at first second third fourth
  let residue := assignment (ChunkRows.residueCol chunk)
  let product0 := assignment (ChunkRows.residueProductCol chunk 0)
  let product1 := assignment (ChunkRows.residueProductCol chunk 1)
  let product2 := assignment (ChunkRows.residueProductCol chunk 2)
  change residue * ((residue + (goldilocksP - 1)) % goldilocksP) %
      goldilocksP = product0 at first
  change product0 * ((residue + (goldilocksP - 2)) % goldilocksP) %
      goldilocksP = product1 at second
  change product1 * ((residue + (goldilocksP - 3)) % goldilocksP) %
      goldilocksP = product2 at third
  change product2 * ((residue + (goldilocksP - 4)) % goldilocksP) %
      goldilocksP = 0 at fourth
  change residue < ProductionAlphabet.alphabetSize
  have residueLt : residue < goldilocksP := canonical _
  have product2Lt : product2 < goldilocksP := canonical _
  have product1Lt : product1 < goldilocksP := canonical _
  have product0Lt : product0 < goldilocksP := canonical _
  have fiveLt : 5 < goldilocksP := by decide
  have onePositive : 0 < 1 := by decide
  have twoPositive : 0 < 2 := by decide
  have threePositive : 0 < 3 := by decide
  have fourPositive : 0 < 4 := by decide
  have oneLt : 1 < goldilocksP :=
    Nat.lt_trans (by decide : 1 < 5) fiveLt
  have twoLt : 2 < goldilocksP :=
    Nat.lt_trans (by decide : 2 < 5) fiveLt
  have threeLt : 3 < goldilocksP :=
    Nat.lt_trans (by decide : 3 < 5) fiveLt
  have fourLt : 4 < goldilocksP :=
    Nat.lt_trans (by decide : 4 < 5) fiveLt
  rcases prime _ _ fourth with product2Zero | residueFour
  · rw [Nat.mod_eq_of_lt product2Lt] at product2Zero
    rw [product2Zero] at third
    rcases prime _ _ third with product1Zero | residueThree
    · rw [Nat.mod_eq_of_lt product1Lt] at product1Zero
      rw [product1Zero] at second
      rcases prime _ _ second with product0Zero | residueTwo
      · rw [Nat.mod_eq_of_lt product0Lt] at product0Zero
        rw [product0Zero] at first
        rcases prime _ _ first with residueZero | residueOne
        · rw [Nat.mod_eq_of_lt residueLt] at residueZero
          change residue < 5
          rw [residueZero]
          decide
        · have equalsOne : residue = 1 := by
            rw [Nat.mod_mod] at residueOne
            exact (fieldSub_eq_zero_iff
              (value := residue)
              (amount := 1) residueLt onePositive oneLt).mp residueOne
          change residue < 5
          rw [equalsOne]
          decide
      · have equalsTwo : residue = 2 := by
          rw [Nat.mod_mod] at residueTwo
          exact (fieldSub_eq_zero_iff
            (value := residue)
            (amount := 2) residueLt twoPositive twoLt).mp residueTwo
        change residue < 5
        rw [equalsTwo]
        decide
    · have equalsThree : residue = 3 := by
        rw [Nat.mod_mod] at residueThree
        exact (fieldSub_eq_zero_iff
          (value := residue)
          (amount := 3) residueLt threePositive threeLt).mp residueThree
      change residue < 5
      rw [equalsThree]
      decide
  · have equalsFour : residue = 4 := by
      rw [Nat.mod_mod] at residueFour
      exact (fieldSub_eq_zero_iff
        (value := residue)
        (amount := 4) residueLt fourPositive fourLt).mp residueFour
    change residue < 5
    rw [equalsFour]
    decide

private theorem decompositionRow_holds
    {assignment : Nat → Nat} {chunk : Nat}
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    RowHolds assignment (ChunkRows.decompositionRow chunk) := by
  apply satisfies
  simp [ChunkRows.chunkRows]

/-- The exact decomposition row is lifted out of the field because both sides
are independently range bounded. -/
theorem decompositionRow_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceBits : BitsBoolean assignment chunk)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    chunkValue assignment chunk =
      5 * quotientValue assignment chunk +
        assignment (ChunkRows.residueCol chunk) := by
  let rightTerms : List (Nat × Nat) :=
    [(ChunkRows.quotientCol chunk, 5),
      (ChunkRows.residueCol chunk, 1)]
  have rightCanonical : Program.CanonicalTerms rightTerms := by
    simp [rightTerms, Program.CanonicalTerms, goldilocksP]
  have holds := decompositionRow_holds satisfies
  have normalized : RowHolds assignment
      ⟨ChunkRows.chunkTerms chunk ++ Program.negateTerms rightTerms,
        [(0, 1)], []⟩ := by
    simpa [ChunkRows.decompositionRow, ChunkRows.zeroEqualityRow,
      rightTerms, Program.negateTerms, Program.negCoeff] using holds
  have equation := linearEquality_sound one
    (ChunkRows.chunkTerms chunk) rightTerms rightCanonical normalized
  have quotientEq := quotientRecompositionRow_sound
    prime canonical one satisfies
  have quotientBits := quotientBits_sound prime canonical one satisfies
  have quotientLt := quotientValue_lt_bound quotientBits
  have residueLt := residueRangeRows_sound prime canonical one satisfies
  have rightLt :
      5 * assignment (ChunkRows.quotientCol chunk) +
          assignment (ChunkRows.residueCol chunk) < goldilocksP := by
    rw [quotientEq]
    have alphabet : ProductionAlphabet.alphabetSize = 5 := rfl
    rw [alphabet] at residueLt
    have bound : 5 * 16384 + 5 < goldilocksP := by decide
    omega
  have rightEval : lcEval assignment rightTerms =
      5 * assignment (ChunkRows.quotientCol chunk) +
        assignment (ChunkRows.residueCol chunk) := by
    simp [rightTerms, lcEval, Nat.mod_eq_of_lt rightLt]
  rw [chunkTerms_value one sourceBits, rightEval, quotientEq] at equation
  exact equation

/-- The decoded residue equals the verifier-owned modulo-five symbol. -/
theorem residue_refines_verifier_symbol
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceBits : BitsBoolean assignment chunk)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    assignment (ChunkRows.residueCol chunk) =
      (ProductionAlphabet.verifier.symbol
        (candidate assignment chunk sourceBits)).val := by
  have decomposition := decompositionRow_sound
    prime canonical one sourceBits satisfies
  have residueLt := residueRangeRows_sound prime canonical one satisfies
  change assignment (ChunkRows.residueCol chunk) =
    chunkValue assignment chunk % ProductionAlphabet.alphabetSize
  symm
  rw [decomposition]
  change (5 * quotientValue assignment chunk +
      assignment (ChunkRows.residueCol chunk)) % 5 =
    assignment (ChunkRows.residueCol chunk)
  have residueLtFive : assignment (ChunkRows.residueCol chunk) < 5 := by
    simpa [ProductionAlphabet.alphabetSize] using residueLt
  simp [Nat.add_mod, Nat.mod_eq_of_lt residueLtFive]

private theorem symbolRow_holds
    {assignment : Nat → Nat} {chunk : Nat}
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    RowHolds assignment (ChunkRows.symbolRow chunk) := by
  apply satisfies
  simp [ChunkRows.chunkRows]

/-- The symbol row is the canonical Goldilocks encoding of the verifier-owned
centered coefficient `symbol - 2`. -/
theorem symbolRow_refines_verifier
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceBits : BitsBoolean assignment chunk)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    assignment (ChunkRows.symbolCol chunk) =
      ((ProductionAlphabet.verifier.symbol
          (candidate assignment chunk sourceBits)).val +
        (goldilocksP - 2)) % goldilocksP := by
  let terms : List (Nat × Nat) :=
    [(ChunkRows.residueCol chunk, 1), (0, goldilocksP - 2)]
  have termsCanonical : Program.CanonicalTerms terms := by
    simp [terms, Program.CanonicalTerms, goldilocksP]
  have holds := symbolRow_holds satisfies
  have builderHolds : RowHolds assignment
      (Program.builderLinearRow (ChunkRows.symbolCol chunk) terms) := by
    simpa [ChunkRows.symbolRow, ChunkRows.zeroEqualityRow, terms,
      Program.builderLinearRow, Program.negateTerms, Program.negCoeff] using holds
  have decoded := Program.builderLinearRow_sound canonical one
    (ChunkRows.symbolCol chunk) terms termsCanonical builderHolds
  have residueEq := residue_refines_verifier_symbol
    prime canonical one sourceBits satisfies
  simpa [terms, lcEval, one, residueEq] using decoded

/-- The four exact acceptance rows force the implementation wire to be one
exactly off the unique rejected bucket. -/
theorem acceptanceRows_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : BitsBoolean assignment chunk)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    assignment (ChunkRows.acceptCol chunk) =
      if chunkValue assignment chunk = 65535 then 0 else 1 := by
  have acceptance := satisfies_acceptanceRows satisfies
  have acceptLe : assignment (ChunkRows.acceptCol chunk) ≤ 1 :=
    bitRow_le_one prime (canonical _) one
      (acceptance _ (by simp [ChunkRows.acceptanceRows]))
  have zeroProduct := acceptance
    ⟨ChunkRows.oneMinusAcceptTerms chunk,
      ChunkRows.differenceTerms chunk, []⟩
    (by simp [ChunkRows.acceptanceRows])
  have inverseProduct := acceptance
    ⟨ChunkRows.differenceTerms chunk,
      [(ChunkRows.inverseCol chunk, 1)],
      [(ChunkRows.acceptCol chunk, 1)]⟩
    (by simp [ChunkRows.acceptanceRows])
  have differenceEq := lcEval_differenceTerms (chunk := chunk) one bits
  have oneMinusEq := lcEval_oneMinusAccept (chunk := chunk) one acceptLe
  have valueLt := chunkValue_lt_bound bits
  have differenceZero := difference_mod_eq_zero_iff valueLt
  by_cases rejected : chunkValue assignment chunk = 65535
  · simp [rejected]
    have differenceIsZero :
        (chunkValue assignment chunk + goldilocksP - 65535) %
            goldilocksP = 0 :=
      differenceZero.mpr rejected
    simp only [RowHolds] at inverseProduct
    rw [differenceEq, differenceIsZero] at inverseProduct
    have acceptCanonical := canonical (ChunkRows.acceptCol chunk)
    simp [lcEval, Nat.mod_eq_of_lt acceptCanonical] at inverseProduct
    exact inverseProduct.symm
  · simp [rejected]
    have differenceNonzero :
        (chunkValue assignment chunk + goldilocksP - 65535) %
            goldilocksP ≠ 0 := by
      exact fun zero => rejected (differenceZero.mp zero)
    simp only [RowHolds] at zeroProduct
    rw [oneMinusEq, differenceEq] at zeroProduct
    simp [lcEval] at zeroProduct
    have factors := prime _ _ zeroProduct
    rcases factors with firstZero | secondZero
    · have firstSmall :
          1 - assignment (ChunkRows.acceptCol chunk) < goldilocksP := by
        have bound : 1 < goldilocksP := by decide
        omega
      rw [Nat.mod_eq_of_lt firstSmall] at firstZero
      omega
    · exact False.elim (differenceNonzero secondZero)

/-- The acceptance wire refines the independently defined production verifier,
not a second Lean transcription of the Rust branch. -/
theorem acceptanceRows_refine_verifier
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : BitsBoolean assignment chunk)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    assignment (ChunkRows.acceptCol chunk) =
      if ProductionAlphabet.verifier.accepts
          (candidate assignment chunk bits) then 1 else 0 := by
  rw [acceptanceRows_sound prime canonical one bits satisfies]
  have acceptedIff :=
    ProductionAlphabet.accepts_eq_true_iff_ne_rejectionBucket
    (candidate assignment chunk bits)
  simp only [candidate] at acceptedIff
  by_cases rejected : chunkValue assignment chunk = 65535
  · have notAccepted :
        ProductionAlphabet.verifier.accepts
            (candidate assignment chunk bits) = false := by
      apply Bool.eq_false_iff.mpr
      intro accepted
      exact (acceptedIff.mp accepted) rejected
    simp [rejected, notAccepted]
  · have accepted :
        ProductionAlphabet.verifier.accepts
            (candidate assignment chunk bits) = true :=
      acceptedIff.mpr rejected
    simp [rejected, accepted]

private theorem cumulativeRow_holds
    {assignment : Nat → Nat} {chunk : Nat}
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    RowHolds assignment (ChunkRows.cumulativeRow chunk) := by
  apply satisfies
  simp [ChunkRows.chunkRows]

/-- Subject to the verifier-visible 64-candidate bound on the incoming count,
the cumulative row advances by exactly the verifier-owned acceptance bit; no
field wrap can disguise another integer count. -/
theorem cumulativeRow_refines_verifier
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {chunk : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceBits : BitsBoolean assignment chunk)
    (priorBound : assignment (ChunkRows.priorCumulativeCol chunk) <
      ProductionAlphabet.candidateBound)
    (satisfies : Satisfies (ChunkRows.chunkRows chunk) assignment) :
    assignment (ChunkRows.cumulativeCol chunk) =
      assignment (ChunkRows.priorCumulativeCol chunk) +
        if ProductionAlphabet.verifier.accepts
            (candidate assignment chunk sourceBits) then 1 else 0 := by
  let terms : List (Nat × Nat) :=
    [(ChunkRows.priorCumulativeCol chunk, 1),
      (ChunkRows.acceptCol chunk, 1)]
  have termsCanonical : Program.CanonicalTerms terms := by
    simp [terms, Program.CanonicalTerms, goldilocksP]
  have holds := cumulativeRow_holds satisfies
  have builderHolds : RowHolds assignment
      (Program.builderLinearRow (ChunkRows.cumulativeCol chunk) terms) := by
    simpa [ChunkRows.cumulativeRow, ChunkRows.zeroEqualityRow, terms,
      Program.builderLinearRow, Program.negateTerms, Program.negCoeff] using holds
  have decoded := Program.builderLinearRow_sound canonical one
    (ChunkRows.cumulativeCol chunk) terms termsCanonical builderHolds
  have acceptEq := acceptanceRows_refine_verifier
    prime canonical one sourceBits satisfies
  have acceptLe : assignment (ChunkRows.acceptCol chunk) ≤ 1 := by
    rw [acceptEq]
    split <;> simp
  have sumLt :
      assignment (ChunkRows.priorCumulativeCol chunk) +
          assignment (ChunkRows.acceptCol chunk) < goldilocksP := by
    have candidateBound : ProductionAlphabet.candidateBound = 64 := rfl
    rw [candidateBound] at priorBound
    have bound : 65 < goldilocksP := by decide
    omega
  have termsEval : lcEval assignment terms =
      assignment (ChunkRows.priorCumulativeCol chunk) +
        assignment (ChunkRows.acceptCol chunk) := by
    simp [terms, lcEval, Nat.mod_eq_of_lt sumLt]
  rw [termsEval, acceptEq] at decoded
  exact decoded

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk
