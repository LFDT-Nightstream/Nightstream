import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate
import Nightstream.Implementation.R1CS.Canonical.KHorner

/-!
Contract: semantic soundness of one Lean-owned `Pi_RLC` candidate
classification.

Satisfaction refines the independent production verifier:

* the complemented source bits determine one canonical 16-bit candidate;
* the accept wire is exactly the verifier's rejection decision;
* the residue is exactly the verifier's modulo-five symbol; and
* the cumulative wire advances by that accept decision.

No generated rows or caller-supplied conclusion are imported.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

private theorem finRange16 :
    List.finRange sourceBitCount =
      [⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩,
        ⟨3, by decide⟩, ⟨4, by decide⟩, ⟨5, by decide⟩,
        ⟨6, by decide⟩, ⟨7, by decide⟩, ⟨8, by decide⟩,
        ⟨9, by decide⟩, ⟨10, by decide⟩, ⟨11, by decide⟩,
        ⟨12, by decide⟩, ⟨13, by decide⟩, ⟨14, by decide⟩,
        ⟨15, by decide⟩] := by
  decide

private theorem range14 :
    List.range quotientBitCount =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] := by
  decide

def rawValue (assignment : Nat → Nat) (layout : Layout) : Nat :=
  (List.finRange sourceBitCount).foldl
    (fun value index =>
      value + 2 ^ index.val * assignment (layout.sourceBit index)) 0

def chunkValue (assignment : Nat → Nat) (layout : Layout) : Nat :=
  ProductionAlphabet.rejectionBucket - rawValue assignment layout

def SourceBitsBoolean (assignment : Nat → Nat) (layout : Layout) : Prop :=
  ∀ index, assignment (layout.sourceBit index) ≤ 1

def quotientValue (assignment : Nat → Nat) (layout : Layout) : Nat :=
  (List.range quotientBitCount).foldl
    (fun value offset =>
      value + 2 ^ offset * assignment (quotientBitColumn layout offset)) 0

def QuotientBitsBoolean (assignment : Nat → Nat) (layout : Layout) : Prop :=
  ∀ offset, offset < quotientBitCount →
    assignment (quotientBitColumn layout offset) ≤ 1

theorem chunkValue_lt_bound
    {assignment : Nat → Nat} {layout : Layout}
    (bits : SourceBitsBoolean assignment layout) :
    chunkValue assignment layout < ProductionAlphabet.chunkModulus := by
  have b0 := bits ⟨0, by decide⟩
  have b1 := bits ⟨1, by decide⟩
  have b2 := bits ⟨2, by decide⟩
  have b3 := bits ⟨3, by decide⟩
  have b4 := bits ⟨4, by decide⟩
  have b5 := bits ⟨5, by decide⟩
  have b6 := bits ⟨6, by decide⟩
  have b7 := bits ⟨7, by decide⟩
  have b8 := bits ⟨8, by decide⟩
  have b9 := bits ⟨9, by decide⟩
  have b10 := bits ⟨10, by decide⟩
  have b11 := bits ⟨11, by decide⟩
  have b12 := bits ⟨12, by decide⟩
  have b13 := bits ⟨13, by decide⟩
  have b14 := bits ⟨14, by decide⟩
  have b15 := bits ⟨15, by decide⟩
  have p3 : 2 ^ ((3 : Fin 16) : Nat) = 8 := by
    decide
  have p4 : 2 ^ ((4 : Fin 16) : Nat) = 16 := by
    decide
  have p5 : 2 ^ ((5 : Fin 16) : Nat) = 32 := by
    decide
  have p6 : 2 ^ ((6 : Fin 16) : Nat) = 64 := by
    decide
  have p7 : 2 ^ ((7 : Fin 16) : Nat) = 128 := by
    decide
  have p8 : 2 ^ ((8 : Fin 16) : Nat) = 256 := by
    decide
  have p9 : 2 ^ ((9 : Fin 16) : Nat) = 512 := by
    decide
  have p10 : 2 ^ ((10 : Fin 16) : Nat) = 1024 := by
    decide
  have p11 : 2 ^ ((11 : Fin 16) : Nat) = 2048 := by
    decide
  have p12 : 2 ^ ((12 : Fin 16) : Nat) = 4096 := by
    decide
  have p13 : 2 ^ ((13 : Fin 16) : Nat) = 8192 := by
    decide
  have p14 : 2 ^ ((14 : Fin 16) : Nat) = 16384 := by
    decide
  have p15 : 2 ^ ((15 : Fin 16) : Nat) = 32768 := by
    decide
  unfold chunkValue rawValue
  rw [finRange16]
  simp [sourceBitCount, ProductionAlphabet.chunkModulus,
    ProductionAlphabet.rejectionBucket,
    p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15] at *
  omega

theorem quotientValue_lt_bound
    {assignment : Nat → Nat} {layout : Layout}
    (bits : QuotientBitsBoolean assignment layout) :
    quotientValue assignment layout < 16384 := by
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
  unfold quotientValue
  rw [range14]
  simp at *
  omega

def candidate
    (assignment : Nat → Nat) (layout : Layout)
    (bits : SourceBitsBoolean assignment layout) :
    ProductionAlphabet.Chunk :=
  ⟨chunkValue assignment layout, chunkValue_lt_bound bits⟩

theorem chunkTerms_eval
    {assignment : Nat → Nat} {layout : Layout}
    (constantWire : assignment 0 = 1)
    (bits : SourceBitsBoolean assignment layout) :
    lcEval assignment (chunkTerms layout) =
      chunkValue assignment layout := by
  have valueLt := chunkValue_lt_bound bits
  have valueGoldilocks : chunkValue assignment layout < goldilocksP := by
    have value16 : chunkValue assignment layout < 65536 := by
      simpa [ProductionAlphabet.chunkModulus] using valueLt
    exact Nat.lt_trans value16 (by decide)
  unfold lcEval
  let oneCount : Nat :=
    (List.finRange sourceBitCount).foldl
      (fun value index => value + assignment (layout.sourceBit index)) 0
  have raw :
      (chunkTerms layout).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
        chunkValue assignment layout + goldilocksP * oneCount := by
    have b0 := bits ⟨0, by decide⟩
    have b1 := bits ⟨1, by decide⟩
    have b2 := bits ⟨2, by decide⟩
    have b3 := bits ⟨3, by decide⟩
    have b4 := bits ⟨4, by decide⟩
    have b5 := bits ⟨5, by decide⟩
    have b6 := bits ⟨6, by decide⟩
    have b7 := bits ⟨7, by decide⟩
    have b8 := bits ⟨8, by decide⟩
    have b9 := bits ⟨9, by decide⟩
    have b10 := bits ⟨10, by decide⟩
    have b11 := bits ⟨11, by decide⟩
    have b12 := bits ⟨12, by decide⟩
    have b13 := bits ⟨13, by decide⟩
    have b14 := bits ⟨14, by decide⟩
    have b15 := bits ⟨15, by decide⟩
    have p3 : 2 ^ ((3 : Fin 16) : Nat) = 8 := by decide
    have p4 : 2 ^ ((4 : Fin 16) : Nat) = 16 := by decide
    have p5 : 2 ^ ((5 : Fin 16) : Nat) = 32 := by decide
    have p6 : 2 ^ ((6 : Fin 16) : Nat) = 64 := by decide
    have p7 : 2 ^ ((7 : Fin 16) : Nat) = 128 := by decide
    have p8 : 2 ^ ((8 : Fin 16) : Nat) = 256 := by decide
    have p9 : 2 ^ ((9 : Fin 16) : Nat) = 512 := by decide
    have p10 : 2 ^ ((10 : Fin 16) : Nat) = 1024 := by decide
    have p11 : 2 ^ ((11 : Fin 16) : Nat) = 2048 := by decide
    have p12 : 2 ^ ((12 : Fin 16) : Nat) = 4096 := by decide
    have p13 : 2 ^ ((13 : Fin 16) : Nat) = 8192 := by decide
    have p14 : 2 ^ ((14 : Fin 16) : Nat) = 16384 := by decide
    have p15 : 2 ^ ((15 : Fin 16) : Nat) = 32768 := by decide
    dsimp only [chunkTerms, chunkValue, rawValue, oneCount]
    rw [finRange16]
    simp [constantWire, sourceBitCount,
      ProductionAlphabet.rejectionBucket, goldilocksP,
      p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15] at *
    omega
  rw [raw]
  have multipleZero : goldilocksP * oneCount % goldilocksP = 0 := by
    rw [Nat.mul_comm]
    exact Nat.mul_mod_left _ _
  rw [Nat.add_mod, Nat.mod_eq_of_lt valueGoldilocks, multipleZero]
  simp [Nat.mod_eq_of_lt valueGoldilocks]

theorem quotientTerms_eval
    {assignment : Nat → Nat} {layout : Layout}
    (bits : QuotientBitsBoolean assignment layout) :
    lcEval assignment (quotientTerms layout) =
      quotientValue assignment layout := by
  have valueLt := quotientValue_lt_bound bits
  have valueGoldilocks : quotientValue assignment layout < goldilocksP :=
    Nat.lt_trans valueLt (by decide)
  unfold lcEval
  have raw :
      (quotientTerms layout).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
        quotientValue assignment layout := by
    simp [quotientTerms, quotientValue, List.foldl_map]
  rw [raw, Nat.mod_eq_of_lt valueGoldilocks]

private theorem singleton_eval
    (assignment : Nat → Nat) (column : Nat)
    (canonical : assignment column < goldilocksP) :
    lcEval assignment [(column, 1)] = assignment column := by
  simp [lcEval, Nat.mod_eq_of_lt canonical]

private theorem one_eval
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1) :
    lcEval assignment [(0, 1)] = 1 := by
  simp [lcEval, constantWire, goldilocksP]

theorem differenceTerms_eval
    {assignment : Nat → Nat} {layout : Layout}
    (constantWire : assignment 0 = 1)
    (bits : SourceBitsBoolean assignment layout) :
    lcEval assignment (differenceTerms layout) =
      (chunkValue assignment layout + goldilocksP -
        ProductionAlphabet.rejectionBucket) % goldilocksP := by
  rw [differenceTerms, KHorner.lcEval_append,
    chunkTerms_eval constantWire bits]
  have shifted :
      chunkValue assignment layout +
          (goldilocksP - ProductionAlphabet.rejectionBucket) =
        chunkValue assignment layout + goldilocksP -
          ProductionAlphabet.rejectionBucket := by
    have rejectionLt :
        ProductionAlphabet.rejectionBucket < goldilocksP := by
      decide
    omega
  simp [lcEval, constantWire, shifted]

theorem oneMinusAccept_eval
    {assignment : Nat → Nat} {layout : Layout}
    (constantWire : assignment 0 = 1)
    (acceptLe : assignment (acceptColumn layout) ≤ 1) :
    lcEval assignment (oneMinusAccept layout) =
      1 - assignment (acceptColumn layout) := by
  have cases :
      assignment (acceptColumn layout) = 0 ∨
        assignment (acceptColumn layout) = 1 := by
    omega
  rcases cases with zero | one
  · simp [oneMinusAccept, lcEval, constantWire, zero, goldilocksP]
  · simp [oneMinusAccept, lcEval, constantWire, one, goldilocksP]

theorem difference_zero_iff
    {value : Nat}
    (valueLt : value < ProductionAlphabet.chunkModulus) :
    (value + goldilocksP - ProductionAlphabet.rejectionBucket) %
        goldilocksP = 0 ↔
      value = ProductionAlphabet.rejectionBucket := by
  change value < 65536 at valueLt
  change (value + 18446744069414584321 - 65535) %
      18446744069414584321 = 0 ↔ value = 65535
  constructor
  · intro zero
    by_cases equal : value = 65535
    · exact equal
    · have valueSmall : value < 65535 := by omega
      have shiftedLt :
          value + 18446744069414584321 - 65535 <
            18446744069414584321 := by
        omega
      rw [Nat.mod_eq_of_lt shiftedLt] at zero
      omega
  · intro equal
    subst value
    simp

private theorem fieldSub_eq_zero_iff
    {value amount : Nat}
    (valueLt : value < goldilocksP)
    (amountPositive : 0 < amount)
    (amountLt : amount < goldilocksP) :
    (value + (goldilocksP - amount)) % goldilocksP = 0 ↔
      value = amount := by
  have shifted :
      value + (goldilocksP - amount) =
        value + goldilocksP - amount := by
    omega
  rw [shifted]
  constructor
  · intro zero
    by_cases small : value < amount
    · have shiftedLt : value + goldilocksP - amount < goldilocksP := by
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

private theorem satisfies_acceptance
    {assignment : Nat → Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (acceptanceRows layout) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem satisfies_residueRange
    {assignment : Nat → Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (residueRangeRows layout) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem satisfies_quotientBits
    {assignment : Nat → Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (quotientBitRows layout) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

theorem quotientBits_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    QuotientBitsBoolean assignment layout := by
  intro offset bounded
  apply bitRow_le_one prime (canonical _) constantWire
  apply satisfies_quotientBits satisfied
  exact List.mem_map.mpr
    ⟨offset, List.mem_range.mpr bounded, rfl⟩

theorem quotientRecomposition_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (quotientColumn layout) =
      quotientValue assignment layout := by
  have bits := quotientBits_sound prime canonical constantWire satisfied
  have holds := satisfied (quotientRecompositionRow layout) (by simp [rows])
  have quotientEval :=
    singleton_eval assignment (quotientColumn layout) (canonical _)
  have bitsEval := quotientTerms_eval bits
  have one := one_eval assignment constantWire
  simpa [RowHolds, quotientRecompositionRow, quotientEval, bitsEval, one,
    Nat.mod_eq_of_lt (canonical _)] using holds

theorem residueRange_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (residueColumn layout) <
      ProductionAlphabet.alphabetSize := by
  have residueRows := satisfies_residueRange satisfied
  have first := residueRows
    ⟨[(residueColumn layout, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 1)],
      [(productColumn layout 0, 1)]⟩
    (by simp [residueRangeRows])
  have second := residueRows
    ⟨[(productColumn layout 0, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 2)],
      [(productColumn layout 1, 1)]⟩
    (by simp [residueRangeRows])
  have third := residueRows
    ⟨[(productColumn layout 1, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 3)],
      [(productColumn layout 2, 1)]⟩
    (by simp [residueRangeRows])
  have fourth := residueRows
    ⟨[(productColumn layout 2, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 4)], []⟩
    (by simp [residueRangeRows])
  simp only [RowHolds, lcEval, List.foldl, constantWire, Nat.one_mul,
    Nat.mul_one, Nat.zero_add, Nat.zero_mod,
    Nat.mod_eq_of_lt (canonical _)] at first second third fourth
  let residue := assignment (residueColumn layout)
  let product0 := assignment (productColumn layout 0)
  let product1 := assignment (productColumn layout 1)
  let product2 := assignment (productColumn layout 2)
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
  have product0Lt : product0 < goldilocksP := canonical _
  have product1Lt : product1 < goldilocksP := canonical _
  have product2Lt : product2 < goldilocksP := canonical _
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
          simpa [ProductionAlphabet.alphabetSize, residueZero]
        · have equalsOne : residue = 1 :=
            (fieldSub_eq_zero_iff residueLt (by decide) (by decide)).mp
              (by simpa only [Nat.mod_mod] using residueOne)
          simpa [ProductionAlphabet.alphabetSize, equalsOne]
      · have equalsTwo : residue = 2 :=
          (fieldSub_eq_zero_iff residueLt (by decide) (by decide)).mp
            (by simpa only [Nat.mod_mod] using residueTwo)
        simpa [ProductionAlphabet.alphabetSize, equalsTwo]
    · have equalsThree : residue = 3 :=
        (fieldSub_eq_zero_iff residueLt (by decide) (by decide)).mp
          (by simpa only [Nat.mod_mod] using residueThree)
      simpa [ProductionAlphabet.alphabetSize, equalsThree]
  · have equalsFour : residue = 4 :=
      (fieldSub_eq_zero_iff residueLt (by decide) (by decide)).mp
        (by simpa only [Nat.mod_mod] using residueFour)
    simpa [ProductionAlphabet.alphabetSize, equalsFour]

theorem decomposition_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (sourceBits : SourceBitsBoolean assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    chunkValue assignment layout =
      5 * quotientValue assignment layout +
        assignment (residueColumn layout) := by
  have holds := satisfied (decompositionRow layout) (by simp [rows])
  have chunkEval := chunkTerms_eval constantWire sourceBits
  have quotientEq :=
    quotientRecomposition_sound prime canonical constantWire satisfied
  have residueLt :=
    residueRange_sound prime canonical constantWire satisfied
  have rightLt :
      5 * assignment (quotientColumn layout) +
          assignment (residueColumn layout) < goldilocksP := by
    rw [quotientEq]
    have quotientBound := quotientValue_lt_bound
      (quotientBits_sound prime canonical constantWire satisfied)
    change assignment (residueColumn layout) < 5 at residueLt
    have bound : 5 * 16384 + 5 < goldilocksP := by decide
    omega
  have rightEval :
      lcEval assignment
          [(quotientColumn layout, 5), (residueColumn layout, 1)] =
        5 * assignment (quotientColumn layout) +
          assignment (residueColumn layout) := by
    simp [lcEval, Nat.mod_eq_of_lt rightLt]
  have one := one_eval assignment constantWire
  have chunkLt : chunkValue assignment layout < goldilocksP :=
    Nat.lt_trans (chunkValue_lt_bound sourceBits) (by decide)
  simp only [RowHolds, decompositionRow, chunkEval, rightEval, one,
    Nat.mul_one, Nat.mod_eq_of_lt chunkLt] at holds
  rw [quotientEq] at holds
  exact holds

theorem residue_refines_verifier
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (sourceBits : SourceBitsBoolean assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (residueColumn layout) =
      (ProductionAlphabet.verifier.symbol
        (candidate assignment layout sourceBits)).val := by
  have decomposition :=
    decomposition_sound prime canonical constantWire sourceBits satisfied
  have residueLt :=
    residueRange_sound prime canonical constantWire satisfied
  change assignment (residueColumn layout) =
    chunkValue assignment layout % ProductionAlphabet.alphabetSize
  symm
  rw [decomposition]
  have residueFive :
      assignment (residueColumn layout) < 5 := by
    simpa [ProductionAlphabet.alphabetSize] using residueLt
  change
    (5 * quotientValue assignment layout +
      assignment (residueColumn layout)) % 5 =
        assignment (residueColumn layout)
  simp [Nat.add_mod, Nat.mod_eq_of_lt residueFive]

theorem acceptance_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (sourceBits : SourceBitsBoolean assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (acceptColumn layout) =
      if ProductionAlphabet.verifier.accepts
          (candidate assignment layout sourceBits) then 1 else 0 := by
  have acceptance := satisfies_acceptance satisfied
  have acceptLe : assignment (acceptColumn layout) ≤ 1 :=
    bitRow_le_one prime (canonical _) constantWire
      (acceptance _ (by simp [acceptanceRows]))
  have zeroProduct := acceptance
    ⟨oneMinusAccept layout, differenceTerms layout, []⟩
    (by simp [acceptanceRows])
  have inverseProduct := acceptance
    ⟨differenceTerms layout, [(inverseColumn layout, 1)],
      [(acceptColumn layout, 1)]⟩
    (by simp [acceptanceRows])
  have differenceEq :=
    differenceTerms_eval constantWire sourceBits
  have oneMinusEq :=
    oneMinusAccept_eval constantWire acceptLe
  have valueLt := chunkValue_lt_bound sourceBits
  have differenceZero := difference_zero_iff valueLt
  have acceptedIff :=
    ProductionAlphabet.accepts_eq_true_iff_ne_rejectionBucket
      (candidate assignment layout sourceBits)
  simp only [candidate] at acceptedIff
  by_cases rejected :
      chunkValue assignment layout = ProductionAlphabet.rejectionBucket
  · have notAccepted :
        ProductionAlphabet.verifier.accepts
            (candidate assignment layout sourceBits) = false := by
      apply Bool.eq_false_iff.mpr
      intro accepted
      exact (acceptedIff.mp accepted) rejected
    simp [notAccepted]
    have differenceIsZero :
        (chunkValue assignment layout + goldilocksP -
          ProductionAlphabet.rejectionBucket) % goldilocksP = 0 :=
      differenceZero.mpr rejected
    simp only [RowHolds] at inverseProduct
    rw [differenceEq, differenceIsZero] at inverseProduct
    have acceptCanonical := canonical (acceptColumn layout)
    simp [lcEval, Nat.mod_eq_of_lt acceptCanonical] at inverseProduct
    exact inverseProduct.symm
  · have accepted :
        ProductionAlphabet.verifier.accepts
            (candidate assignment layout sourceBits) = true :=
      acceptedIff.mpr rejected
    simp [accepted]
    have differenceNonzero :
        (chunkValue assignment layout + goldilocksP -
          ProductionAlphabet.rejectionBucket) % goldilocksP ≠ 0 := by
      exact fun zero => rejected (differenceZero.mp zero)
    simp only [RowHolds] at zeroProduct
    rw [oneMinusEq, differenceEq] at zeroProduct
    simp [lcEval] at zeroProduct
    rcases prime _ _ zeroProduct with firstZero | secondZero
    · have firstSmall :
          1 - assignment (acceptColumn layout) < goldilocksP := by
        have bound : 1 < goldilocksP := by decide
        omega
      rw [Nat.mod_eq_of_lt firstSmall] at firstZero
      omega
    · exact False.elim (differenceNonzero secondZero)

theorem cumulative_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (sourceBits : SourceBitsBoolean assignment layout)
    (priorBound : lcEval assignment layout.prior <
      ProductionAlphabet.candidateBound)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (cumulativeColumn layout) =
      lcEval assignment layout.prior +
        if ProductionAlphabet.verifier.accepts
            (candidate assignment layout sourceBits) then 1 else 0 := by
  have holds := satisfied (cumulativeRow layout) (by simp [rows])
  have acceptEq :=
    acceptance_sound prime canonical constantWire sourceBits satisfied
  have acceptLe : assignment (acceptColumn layout) ≤ 1 := by
    rw [acceptEq]
    split <;> simp
  have sumLt :
      lcEval assignment layout.prior +
          assignment (acceptColumn layout) < goldilocksP := by
    change lcEval assignment layout.prior < 64 at priorBound
    have bound : 65 < goldilocksP := by decide
    omega
  have rightEval :
      lcEval assignment
          (layout.prior ++ [(acceptColumn layout, 1)]) =
        lcEval assignment layout.prior +
          assignment (acceptColumn layout) := by
    rw [KHorner.lcEval_append]
    rw [singleton_eval assignment (acceptColumn layout) (canonical _),
      Nat.mod_eq_of_lt sumLt]
  have cumulativeEval :=
    singleton_eval assignment (cumulativeColumn layout) (canonical _)
  have one := one_eval assignment constantWire
  simp only [RowHolds, cumulativeRow, cumulativeEval, one, rightEval,
    Nat.mul_one, Nat.mod_eq_of_lt (canonical _)] at holds
  rw [acceptEq] at holds
  exact holds

/-- Complete semantic result for one candidate occurrence. -/
structure Refines
    (assignment : Nat → Nat) (layout : Layout)
    (sourceBits : SourceBitsBoolean assignment layout) : Prop where
  accepted :
    assignment (acceptColumn layout) =
      if ProductionAlphabet.verifier.accepts
          (candidate assignment layout sourceBits) then 1 else 0
  symbol :
    assignment (residueColumn layout) =
      (ProductionAlphabet.verifier.symbol
        (candidate assignment layout sourceBits)).val
  cumulative :
    assignment (cumulativeColumn layout) =
      lcEval assignment layout.prior +
        if ProductionAlphabet.verifier.accepts
            (candidate assignment layout sourceBits) then 1 else 0

theorem sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (sourceBits : SourceBitsBoolean assignment layout)
    (priorBound : lcEval assignment layout.prior <
      ProductionAlphabet.candidateBound)
    (satisfied : Satisfies (rows layout) assignment) :
    Refines assignment layout sourceBits where
  accepted :=
    acceptance_sound prime canonical constantWire sourceBits satisfied
  symbol :=
    residue_refines_verifier prime canonical constantWire sourceBits satisfied
  cumulative :=
    cumulative_sound prime canonical constantWire sourceBits priorBound
      satisfied

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateSound
