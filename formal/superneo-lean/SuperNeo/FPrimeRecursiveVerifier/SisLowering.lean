import SuperNeo.Primitives.Field

/-!
Owns: the model-level canonical 41-trit encoding used at the SIS boundary,
its unique decoder, and structural composition through two deterministic SIS
maps and a metadata-binding Poseidon2 envelope.

Does not own: Rust trace conformance, R1CS/CCS row equivalence, seeded-Phi81
coefficient generation, row-major padding, concrete Poseidon2 semantics, or
Module-SIS/collision-resistance reductions.

Emits constraints: no.

Authority boundary: source fields determine one canonical centered-unit
message. Both linear maps and the final envelope are functions of that
message; no carried commitment or digest is accepted as authority.

| Model branch | Mathematical obligation | Production concept |
|---|---|---|
| `SourceFieldRows` | Centered-unit word, reconstruction, and `< q` canonicality | 124 shifted-ternary rows per field |
| `canonicalWord` | Exact 41-digit low-norm encoding | balanced-ternary witness slots |
| `decodeWord_canonicalWord` | Unique recovery of the source field | shared field/digit slot reconstruction |
| `Pipeline.primaryMap` | First deterministic linear map | rank-2 `SeededPhi81` block |
| `Pipeline.shortMap` | Independent short deterministic map | rank-1 `SeededPhi81` block |
| `Pipeline.poseidon2Envelope` | Bind role, field count, rank, and short output | SIS digest envelope |
| `sourceSisRows_iff_lowNormEncoding` | Source semantics equal canonical low-norm semantics | low-norm representation target |

A concrete refinement must separately prove: the exact Rust digit order and
Goldilocks representatives; soundness and completeness of every represented
alphabet/reconstruction/borrow row; source-field-to-digit alias coverage;
row-major SIS padding and word starts; both seeded-Phi81 schedules and their
independent domains; the exact Poseidon2 v4 envelope serialization; and that no
aliased or projected column escapes its validated boundary. None is inferred
from metadata or from this model theorem.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.SisLowering

/-- Number of centered ternary digits in one canonical Goldilocks opening. -/
def digitCount : Nat := 41

/-- Full unsigned range of a 41-trit word. -/
def ternaryRadix : Nat := 3 ^ digitCount

/-- All-ones ternary word, interpreted as the offset from trits to centered units. -/
def shift : Nat := (ternaryRadix - 1) / 2

theorem goldilocks_lt_ternaryRadix : Goldilocks.q < ternaryRadix := by
  native_decide

theorem shift_lt_goldilocks : shift < Goldilocks.q := by
  native_decide

/-- A low-norm SIS coefficient. Invalid field representatives are unrepresentable. -/
inductive CenteredUnit where
  | neg
  | zero
  | pos
deriving DecidableEq, Repr

namespace CenteredUnit

/-- Ordinary base-three digit obtained by adding one to a centered unit. -/
def trit : CenteredUnit → Nat
  | .neg => 0
  | .zero => 1
  | .pos => 2

/-- Goldilocks representative consumed by the SIS linear map. -/
def residue : CenteredUnit → F
  | .neg => -1
  | .zero => 0
  | .pos => 1

theorem trit_lt_three (digit : CenteredUnit) : digit.trit < 3 := by
  cases digit <;> decide

theorem residue_centeredAbs_le_one (digit : CenteredUnit) :
    F.centeredAbs digit.residue ≤ 1 := by
  cases digit <;> native_decide

/-- Canonical centered unit for one ordinary trit; the input is reduced modulo three. -/
def ofTrit (value : Nat) : CenteredUnit :=
  match value % 3 with
  | 0 => .neg
  | 1 => .zero
  | _ => .pos

@[simp] theorem trit_ofTrit (value : Nat) :
    (ofTrit value).trit = value % 3 := by
  have hLt : value % 3 < 3 := Nat.mod_lt _ (by decide)
  have hCases : value % 3 = 0 ∨ value % 3 = 1 ∨ value % 3 = 2 := by
    omega
  rcases hCases with hZero | hOne | hTwo
  · simp [ofTrit, trit, hZero]
  · simp [ofTrit, trit, hOne]
  · simp [ofTrit, trit, hTwo]

@[simp] theorem ofTrit_trit (digit : CenteredUnit) :
    ofTrit digit.trit = digit := by
  cases digit <;> rfl

end CenteredUnit

/-- Least-significant-trit-first encoding of a natural number. -/
def encodeNat : Nat → Nat → List CenteredUnit
  | 0, _ => []
  | count + 1, value =>
      CenteredUnit.ofTrit value :: encodeNat count (value / 3)

/-- Unsigned ordinary-base-three value of a centered-unit word. -/
def decodeNat : List CenteredUnit → Nat
  | [] => 0
  | digit :: tail => digit.trit + 3 * decodeNat tail

@[simp] theorem encodeNat_length (count value : Nat) :
    (encodeNat count value).length = count := by
  induction count generalizing value with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [encodeNat, inductionHypothesis]

/-- Encoding and decoding round-trip for every value covered by the chosen width. -/
theorem decodeNat_encodeNat_of_lt
    (count value : Nat) (hValue : value < 3 ^ count) :
    decodeNat (encodeNat count value) = value := by
  induction count generalizing value with
  | zero =>
      simp only [pow_zero] at hValue
      have hZero : value = 0 := by omega
      simp [encodeNat, decodeNat, hZero]
  | succ count inductionHypothesis =>
      have hTail : value / 3 < 3 ^ count := by
        apply (Nat.div_lt_iff_lt_mul (by decide : 0 < 3)).2
        simpa [pow_succ, Nat.mul_comm] using hValue
      rw [encodeNat, decodeNat, CenteredUnit.trit_ofTrit,
        inductionHypothesis (value / 3) hTail]
      exact Nat.mod_add_div value 3

/-- Every fixed-width centered-unit word is the encoding of its decoded value. -/
theorem encodeNat_decodeNat_of_length
    (count : Nat) (digits : List CenteredUnit)
    (hLength : digits.length = count) :
    encodeNat count (decodeNat digits) = digits := by
  induction count generalizing digits with
  | zero =>
      have hNil : digits = [] := List.eq_nil_of_length_eq_zero hLength
      simp [hNil, encodeNat]
  | succ count inductionHypothesis =>
      cases digits with
      | nil => simp at hLength
      | cons digit tail =>
          have hTailLength : tail.length = count := by
            simpa using hLength
          have hMod : decodeNat (digit :: tail) % 3 = digit.trit := by
            simp [decodeNat, Nat.add_mul_mod_self_left,
              Nat.mod_eq_of_lt digit.trit_lt_three]
          have hDiv : decodeNat (digit :: tail) / 3 = decodeNat tail := by
            simp [decodeNat, Nat.add_mul_div_left,
              Nat.div_eq_of_lt digit.trit_lt_three]
          have hHead :
              CenteredUnit.ofTrit (decodeNat (digit :: tail)) = digit := by
            rw [CenteredUnit.ofTrit, hMod]
            cases digit <;> rfl
          rw [encodeNat, hHead, hDiv,
            inductionHypothesis tail hTailLength]

/-- Shift one canonical field value into the unsigned ternary interval. -/
def targetValue (source : F) : Nat :=
  (source.val + shift) % Goldilocks.q

/-- Undo the all-ones ternary offset inside the Goldilocks field. -/
def unshiftValue (value : Nat) : Nat :=
  (value + Goldilocks.q - shift) % Goldilocks.q

theorem targetValue_lt_goldilocks (source : F) :
    targetValue source < Goldilocks.q := by
  exact Nat.mod_lt _ Goldilocks.q_pos

theorem unshiftValue_lt_goldilocks (value : Nat) :
    unshiftValue value < Goldilocks.q := by
  exact Nat.mod_lt _ Goldilocks.q_pos

/-- Shifting then unshifting recovers the canonical field representative. -/
theorem unshiftValue_targetValue (source : F) :
    unshiftValue (targetValue source) = source.val := by
  unfold unshiftValue targetValue
  have hShift : shift < Goldilocks.q := shift_lt_goldilocks
  have hSource : source.val < Goldilocks.q := source.isLt
  by_cases hBelow : source.val + shift < Goldilocks.q
  · rw [Nat.mod_eq_of_lt hBelow]
    have hSum :
        source.val + shift + Goldilocks.q - shift =
          source.val + Goldilocks.q := by
      omega
    rw [hSum]
    simp [Nat.mod_eq_of_lt source.isLt]
  · have hModulusLe : Goldilocks.q ≤ source.val + shift :=
      Nat.le_of_not_gt hBelow
    rw [Nat.mod_eq_sub_mod hModulusLe]
    have hReduced :
        source.val + shift - Goldilocks.q < Goldilocks.q := by
      omega
    rw [Nat.mod_eq_of_lt hReduced]
    have hSum :
        source.val + shift - Goldilocks.q + Goldilocks.q - shift =
          source.val := by
      omega
    rw [hSum, Nat.mod_eq_of_lt source.isLt]

/-- Within `[0,q)`, unshifting then shifting recovers the unsigned value. -/
theorem targetValue_unshiftValue_of_lt
    (value : Nat) (hValue : value < Goldilocks.q) :
    (unshiftValue value + shift) % Goldilocks.q = value := by
  unfold unshiftValue
  have hShift : shift < Goldilocks.q := shift_lt_goldilocks
  by_cases hBelow : value < shift
  · have hReduced : value + Goldilocks.q - shift < Goldilocks.q := by
      omega
    rw [Nat.mod_eq_of_lt hReduced]
    have hSum : value + Goldilocks.q - shift + shift =
        value + Goldilocks.q := by
      omega
    rw [hSum]
    simp [Nat.mod_eq_of_lt hValue]
  · have hShiftLe : shift ≤ value := Nat.le_of_not_gt hBelow
    have hModulusLe :
        Goldilocks.q ≤ value + Goldilocks.q - shift := by
      omega
    rw [Nat.mod_eq_sub_mod hModulusLe]
    have hReduced :
        value + Goldilocks.q - shift - Goldilocks.q < Goldilocks.q := by
      omega
    rw [Nat.mod_eq_of_lt hReduced]
    have hDifference :
        value + Goldilocks.q - shift - Goldilocks.q = value - shift := by
      omega
    rw [hDifference, Nat.sub_add_cancel hShiftLe,
      Nat.mod_eq_of_lt hValue]

abbrev DigitWord := List CenteredUnit

/-- Canonical low-norm word for one source field. -/
def canonicalWord (source : F) : DigitWord :=
  encodeNat digitCount (targetValue source)

/-- Decoder used after the full-field source slot has been removed. -/
def decodeWord (digits : DigitWord) : F :=
  F.ofNat (unshiftValue (decodeNat digits))

@[simp] theorem canonicalWord_length (source : F) :
    (canonicalWord source).length = digitCount := by
  simp [canonicalWord]

theorem decodeNat_canonicalWord (source : F) :
    decodeNat (canonicalWord source) = targetValue source := by
  apply decodeNat_encodeNat_of_lt
  exact Nat.lt_trans (targetValue_lt_goldilocks source)
    goldilocks_lt_ternaryRadix

/-- The model-level low-norm decoder uniquely recovers the source field. -/
@[simp] theorem decodeWord_canonicalWord (source : F) :
    decodeWord (canonicalWord source) = source := by
  unfold decodeWord
  rw [decodeNat_canonicalWord, unshiftValue_targetValue]
  exact F.ofNat_val source

/-- Semantic content of one source field's alphabet, reconstruction, and borrow rows. -/
def SourceFieldRows (source : F) (digits : DigitWord) : Prop :=
  digits.length = digitCount ∧
    decodeNat digits < Goldilocks.q ∧
    source.val = unshiftValue (decodeNat digits)

theorem sourceFieldRows_canonicalWord (source : F) :
    SourceFieldRows source (canonicalWord source) := by
  refine ⟨canonicalWord_length source, ?_, ?_⟩
  · rw [decodeNat_canonicalWord]
    exact targetValue_lt_goldilocks source
  · rw [decodeNat_canonicalWord]
    exact (unshiftValue_targetValue source).symm

/-- The source-row semantics select exactly the canonical low-norm word. -/
theorem sourceFieldRows_iff_canonicalWord
    (source : F) (digits : DigitWord) :
    SourceFieldRows source digits ↔ digits = canonicalWord source := by
  constructor
  · intro hRows
    have hTarget : targetValue source = decodeNat digits := by
      unfold targetValue
      rw [hRows.2.2]
      exact targetValue_unshiftValue_of_lt (decodeNat digits) hRows.2.1
    calc
      digits = encodeNat digitCount (decodeNat digits) :=
        (encodeNat_decodeNat_of_length digitCount digits hRows.1).symm
      _ = canonicalWord source := by simp [canonicalWord, hTarget]
  · rintro rfl
    exact sourceFieldRows_canonicalWord source

/-- Source-row semantics for an ordered sequence of fields. -/
def SourceRows (sources : List F) (words : List DigitWord) : Prop :=
  List.Forall₂ SourceFieldRows sources words

/-- Ordered canonical low-norm words for a source sequence. -/
def canonicalWords (sources : List F) : List DigitWord :=
  sources.map canonicalWord

theorem sourceRows_iff_canonicalWords
    (sources : List F) (words : List DigitWord) :
    SourceRows sources words ↔ words = canonicalWords sources := by
  constructor
  · intro hRows
    induction hRows with
    | nil => rfl
    | cons hHead _ inductionHypothesis =>
        rw [(sourceFieldRows_iff_canonicalWord _ _).mp hHead,
          inductionHypothesis]
        rfl
  · intro hWords
    subst words
    induction sources with
    | nil => exact .nil
    | cons source tail inductionHypothesis =>
        change List.Forall₂ SourceFieldRows (source :: tail)
          (canonicalWord source :: List.map canonicalWord tail)
        change List.Forall₂ SourceFieldRows tail
          (List.map canonicalWord tail) at inductionHypothesis
        exact .cons (sourceFieldRows_canonicalWord source)
          inductionHypothesis

/-- Decode each low-norm word independently. -/
def decodeWords (words : List DigitWord) : List F :=
  words.map decodeWord

@[simp] theorem decodeWords_canonicalWords (sources : List F) :
    decodeWords (canonicalWords sources) = sources := by
  induction sources with
  | nil => rfl
  | cons source tail inductionHypothesis =>
      change decodeWord (canonicalWord source) ::
          decodeWords (canonicalWords tail) = source :: tail
      rw [decodeWord_canonicalWord, inductionHypothesis]

theorem sourceRows_unique_decoding
    {sources : List F} {words : List DigitWord}
    (hRows : SourceRows sources words) :
    words = canonicalWords sources ∧ decodeWords words = sources := by
  have hCanonical := (sourceRows_iff_canonicalWords sources words).mp hRows
  exact ⟨hCanonical, by simp [hCanonical]⟩

universe uPrimary uShort uDigest

/-- Deterministic functions surrounding the low-norm message.

The names describe the production roles, but the structure assumes no
linearity, seed correctness, or collision resistance. -/
structure Pipeline
    (Primary : Type uPrimary) (Short : Type uShort) (Digest : Type uDigest) where
  primaryMap : List CenteredUnit → Primary
  shortMap : Primary → Short
  poseidon2Envelope : Nat → Nat → Nat → Short → Digest

structure BindingOutput
    (Primary : Type uPrimary) (Short : Type uShort) (Digest : Type uDigest) where
  primary : Primary
  short : Short
  digest : Digest
deriving DecidableEq

def Pipeline.evaluate
    {Primary : Type uPrimary} {Short : Type uShort} {Digest : Type uDigest}
    (pipeline : Pipeline Primary Short Digest)
    (role fieldCount primaryRank : Nat)
    (message : List CenteredUnit) : BindingOutput Primary Short Digest :=
  let primary := pipeline.primaryMap message
  let short := pipeline.shortMap primary
  { primary := primary
    short := short
    digest := pipeline.poseidon2Envelope role fieldCount primaryRank short }

/-- Source-row formulation before selective low-norm lowering. -/
def SourceSisRows
    {Primary : Type uPrimary} {Short : Type uShort} {Digest : Type uDigest}
    (pipeline : Pipeline Primary Short Digest)
    (role primaryRank : Nat)
    (sources : List F) (words : List DigitWord)
    (output : BindingOutput Primary Short Digest) : Prop :=
  SourceRows sources words ∧
    output = pipeline.evaluate role sources.length primaryRank words.flatten

/-- Canonical low-norm formulation after the source field slots are decoded away. -/
def LowNormEncoding
    {Primary : Type uPrimary} {Short : Type uShort} {Digest : Type uDigest}
    (pipeline : Pipeline Primary Short Digest)
    (role primaryRank : Nat)
    (sources : List F) (words : List DigitWord)
    (output : BindingOutput Primary Short Digest) : Prop :=
  words = canonicalWords sources ∧
    output = pipeline.evaluate role sources.length primaryRank words.flatten

/-- Model-level lowering theorem: source SIS rows and canonical low-norm encoding are exact. -/
theorem sourceSisRows_iff_lowNormEncoding
    {Primary : Type uPrimary} {Short : Type uShort} {Digest : Type uDigest}
    (pipeline : Pipeline Primary Short Digest)
    (role primaryRank : Nat)
    (sources : List F) (words : List DigitWord)
    (output : BindingOutput Primary Short Digest) :
    SourceSisRows pipeline role primaryRank sources words output ↔
      LowNormEncoding pipeline role primaryRank sources words output := by
  simp only [SourceSisRows, LowNormEncoding,
    sourceRows_iff_canonicalWords]

/-- For fixed authoritative fields, two accepted source witnesses have the same
low-norm message and therefore the same recomputed two-map/envelope output. -/
theorem sourceSisRows_unique
    {Primary : Type uPrimary} {Short : Type uShort} {Digest : Type uDigest}
    (pipeline : Pipeline Primary Short Digest)
    (role primaryRank : Nat) (sources : List F)
    {leftWords rightWords : List DigitWord}
    {leftOutput rightOutput : BindingOutput Primary Short Digest}
    (hLeft : SourceSisRows pipeline role primaryRank sources leftWords leftOutput)
    (hRight : SourceSisRows pipeline role primaryRank sources rightWords rightOutput) :
    leftWords = rightWords ∧ leftOutput = rightOutput := by
  have hWords : leftWords = rightWords :=
    ((sourceRows_iff_canonicalWords sources leftWords).mp hLeft.1).trans
      ((sourceRows_iff_canonicalWords sources rightWords).mp hRight.1).symm
  refine ⟨hWords, ?_⟩
  rw [hLeft.2, hRight.2, hWords]

end SuperNeo.FPrimeRecursiveVerifier.SisLowering
