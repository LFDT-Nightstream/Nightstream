import Mathlib.Data.List.Sort
import Nightstream.HyperNova.NIVCCompatibility
import Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentity
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteAlgebra

/-!
Contract: `nightstream-sparse-structure-v1` for the selected rectangular
thirteen-matrix relation.

Owns: the exact ten-field profile header; thirteen indexed, row-major sparse
matrix sections; the fixed 74-term polynomial section in strict exponent-
tuple order; the outer stream-field count used by the verifier-key preimage;
deterministic expansion to the exact dense matrices used by
`PaddedRowIdentity`; and the proof that an equal canonical stream determines
equal dense matrices.

Does not own: Rust provenance, a selected application compiler, a generated
matrix payload, or a cryptographic collision assumption.

Emits constraints: no.

Assurance tier: model-level compiler-description boundary. The raw structure
stream has `1073 + 3N` fields for `N` nonzero matrix entries. Its counted
verifier-key form has `1074 + 3N` fields. It never serializes the dense
`13 * logicalRows * assignmentColumns` table.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompilerDescription

open Nightstream.HyperNova.NIVCCompatibility
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable

abbrev PolynomialMonomial := Monomial F applicationMatrixCount

/-! ## Canonical sparse matrix sections -/

/-- One nonzero coefficient in a matrix-local row-major stream. -/
structure MatrixEntry where
  row : Fin logicalRows
  column : Fin assignmentColumns
  value : F
deriving DecidableEq, Repr

namespace MatrixEntry

/-- Overflow-free row-major coordinate key. -/
def position (entry : MatrixEntry) : Nat :=
  entry.row.val * assignmentColumns + entry.column.val

/-- The three exact field coordinates used by one sparse triple. -/
def data (entry : MatrixEntry) : Nat × (Nat × F) :=
  (entry.row.val, (entry.column.val, entry.value))

theorem data_injective : Function.Injective data := by
  intro left right equal
  cases left
  cases right
  simp only [data, Prod.mk.injEq] at equal
  rcases equal with ⟨row, column, value⟩
  congr
  · exact Fin.ext row
  · exact Fin.ext column

end MatrixEntry

/-- Strict order excludes duplicate coordinates. Zero coefficients are not
stored. The entry count must have one canonical base-field encoding. -/
def CanonicalMatrixEntries (entries : List MatrixEntry) : Prop :=
  entries.length < goldilocksModulus /\
  entries.Pairwise (fun left right => left.position < right.position) /\
  forall entry, entry ∈ entries -> entry.value ≠ 0

/-- One proof-carrying matrix-local sparse section. -/
structure MatrixDescription where
  entries : List MatrixEntry
  canonical : CanonicalMatrixEntries entries

namespace MatrixDescription

@[ext] theorem ext
    {left right : MatrixDescription}
    (entries : left.entries = right.entries) : left = right := by
  cases left
  cases right
  cases entries
  rfl

end MatrixDescription

/-- The total number of nonzero coefficients in all thirteen sections. -/
def totalEntries (sections : Fin applicationMatrixCount -> MatrixDescription) : Nat :=
  ((List.ofFn sections).map fun matrixDescription =>
    matrixDescription.entries.length).sum

/-- One selected sparse structure. `streamFits` makes the outer
`structure_stream_field_count` fail closed in the base field. -/
structure Description where
  sections : Fin applicationMatrixCount -> MatrixDescription
  streamFits : 1073 + 3 * totalEntries sections < goldilocksModulus

namespace Description

@[ext] theorem ext
    {left right : Description}
    (sections : left.sections = right.sections) : left = right := by
  cases left
  cases right
  cases sections
  rfl

/-- Total stored nonzero count for the exact structure. -/
def entryCount (description : Description) : Nat :=
  totalEntries description.sections

end Description

noncomputable def matrixEntryGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec MatrixEntry :=
  Codec.pullback
    (Codec.product boundedNatCodec
      (Codec.product boundedNatCodec fieldCodec))
    MatrixEntry.data MatrixEntry.data_injective

theorem matrixEntryGoldCodec_admissible (entry : MatrixEntry) :
    matrixEntryGoldCodec.Admissible entry := by
  exact ⟨Nat.lt_trans entry.row.isLt (by decide),
    Nat.lt_trans entry.column.isLt (by decide), True.intro⟩

@[simp] theorem matrixEntryGoldCodec_width :
    matrixEntryGoldCodec.width = 3 := by
  rfl

@[simp] theorem boundedNatCodec_width_exact : boundedNatCodec.width = 1 := by
  rfl

/-- Concatenate fixed-width row-column-value triples. -/
noncomputable def encodeMatrixEntries : List MatrixEntry -> List F
  | [] => []
  | entry :: rest =>
      matrixEntryGoldCodec.encode entry ++ encodeMatrixEntries rest

@[simp] theorem encodeMatrixEntries_length (entries : List MatrixEntry) :
    (encodeMatrixEntries entries).length = 3 * entries.length := by
  induction entries with
  | nil => rfl
  | cons entry rest inductionHypothesis =>
      simp [encodeMatrixEntries, matrixEntryGoldCodec.encode_length,
        inductionHypothesis, Nat.mul_add]
      omega

theorem encodeMatrixEntries_injective :
    Function.Injective encodeMatrixEntries := by
  intro left
  induction left with
  | nil =>
      intro right equal
      have lengths := congrArg List.length equal
      simp only [encodeMatrixEntries_length, List.length_nil,
        Nat.mul_zero] at lengths
      exact (List.length_eq_zero_iff.mp (by omega : right.length = 0)).symm
  | cons head tail inductionHypothesis =>
      intro right equal
      cases right with
      | nil =>
          have lengths := congrArg List.length equal
          simp only [encodeMatrixEntries_length, List.length_cons,
            List.length_nil, Nat.mul_zero] at lengths
          omega
      | cons other rest =>
          have headEncoded := congrArg (List.take 3) equal
          have headEqual : head = other := by
            apply matrixEntryGoldCodec.encode_injective_of_admissible
              (matrixEntryGoldCodec_admissible head)
              (matrixEntryGoldCodec_admissible other)
            simpa [encodeMatrixEntries,
              matrixEntryGoldCodec.encode_length] using headEncoded
          subst other
          have tailEncoded := congrArg (List.drop 3) equal
          have tailEqual : tail = rest :=
            inductionHypothesis (by
              simpa [encodeMatrixEntries,
                matrixEntryGoldCodec.encode_length] using tailEncoded)
          subst rest
          rfl

def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

/-- One exact matrix section:
`one-based-index || nonzero-count || row-column-value-triples`. -/
noncomputable def matrixFields
    (zeroBasedIndex : Nat)
    (matrixDescription : MatrixDescription) : List F :=
  [fieldOfNat (zeroBasedIndex + 1)] ++
    boundedNatCodec.encode matrixDescription.entries.length ++
    encodeMatrixEntries matrixDescription.entries

@[simp] theorem matrixFields_length
    (zeroBasedIndex : Nat)
    (matrixDescription : MatrixDescription) :
    (matrixFields zeroBasedIndex matrixDescription).length =
      2 + 3 * matrixDescription.entries.length := by
  simp [matrixFields, boundedNatCodec.encode_length]
  omega

private theorem matrixFields_count_eq
    (zeroBasedIndex : Nat)
    {left right : MatrixDescription}
    (equal : matrixFields zeroBasedIndex left =
      matrixFields zeroBasedIndex right) :
    left.entries.length = right.entries.length := by
  apply boundedNatCodec.encode_injective_of_admissible
    left.canonical.1 right.canonical.1
  have firstTwo := congrArg (List.take 2) equal
  simpa [matrixFields, boundedNatCodec.encode_length] using
    congrArg (List.drop 1) firstTwo

theorem matrixFields_injective (zeroBasedIndex : Nat) :
    Function.Injective (matrixFields zeroBasedIndex) := by
  intro left right equal
  have entriesEncoded := congrArg (List.drop 2) equal
  apply MatrixDescription.ext
  apply encodeMatrixEntries_injective
  simpa [matrixFields, boundedNatCodec.encode_length] using entriesEncoded

theorem matrixFields_prefixFree (zeroBasedIndex : Nat) :
    PrefixFree (matrixFields zeroBasedIndex) := by
  intro left right suffix prefixed
  have firstTwo := congrArg (List.take 2) prefixed
  have countEncoded :
      boundedNatCodec.encode right.entries.length =
        boundedNatCodec.encode left.entries.length := by
    simpa [matrixFields, boundedNatCodec.encode_length] using
      congrArg (List.drop 1) firstTwo
  have countEqual : right.entries.length = left.entries.length :=
    boundedNatCodec.encode_injective_of_admissible
      right.canonical.1 left.canonical.1 countEncoded
  have lengths := congrArg List.length prefixed
  simp only [matrixFields_length, List.length_append] at lengths
  rw [countEqual] at lengths
  have suffixEmpty : suffix = [] :=
    List.length_eq_zero_iff.mp (by omega)
  subst suffix
  simp only [List.append_nil] at prefixed
  exact (matrixFields_injective zeroBasedIndex prefixed).symm

/-- Encode matrix sections in the supplied order. The index starts at zero,
while each encoded matrix identifier is one based. -/
noncomputable def encodeMatrixSectionsFrom :
    Nat -> List MatrixDescription -> List F
  | _, [] => []
  | index, matrixDescription :: rest =>
      matrixFields index matrixDescription ++
        encodeMatrixSectionsFrom (index + 1) rest

@[simp] theorem encodeMatrixSectionsFrom_length
    (start : Nat)
    (sections : List MatrixDescription) :
    (encodeMatrixSectionsFrom start sections).length =
      2 * sections.length +
        3 * ((sections.map fun matrixDescription =>
          matrixDescription.entries.length).sum) := by
  induction sections generalizing start with
  | nil => rfl
  | cons matrixDescription rest inductionHypothesis =>
      simp [encodeMatrixSectionsFrom, matrixFields_length,
        inductionHypothesis, Nat.mul_add]
      omega

theorem encodeMatrixSectionsFrom_injective_of_length_eq
    (start : Nat)
    {left right : List MatrixDescription}
    (sameLength : left.length = right.length)
    (equal : encodeMatrixSectionsFrom start left =
      encodeMatrixSectionsFrom start right) :
    left = right := by
  induction left generalizing start right with
  | nil =>
      have rightEmpty : right = [] :=
        List.length_eq_zero_iff.mp (by simpa using sameLength.symm)
      subst right
      rfl
  | cons head tail inductionHypothesis =>
      cases right with
      | nil => simp at sameLength
      | cons other rest =>
          have firstTwo := congrArg (List.take 2) equal
          have countEncoded :
              boundedNatCodec.encode head.entries.length =
                boundedNatCodec.encode other.entries.length := by
            simpa [encodeMatrixSectionsFrom, matrixFields,
              boundedNatCodec.encode_length] using
                congrArg (List.drop 1) firstTwo
          have countEqual : head.entries.length = other.entries.length :=
            boundedNatCodec.encode_injective_of_admissible
              head.canonical.1 other.canonical.1 countEncoded
          have headEncoded :=
            congrArg (List.take (matrixFields start head).length) equal
          have headEqual : head = other := by
            apply matrixFields_injective start
            simpa [encodeMatrixSectionsFrom, matrixFields_length,
              countEqual] using headEncoded
          subst other
          have tailEncoded :=
            congrArg (List.drop (matrixFields start head).length) equal
          have tailEqual : tail = rest :=
            inductionHypothesis (start + 1)
              (by simpa using Nat.succ.inj sameLength)
              (by
                simpa [encodeMatrixSectionsFrom, matrixFields_length] using
                  tailEncoded)
          subst rest
          rfl

def sectionList (description : Description) : List MatrixDescription :=
  List.ofFn description.sections

@[simp] theorem sectionList_length (description : Description) :
    (sectionList description).length = applicationMatrixCount := by
  simp [sectionList]

@[simp] theorem sectionList_entry_sum (description : Description) :
    ((sectionList description).map fun matrixDescription =>
      matrixDescription.entries.length).sum =
      description.entryCount := by
  rfl

theorem sectionList_injective : Function.Injective sectionList := by
  intro left right equal
  apply Description.ext
  exact List.ofFn_injective equal

/-! ## Exact fixed polynomial section -/

def exponentTuple (term : PolynomialMonomial) : List Nat :=
  (List.finRange applicationMatrixCount).map term.exponents

/-- Executable strict lexicographic order on natural-number lists. -/
def exponentTupleLexLess : List Nat -> List Nat -> Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | left :: leftTail, right :: rightTail =>
      if left < right then true
      else if right < left then false
      else exponentTupleLexLess leftTail rightTail

def exponentTupleLess (left right : PolynomialMonomial) : Prop :=
  exponentTupleLexLess (exponentTuple left) (exponentTuple right) = true

instance exponentTupleLessDecidable : DecidableRel exponentTupleLess :=
  by
    intro left right
    unfold exponentTupleLess
    infer_instance

/-- The exact production terms, reordered only for the contract's canonical
strict lexicographic exponent-tuple stream. -/
def polynomialTerms : List PolynomialMonomial :=
  Polynomial.Semantics.terms.insertionSort exponentTupleLess

theorem polynomialTerms_perm :
    polynomialTerms.Perm Polynomial.Semantics.terms := by
  exact List.perm_insertionSort exponentTupleLess _

theorem polynomialTerms_count_exact : polynomialTerms.length = 74 := by
  simpa [polynomialTerms, List.length_insertionSort] using
    Polynomial.Semantics.term_count_exact

theorem polynomialTerms_strictLex :
    polynomialTerms.Pairwise exponentTupleLess := by
  native_decide

theorem polynomialTerms_constraints_exact :
    (forall term, term ∈ polynomialTerms ->
      term.coefficient ≠ 0 /\
      (forall index, term.exponents index <= 8) /\
      1 <= term.totalDegree /\ term.totalDegree <= 8) /\
    (exists term, term ∈ polynomialTerms /\ term.totalDegree = 8) := by
  native_decide

/-- One coefficient followed by all thirteen exponents. -/
def polynomialTermFields (term : PolynomialMonomial) : List F :=
  term.coefficient :: (exponentTuple term).map fieldOfNat

@[simp] theorem exponentTuple_length (term : PolynomialMonomial) :
    (exponentTuple term).length = applicationMatrixCount := by
  simp [exponentTuple]

@[simp] theorem polynomialTermFields_length (term : PolynomialMonomial) :
    (polynomialTermFields term).length = 14 := by
  simp [polynomialTermFields, exponentTuple_length, applicationMatrixCount]

/-- Exact fixed polynomial suffix:
`term-count || coefficient-and-13-exponents-per-term`. -/
def polynomialFields : List F :=
  boundedNatCodec.encode polynomialTerms.length ++
    polynomialTerms.flatMap polynomialTermFields

@[simp] theorem polynomialFields_length : polynomialFields.length = 1037 := by
  simp [polynomialFields, boundedNatCodec.encode_length,
    polynomialTerms_count_exact, polynomialTermFields_length]

/-! ## `nightstream-sparse-structure-v1` and counted verifier-key stream -/

def encodingVersion : Nat := 1
def identityVariantCode : Nat := 1
def polynomialTotalDegree : Nat := 8

/-- Exact ten-field contract header. `M_0` and zero-row padding are implicit
in `identityVariantCode`. -/
def structureHeader : List F :=
  [fieldOfNat encodingVersion,
    fieldOfNat logicalRows,
    fieldOfNat (2 ^ rowVariables),
    fieldOfNat rowVariables,
    fieldOfNat assignmentColumns,
    fieldOfNat PaddedRowIdentityConcreteAlgebra.relationShape.publicWidth,
    fieldOfNat applicationMatrixCount,
    fieldOfNat jointMatrixCount,
    fieldOfNat polynomialTotalDegree,
    fieldOfNat identityVariantCode]

@[simp] theorem structureHeader_length : structureHeader.length = 10 := by
  rfl

/-- Exact raw `nightstream-sparse-structure-v1` payload. -/
noncomputable def structureFields (description : Description) : List F :=
  structureHeader ++
    encodeMatrixSectionsFrom 0 (sectionList description) ++
    polynomialFields

@[simp] theorem structureFields_length (description : Description) :
    (structureFields description).length =
      1073 + 3 * description.entryCount := by
  simp [structureFields, encodeMatrixSectionsFrom_length,
    sectionList_entry_sum, applicationMatrixCount]
  omega

theorem structureFields_injective : Function.Injective structureFields := by
  intro left right equal
  change
    structureHeader ++
        (encodeMatrixSectionsFrom 0 (sectionList left) ++ polynomialFields) =
      structureHeader ++
        (encodeMatrixSectionsFrom 0 (sectionList right) ++ polynomialFields)
    at equal
  have withoutHeader :
      encodeMatrixSectionsFrom 0 (sectionList left) ++ polynomialFields =
        encodeMatrixSectionsFrom 0 (sectionList right) ++ polynomialFields :=
    List.append_cancel_left equal
  have encodedEqual :
      encodeMatrixSectionsFrom 0 (sectionList left) =
        encodeMatrixSectionsFrom 0 (sectionList right) := by
    exact List.append_cancel_right withoutHeader
  apply sectionList_injective
  exact encodeMatrixSectionsFrom_injective_of_length_eq 0
    (by simp) encodedEqual

/-- The verifier-key preimage uses the raw stream field count immediately
before the raw structure stream. -/
noncomputable def fields (description : Description) : List F :=
  boundedNatCodec.encode (structureFields description).length ++
    structureFields description

@[simp] theorem fields_length (description : Description) :
    (fields description).length = 1074 + 3 * description.entryCount := by
  simp [fields, boundedNatCodec.encode_length, structureFields_length]
  omega

private theorem structureLength_admissible (description : Description) :
    (structureFields description).length < goldilocksModulus := by
  simpa [structureFields_length] using description.streamFits

theorem fields_injective : Function.Injective fields := by
  intro left right equal
  apply structureFields_injective
  have tails := congrArg (List.drop 1) equal
  simpa [fields, boundedNatCodec.encode_length] using tails

theorem fields_prefixFree : PrefixFree fields := by
  intro left right suffix prefixed
  have first := congrArg (List.take 1) prefixed
  have lengthEncoded :
      boundedNatCodec.encode (structureFields right).length =
        boundedNatCodec.encode (structureFields left).length := by
    simpa [fields, boundedNatCodec.encode_length] using first
  have rawLengthEqual :
      (structureFields right).length = (structureFields left).length :=
    boundedNatCodec.encode_injective_of_admissible
      (structureLength_admissible right)
      (structureLength_admissible left) lengthEncoded
  have lengths := congrArg List.length prefixed
  simp only [fields, List.length_append, boundedNatCodec.encode_length] at lengths
  rw [rawLengthEqual] at lengths
  have suffixEmpty : suffix = [] :=
    List.length_eq_zero_iff.mp (by omega)
  subst suffix
  simp only [List.append_nil] at prefixed
  exact (fields_injective prefixed).symm

noncomputable def codec :
    Nightstream.HyperNova.NIVCCompatibility.Codec Description F :=
  Codec.withClassicalDecoder fields

theorem codec_canonical : codec.Canonical :=
  Codec.injectivePrefixFree_canonical fields fields_injective fields_prefixFree

/-! ## Deterministic expansion to the paper relation -/

def MatrixEntry.matches
    (entry : MatrixEntry)
    (row : Fin logicalRows)
    (column : Fin assignmentColumns) : Bool :=
  decide (entry.row = row /\ entry.column = column)

/-- Look up one canonical nonzero. Absent coordinates are exactly zero. -/
def coefficientAt
    (description : Description)
    (matrix : Fin applicationMatrixCount)
    (row : Fin logicalRows)
    (column : Fin assignmentColumns) : F :=
  match (description.sections matrix).entries.find? fun entry =>
      entry.matches row column with
  | some entry => entry.value
  | none => 0

/-- The exact thirteen dense matrices used by the Lean relation, derived only
from the canonical sparse description. -/
def matrices (description : Description) : ApplicationMatrices where
  matrices := fun role row column =>
    coefficientAt description role.index row column

@[simp] theorem matrices_role
    (description : Description)
    (role : Role)
    (row : Fin logicalRows)
    (column : Fin assignmentColumns) :
    (matrices description).matrices role row column =
      coefficientAt description role.index row column :=
  rfl

/-- Equal exact raw structure streams determine the same thirteen dense
matrices used by the relation. -/
theorem matrices_eq_of_structureFields_eq
    {left right : Description}
    (equal : structureFields left = structureFields right) :
    matrices left = matrices right := by
  exact congrArg matrices (structureFields_injective equal)

/-- Equal counted verifier-key streams determine the same thirteen dense
matrices used by the relation. -/
theorem matrices_eq_of_fields_eq
    {left right : Description}
    (equal : fields left = fields right) :
    matrices left = matrices right := by
  exact congrArg matrices (fields_injective equal)

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompilerDescription
