import Nightstream.Implementation.Nebula.FPrime.State.SeedSchedule
import Nightstream.Implementation.Nebula.Commitment.Lanes.ShiftedTernaryEncodingBridge
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction

/-!
Contract: exact R1CS relation for one seeded Ajtai stage of a Nebula V2
compact commitment token.

Assurance tier: implementation-to-protocol bridge.

Owns canonical shifted-ternary rows for every input field, the exact
ring-column-major `Packing.indexEquiv` order, verifier-key-derived Phi81
linear coefficients, one output equation per field coordinate, and a
row-derived equality to an independently executable seeded stage.

Does not own composition of the primary and short stages, Poseidon2 lane
chains, prechallenge-root knowledge, transcript derivation, Rust refinement,
Module-SIS hardness, or absolute generated columns.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.CompactAjtaiStageRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.Nebula.ShiftedTernaryEncodingBridge
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

/-- Canonical Goldilocks residue of the only signed digits accepted by the
V2 shifted-ternary encoder. Values outside `{-1,0,1}` fail closed to zero;
the source-row theorem proves that production inputs never take that branch. -/
def unitDigitResidue : Int → Nat
  | -1 => modulus - 1
  | 0 => 0
  | 1 => 1
  | _ => 0

/-- Exact bridge from the protocol's signed integer coefficient to its R1CS
field representative. -/
theorem unitDigitResidue_signedDigit
    (value : CanonicalGoldilocks)
    (index : Fin ShiftedTernary41V1.digitCount) :
    unitDigitResidue (signedDigit value index) =
      fieldDigit (tritAt value index) := by
  have bound := tritAt_lt_three value index
  have alternatives :
      tritAt value index = 0 ∨ tritAt value index = 1 ∨
        tritAt value index = 2 := by
    omega
  rcases alternatives with equal | equal | equal <;>
    simp [signedDigit, equal, unitDigitResidue, fieldDigit]

abbrev Entry (messageColumns : Nat) :=
  Fin messageColumns × Fin CompactCommit.ringDegree

def entries (messageColumns : Nat) : List (Entry messageColumns) :=
  (List.ofFn fun column : Fin messageColumns =>
    List.ofFn fun coefficient : Fin CompactCommit.ringDegree =>
      (column, coefficient)).flatten

private theorem flatten_ofFn_length
    {alpha : Type} {count width : Nat} (blocks : Fin count → List alpha)
    (each : ∀ index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten]
  have constant : ∀ value ∈ (List.ofFn blocks).map List.length,
      value = width := by
    intro value member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
    exact each index
  rw [List.sum_eq_card_nsmul _ width constant]
  simp

theorem entries_length (messageColumns : Nat) :
    (entries messageColumns).length =
      messageColumns * CompactCommit.ringDegree := by
  exact flatten_ofFn_length _ (fun _ => by simp)

def outputPair {verifierRows : Nat}
    (output : Fin (verifierRows * CompactCommit.ringDegree)) :
    Fin verifierRows × Fin CompactCommit.ringDegree :=
  (finProdFinEquiv (m := verifierRows)
    (n := CompactCommit.ringDegree)).symm output

/-- One exact base-field coefficient in the dense R1CS linear equation. It
is derived from the verifier-key ring element and executable Phi81
multiplication by a coefficient basis vector. -/
def coefficient
    {verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (output : Fin (verifierRows * CompactCommit.ringDegree))
    (entry : Entry messageColumns) : Nat :=
  (Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.rightCoefficient
    (setup.verifierKey (outputPair output).1 entry.1)
    (outputPair output).2 entry.2).val

/-- The row coefficient is not a caller-selected matrix entry. It is exactly
the coefficient of the seeded verifier-key ring element multiplied by the
corresponding Phi81 basis monomial. -/
theorem coefficient_eq_seeded_phi81_basis_action
    {verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (output : Fin (verifierRows * CompactCommit.ringDegree))
    (entry : Entry messageColumns) :
    coefficient setup output entry =
      (Nightstream.SuperNeo.Concrete.ringFMul
        (setup.verifierKey (outputPair output).1 entry.1)
        (Nightstream.SuperNeo.Concrete.ringFMonomial entry.2.val 1)
        (outputPair output).2).val := by
  rfl

/-- Independent executable value of one stage output. The input is the
protocol `RingMessage`, not an R1CS assignment or a claimed output. -/
def semanticValue
    {verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (message : RingMessage messageColumns)
    (output : Fin (verifierRows * CompactCommit.ringDegree)) : Nat :=
  ((entries messageColumns).foldl (fun accumulated entry =>
    accumulated + coefficient setup output entry *
      unitDigitResidue (message entry.1 entry.2)) 0) % goldilocksP

theorem semanticValue_lt
    {verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (message : RingMessage messageColumns)
    (output : Fin (verifierRows * CompactCommit.ringDegree)) :
    semanticValue setup message output < modulus := by
  exact Nat.mod_lt _ (by norm_num [modulus, goldilocksP])

def semanticOutput
    {verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (message : RingMessage messageColumns) :
    FieldVector (verifierRows * CompactCommit.ringDegree) :=
  fun output => ⟨semanticValue setup message output,
    semanticValue_lt setup message output⟩

/-- Relative column placement for one stage. The 122-column auxiliary window
for each shifted-ternary word starts at `digitStart field`; its first 41
columns are the authority-bearing centered digits. -/
structure Layout (fieldCount verifierRows : Nat) where
  fieldColumn : Fin fieldCount → Nat
  digitStart : Fin fieldCount → Nat
  outputColumn : Fin (verifierRows * CompactCommit.ringDegree) → Nat

def sourceIndex
    {fieldCount messageColumns : Nat}
    (packing : Packing fieldCount messageColumns)
    (entry : Entry messageColumns) :
    Fin fieldCount × Fin ShiftedTernary41V1.digitCount :=
  packing.indexEquiv entry

def sourceDigitColumn
    {fieldCount verifierRows messageColumns : Nat}
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows)
    (entry : Entry messageColumns) : Nat :=
  layout.digitStart (sourceIndex packing entry).1 +
    (sourceIndex packing entry).2.val

def encodingBlockRows
    {fieldCount verifierRows : Nat}
    (layout : Layout fieldCount verifierRows)
    (field : Fin fieldCount) : List Row :=
  canonicalRows.map (Relabel.row
    (OwnerCertificate.shiftedTernaryColumnMap
      (layout.fieldColumn field) (layout.digitStart field)))

def encodingRows
    {fieldCount verifierRows : Nat}
    (layout : Layout fieldCount verifierRows) : List Row :=
  (List.ofFn fun field : Fin fieldCount =>
    encodingBlockRows layout field).flatten

def linearTerms
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows)
    (output : Fin (verifierRows * CompactCommit.ringDegree)) :
    List (Nat × Nat) :=
  (entries messageColumns).map fun entry =>
    (sourceDigitColumn packing layout entry,
      coefficient setup output entry)

def linearDefinition
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows)
    (output : Fin (verifierRows * CompactCommit.ringDegree)) : Definition where
  output := layout.outputColumn output
  rhs := .linear (linearTerms setup packing layout output)

def linearDefinitions
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows) : List Definition :=
  List.ofFn fun output : Fin (verifierRows * CompactCommit.ringDegree) =>
    linearDefinition setup packing layout output

def linearRows
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows) : List Row :=
  (linearDefinitions setup packing layout).map Definition.row

def rows
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows) : List Row :=
  encodingRows layout ++ linearRows setup packing layout

theorem encodingBlockRows_length
    {fieldCount verifierRows : Nat}
    (layout : Layout fieldCount verifierRows) (field : Fin fieldCount) :
    (encodingBlockRows layout field).length = 124 := by
  simp [encodingBlockRows]
  decide

theorem encodingRows_length
    {fieldCount verifierRows : Nat}
    (layout : Layout fieldCount verifierRows) :
    (encodingRows layout).length = fieldCount * 124 := by
  exact flatten_ofFn_length _ (encodingBlockRows_length layout)

theorem linearRows_length
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows) :
    (linearRows setup packing layout).length =
      verifierRows * CompactCommit.ringDegree := by
  simp [linearRows, linearDefinitions]

theorem rows_length
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows) :
    (rows setup packing layout).length =
      fieldCount * 124 + verifierRows * CompactCommit.ringDegree := by
  simp [rows, encodingRows_length, linearRows_length]

def FieldsPlaced
    {fieldCount verifierRows : Nat}
    (layout : Layout fieldCount verifierRows)
    (assignment : Nat → Nat) (fields : FieldVector fieldCount) : Prop :=
  ∀ field, assignment (layout.fieldColumn field) = (fields field).val

private theorem encoding_holds_of_rows
    {fieldCount verifierRows messageColumns : Nat}
    {setup : SeededAjtai.Setup verifierRows messageColumns}
    {packing : Packing fieldCount messageColumns}
    {layout : Layout fieldCount verifierRows}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows setup packing layout) assignment) :
    Satisfies (encodingRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem linear_holds_of_rows
    {fieldCount verifierRows messageColumns : Nat}
    {setup : SeededAjtai.Setup verifierRows messageColumns}
    {packing : Packing fieldCount messageColumns}
    {layout : Layout fieldCount verifierRows}
    {assignment : Nat → Nat}
    (holds : Satisfies (rows setup packing layout) assignment) :
    Satisfies (linearRows setup packing layout) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

private theorem encodingBlock_holds
    {fieldCount verifierRows : Nat}
    {layout : Layout fieldCount verifierRows}
    {assignment : Nat → Nat}
    (holds : Satisfies (encodingRows layout) assignment)
    (field : Fin fieldCount) :
    Satisfies (encodingBlockRows layout field) assignment := by
  exact (satisfies_flatten_iff _ _).mp holds _
    (List.mem_ofFn.mpr ⟨field, rfl⟩)

def localAssignment
    {fieldCount verifierRows : Nat}
    (layout : Layout fieldCount verifierRows)
    (assignment : Nat → Nat) (field : Fin fieldCount) : Nat → Nat :=
  ShiftedTernaryCanonicalWord.localAssignment assignment
    (layout.fieldColumn field) (layout.digitStart field)

theorem opening_of_rows
    {fieldCount verifierRows messageColumns : Nat}
    {setup : SeededAjtai.Setup verifierRows messageColumns}
    {packing : Packing fieldCount messageColumns}
    {layout : Layout fieldCount verifierRows}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows setup packing layout) assignment)
    (field : Fin fieldCount) :
    CanonicalOpening (localAssignment layout assignment field) := by
  apply canonicalOpening_of_canonicalRows
    Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
    (Relabel.canonical canonical)
  · calc
      Relabel.assignment
          (OwnerCertificate.shiftedTernaryColumnMap
            (layout.fieldColumn field) (layout.digitStart field))
          assignment 0 = assignment 0 :=
        ShiftedTernaryCanonicalWord.localAssignment_zero assignment
          (layout.fieldColumn field) (layout.digitStart field)
      _ = 1 := one
  · exact (Relabel.satisfies_mapped_iff _ _ _).mp
      (encodingBlock_holds (encoding_holds_of_rows holds) field)

/-- All R1CS input digit columns equal the exact protocol packing. This
theorem rules out the transposed `messageRow * messageColumns + column`
schedule used by the older generic block. -/
theorem sourceDigit_exact
    {fieldCount verifierRows messageColumns : Nat}
    {setup : SeededAjtai.Setup verifierRows messageColumns}
    {packing : Packing fieldCount messageColumns}
    {layout : Layout fieldCount verifierRows}
    {assignment : Nat → Nat} {fields : FieldVector fieldCount}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FieldsPlaced layout assignment fields)
    (holds : Satisfies (rows setup packing layout) assignment)
    (entry : Entry messageColumns) :
    assignment (sourceDigitColumn packing layout entry) =
      unitDigitResidue
        (packFields packing fields entry.1 entry.2) := by
  let source := sourceIndex packing entry
  have protocolDigit := productionDigit_eq_protocolDigit
    (fields source.1) (placed source.1)
    (opening_of_rows canonical one holds source.1) source.2
  calc
    assignment (sourceDigitColumn packing layout entry) =
        assignment (layout.digitStart source.1 + source.2.val) := by rfl
    _ = fieldDigit (tritAt (fields source.1) source.2) := protocolDigit
    _ = unitDigitResidue (signedDigit (fields source.1) source.2) :=
      (unitDigitResidue_signedDigit (fields source.1) source.2).symm
    _ = unitDigitResidue
        (packFields packing fields entry.1 entry.2) := by rfl

private theorem foldl_terms_exact_for
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows)
    (assignment : Nat → Nat) (message : RingMessage messageColumns)
    (output : Fin (verifierRows * CompactCommit.ringDegree))
    (inputExact : ∀ entry,
      assignment (sourceDigitColumn packing layout entry) =
        unitDigitResidue (message entry.1 entry.2))
    (source : List (Entry messageColumns)) :
    ∀ initial,
      (source.map fun entry =>
        (sourceDigitColumn packing layout entry,
          coefficient setup output entry)).foldl
          (fun accumulated term =>
            accumulated + term.2 * assignment term.1) initial =
        source.foldl (fun accumulated entry =>
          accumulated + coefficient setup output entry *
            unitDigitResidue (message entry.1 entry.2)) initial := by
  intro initial
  induction source generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.foldl]
      rw [inputExact head]
      exact inductionHypothesis _

private theorem foldl_terms_exact
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows)
    (assignment : Nat → Nat) (message : RingMessage messageColumns)
    (output : Fin (verifierRows * CompactCommit.ringDegree))
    (inputExact : ∀ entry,
      assignment (sourceDigitColumn packing layout entry) =
        unitDigitResidue (message entry.1 entry.2)) :
    ∀ initial,
      ((entries messageColumns).map fun entry =>
        (sourceDigitColumn packing layout entry,
          coefficient setup output entry)).foldl
          (fun accumulated term =>
            accumulated + term.2 * assignment term.1) initial =
        (entries messageColumns).foldl (fun accumulated entry =>
          accumulated + coefficient setup output entry *
            unitDigitResidue (message entry.1 entry.2)) initial := by
  exact foldl_terms_exact_for setup packing layout assignment message output
    inputExact (entries messageColumns)

theorem linearValue_exact
    {fieldCount verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (packing : Packing fieldCount messageColumns)
    (layout : Layout fieldCount verifierRows)
    (assignment : Nat → Nat) (message : RingMessage messageColumns)
    (output : Fin (verifierRows * CompactCommit.ringDegree))
    (inputExact : ∀ entry,
      assignment (sourceDigitColumn packing layout entry) =
        unitDigitResidue (message entry.1 entry.2)) :
    lcEval assignment (linearTerms setup packing layout output) =
      semanticValue setup message output := by
  unfold linearTerms semanticValue lcEval
  exact congrArg (fun value => value % goldilocksP)
    (foldl_terms_exact setup packing layout assignment message output
      inputExact 0)

/-- Main stage soundness theorem. Its premises are exact field placement and
the stage rows. It does not assume an output, compact token, or commitment
equality. -/
theorem output_exact
    {fieldCount verifierRows messageColumns : Nat}
    {setup : SeededAjtai.Setup verifierRows messageColumns}
    {packing : Packing fieldCount messageColumns}
    {layout : Layout fieldCount verifierRows}
    {assignment : Nat → Nat} {fields : FieldVector fieldCount}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FieldsPlaced layout assignment fields)
    (holds : Satisfies (rows setup packing layout) assignment) :
    ∀ output,
      assignment (layout.outputColumn output) =
        (semanticOutput setup (packFields packing fields) output).val := by
  intro output
  have definitionsHold := definitions_sound canonical one
    (linear_holds_of_rows holds)
  have outputHolds := definitionsHold
    (linearDefinition setup packing layout output)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)
  change assignment (layout.outputColumn output) =
    lcEval assignment (linearTerms setup packing layout output) at outputHolds
  rw [outputHolds]
  exact linearValue_exact setup packing layout assignment
    (packFields packing fields) output
    (sourceDigit_exact canonical one placed holds)

end Nightstream.Implementation.Nebula.CompactAjtaiStageRows
