import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
import Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-!
Contract: pull one canonical numeric NIFS witness back through the exact typed
call-frame column map.

The completion writes only the frame's declared temporary identities. Visible
inputs, outputs, the activation column, and the constant-one wire are
preserved.  Within the exact global numeric namespace, the pulled assignment
is definitionally the supplied canonical numeric witness.

This module owns no verifier semantics, row construction, activation, output
materialization, Rust layout, or generated artifact.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNumericCompletion

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

universe u

/-- Pointwise canonical representative of a numeric assignment. -/
def canonicalize (assignment : Nat → Nat) : Nat → Nat :=
  fun column => assignment column % goldilocksP

theorem canonicalize_lt
    (assignment : Nat → Nat) (column : Nat) :
    canonicalize assignment column < goldilocksP := by
  exact Nat.mod_lt _ (by decide)

private theorem rawSum_canonicalize
    (assignment : Nat → Nat) :
    ∀ terms,
      rawSum (canonicalize assignment) terms % goldilocksP =
        rawSum assignment terms % goldilocksP := by
  intro terms
  induction terms with
  | nil => rfl
  | cons term rest hypothesis =>
      rw [rawSum_cons, rawSum_cons, Nat.add_mod, Nat.add_mod,
        Nat.mul_mod, hypothesis]
      simp [canonicalize]

/-- R1CS linear-combination evaluation depends only on canonical residue
classes. -/
theorem lcEval_canonicalize
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) :
    lcEval (canonicalize assignment) terms = lcEval assignment terms := by
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum]
  exact rawSum_canonicalize assignment terms

/-- Pointwise canonicalization preserves every R1CS row equation. -/
theorem satisfies_canonicalize
    (rows : List Nightstream.Implementation.R1CS.Row)
    (assignment : Nat → Nat)
    (satisfied :
      Nightstream.Implementation.R1CS.Satisfies rows assignment) :
    Nightstream.Implementation.R1CS.Satisfies
      rows (canonicalize assignment) := by
  intro row member
  have holds := satisfied row member
  unfold RowHolds at holds ⊢
  simpa only [lcEval_canonicalize] using holds

private abbrev FrameFor
    {parameters : Parameters}
    (family : Family (typeSystem parameters))
    {context : Schema (typeSystem parameters)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    {fresh : Ref (typeSystem parameters) context (.data .fresh)}
    {proof : Ref (typeSystem parameters) context (.data .nifsProof)} :=
  CallFrame (signature := signature parameters) family Call.nifsVerify
    (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))

/-- Canonical values of the supplied numeric witness at every declared
temporary source, in the frame's exact temporary order. -/
def temporaryValues
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    {fresh : Ref (typeSystem parameters) context (.data .fresh)}
    {proof : Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame : FrameFor family
      (context := context) (running := running) (fresh := fresh)
      (proof := proof))
    (numeric : Nat → Nat) : List Field :=
  List.ofFn fun index : Fin frame.temporaries.ids.length =>
    residue (numeric (temporarySource frame index.val))

@[simp] theorem temporaryValues_length
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    {fresh : Ref (typeSystem parameters) context (.data .fresh)}
    {proof : Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame : FrameFor family
      (context := context) (running := running) (fresh := fresh)
      (proof := proof))
    (numeric : Nat → Nat) :
    (temporaryValues frame numeric).length =
      frame.temporaries.ids.length := by
  simp [temporaryValues]

/-- Typed assignment obtained by installing the canonical numeric witness on
the frame's declared temporary suffix. -/
def complete
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    {fresh : Ref (typeSystem parameters) context (.data .fresh)}
    {proof : Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame : FrameFor family
      (context := context) (running := running) (fresh := fresh)
      (proof := proof))
    (assignment : ColumnId → Field)
    (numeric : Nat → Nat) : ColumnId → Field :=
  writeColumns assignment frame.temporaries.ids
    (temporaryValues frame numeric)

/-- Pulling a numeric witness back into the typed frame changes only the
declared temporary bundle. -/
theorem complete_changesOnly
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    {fresh : Ref (typeSystem parameters) context (.data .fresh)}
    {proof : Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame : FrameFor family
      (context := context) (running := running) (fresh := fresh)
      (proof := proof))
    (assignment : ColumnId → Field)
    (numeric : Nat → Nat) :
    ChangesOnly frame.temporaries.ids assignment
      (complete frame assignment numeric) := by
  exact writeColumns_changesOnly assignment frame.temporaries.ids
    (temporaryValues frame numeric)

/-- Numeric completion changes no visible identity. -/
theorem complete_agrees_visible
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    {fresh : Ref (typeSystem parameters) context (.data .fresh)}
    {proof : Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame : FrameFor family
      (context := context) (running := running) (fresh := fresh)
      (proof := proof))
    (assignment : ColumnId → Field)
    (numeric : Nat → Nat) :
    AgreesOn frame.visibleIds assignment
      (complete frame assignment numeric) := by
  exact writeColumns_agreesOn assignment frame.temporaries.ids
    frame.visibleIds (temporaryValues frame numeric)
    frame.temporariesDisjointVisible

/-- One declared temporary recovers exactly the matching canonical numeric
value. -/
theorem complete_temporary
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    {fresh : Ref (typeSystem parameters) context (.data .fresh)}
    {proof : Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame : FrameFor family
      (context := context) (running := running) (fresh := fresh)
      (proof := proof))
    (assignment : ColumnId → Field)
    (numeric : Nat → Nat)
    (index : Nat) (indexLt : index < frame.temporaries.ids.length) :
    complete frame assignment numeric frame.temporaries.ids[index] =
      residue (numeric (temporarySource frame index)) := by
  have recovered :
      frame.temporaries.ids.map (complete frame assignment numeric) =
        temporaryValues frame numeric := by
    apply writeColumns_map_eq
    · rw [temporaryValues_length]
    · exact (List.nodup_append.1 frame.allocationsNodup).2.1
  have leftBound :
      index <
        (frame.temporaries.ids.map
          (complete frame assignment numeric)).length := by
    simpa using indexLt
  have rightBound : index < (temporaryValues frame numeric).length := by
    simpa using indexLt
  have atIndex := congrArg
    (fun values : List Field =>
      values.getD index (complete frame assignment numeric frame.one))
    recovered
  change
    (frame.temporaries.ids.map
        (complete frame assignment numeric)).getD index
          (complete frame assignment numeric frame.one) =
      (temporaryValues frame numeric).getD index
        (complete frame assignment numeric frame.one) at atIndex
  rw [← List.getElem_eq_getD
      (l := frame.temporaries.ids.map
        (complete frame assignment numeric))
      (i := index) (h := leftBound)
      (complete frame assignment numeric frame.one),
    ← List.getElem_eq_getD
      (l := temporaryValues frame numeric)
      (i := index) (h := rightBound)
      (complete frame assignment numeric frame.one)] at atIndex
  simp only [List.getElem_map] at atIndex
  simpa [temporaryValues] using atIndex

/-- Inside the exact global namespace, pulling a canonical numeric witness
back through the typed completion recovers it pointwise. The only premise
about the visible prefix is explicit agreement with the assignment being
completed. -/
theorem numericAssignment_complete_of_lt
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    {fresh : Ref (typeSystem parameters) context (.data .fresh)}
    {proof : Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame : FrameFor family
      (context := context) (running := running) (fresh := fresh)
      (proof := proof))
    (assignment : ColumnId → Field)
    (numeric : Nat → Nat)
    (canonical : ∀ source, numeric source < goldilocksP)
    (visibleAgreement :
      ∀ source, source < temporaryBase frame →
        numeric source =
          numericAssignment (columnMap frame) assignment source)
    (source : Nat) (sourceBound : source < (orderedIds frame).length) :
    numericAssignment (columnMap frame)
        (complete frame assignment numeric) source =
      numeric source := by
  by_cases before : source < temporaryBase frame
  · have visible :=
      columnMap_before_temporaryBase frame source before
    have preserved :=
      complete_agrees_visible frame assignment numeric
        (columnMap frame source) visible
    change
      (complete frame assignment numeric (columnMap frame source)).val =
        numeric source
    rw [preserved]
    exact (visibleAgreement source before).symm
  · have sourceGe : temporaryBase frame ≤ source := Nat.le_of_not_gt before
    let index := source - temporaryBase frame
    have sourceEq : temporaryBase frame + index = source := by
      simp only [index]
      omega
    have indexLt : index < frame.temporaries.ids.length := by
      rw [orderedIds_eq_visible_append_temporaries,
        List.length_append] at sourceBound
      simp only [index, temporaryBase] at sourceGe sourceBound ⊢
      omega
    have mapped :
        columnMap frame source = frame.temporaries.ids[index] := by
      rw [← sourceEq]
      exact columnMap_temporarySource frame indexLt
    change
      (complete frame assignment numeric (columnMap frame source)).val =
        numeric source
    rw [mapped, complete_temporary frame assignment numeric index indexLt]
    rw [show temporarySource frame index = source by
      simpa [temporarySource] using sourceEq]
    change numeric source % Nightstream.SuperNeo.Concrete.goldilocksModulus =
      numeric source
    apply Nat.mod_eq_of_lt
    simpa [goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using
      canonical source

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNumericCompletion
