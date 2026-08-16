import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplaySchema

/-!
Contract: compact Boolean checks and kernel proofs for bounded claim-replay leaf
certificates.

Assurance tier: structural certificate support.

Owns the generic proof that a successful compact glue-row geometry check gives
the typed glue-row geometry relation.

Does not own any generated data, concrete artifact identity, or row semantics.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayLeafCertificateSupport

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

def termsBelowCheck (columnCount : Nat) (terms : List (Nat × Nat)) : Bool :=
  terms.all fun term => decide (term.1 < columnCount)

theorem termsBelowCheck_sound
    {columnCount : Nat} {terms : List (Nat × Nat)}
    (checked : termsBelowCheck columnCount terms = true) :
    ∀ term ∈ terms, term.1 < columnCount := by
  intro term member
  exact of_decide_eq_true
    ((List.all_eq_true.mp checked) term member)

def rowColumnsBelowCheck (columnCount : Nat) (row : Row) : Bool :=
  termsBelowCheck columnCount row.a &&
    (termsBelowCheck columnCount row.b && termsBelowCheck columnCount row.c)

theorem rowColumnsBelowCheck_sound
    {columnCount : Nat} {row : Row}
    (checked : rowColumnsBelowCheck columnCount row = true) :
    rowColumnsBelow columnCount row := by
  simp only [rowColumnsBelowCheck, Bool.and_eq_true] at checked
  exact ⟨termsBelowCheck_sound checked.1,
    termsBelowCheck_sound checked.2.1,
    termsBelowCheck_sound checked.2.2⟩

def glueRowGeometryCheck
    (rowCount columnCount : Nat) (indexed : IndexedRow) : Bool :=
  decide (indexed.index < rowCount) &&
    rowColumnsBelowCheck columnCount indexed.row

def glueRowsGeometryCheck
    (rowCount columnCount : Nat) (rows : List IndexedRow) : Bool :=
  rows.all (glueRowGeometryCheck rowCount columnCount)

theorem glueRowsGeometryCheck_sound
    {rowCount columnCount : Nat} {rows : List IndexedRow}
    (checked : glueRowsGeometryCheck rowCount columnCount rows = true) :
    ∀ indexed ∈ rows,
      indexed.index < rowCount ∧ rowColumnsBelow columnCount indexed.row := by
  intro indexed member
  have rowChecked := (List.all_eq_true.mp checked) indexed member
  simp only [glueRowGeometryCheck, Bool.and_eq_true] at rowChecked
  exact ⟨of_decide_eq_true rowChecked.1,
    rowColumnsBelowCheck_sound rowChecked.2⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayLeafCertificateSupport
