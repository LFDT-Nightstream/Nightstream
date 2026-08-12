import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.TranscriptRows
import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TranscriptSemantics
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

/-!
Contract: row-satisfaction semantics for the exact V2 PiRLC candidate calls.

Owns the proof that every one of the 15 x 54 x 3 candidate expressions is
the exact full-field `ProductPoseidon2.candidateValue` derived from the same
post-PiCCS Poseidon2 state. The proof starts only from canonical physical
fields, the shared constant wire, and satisfaction of the indexed rows.

Does not own candidate acceptance, modulo-five decoding, first-accepted
selection, PiRLC algebra, honest witness construction, random-oracle
security, Rust, or the surrounding NIFS verifier.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptSemantics

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

abbrev ValueState := Poseidon2Duplex.State

/-- The value-level post-PiCCS state denoted by the candidate input. -/
def valueStart (assignment : Nat -> Nat) (input : Input) : ValueState :=
  decodedBuilder assignment (start input)

def fieldValues (assignment : Nat -> Nat) (fields : List LinComb) :
    List Nat :=
  fields.map (lcEval assignment)

theorem fieldValues_candidateFields
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (index : CandidateIndex) :
    fieldValues assignment (candidateFields index) =
      (ProductPoseidon2.candidateFields
        (Fin.cast scalarCount_profile index.source)
        (Fin.cast coefficientCount_profile index.coefficient)
        (Fin.cast attemptCount_profile index.attempt)).map
          (fun value => value % goldilocksP) := by
  unfold fieldValues candidateFields
  exact ProductPiCcsTranscriptSemantics.fieldValues_words assignment one _

/-- Absorption already reduces every overwritten value, so reducing a list
before absorption does not change the duplex state. -/
theorem absorbList_map_mod (values : List Nat) (state : ValueState) :
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (values.map fun value => value % goldilocksP) state =
      Poseidon2Duplex.absorbList ProductPoseidon2.constants values state := by
  induction values generalizing state with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, Poseidon2Duplex.absorbList]
      simpa [Poseidon2Duplex.absorbElem, Nat.mod_mod] using
        inductionHypothesis
          (Poseidon2Duplex.absorbElem ProductPoseidon2.constants head state)

/-- Satisfaction of one candidate window computes that exact candidate from
the common post-PiCCS state. No candidate or challenge value is supplied as
an assumption. -/
theorem candidate_rows_sound
    (input : Input) (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : RowsHold input assignment)
    (index : CandidateIndex) :
    lcEval assignment (candidate input index) =
      (ProductPoseidon2.candidateValue
        (valueStart assignment input)
        (Fin.cast scalarCount_profile index.source)
        (Fin.cast coefficientCount_profile index.coefficient)
        (Fin.cast attemptCount_profile index.attempt)).val := by
  let base := candidateBase input index
  let absorbed :=
    SymbolicDuplex.absorbMany base (candidateFields index) (start input)
  have finalValid :
      Valid base ProductPoseidon2.constants assignment
        (SymbolicDuplex.gate base absorbed) := by
    apply valid_of_satisfied base ProductPoseidon2.constants
      (SymbolicDuplex.gate base absorbed) assignment residues one
    simpa [rows, builder, base, absorbed] using holds index
  have absorbedValid :
      Valid base ProductPoseidon2.constants assignment absorbed :=
    finalValid.of_extends (gate_extends base absorbed)
  have absorbedEq := decodedBuilder_absorbMany base
    ProductPoseidon2.constants assignment (candidateFields index)
    (start input) absorbedValid
  change decodedBuilder assignment absorbed =
    Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (fieldValues assignment (candidateFields index))
      (valueStart assignment input) at absorbedEq
  rw [fieldValues_candidateFields assignment one index,
    absorbList_map_mod] at absorbedEq
  have gateEq := decodedBuilder_gate base ProductPoseidon2.constants
    assignment absorbed one finalValid
  rw [absorbedEq] at gateEq
  have laneEq := congrArg
    (fun state => state.lanes ⟨0, by decide⟩) gateEq
  change lcEval assignment (candidate input index) = _ at laneEq
  have laneEq' :
      lcEval assignment (candidate input index) =
        (Poseidon2Duplex.gate ProductPoseidon2.constants
          (Poseidon2Duplex.absorbList ProductPoseidon2.constants
            (ProductPoseidon2.candidateFields
              (Fin.cast scalarCount_profile index.source)
              (Fin.cast coefficientCount_profile index.coefficient)
              (Fin.cast attemptCount_profile index.attempt))
            (valueStart assignment input))).lanes ⟨0, by decide⟩ := by
    simpa only [] using laneEq
  have sampledLt :
      (Poseidon2Duplex.gate ProductPoseidon2.constants
        (Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (ProductPoseidon2.candidateFields
            (Fin.cast scalarCount_profile index.source)
            (Fin.cast coefficientCount_profile index.coefficient)
            (Fin.cast attemptCount_profile index.attempt))
          (valueStart assignment input))).lanes ⟨0, by decide⟩ <
        goldilocksP := by
    rw [← laneEq']
    exact lcEval_lt assignment _
  rw [ProductPoseidon2.candidateValue]
  change lcEval assignment (candidate input index) =
    (Poseidon2Duplex.gate ProductPoseidon2.constants
      (Poseidon2Duplex.absorbList ProductPoseidon2.constants
        (ProductPoseidon2.candidateFields
          (Fin.cast scalarCount_profile index.source)
          (Fin.cast coefficientCount_profile index.coefficient)
          (Fin.cast attemptCount_profile index.attempt))
        (valueStart assignment input))).lanes ⟨0, by decide⟩ %
          goldilocksP
  rw [Nat.mod_eq_of_lt sampledLt]
  exact laneEq'

/-- All indexed physical rows refine all exact sampler candidates. -/
theorem all_candidates_sound
    (input : Input) (assignment : Nat -> Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : RowsHold input assignment) :
    forall index,
      lcEval assignment (candidate input index) =
        (ProductPoseidon2.candidateValue
          (valueStart assignment input)
          (Fin.cast scalarCount_profile index.source)
          (Fin.cast coefficientCount_profile index.coefficient)
          (Fin.cast attemptCount_profile index.attempt)).val :=
  candidate_rows_sound input assignment residues one holds

end Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptSemantics
