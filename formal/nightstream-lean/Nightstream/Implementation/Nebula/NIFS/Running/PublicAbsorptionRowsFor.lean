import Nightstream.Implementation.Nebula.NIFS.Core.Poseidon2
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

/-!
Contract: shape-independent rows for the paper-NIFS public-input absorption.

The field count is a type parameter. Production obtains it from the exact
augmented-relation exponent. This module proves only the public-prefix replay.
The complete PiCCS, SumCheck, PiRLC, and PiDEC rows remain separate sections
of the generated relation.

Assurance tier: row-semantics component.

Emits constraints: through `SymbolicDuplex.rows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductNifsPublicAbsorptionRowsFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

def word (value : Nat) : LinComb := [(0, value % goldilocksP)]

/-- Physical input to one exact public-prefix row section. -/
structure Input (fieldCount : Nat) where
  statementId : ProductPoseidon2.StatementId
  fields : List LinComb
  fields_length : fields.length = fieldCount
  transcriptBase : Nat

def initialLanes
    (statementId : ProductPoseidon2.StatementId) : Poseidon2Core.State :=
  fun lane => word
    ((ProductPoseidon2.initialStateForStatement statementId).lanes lane)

def initialBuilder {fieldCount : Nat} (input : Input fieldCount) :
    SymbolicDuplex.Builder :=
  SymbolicDuplex.start (initialLanes input.statementId)
    (ProductPoseidon2.initialStateForStatement input.statementId).absorbed

def absorbPublicInput {fieldCount : Nat} (input : Input fieldCount) :
    SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany input.transcriptBase input.fields
    (initialBuilder input)

def rows {fieldCount : Nat} (input : Input fieldCount) : List Row :=
  SymbolicDuplex.rows input.transcriptBase ProductPoseidon2.constants
    (absorbPublicInput input)

theorem rows_length {fieldCount : Nat} (input : Input fieldCount) :
    (rows input).length =
      (absorbPublicInput input).entries.length * SymbolicDuplex.stride := by
  simp [rows, SymbolicDuplex.rows_length, SymbolicDuplex.stride]

def fieldValues (assignment : Nat -> Nat) (fields : List LinComb) :
    List Nat :=
  fields.map (lcEval assignment)

theorem lcEval_word (assignment : Nat -> Nat) (one : assignment 0 = 1)
    (value : Nat) :
    lcEval assignment (word value) = value % goldilocksP := by
  simp [word, lcEval, rawSum, one, goldilocksP]

def StateCanonical (state : Poseidon2Duplex.State) : Prop :=
  forall lane, state.lanes lane < goldilocksP

theorem empty_canonical : StateCanonical Poseidon2Duplex.empty := by
  intro lane
  simp [StateCanonical, Poseidon2Duplex.empty, goldilocksP]

theorem permute_canonical (state : Poseidon2Duplex.State) :
    StateCanonical
      (Poseidon2Duplex.permute ProductPoseidon2.constants state) := by
  intro lane
  exact Poseidon2Honest.refTerminal_lt _ _ _ _

theorem guarded_canonical (state : Poseidon2Duplex.State)
    (canonical : StateCanonical state) :
    StateCanonical
      (Poseidon2Duplex.guarded ProductPoseidon2.constants state) := by
  unfold Poseidon2Duplex.guarded
  split
  · exact permute_canonical state
  · exact canonical

theorem absorbElem_canonical (value : Nat) (state : Poseidon2Duplex.State)
    (canonical : StateCanonical state) :
    StateCanonical
      (Poseidon2Duplex.absorbElem ProductPoseidon2.constants value state) := by
  intro lane
  unfold Poseidon2Duplex.absorbElem
  let target := Poseidon2Duplex.guarded ProductPoseidon2.constants state
  have targetCanonical : StateCanonical target :=
    guarded_canonical state canonical
  change (if lane.val = target.absorbed then value % goldilocksP
    else target.lanes lane) < goldilocksP
  split
  · exact Nat.mod_lt _ (by decide)
  · exact targetCanonical lane

theorem absorbList_canonical (values : List Nat)
    (state : Poseidon2Duplex.State) (canonical : StateCanonical state) :
    StateCanonical
      (Poseidon2Duplex.absorbList ProductPoseidon2.constants values state) := by
  induction values generalizing state with
  | nil => exact canonical
  | cons head tail inductionHypothesis =>
      exact inductionHypothesis _ (absorbElem_canonical head state canonical)

theorem initialStateForStatement_canonical
    (statementId : ProductPoseidon2.StatementId) :
    StateCanonical (ProductPoseidon2.initialStateForStatement statementId) := by
  unfold ProductPoseidon2.initialStateForStatement
  apply absorbList_canonical
  exact empty_canonical

theorem decoded_initialBuilder
    (assignment : Nat -> Nat) (one : assignment 0 = 1)
    {fieldCount : Nat} (input : Input fieldCount) :
    decodedBuilder assignment (initialBuilder input) =
      ProductPoseidon2.initialStateForStatement input.statementId := by
  rw [Poseidon2Duplex.State.mk.injEq]
  refine ⟨?_, rfl⟩
  funext lane
  change lcEval assignment
      (word ((ProductPoseidon2.initialStateForStatement
        input.statementId).lanes lane)) =
    (ProductPoseidon2.initialStateForStatement input.statementId).lanes lane
  rw [lcEval_word assignment one]
  exact Nat.mod_eq_of_lt
    (initialStateForStatement_canonical input.statementId lane)

def valueAbsorbPublic
    (assignment : Nat -> Nat) {fieldCount : Nat} (input : Input fieldCount) :
    Poseidon2Duplex.State :=
  Poseidon2Duplex.absorbList ProductPoseidon2.constants
    (fieldValues assignment input.fields)
    (ProductPoseidon2.initialStateForStatement input.statementId)

/-- Satisfying the complete public-prefix row list fixes the decoded state to
the value-level replay of every placed field. -/
theorem rows_semantics
    (assignment : Nat -> Nat) {fieldCount : Nat} (input : Input fieldCount)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decodedBuilder assignment (absorbPublicInput input) =
      valueAbsorbPublic assignment input := by
  have valid : Valid input.transcriptBase ProductPoseidon2.constants assignment
      (absorbPublicInput input) := by
    exact valid_of_satisfied input.transcriptBase ProductPoseidon2.constants
      (absorbPublicInput input) assignment residues one satisfied
  have replay := decodedBuilder_absorbMany input.transcriptBase
    ProductPoseidon2.constants assignment input.fields
    (initialBuilder input) valid
  rw [decoded_initialBuilder assignment one input] at replay
  exact replay

end Nightstream.Implementation.Nebula.ProductNifsPublicAbsorptionRowsFor
