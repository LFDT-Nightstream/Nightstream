import Nightstream.Implementation.R1CS.Canonical.KTraceProgramHonest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PublicColumns

/-!
Contract: construct the canonical public PiRLC trace batch directly from
decoded NIFS coefficient columns.

The input is the protocol-shaped public batch: one challenge and one input
coefficient vector per fold input, one output vector per public role, one
quotient witness per role, and one verifier-selected projection point.  It is
not a generated artifact and contains no legacy power-ladder or evaluation
trace.

Transcript derivation of `beta`, private PiCCS/PiDEC rows, and the application
proof codec remain separate boundaries.
-/

set_option autoImplicit false
set_option maxRecDepth 4096

namespace Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace

open Nightstream.Implementation.R1CS.Canonical.KProjectionTrace
open Nightstream.Implementation.R1CS.Canonical.KTraceProgram
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

/-- The complete coefficient-column input to the public PiRLC projection
subprogram.  `quotients` are prover witness columns checked by the emitted
identity rows; they are not treated as semantic authority. -/
structure Columns (arity matrixCount : Nat) where
  beta : KColumns
  challenges : Fin arity → List Nat
  inputs : Fin arity → ProjectionColumns matrixCount
  output : ProjectionColumns matrixCount
  quotients : PublicRole matrixCount → List Nat

/-- Static eligibility required by the Phi81 quotient identity. -/
structure Columns.Valid
    {arity matrixCount : Nat} (columns : Columns arity matrixCount) : Prop where
  arityPositive : 0 < arity
  challengeWidth : ∀ index, (columns.challenges index).length = 54
  inputWidth : ∀ index role, ((columns.inputs index).at role).length = 54
  outputWidth : ∀ role, (columns.output.at role).length = 54
  quotientWidth : ∀ role, (columns.quotients role).length = 53

/-- Physical placement required before the coefficient program may allocate
its auxiliary block at `base`.  Every verifier challenge, public PiCCS input,
public PiRLC output, and prover quotient coefficient must already be allocated
strictly below that base.  This prevents an authoritative read from aliasing a
fresh Horner or multiplication witness column. -/
structure Columns.BelowBase
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    (base : Nat) : Prop where
  betaLow : columns.beta.c0 < base
  betaHigh : columns.beta.c1 < base
  challenge :
    ∀ index column, column ∈ columns.challenges index -> column < base
  input :
    ∀ index role column,
      column ∈ (columns.inputs index).at role -> column < base
  output :
    ∀ role column, column ∈ columns.output.at role -> column < base
  quotient :
    ∀ role column, column ∈ columns.quotients role -> column < base

def pair
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    (role : PublicRole matrixCount) (index : Fin arity) : PairColumns where
  rho := columns.challenges index
  input := (columns.inputs index).at role

/-- One canonical identity trace for one public PiRLC role. -/
def trace
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    (role : PublicRole matrixCount) : KProjectionTrace.Trace where
  beta := columns.beta
  pairs := List.ofFn fun index => pair columns role index
  output := columns.output.at role
  quotient := columns.quotients role
  maxDegree := 106

/-- Public traces in the paper-owned role order. -/
def traces
    {arity matrixCount : Nat} (columns : Columns arity matrixCount) :
    List KProjectionTrace.Trace :=
  (publicOrder matrixCount).map (trace columns)

theorem pair_rho
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    (role : PublicRole matrixCount) (index : Fin arity) :
    (pair columns role index).rho = columns.challenges index :=
  rfl

theorem pair_input
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    (role : PublicRole matrixCount) (index : Fin arity) :
    (pair columns role index).input = (columns.inputs index).at role :=
  rfl

theorem trace_valid
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    (valid : columns.Valid) (role : PublicRole matrixCount) :
    (trace columns role).Valid := by
  refine ⟨?_, ?_, valid.outputWidth role, valid.quotientWidth role, rfl⟩
  · apply List.ne_nil_of_length_pos
    simpa only [trace, List.length_ofFn] using valid.arityPositive
  · intro item member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    exact ⟨valid.challengeWidth index, valid.inputWidth index role⟩

theorem trace_columns_belowBase
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    {base : Nat} (placed : columns.BelowBase base)
    (role : PublicRole matrixCount) :
    let selected := trace columns role
    selected.beta.c0 < base ∧
      selected.beta.c1 < base ∧
      (∀ item ∈ selected.pairs,
        (∀ column ∈ item.rho, column < base) ∧
        (∀ column ∈ item.input, column < base)) ∧
      (∀ column ∈ selected.output, column < base) ∧
      (∀ column ∈ selected.quotient, column < base) := by
  refine ⟨placed.betaLow, placed.betaHigh, ?_, placed.output role,
    placed.quotient role⟩
  intro item member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact ⟨placed.challenge index, placed.input index role⟩

theorem trace_coefficients_belowBase
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    {base : Nat} (placed : columns.BelowBase base)
    (role : PublicRole matrixCount) :
    (trace columns role).CoefficientsBelow base := by
  refine ⟨?_, placed.output role, placed.quotient role⟩
  intro item member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact ⟨placed.challenge index, placed.input index role⟩

theorem traces_length
    {arity matrixCount : Nat} (columns : Columns arity matrixCount) :
    (traces columns).length = 23 + 2 * matrixCount := by
  simp only [traces, List.length_map, public_role_count]

@[simp] theorem trace_pairs_length
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    (role : PublicRole matrixCount) :
    (trace columns role).pairs.length = arity := by
  simp only [trace, List.length_ofFn]

/-- The selected batch layout is constructed, not supplied.  Shared-beta and
per-trace validity follow from the column carrier itself. -/
def batchLayout
    {arity matrixCount : Nat} (columns : Columns arity matrixCount)
    (valid : columns.Valid) : BatchLayout where
  traces := traces columns
  sharedBeta := columns.beta
  betaShared := by
    intro item member
    rcases List.mem_map.mp member with ⟨role, _, rfl⟩
    rfl
  valid := by
    intro item member
    rcases List.mem_map.mp member with ⟨role, _, rfl⟩
    exact trace_valid columns valid role

/-- One occurrence of the public PiRLC quotient checks. -/
def occurrence
    {arity matrixCount : Nat} (base : Nat)
    (columns : Columns arity matrixCount) (valid : columns.Valid) :
    Occurrence :=
  ⟨base, batchLayout columns valid⟩

/-- The canonical honest auxiliary assignment for the complete public batch. -/
def honestWitness
    {arity matrixCount : Nat} (source : Nat → Nat) (base : Nat)
    (columns : Columns arity matrixCount) : Nat → Nat :=
  KTraceProgramHonest.batchWitness source base (traces columns)

/-- The exact public-PiRLC row subtotal selected by this occurrence.  The
remaining PiCCS, PiDEC, transcript, point-binding, and accumulator rows of a
complete NIFS verifier are deliberately not counted here. -/
theorem occurrence_rows_length
    {arity matrixCount : Nat} (base : Nat)
    (columns : Columns arity matrixCount) (valid : columns.Valid) :
    (occurrence base columns valid).rows.length =
      (23 + 2 * matrixCount) * (321 * arity + 482) := by
  change
    (KTraceProgram.rows base (batchLayout columns valid)).length =
      (23 + 2 * matrixCount) * (321 * arity + 482)
  rw [KTraceProgram.rows_length]
  change
    ((traces columns).map fun trace =>
      321 * trace.pairs.length + 482).sum =
        (23 + 2 * matrixCount) * (321 * arity + 482)
  have sumConstant :
      ∀ roles : List (PublicRole matrixCount),
        (roles.map fun role =>
          321 * (trace columns role).pairs.length + 482).sum =
            roles.length * (321 * arity + 482) := by
    intro roles
    induction roles with
    | nil => simp
    | cons role rest inductionHypothesis =>
        have headLength : (trace columns role).pairs.length = arity := by
          exact trace_pairs_length columns role
        rw [List.map_cons, List.sum_cons, List.length_cons,
          inductionHypothesis, headLength]
        change
          321 * arity + 482 + rest.length * (321 * arity + 482) =
            (rest.length + 1) * (321 * arity + 482)
        rw [Nat.add_mul, Nat.one_mul, Nat.add_comm]
  unfold traces
  rw [List.map_map]
  change
    ((publicOrder matrixCount).map fun role =>
      321 * (trace columns role).pairs.length + 482).sum =
        (23 + 2 * matrixCount) * (321 * arity + 482)
  rw [sumConstant, public_role_count]

theorem occurrence_exact_or_badRoot
    {arity matrixCount : Nat} (base : Nat)
    (columns : Columns arity matrixCount) (valid : columns.Valid)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (satisfied :
      Nightstream.Implementation.R1CS.Satisfies
        (occurrence base columns valid).rows assignment) :
    (occurrence base columns valid).Exact assignment
      ∨ (occurrence base columns valid).BadRoot assignment :=
  Occurrence.exact_or_badRoot
    (occurrence base columns valid) assignment constantWire satisfied

/-- Coefficient-exact public identities have a concrete satisfying assignment
for the complete concatenated occurrence.  The construction preserves every
authoritative column below `base`. -/
theorem occurrence_rows_honest
    {arity matrixCount : Nat} (source : Nat → Nat) (base : Nat)
    (columns : Columns arity matrixCount) (valid : columns.Valid)
    (basePositive : 0 < base)
    (placed : columns.BelowBase base)
    (constantWire : source 0 = 1)
    (exact : (occurrence base columns valid).Exact source) :
    Nightstream.Implementation.R1CS.Satisfies
      (occurrence base columns valid).rows
      (honestWitness source base columns) := by
  apply KTraceProgramHonest.rowsFrom_honest source columns.beta base
      (traces columns) basePositive constantWire
  · intro item member
    rcases List.mem_map.mp member with ⟨role, _, rfl⟩
    rfl
  · exact (batchLayout columns valid).valid
  · intro item member
    rcases List.mem_map.mp member with ⟨role, _, rfl⟩
    exact ⟨placed.betaLow, placed.betaHigh⟩
  · intro item member
    rcases List.mem_map.mp member with ⟨role, _, rfl⟩
    exact trace_coefficients_belowBase columns placed role
  · intro item member
    exact exact (item.identity source) (by
      unfold Occurrence.identities KProjectionTrace.BatchIdentity
      exact List.mem_map.mpr ⟨item, member, rfl⟩)

theorem honestWitness_preserves_source
    {arity matrixCount : Nat} (source : Nat → Nat) (base : Nat)
    (columns : Columns arity matrixCount)
    (column : Nat) (below : column < base) :
    honestWitness source base columns column = source column :=
  KTraceProgramHonest.batchWitness_preserves_below
    source base (traces columns) column below

end Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace
