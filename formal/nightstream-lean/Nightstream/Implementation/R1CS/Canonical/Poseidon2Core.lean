import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.Implementation.Lowering.Typed.Cost
import Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-!
Contract: canonical Lean-owned encoding of the Goldilocks width-8 Poseidon2
permutation.

## Why this module was rebuilt

An earlier version typed the S-box input as a single column
(`SboxFrame.input : Nat`).  That representation *cannot express* the normal
form it claimed: a linear layer produces a linear combination of lanes, not a
column, so "the linear map folds into the operand vectors of the following
multiplication" was not implementable, and the terminal linear layer had no
following S-box to fold into at all.  The row count that followed was the
nonlinear subtotal, not a permutation cost.

The repair is at the type level: an S-box consumes a `LinComb`.  Folding then
becomes real, and the terminal layer's binding cost becomes visible instead of
being assumed away.

Owns: the S-box over arbitrary linear combinations; linear layers as pure
combination arithmetic emitting no rows; terminal output binding; and the
derived *row* subtotals.

**This is not a certified permutation cost.**  The row figures are the exact
structural cost of the selected width-8 / 8 / 22 / `x^7` folding normal form,
**conditional on a conforming Poseidon2 schedule**.  `sboxRows_chain` proves
each S-box computes `x^7` on whatever combination it is handed; it does not
prove those inputs are Poseidon2's successive states.  That is
`POSEIDON2-ROUND-INDUCTION`.

Independent corroboration of the normal form: the Rust gadget also carries its
state as linear combinations (`[Lc; WIDTH]`, `external_linear_layer`,
`internal_linear_layer`) rather than materializing every round.  Its extra rows
come from explicit `materialize_state` calls, which is the first place to look
when the encodings are eventually compared.

Scope: the **width-8 F'/`neo_ccs` permutation only**.  A separate width-16
Poseidon2 exists for CCS digest machinery
(`neo-reductions/src/engines/utils.rs`, `Poseidon2Goldilocks::<16>`); it is out
of scope here and shares nothing with this profile but the algorithm family.

Does not own:
  * absorption, padding, rate, capacity — Phase 3 sponge concerns;
  * the activation/output-copy wrapper;
  * any generated artifact.  The 600-row `Poseidon2PermutationArtifact` is
    never read and is not authority.

Authority boundary — corrected 2026-07-25.

Neither SuperNeo nor HyperNova selects Poseidon2.  HyperNova Construction 2
takes "a cryptographic hash function" (§6.3) and Appendix B takes "a random
oracle"; both are abstract parameters the papers deliberately delegate.  The
only Poseidon mentions in either paper are cost comparisons against other
schemes.  Contrast SuperNeo Appendix B.2, which *is* concrete about
`d = 54`, `b = 2`, `k = 14`, `T = 216` — those are paper-derived; these are
not.

**Poseidon2 and its parameters are implementation choices made by the Rust
prover**, verified against
`crates/neo-fold-clean/src/engine/r1cs_circuit/poseidon2.rs`:

    WIDTH             = 8
    HALF_FULL_ROUNDS  = 4     (so eight external rounds)
    PARTIAL_ROUNDS    = 22
    S-box             = x^7   (`enforce_sbox_x7`)

That selection is legitimate and delegated, but it is not paper authority and
must not be described as such.

Parameter ownership splits across two sources and the profile must freeze
both: `neo-params` owns width, capacity, rate, digest length and seed; **p3 and
the Rust circuit** own `x^7`, the 8/22 round selection, and the two linear-layer
definitions.

Round constants and both linear matrices remain explicit parameters.  Because
the goal is to re-encode *the selected permutation*, taking their values from
the Rust implementation is correct rather than contaminating: the canonical
encoding must compute the same function.  What may **not** be taken from the
generated artifact is its row count or row layout — that is what the canonical
encoding exists to derive independently.

This file publishes only the core definitions and local structural lemmas.
Round induction, concrete layout, ownership, conservation, exact cost, and
honest completeness are proved in their dedicated canonical modules; their
fail-closed status is owned by the corresponding guard files rather than an
editable source-header checklist.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Core

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Lowering
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-! ## Published permutation parameters -/

def width : Nat := 8
def externalRounds : Nat := 8
def partialRounds : Nat := 22
def sboxDegree : Nat := 7

/-! ## Linear combinations

`Row` already carries sparse linear combinations in `a`, `b`, `c`.  Using that
same representation for an S-box input is what makes folding expressible. -/

/-- A sparse linear combination over columns.  Column `0` is the constant wire,
so round constants appear as `(0, c)` terms rather than as separate rows. -/
abbrev LinComb := List (Nat × Nat)

/-- The permutation state: one linear combination per lane. -/
abbrev State := Fin width → LinComb

/-- Scale a combination by a field constant. -/
def scale (factor : Nat) (comb : LinComb) : LinComb :=
  comb.map (fun term => (term.1, factor * term.2 % goldilocksP))

/-- Add a round constant on the constant wire. -/
def addConstant (constant : Nat) (comb : LinComb) : LinComb :=
  (0, constant) :: comb

/-! ## Explicit parameters

Constant and matrix *values* do not affect the row count, but they appear in
row coefficients and in any semantic proof, so they are carried rather than
omitted. -/

structure Parameters where
  externalMatrix : Fin width → Fin width → Nat
  internalMatrix : Fin width → Fin width → Nat
  roundConstant : Nat → Fin width → Nat

/-- Apply a linear matrix to the state: pure combination arithmetic.

The raw matrix product concatenates via `flatMap`; `normalize` is applied
immediately so the live symbolic state has one entry per referenced column
after every layer.  This is the executable representation required by the
support recurrence: a full round resets to eight fresh columns and a partial
round grows support by at most one.  Normalization is semantics-preserving and
keeps zero coefficients until final field normalization, so support reasoning
does not silently assume no cancellation. -/
def applyMatrix (matrix : Fin width → Fin width → Nat) (state : State) : State :=
  fun target =>
    normalize <|
      (List.finRange width).flatMap
        (fun source => scale (matrix target source) (state source))

/-- Applying a matrix yields an empty row program.

**This is definitional**, since `linearLayerRows` is defined as `[]`.  It
records that the chosen normal form emits no row for a linear layer; it does
**not** prove that the carried combinations semantically implement the matrix.
That obligation belongs to `POSEIDON2-ROUND-INDUCTION`, which must relate
`lcEval z (applyMatrix matrix state lane)` to the matrix-vector product. -/
def linearLayerRows (matrix : Fin width → Fin width → Nat) (_state : State) :
    List Row := []

theorem applyMatrix_emits_no_rows
    (matrix : Fin width → Fin width → Nat) (state : State) :
    linearLayerRows matrix state = [] := rfl

/-! ## S-box over a linear combination

`x^7` by the addition chain `1 → 2 → 4 → 6 → 7`, one R1CS multiplication per
step.  The input is an arbitrary combination, so whatever the preceding linear
layer produced is consumed directly.

This is production's chain (`enforce_sbox_x7`), not the `1 → 2 → 3 → 6 → 7`
variant.  Both cost four multiplications, so the row and column counts are
identical, but production's places the input combination in **three** operand
positions rather than four.  Since a scheduled input reaches 31 columns in the
terminal block, that is `3·|c| + 9` against `4·|c| + 8` per S-box — strictly
cheaper, and it aligns the intermediate trace. -/

structure SboxFrame where
  /-- An arbitrary linear combination, not a column.  This is the repair. -/
  input : LinComb
  square : Nat
  fourth : Nat
  sixth : Nat
  output : Nat

def rowSquare (frame : SboxFrame) : Row where
  a := frame.input
  b := frame.input
  c := [(frame.square, 1)]

def rowFourth (frame : SboxFrame) : Row where
  a := [(frame.square, 1)]
  b := [(frame.square, 1)]
  c := [(frame.fourth, 1)]

def rowSixth (frame : SboxFrame) : Row where
  a := [(frame.square, 1)]
  b := [(frame.fourth, 1)]
  c := [(frame.sixth, 1)]

def rowSeventh (frame : SboxFrame) : Row where
  a := frame.input
  b := [(frame.sixth, 1)]
  c := [(frame.output, 1)]

def sboxRows (frame : SboxFrame) : List Row :=
  [rowSquare frame, rowFourth frame, rowSixth frame, rowSeventh frame]

theorem sboxRows_length (frame : SboxFrame) :
    (sboxRows frame).length = 4 := by
  simp [sboxRows]

/-- Temporaries introduced by one S-box. -/
def sboxTemporaries : Nat := 3

/-- **The S-box implements the addition chain on its input combination.**  Each
conjunct is one row read back under canonical residues, with `lcEval` of the
input combination in place of a bare column value. -/
theorem sboxRows_chain
    (frame : SboxFrame) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP)
    (satisfied : Satisfies (sboxRows frame) z) :
    z frame.square = lcEval z frame.input * lcEval z frame.input % goldilocksP ∧
      z frame.fourth = z frame.square * z frame.square % goldilocksP ∧
      z frame.sixth = z frame.square * z frame.fourth % goldilocksP ∧
      z frame.output = lcEval z frame.input * z frame.sixth % goldilocksP := by
  have squareRow := satisfied (rowSquare frame) (by simp [sboxRows])
  have fourthRow := satisfied (rowFourth frame) (by simp [sboxRows])
  have sixthRow := satisfied (rowSixth frame) (by simp [sboxRows])
  have seventhRow := satisfied (rowSeventh frame) (by simp [sboxRows])
  simp only [RowHolds, rowSquare, rowFourth, rowSixth, rowSeventh, lcEval,
    List.foldl, Nat.zero_add, Nat.one_mul,
    Nat.mod_eq_of_lt (residues frame.square),
    Nat.mod_eq_of_lt (residues frame.fourth),
    Nat.mod_eq_of_lt (residues frame.sixth),
    Nat.mod_eq_of_lt (residues frame.output)] at squareRow fourthRow sixthRow seventhRow
  exact ⟨squareRow.symm, fourthRow.symm, sixthRow.symm, seventhRow.symm⟩

/-! ## Terminal output binding

After the final linear layer the state is a vector of combinations with no
following S-box.  Exposing them as declared output ports costs one row per
lane.  The earlier version omitted this entirely. -/

def bindRow (comb : LinComb) (port : Nat) : Row where
  a := comb
  b := [(0, 1)]
  c := [(port, 1)]

def terminalBindingRows (state : State) (ports : Fin width → Nat) : List Row :=
  (List.finRange width).map (fun lane => bindRow (state lane) (ports lane))

/-- **Derived terminal binding cost:** one row per lane. -/
theorem terminalBindingRows_length (state : State) (ports : Fin width → Nat) :
    (terminalBindingRows state ports).length = width := by
  simp [terminalBindingRows, width]

/-! ## Round structure and derived cost -/

def sboxCount : Nat := externalRounds * width + partialRounds

theorem sboxCount_eq : sboxCount = 86 := by decide

/-- Nonlinear rows: four per S-box. -/
def nonlinearRows : Nat := sboxCount * 4

theorem nonlinearRows_eq : nonlinearRows = 344 := by decide

/-- Rows binding the terminal linear layer to output ports. -/
def terminalRows : Nat := width

/-- **Complete permutation row count.**  Nonlinear rows plus terminal binding.
Linear layers contribute nothing, and that is now a fact about the encoding
rather than an assumption. -/
def permutationRows : Nat := nonlinearRows + terminalRows

theorem permutationRows_eq : permutationRows = 352 := by decide

/-! ### Auxiliary columns are UNRESOLVED

`sboxTemporaries = 3` counts `square`, `fourth`, `sixth` and omits `output`.
Inside an assembled permutation `output` is not a declared permutation port: it
feeds the following symbolic linear layer, and even the final eight S-box
outputs precede the terminal matrix.  So every S-box contributes four physical
columns unless an assembled receipt proves a valid reuse or alternate ownership
scheme.

The count is therefore **not** published here.  It must be derived from the
assembled receipt, not written by hand. -/

/-- Columns one S-box frame names.  Whether all four are distinct auxiliaries
is exactly what the assembled receipt must settle. -/
def sboxFrameColumns : Nat := 4

/-! ## Derived rows, and what is only forecast

Rows are derived.  The auxiliary-column count is not. -/

/-- Nonlinear row subtotal: derived and solid. -/
theorem nonlinearRowSubtotal : nonlinearRows = 344 := nonlinearRows_eq

/-- Terminal binding rows: one per lane. -/
theorem terminalBindingForecast : terminalRows = 8 := by decide

/-- Total row forecast for the selected folding normal form.  It is a forecast
rather than a certified permutation cost because the assembled round-ordered
program, its receipts, and round-induction soundness are not yet built. -/
theorem totalRowForecast : permutationRows = 352 := permutationRows_eq

/-- The nonlinear subtotal is strictly below the row forecast, so the two
cannot be conflated as they were before this module was rebuilt. -/
theorem nonlinearRows_lt_permutationRows : nonlinearRows < permutationRows := by
  decide

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
