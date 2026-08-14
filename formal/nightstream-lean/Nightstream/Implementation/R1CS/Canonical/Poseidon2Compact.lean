import Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest

/-!
Contract: the compact degree-seven Poseidon2 relation used after selective
lowering.

Owns: the exact 86 S-box-output equations, the eight final output bindings,
their end-to-end soundness against the independent reference permutation, an
honest witness, and the exact 103-column logical carrier.

Does not own: Rust trace correspondence, selector authority, outer low-norm
encoding, a production call manifest, or Poseidon2 collision security.

Emits constraints: no. This file proves that the compact relation preserves
the semantic authority of the 352-row canonical R1CS program while it omits
the three intermediate multiplication columns of every S-box.

Assurance tier: model-level.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest

/-- The compact relation keeps only each seventh-power output and the final
eight bindings. Linear round state remains an expression, not a witness
column. -/
structure Holds (layout : Layout) (constants : Constants)
    (assignment : Nat → Nat) : Prop where
  constantWire : assignment 0 = 1
  sboxChain : SboxChain layout constants assignment
  outputBinding : ∀ lane : Fin width,
    assignment (layout.outputPort lane) =
      lcEval assignment (finalState layout lane)

/-- The compact relation has the same sound output as the full canonical
program. No omitted square, fourth-power, or sixth-power column is needed by
the round induction. -/
theorem computes_reference
    (layout : Layout) (constants : Constants) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (holds : Holds layout constants assignment)
    (lane : Fin width) :
    assignment (layout.outputPort lane) =
      referencePermutation constants (inputValues layout assignment) lane := by
  rw [holds.outputBinding lane]
  exact terminalState_eval layout constants assignment canonical
    holds.constantWire holds.sboxChain halfFullRounds (Nat.le_refl _) lane

/-- Satisfaction of the full canonical program implies the compact relation.
This direction proves that compact execution does not reject a source witness. -/
theorem of_canonical_satisfies
    (layout : Layout) (constants : Constants) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (canonicalProgram layout constants) assignment) :
    Holds layout constants assignment := by
  refine ⟨constantWire,
    satisfies_sboxChain layout constants assignment canonical satisfied, ?_⟩
  intro lane
  have bindingRow :
      RowHolds assignment
        (bindRow (finalState layout lane) (layout.outputPort lane)) := by
    refine satisfied _ (List.mem_append.2 (Or.inr ?_))
    exact List.mem_map.2 ⟨lane, List.mem_finRange lane, rfl⟩
  simp only [RowHolds, bindRow] at bindingRow
  rw [lcEval_singleton assignment 0 (canonical 0),
    lcEval_singleton assignment _ (canonical _), constantWire,
    Nat.mul_one] at bindingRow
  have combinationCanonical :
      lcEval assignment (finalState layout lane) < goldilocksP :=
    Nat.mod_lt _ (by decide)
  rw [Nat.mod_eq_of_lt combinationCanonical] at bindingRow
  exact bindingRow.symm

/-- A canonical input has a compact satisfying witness. This is the
completeness half for the compact relation. -/
theorem honest_holds
    (constants : Constants) (input : Values)
    (inputCanonical : ∀ lane, input lane < goldilocksP) :
    Holds canonicalLayout constants (honestAssignment constants input) := by
  apply of_canonical_satisfies canonicalLayout constants
    (honestAssignment constants input)
  · exact honest_residues constants input inputCanonical
  · exact honest_constantWire constants input
  · exact honest_satisfies constants input inputCanonical

/-- Logical columns retained by the compact relation: constant one, eight
inputs, 86 S-box outputs, and eight declared outputs. -/
def activeColumns (layout : Layout) : List Nat :=
  [0] ++
    (List.finRange width).map layout.inputPort ++
    ((List.finRange sboxCount).map fun index => sboxOutput layout index.val) ++
    (List.finRange width).map layout.outputPort

theorem activeColumns_length (layout : Layout) :
    (activeColumns layout).length = 1 + width + sboxCount + width := by
  simp [activeColumns]
  omega

/-- The canonical compact carrier uses 103 logical columns. The ordinary
canonical R1CS carrier uses 361; the production source trace uses 609. -/
theorem canonical_activeColumns_exact :
    (activeColumns canonicalLayout).length = 103 ∧
      (activeColumns canonicalLayout).Nodup := by
  native_decide

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact
