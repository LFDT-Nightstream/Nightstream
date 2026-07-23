import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Provenance
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
import Nightstream.Implementation.R1CS.Core.Program

/-!
Deterministic source-column reconstruction for the production combined-NC
selective artifact.

Owns: the executable retained-slot seed, ordered compiler-linear program,
and final-column view of compiler-introduced product-sum slots.

Does not own: generated-data validity, uniqueness of retained or derived
indices, selected-row satisfaction, rewrite soundness, source-program
semantics, transcript authority, raw-child authority, commitment binding,
costs, or row removal.

Emits constraints: none.

The generated records remain untrusted inputs here.  Later bounded artifact
certificates must prove their exact partition, SSA discipline, canonical
coefficients, and agreement with the fail-closed typed decoder.  This leaf
only gives those certificates a small deterministic interpreter.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_assignment` | Materialize generated source columns from authoritative boundary inputs and derived definitions. | direct dataflow |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceAssignment

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

/-- Rust's low-norm slot radix.  This profile currently exports width 41,
but the definition remains exact for the two supported compiler encodings. -/
def slotRadix (width : Nat) : Nat :=
  if width = 41 then 3 else 2

def slotExpansionTerms (start width : Nat) : List (Nat × Nat) :=
  (List.range width).map fun offset =>
    (start + offset, slotRadix width ^ offset % goldilocksP)

def RawSourceSlot.expansionTerms (slot : RawSourceSlot) : List (Nat × Nat) :=
  slotExpansionTerms slot.start slot.width

def RawTerm.asNatTerm (term : RawTerm) : Nat × Nat :=
  (term.column, term.coefficient)

/-- Convert the separately stored constant to the repository's standard
constant-one column without introducing a zero sparse term. -/
def RawLinearCombination.programTerms
    (linear : RawLinearCombination) : List (Nat × Nat) :=
  (if linear.constant = 0 then [] else [(0, linear.constant)]) ++
    linear.terms.map RawTerm.asNatTerm

def RawSourceDefinition.programDefinition
    (definition : RawSourceDefinition) : Program.Definition :=
  { output := definition.target
    rhs := .linear (RawLinearCombination.programTerms definition.value) }

def compilerDefinitions : List Program.Definition :=
  Provenance.linearDefinitions.map RawSourceDefinition.programDefinition

def retainedSlot? (column : Nat) : Option RawSourceSlot :=
  Provenance.retainedSlots.find? fun slot => decide (slot.column = column)

/-- Canonical candidate values for the compiler's retained source inputs.
Unknown source columns are zero until an exact partition certificate proves
that every read is either retained, constant one, or defined earlier. -/
def retainedSeed (assignment : Nat → Nat) : Nat → Nat :=
  fun column =>
    if column = 0 then
      assignment 0 % goldilocksP
    else
      match retainedSlot? column with
      | some slot => lcEval assignment (RawSourceSlot.expansionTerms slot)
      | none => 0

theorem retainedSeedCanonical (assignment : Nat → Nat) :
    ∀ column, retainedSeed assignment column < goldilocksP := by
  intro column
  unfold retainedSeed
  split
  · exact Nat.mod_lt _ (by decide)
  · split
    · unfold lcEval
      exact Nat.mod_lt _ (by decide)
    · decide

theorem retainedSeedConstantOne {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1) :
    retainedSeed assignment 0 = 1 := by
  simp [retainedSeed, constantOne, goldilocksP]

/-- Execute the exact ordered generated compiler definitions.  Artifact
soundness will supply `Program.WellFormed`; execution itself is total and
does not trust that property. -/
def compilerAssignment (assignment : Nat → Nat) : Nat → Nat :=
  Program.run (retainedSeed assignment) compilerDefinitions

theorem compilerAssignmentCanonical (assignment : Nat → Nat) :
    ∀ column, compilerAssignment assignment column < goldilocksP := by
  exact Program.run_canonical (retainedSeedCanonical assignment)

def derivedSlot? (compilerIndex : Nat) : Option RawDerivedProductSum :=
  Provenance.derivedProductSums.find? fun slot =>
    decide (slot.compilerIndex = compilerIndex)

/-- Final selective-assignment value of one compiler-introduced product-sum
slot.  Missing indices map to zero until the generated registry certificate
proves exact lookup coverage. -/
def derivedValue (assignment : Nat → Nat) (compilerIndex : Nat) : F :=
  match derivedSlot? compilerIndex with
  | none => 0
  | some slot =>
      fieldResidue (lcEval assignment (slotExpansionTerms slot.start slot.width))

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceAssignment
