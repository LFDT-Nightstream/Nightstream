import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Spec

/-!
Contract: expose native selected-CCS programs through the generic replacement
interface.

Assurance tier: model-level.

Owns: the degree-three system adapter, the proof-free native manifest
replacement, and exact manifest cost, row, and allocation preservation.

Does not own: R1CS-to-native lowering, protocol observables, a selected
application, JSON, Rust, or a security reduction.

Emits constraints: none. It adapts an existing native CCS program.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.NativeCcs

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest

universe u

private abbrev Field := Nightstream.SuperNeo.Concrete.F
abbrev Assignment := ColumnId -> Field

/-- Native selected CCS has total degree three. -/
def degree : Nat :=
  NativeCcsSelector.polynomialDegree

def system
    {Observable : Type u}
    (program : NativeCcsProgram.Program)
    (observe : Assignment -> Observable) :
    Optimization.System Assignment Observable where
  Accepts := program.Satisfies
  observe := observe
  degree := degree

def manifestSystem
    {Observable : Type u}
    (program : NativeCcsManifest.Program)
    (observe : Assignment -> Observable) :
    Optimization.System Assignment Observable where
  Accepts := program.decode.Satisfies
  observe := observe
  degree := degree

/-- Serialization through the native manifest is an exact replacement. -/
def ofProgramReplacement
    {Observable : Type u}
    (program : NativeCcsProgram.Program)
    (observe : Assignment -> Observable)
    (degreeLimit : Nat)
    (withinLimit : degree <= degreeLimit) :
    Optimization.Replacement
      (system program observe)
      (manifestSystem (NativeCcsManifest.Program.ofProgram program) observe)
      degreeLimit where
  recover := fun assignment => assignment
  derive := fun assignment => assignment
  sound := by
    intro assignment accepted
    exact
      (NativeCcsManifest.Program.decoded_program_satisfies_iff
        program assignment).mp accepted
  complete := by
    intro assignment accepted
    exact
      (NativeCcsManifest.Program.decoded_program_satisfies_iff
        program assignment).mpr accepted
  recover_observes := fun _ _ => rfl
  derive_observes := fun _ _ => rfl
  source_degree := withinLimit
  target_degree := withinLimit

theorem manifest_cost_exact
    (program : NativeCcsProgram.Program) :
    (NativeCcsManifest.Program.ofProgram program).cost = program.cost :=
  NativeCcsManifest.Program.cost_ofProgram program

theorem manifest_rows_exact
    (program : NativeCcsProgram.Program) :
    (NativeCcsManifest.Program.ofProgram program).rows.length =
      program.rows.length :=
  NativeCcsManifest.Program.rows_length_ofProgram program

theorem manifest_columns_exact
    (program : NativeCcsProgram.Program) :
    (NativeCcsManifest.Program.ofProgram program).columns =
      program.allocations :=
  NativeCcsManifest.Program.columns_ofProgram program

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.NativeCcs
