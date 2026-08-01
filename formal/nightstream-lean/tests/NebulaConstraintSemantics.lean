import Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics

set_option autoImplicit false

namespace tests.NebulaConstraintSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics

/-- The selected 42-times-6 profile uses the same exact correspondence as
the generic stackless compiler. -/
theorem selected_rows_iff_constraints (assignment : Nat → F) :
    assignment 0 = 1 ∧ Satisfies (rows wasm42x6) assignment ↔
      Accepted assignment wasm42x6 :=
  satisfies_iff_accepted assignment wasm42x6

end tests.NebulaConstraintSemantics
