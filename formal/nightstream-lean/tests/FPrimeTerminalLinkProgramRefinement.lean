import Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

/-!
Focused elaboration boundary for the Rust-emitted terminal-link program.
-/

namespace NightstreamTests.FPrimeTerminalLinkProgramRefinement

open Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

#check generated_plain_eq_canonical
#check generated_plain_cost
#check generated_plain_expansion
#check generated_plain_compile
#check generated_plain_accepts_iff_selectedRows
#check generated_batchCost_eq_rowCount
#check generated_program_exact_row_ownership

example :
    batchCost generatedPlain 3 = 810 :=
  generated_batchCost_eq_rowCount 3

example :
    compile generatedPlain 3 = some (rows 3) :=
  generated_plain_compile 3

example (assignment : Nat -> Nat) :
    Accepts generatedPlain 3 assignment <->
      Nightstream.Implementation.R1CS.Satisfies (rows 3) assignment :=
  generated_plain_accepts_iff_selectedRows 3 assignment

end NightstreamTests.FPrimeTerminalLinkProgramRefinement
