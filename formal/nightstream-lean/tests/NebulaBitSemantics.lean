import Nightstream.Implementation.Lowering.Nebula.BitSemantics

set_option autoImplicit false

namespace tests.NebulaBitSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.BitSemantics
open Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics

theorem physical_bit_has_exact_roots
    (assignment : Nat -> F) (column : Nat) :
    IsBit assignment column ↔
      assignment column = 0 ∨ assignment column = 1 :=
  isBit_iff_zero_or_one assignment column

theorem selected_word_does_not_wrap
    (assignment : Nat -> F) (start : Nat)
    (canonical : CanonicalBits assignment start 44) :
    (Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination.eval
      assignment
      (Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination.word
        start 44)).val = bitsValue assignment start 44 := by
  exact eval_word_val_exact assignment start 44 canonical (by decide)

end tests.NebulaBitSemantics
