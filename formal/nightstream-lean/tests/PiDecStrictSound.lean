import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictSound
import Nightstream.Implementation.R1CS.Correspondence.PiDecStrict.ShapeNecessity

/-! Regression checks for exact strict-PiDEC row soundness and its shape gates. -/

open Nightstream.Implementation.R1CS.PiDecStrictShapeNecessity

example :
    (Nightstream.Implementation.R1CS.Satisfies
        (Nightstream.Implementation.R1CS.PiDecStrictCompiler.rows
          radixThreeLayout) assignment ∧
      ¬ Nightstream.Implementation.R1CS.PiDecStrictCompiler.Accepted
        radixThreeLayout assignment) ∧
    (Nightstream.Implementation.R1CS.Satisfies
        (Nightstream.Implementation.R1CS.PiDecStrictCompiler.rows
          mismatchedPresenceLayout) assignment ∧
      ¬ Nightstream.Implementation.R1CS.PiDecStrictCompiler.Accepted
        mismatchedPresenceLayout assignment) :=
  rows_alone_do_not_imply_strict_acceptance

