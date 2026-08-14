import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.GroupedProduct

namespace Tests.SelectiveCcsGroupedProduct

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct

private def specs : List StepSpec :=
  [ { base := 3, products := [{ left := 4, right := 5 }] }
  , { base := 6, products := [{ left := 7, right := 8 }] }
  ]

example : ChainHolds 2 (compile 2 specs) :=
  compile_chainHolds 2 specs

example :
    finalValue 2 (compile 2 specs) =
      2 + totalContribution (compile 2 specs) :=
  chainHolds_sound 2 (compile 2 specs) (compile_chainHolds 2 specs)

example :
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
        (Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluationPoint
          1 2 3 4 5 6 7 8 9 10 11 (79 - 12 - 13)) = 0 ↔
      (79 : F) = 13 + (12 + fiveProductSum 2 3 4 5 6 7 8 9 10 11) :=
  evaluationRow_zero_iff_stepHolds 12 13 2 3 4 5 6 7 8 9 10 11 79

end Tests.SelectiveCcsGroupedProduct
