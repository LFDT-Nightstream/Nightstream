import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCanonicalX

/-! Focused interface regression for the model-level common-sign R1CS family. -/

namespace Nightstream.Tests.PiDecStrictCanonicalX

open Nightstream.Implementation.R1CS.PiDecStrictCanonicalX

#check digitInstruction_sound
#check digitInstruction_complete
#check decodedRecomposition_of_recomposes
#check canonicality_sound
#check rows_sound
#check rows_force_splitScalar
#check materializedSign_complete
#check honest_complete_rows

example : materializedSign (0 : Nightstream.SuperNeo.Concrete.F) = 0 := by
  exact materializedSign_zero

example (layout : Layout) :
    (Nightstream.Implementation.R1CS.CheckedProgram.rows
      (canonicalityInstructions layout)).length = 16 := by
  exact canonicality_rows_exact layout

example (layout : Layout) : (rows layout).length = 17 := by
  exact total_rows_exact layout

example : 28 - 16 = 12 /\ 29 - 17 = 12 := by
  exact exact_saving

end Nightstream.Tests.PiDecStrictCanonicalX
