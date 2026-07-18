import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Sis.Refinement

/-!
Public theorem-shape regression for generic flattened `Pi_CCS` output SIS
refinement.
-/

namespace tests.PiCcsOutputSisRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis

example
    {block : SeededPhi81.Block}
    {fields : List Nat}
    {assignment : Nat -> Nat}
    (valid : block.Valid)
    (holds : block.Holds assignment)
    (agreement : Refinement.WordAgreement block fields assignment) :
    block.outputColumns.map assignment =
      Semantics.apply (Refinement.mapOfBlock block) fields := by
  exact Refinement.outputs_eq_apply valid holds agreement

end tests.PiCcsOutputSisRefinement
