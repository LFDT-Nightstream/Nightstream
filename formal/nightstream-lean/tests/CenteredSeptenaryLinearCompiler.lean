import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryLinearCompiler

set_option autoImplicit false

namespace NightstreamTests.CenteredSeptenaryLinearCompiler

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler

example {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row) (encoded : Nat → Nat) :
    Satisfies (loweredRows layout sourceRows) encoded ↔
      Satisfies sourceRows (decodedAssignment layout encoded) := by
  exact loweredRows_iff_sourceRows layout sourceRows encoded

example {fieldCount : Nat} (layout : Layout fieldCount)
    (sourceRows : List Row) {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundFour layout encoded)
    (accepted : Satisfies (loweredRows layout sourceRows) encoded) :
    Satisfies sourceRows (decodedAssignment layout encoded) := by
  exact (loweredRows_sound layout sourceRows norm accepted).1

end NightstreamTests.CenteredSeptenaryLinearCompiler
