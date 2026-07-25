import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeEncodingCanonicalBits

/-!
Focused elaboration boundary for canonical bit recovery from encoding rows.
-/

namespace NightstreamTests.FPrimeEncodingCanonicalBits

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeEncoding
open Nightstream.Implementation.R1CS.FPrimeEncodingCanonicalBits
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

example {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (holds : FPrimeEncodingSound.Holds z)
    (lane : Fin 4) (bit : Fin 64) :
    z (publicBitCol lane.val bit.val) =
      CanonicalPlainCarrierLink.encodedBit
        (digestOfAssignment z canonical) lane bit :=
  publicBit_eq_encodedBit canonical holds lane bit

end NightstreamTests.FPrimeEncodingCanonicalBits
