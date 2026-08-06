import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConstruction2Encoding

/-! Focused interface gate for the selected Construction 2 instance encoding. -/

set_option autoImplicit false

namespace tests.PaddedRowIdentityConstruction2Encoding

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConstruction2Encoding

#check encHash_affine
#check encHash_digestColumn
#check encHash_padding
#check encHash_injective
#check freshLinked_iff

example : relationShape.publicWidth = 270 :=
  relationShape_publicWidth

example (lane : Fin 4) (bit : Fin 64) :
    (digestColumn lane bit).val = 1 + bit.val + 64 * lane.val :=
  digestColumn_val lane bit

end tests.PaddedRowIdentityConstruction2Encoding
