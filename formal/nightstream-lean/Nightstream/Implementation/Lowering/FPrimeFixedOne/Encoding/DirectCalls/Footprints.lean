import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Profile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.AffineMap
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Equality
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Zero

/-!
Contract: codec-derived footprints and semantic coordinate maps for the five
open direct fixed-one calls.

Footprints are computed from the actual row programs:
- zero test: three rows, two one-coordinate temporaries;
- vector equality: two coordinate rows, a `width - 1` product chain, one
  final row, and exactly the matching inverse/flag/product temporaries;
- affine encoders: one row per target coordinate and no temporaries.

The equalities back to `Vocabulary.Parameters` are conformance requirements;
the editable footprint fields are never used to choose these costs.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

def zeroFootprint : CallFootprint where
  recurringRows := 3
  temporaries := [auxiliaryLayout 1, auxiliaryLayout 1]

def equalityFootprint (width : Nat) : CallFootprint where
  recurringRows := 2 * width + width.pred + 1
  temporaries :=
    [auxiliaryLayout width, auxiliaryLayout width,
      auxiliaryLayout width.pred]

def affineFootprint (targetWidth : Nat) : CallFootprint where
  recurringRows := targetWidth
  temporaries := []

/-- Exact direct-call encoding profile.  It adds only executable coordinate
maps, field laws, and equalities from the legacy parameter record back to
costs computed above. -/
structure DirectProfile (parameters : Parameters)
    extends Encoding.Profile parameters where
  fieldLaws : FieldLaws
  inverseLaw : InverseLaw
  freshPublicMap :
    AffineEncodingMap codecs.fresh codecs.encoded
      parameters.machine.freshPublic
  encodeInstanceMap :
    AffineEncodingMap codecs.digest codecs.encoded
      parameters.machine.encodeInstance
  iterationZeroFootprint :
    parameters.footprints.iterationZero = zeroFootprint
  stateEqualFootprint :
    parameters.footprints.stateEqual =
      equalityFootprint codecs.state.width
  freshPublicFootprint :
    parameters.footprints.freshPublic =
      affineFootprint codecs.encoded.width
  encodeInstanceFootprint :
    parameters.footprints.encodeInstance =
      affineFootprint codecs.encoded.width
  encodedEqualFootprint :
    parameters.footprints.encodedEqual =
      equalityFootprint codecs.encoded.width

namespace DirectProfile

def family
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    Family (typeSystem parameters) :=
  profile.toProfile.family parameters

end DirectProfile

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
