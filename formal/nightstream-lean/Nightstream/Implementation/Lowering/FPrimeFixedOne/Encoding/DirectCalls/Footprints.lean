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

/-- Minimal selected data needed to certify only the `iterationZero` call. -/
structure IterationZeroProfile (parameters : Parameters)
    extends Encoding.Profile parameters where
  fieldLaws : FieldLaws
  inverseLaw : InverseLaw
  iterationZeroFootprint :
    parameters.footprints.iterationZero = zeroFootprint

namespace IterationZeroProfile

def family
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters) :
    Family (typeSystem parameters) :=
  profile.toProfile.family parameters

end IterationZeroProfile

/-- Minimal selected data needed to certify only encoded-value equality. -/
structure EncodedEqualProfile (parameters : Parameters)
    extends Encoding.Profile parameters where
  fieldLaws : FieldLaws
  inverseLaw : InverseLaw
  encodedEqualFootprint :
    parameters.footprints.encodedEqual =
      equalityFootprint codecs.encoded.width

namespace EncodedEqualProfile

def family
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters) :
    Family (typeSystem parameters) :=
  profile.toProfile.family parameters

end EncodedEqualProfile

/-- Minimal selected data needed to certify only the `encodeInstance` call.
This avoids making that affine recipe depend on the unrelated nonlinear
`freshPublic` encoding. -/
structure EncodeInstanceProfile (parameters : Parameters)
    extends Encoding.Profile parameters where
  encodeInstanceMap :
    AffineEncodingMap codecs.digest codecs.encoded
      parameters.machine.encodeInstance
  encodeInstanceFootprint :
    parameters.footprints.encodeInstance =
      affineFootprint codecs.encoded.width

namespace EncodeInstanceProfile

def family
    (parameters : Parameters)
    (profile : EncodeInstanceProfile parameters) :
    Family (typeSystem parameters) :=
  profile.toProfile.family parameters

end EncodeInstanceProfile

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

/-- Forget every direct-call choice except the exact zero-test slice. -/
def iterationZeroProfile
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    IterationZeroProfile parameters where
  toProfile := profile.toProfile
  fieldLaws := profile.fieldLaws
  inverseLaw := profile.inverseLaw
  iterationZeroFootprint := profile.iterationZeroFootprint

/-- Forget every direct-call choice except encoded-value equality. -/
def encodedEqualProfile
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    EncodedEqualProfile parameters where
  toProfile := profile.toProfile
  fieldLaws := profile.fieldLaws
  inverseLaw := profile.inverseLaw
  encodedEqualFootprint := profile.encodedEqualFootprint

/-- Forget every direct-call choice except the exact affine
`encodeInstance` slice. -/
def encodeInstanceProfile
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    EncodeInstanceProfile parameters where
  toProfile := profile.toProfile
  encodeInstanceMap := profile.encodeInstanceMap
  encodeInstanceFootprint := profile.encodeInstanceFootprint

end DirectProfile

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
