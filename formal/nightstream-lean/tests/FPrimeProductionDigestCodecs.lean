import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs

/-!
Focused elaboration boundary for the production digest codecs and the compact
fixed-one `encodeInstance` affine map.
-/

namespace NightstreamTests.FPrimeProductionDigestCodecs

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs

#check digestCodec
#check digestCodec_encode_exact
#check digestCodec_roundtrip
#check optionalDigestCodec
#check optionalDigestCodec_encode_none
#check optionalDigestCodec_encode_some
#check optionalDigestCodec_roundtrip
#check adapterEncodedCodec
#check adapterEncodedCodec_roundtrip
#check encodeInstanceAffineMap
#check encodeInstance_coordinates_exact

end NightstreamTests.FPrimeProductionDigestCodecs
