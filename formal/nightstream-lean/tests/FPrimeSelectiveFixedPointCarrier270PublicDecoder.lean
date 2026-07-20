import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270

/-! Focused interface regression for the bounded fixed-point public decoder. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270PublicDecoder

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

#check PublicDecoder.generated_chunk0_exact
#check PublicDecoder.generated_chunk1_exact
#check PublicDecoder.generated_chunk_lengths
#check PublicDecoder.generated_totalColumns_exact
#check PublicDecoder.generatedCoordinate_exact
#check PublicAssignment.artifactPublicValue_exact
#check PublicAssignment.artifactPublicInput_eq_expectedPublicInput
#check PublicAssignment.artifactPublicInput_eq_projectPublicInput

example : PublicDecoder.firstChunkWidth = 256 ∧
    PublicDecoder.secondChunkWidth = 14 ∧
    PublicDecoder.alignedPublicWidth = 270 := by
  decide

example : PublicDecoder.rawChunk0.length = 256 ∧
    PublicDecoder.rawChunk1.length = 14 := by
  exact ⟨PublicDecoder.generated_chunk_lengths.1,
    PublicDecoder.generated_chunk_lengths.2.1⟩

end Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270PublicDecoder
