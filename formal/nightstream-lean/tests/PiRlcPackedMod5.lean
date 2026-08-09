import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5

/-! Public theorem checks for the packed Mod-5 sampler leaf. -/

namespace tests.PiRlcPackedMod5

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5

#check quadraticZeroPair_iff
#check packedRows_iff_directRows
#check PackedMod5Artifact.SourceRole.column_injective
#check PackedMod5Artifact.DecoderAtom.column_injective
#check generated_shape_exact
#check generated_polynomial_degrees_exact
#check generated_polynomial_degree_at_most_eight
#check generated_source_rows_exact
#check generatedSourceAccepts_iff_candidateZero
#check generated_bit_polynomial
#check generated_residue_polynomial
#check witnessOfCoordinates_low_eq_source
#check generatedHighDecoder_shape
#check generatedHighDecoder_fieldTerms_exact
#check highFormulaFieldTerms_eval_derived
#check generatedHighDecoderRhs_eq_derived
#check generatedHighDecoder_output_eq_derived

end tests.PiRlcPackedMod5
