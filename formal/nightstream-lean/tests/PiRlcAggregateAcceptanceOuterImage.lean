import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage

/-! Public theorem checks for the recursive aggregate-acceptance outer image. -/

namespace tests.PiRlcAggregateAcceptanceOuterImage

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage

#check booleanOwner_holds_iff
#check decodedChunkBits_are_boolean
#check activeRowsHold_iff_sourceMeaning
#check generated_outer_image_shape_exact
#check generated_decoder_tree_exact
#check generated_physical_row_tree_exact
#check generated_source_row_tree_exact

end tests.PiRlcAggregateAcceptanceOuterImage
