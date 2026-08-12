import Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound

/-! Regression surface for the exact row-derived V2 paper NIFS result. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductNifsPaperRowsSound

#check Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.parentBundle_decode_eq
#check Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.parentEvaluation_decode_eq
#check Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.piDecPlacement_of_parentFields
#check Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.rows_imply_exact_result

end tests.NebulaV2ProductNifsPaperRowsSound
