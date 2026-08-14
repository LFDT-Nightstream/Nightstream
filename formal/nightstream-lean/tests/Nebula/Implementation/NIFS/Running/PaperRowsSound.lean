import Nightstream.Implementation.Nebula.NIFS.Running.PaperRowsSound

/-! Regression surface for the exact row-derived V2 paper NIFS result. -/

set_option autoImplicit false

namespace tests.NebulaProductNifsPaperRowsSound

#check Nightstream.Implementation.Nebula.ProductNifsPaperRowsSound.parentBundle_decode_eq
#check Nightstream.Implementation.Nebula.ProductNifsPaperRowsSound.parentEvaluation_decode_eq
#check Nightstream.Implementation.Nebula.ProductNifsPaperRowsSound.piDecPlacement_of_parentFields
#check Nightstream.Implementation.Nebula.ProductNifsPaperRowsSound.rows_imply_exact_result

end tests.NebulaProductNifsPaperRowsSound
