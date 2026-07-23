import Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction

/-!
Focused interface regression for operational paper `Pi_RLC` coordinate-fork
extraction.
-/

namespace tests.PiRLCPaperForkExtraction

open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction

#check InputBatch
#check Response
#check Response.output
#check Response.Success
#check StrongSetUnits
#check LinearMapLaws
#check ExtractionAlgebra
#check CompleteFork
#check CompleteFork.coordinateUnit
#check extractedAssignment
#check completeFork_implies_correctedAmbientHolds

end tests.PiRLCPaperForkExtraction
