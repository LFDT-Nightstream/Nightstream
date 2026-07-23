import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.PaperBridge
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.ArtifactRows
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.ArtifactSemantics

/-! Focused interface regression for the reduced production strict-`PiDEC`
compiler and its typed paper bridge. -/

namespace Nightstream.Tests.PiDecStrictProductionCompiler

open Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler

#check Accepted.childXExact
#check UniformXAccepted.childXExact
#check uniformXRows_sound
#check uniformXRows_complete
#check uniformXRows_count
#check sound_noAdv
#check complete_noAdv
#check canonicalX_saving
#check combined_source_saving

#check PaperBridge.commitmentEquation
#check PaperBridge.evaluationEquation
#check PaperBridge.accepted_refines_typed
#check PaperBridge.accepted_refines_paper
#check PaperBridge.active_logicalXCount_270
#check PaperBridge.active_uniformXRows_count_4590
#check PaperBridge.active_source_rows_saved_3500

#check ArtifactRows.coefficients_exact
#check ArtifactRows.ownership_exact
#check ArtifactRows.physicalIndices_exact
#check ArtifactRows.physicalIndices_unique
#check ArtifactRows.physical_owner_partition

#check ArtifactSemantics.RowsSatisfied
#check ArtifactSemantics.rows_sound
#check ArtifactSemantics.rows_complete

end Nightstream.Tests.PiDecStrictProductionCompiler
