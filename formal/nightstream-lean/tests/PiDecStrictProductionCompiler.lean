import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.PaperBridge

/-! Focused interface regression for the reduced production strict-`PiDEC`
compiler and its typed paper bridge. -/

namespace Nightstream.Tests.PiDecStrictProductionCompiler

open Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler

#check Accepted.childXExact
#check sound_noAdv
#check complete_noAdv
#check canonicalX_saving
#check combined_source_saving

#check PaperBridge.commitmentEquation
#check PaperBridge.evaluationEquation
#check PaperBridge.accepted_refines_typed
#check PaperBridge.accepted_refines_paper
#check PaperBridge.active_source_rows_saved_3500

end Nightstream.Tests.PiDecStrictProductionCompiler
