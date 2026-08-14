import Nightstream.Implementation.R1CS.Correspondence.Nebula.NebulaProgramBindingSound

namespace NightstreamTests.NebulaProgramBinding

open Nightstream.Implementation.R1CS.NebulaProgramBinding
open Nightstream.Implementation.R1CS.NebulaProgramBindingSound

example : rowCount = 3682 := by decide
example : definitionCount + checkCount = rowCount := by decide
example : computedBindingColumns.length = 4 := by decide

#check program_binding_sound

end NightstreamTests.NebulaProgramBinding
