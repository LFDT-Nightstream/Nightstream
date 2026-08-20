import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeafImages
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeafRows0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeafRows1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeafSteps

/-!
Contract: facade for the exact first recursive-terminal XOut Poseidon2 leaf.

Rust owns the source steps and final selective-row ports. This facade only
joins the two bounded row shards.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeaf

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

abbrev rawSteps : List RawStep :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf.rawSteps
abbrev rawRowHead : List RawRow :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf.rawRows0
abbrev rawRowTail : List RawRow :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf.rawRows1
abbrev rawImages : List RawSourceImage :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf.rawImages
def rawRows : List RawRow := rawRowHead ++ rawRowTail

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeaf
