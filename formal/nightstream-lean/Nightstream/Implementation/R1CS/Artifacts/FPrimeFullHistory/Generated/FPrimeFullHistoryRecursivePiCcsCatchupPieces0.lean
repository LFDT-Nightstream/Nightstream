import Nightstream.Implementation.R1CS.Ownership.Core.OwnerCertificate
import Nightstream.Implementation.R1CS.Ownership.AlphabetSampling.AlphabetSamplingResidualTemplate
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Artifact

/-! Generated exact ordered owner pieces, shard 0. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiCcsCatchup.Generated

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

def pieces0 : List Piece :=
  [{ rowStart := 273040, rowEnd := 273041, payload := .ordinary [⟨[(271745, 1), (0, 18446744069414584320)], [(0, 1)], []⟩] },
   { rowStart := 273041, rowEnd := 273641, payload := .poseidon { rowStart := 1, rowEnd := 601, inputColumns := [271745, 269767, 269768, 269769, 269770, 269771, 269772, 269773], firstAllocatedColumn := 271746 } },
   { rowStart := 273641, rowEnd := 273649, payload := .ordinary [⟨[(272338, 1), (271741, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(272339, 1), (271742, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(272340, 1), (271743, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(272341, 1), (271744, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(33310, 1), (271741, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(33311, 1), (271742, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(33312, 1), (271743, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(33313, 1), (271744, 18446744069414584320)], [(0, 1)], []⟩] }]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiCcsCatchup.Generated
