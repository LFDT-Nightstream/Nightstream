import Nightstream.Implementation.R1CS.Ownership.Core.OwnerCertificate
import Nightstream.Implementation.R1CS.Ownership.AlphabetSampling.AlphabetSamplingResidualTemplate
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Artifact

/-! Generated exact ordered owner pieces, shard 0. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiCcsCatchup.Generated

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def pieces0 : List Piece :=
  [{ rowStart := 270647, rowEnd := 270648, payload := .ordinary [⟨[(269330, 1), (0, 18446744069414584320)], [(0, 1)], []⟩] },
   { rowStart := 270648, rowEnd := 271248, payload := .poseidon { rowStart := 1, rowEnd := 601, inputColumns := [269330, 267352, 267353, 267354, 267355, 267356, 267357, 267358], firstAllocatedColumn := 269331 } },
   { rowStart := 271248, rowEnd := 271256, payload := .ordinary [⟨[(269923, 1), (269326, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(269924, 1), (269327, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(269925, 1), (269328, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(269926, 1), (269329, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(33310, 1), (269326, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(33311, 1), (269327, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(33312, 1), (269328, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(33313, 1), (269329, 18446744069414584320)], [(0, 1)], []⟩] }]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiCcsCatchup.Generated
