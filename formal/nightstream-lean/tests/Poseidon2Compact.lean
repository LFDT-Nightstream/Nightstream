import Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact

namespace Nightstream.Tests.Poseidon2Compact

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact

example :
    (activeColumns
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.canonicalLayout).length =
      103 :=
  canonical_activeColumns_exact.1

end Nightstream.Tests.Poseidon2Compact
