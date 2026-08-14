import Nightstream.Implementation.Nebula.Commitment.Compact.AjtaiStageRows

set_option autoImplicit false

namespace tests.NebulaCompactAjtaiStageRows

open Nightstream.Implementation.Nebula.CompactAjtaiStageRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.CompactCommit

example
    (setup : SeededAjtai.Setup primaryRank primaryMessageRingColumns)
    (layout : Layout commitmentFieldCount primaryRank) :
    (rows setup primaryPacking layout).length = 120636 := by
  simpa [commitmentFieldCount, primaryRank, ringDegree] using
    rows_length setup primaryPacking layout

example
    (setup : SeededAjtai.Setup shortRank shortMessageRingColumns)
    (layout : Layout primaryOutputFieldCount shortRank) :
    (rows setup shortPacking layout).length = 13446 := by
  simpa [primaryOutputFieldCount, shortRank, ringDegree] using
    rows_length setup shortPacking layout

example
    {verifierRows messageColumns : Nat}
    (setup : SeededAjtai.Setup verifierRows messageColumns)
    (output : Fin (verifierRows * ringDegree))
    (entry : Entry messageColumns) :
    coefficient setup output entry =
      (Nightstream.SuperNeo.Concrete.ringFMul
        (setup.verifierKey (outputPair output).1 entry.1)
        (Nightstream.SuperNeo.Concrete.ringFMonomial entry.2.val 1)
        (outputPair output).2).val :=
  coefficient_eq_seeded_phi81_basis_action setup output entry

end tests.NebulaCompactAjtaiStageRows
