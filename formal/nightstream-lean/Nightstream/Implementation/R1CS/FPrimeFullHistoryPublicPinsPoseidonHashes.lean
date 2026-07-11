import Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsArtifact
import Nightstream.Implementation.R1CS.Poseidon2Sponge

/-! Generated sponge certificate for the exact public-image pins. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsPoseidonHashes

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288

def xOutTrace : Trace :=
  { inputColumns := [3207985, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 3207986, 3207987, 3207988, 3207989, 3207990, 3207991, 10864, 10865, 10866, 10867, 3207977, 3207978, 3207979, 3207980], zeroColumn := 3207992, zeroRow := 50, rounds := [
      { kind := .absorb [3207985, 10834, 10835, 10836], stateBeforeColumns := [3207992, 3207992, 3207992, 3207992, 3207992, 3207992, 3207992, 3207992], permutationInputColumns := [3207993, 3207994, 3207995, 3207996, 3207992, 3207992, 3207992, 3207992], permutationOutputColumns := [3208589, 3208590, 3208591, 3208592, 3208593, 3208594, 3208595, 3208596], definingRows := [51, 52, 53, 54], call := { rowStart := 55, rowEnd := 655, inputColumns := [3207993, 3207994, 3207995, 3207996, 3207992, 3207992, 3207992, 3207992], firstAllocatedColumn := 3207997 } }
    , { kind := .absorb [10837, 10838, 10839, 10840], stateBeforeColumns := [3208589, 3208590, 3208591, 3208592, 3208593, 3208594, 3208595, 3208596], permutationInputColumns := [3208597, 3208598, 3208599, 3208600, 3208593, 3208594, 3208595, 3208596], permutationOutputColumns := [3209193, 3209194, 3209195, 3209196, 3209197, 3209198, 3209199, 3209200], definingRows := [655, 656, 657, 658], call := { rowStart := 659, rowEnd := 1259, inputColumns := [3208597, 3208598, 3208599, 3208600, 3208593, 3208594, 3208595, 3208596], firstAllocatedColumn := 3208601 } }
    , { kind := .absorb [10841, 3207986, 3207987, 3207988], stateBeforeColumns := [3209193, 3209194, 3209195, 3209196, 3209197, 3209198, 3209199, 3209200], permutationInputColumns := [3209201, 3209202, 3209203, 3209204, 3209197, 3209198, 3209199, 3209200], permutationOutputColumns := [3209797, 3209798, 3209799, 3209800, 3209801, 3209802, 3209803, 3209804], definingRows := [1259, 1260, 1261, 1262], call := { rowStart := 1263, rowEnd := 1863, inputColumns := [3209201, 3209202, 3209203, 3209204, 3209197, 3209198, 3209199, 3209200], firstAllocatedColumn := 3209205 } }
    , { kind := .absorb [3207989, 3207990, 3207991, 10864], stateBeforeColumns := [3209797, 3209798, 3209799, 3209800, 3209801, 3209802, 3209803, 3209804], permutationInputColumns := [3209805, 3209806, 3209807, 3209808, 3209801, 3209802, 3209803, 3209804], permutationOutputColumns := [3210401, 3210402, 3210403, 3210404, 3210405, 3210406, 3210407, 3210408], definingRows := [1863, 1864, 1865, 1866], call := { rowStart := 1867, rowEnd := 2467, inputColumns := [3209805, 3209806, 3209807, 3209808, 3209801, 3209802, 3209803, 3209804], firstAllocatedColumn := 3209809 } }
    , { kind := .absorb [10865, 10866, 10867, 3207977], stateBeforeColumns := [3210401, 3210402, 3210403, 3210404, 3210405, 3210406, 3210407, 3210408], permutationInputColumns := [3210409, 3210410, 3210411, 3210412, 3210405, 3210406, 3210407, 3210408], permutationOutputColumns := [3211005, 3211006, 3211007, 3211008, 3211009, 3211010, 3211011, 3211012], definingRows := [2467, 2468, 2469, 2470], call := { rowStart := 2471, rowEnd := 3071, inputColumns := [3210409, 3210410, 3210411, 3210412, 3210405, 3210406, 3210407, 3210408], firstAllocatedColumn := 3210413 } }
    , { kind := .absorb [3207978, 3207979, 3207980], stateBeforeColumns := [3211005, 3211006, 3211007, 3211008, 3211009, 3211010, 3211011, 3211012], permutationInputColumns := [3211013, 3211014, 3211015, 3211008, 3211009, 3211010, 3211011, 3211012], permutationOutputColumns := [3211608, 3211609, 3211610, 3211611, 3211612, 3211613, 3211614, 3211615], definingRows := [3071, 3072, 3073], call := { rowStart := 3074, rowEnd := 3674, inputColumns := [3211013, 3211014, 3211015, 3211008, 3211009, 3211010, 3211011, 3211012], firstAllocatedColumn := 3211016 } }
    , { kind := .pad, stateBeforeColumns := [3211608, 3211609, 3211610, 3211611, 3211612, 3211613, 3211614, 3211615], permutationInputColumns := [3211616, 3211609, 3211610, 3211611, 3211612, 3211613, 3211614, 3211615], permutationOutputColumns := [3212209, 3212210, 3212211, 3212212, 3212213, 3212214, 3212215, 3212216], definingRows := [3674], call := { rowStart := 3675, rowEnd := 4275, inputColumns := [3211616, 3211609, 3211610, 3211611, 3211612, 3211613, 3211614, 3211615], firstAllocatedColumn := 3211617 } }
    ], outputColumns := [3212209, 3212210, 3212211, 3212212] }

theorem xOutTrace_valid :
xOutTrace.Valid FPrimeFullHistoryPublicPins.rows := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsPoseidonHashes
