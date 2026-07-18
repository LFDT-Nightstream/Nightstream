import Nightstream.Implementation.R1CS.Ownership.Core.OwnerCertificate
import Nightstream.Implementation.R1CS.Ownership.AlphabetSampling.AlphabetSamplingResidualTemplate
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Artifact

/-! Generated exact ordered owner pieces, shard 0. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def pieces0 : List Piece :=
  [{ rowStart := 351287, rowEnd := 351292, payload := .ordinary [⟨[(348840, 1), (0, 18446744069414584295)], [(0, 1)], []⟩,
      ⟨[(348841, 1), (0, 18433405428082710161)], [(0, 1)], []⟩,
      ⟨[(348842, 1), (0, 18418773092929081752)], [(0, 1)], []⟩,
      ⟨[(348843, 1), (0, 18418491622382018197)], [(0, 1)], []⟩,
      ⟨[(348844, 1), (0, 18446743569262352536)], [(0, 1)], []⟩] },
   { rowStart := 351292, rowEnd := 351892, payload := .poseidon { rowStart := 5, rowEnd := 605, inputColumns := [348840, 348841, 348842, 348843, 269927, 269928, 269929, 269930], firstAllocatedColumn := 348845 } },
   { rowStart := 351892, rowEnd := 351893, payload := .ordinary [⟨[(349445, 1), (0, 18446744069414584317)], [(0, 1)], []⟩] },
   { rowStart := 351893, rowEnd := 352493, payload := .poseidon { rowStart := 606, rowEnd := 1206, inputColumns := [348844, 349445, 348828, 348829, 349441, 349442, 349443, 349444], firstAllocatedColumn := 349446 } },
   { rowStart := 352493, rowEnd := 352496, payload := .ordinary [⟨[(350046, 1), (0, 18446744069414584319)], [(0, 1)], []⟩,
      ⟨[(350047, 1)], [(0, 1)], []⟩,
      ⟨[(350048, 1)], [(0, 1)], []⟩] },
   { rowStart := 352496, rowEnd := 353096, payload := .poseidon { rowStart := 1209, rowEnd := 1809, inputColumns := [348830, 348831, 350046, 350047, 350042, 350043, 350044, 350045], firstAllocatedColumn := 350049 } },
   { rowStart := 353096, rowEnd := 353101, payload := .ordinary [⟨[(350649, 1)], [(0, 1)], []⟩,
      ⟨[(350650, 1), (0, 18446744069414584319)], [(0, 1)], []⟩,
      ⟨[(350651, 1), (0, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(350652, 1)], [(0, 1)], []⟩,
      ⟨[(350653, 1), (0, 18446744069414584320)], [(0, 1)], []⟩] },
   { rowStart := 353101, rowEnd := 353701, payload := .poseidon { rowStart := 1814, rowEnd := 2414, inputColumns := [350048, 350650, 350651, 350652, 350645, 350646, 350647, 350648], firstAllocatedColumn := 350654 } },
   { rowStart := 353701, rowEnd := 354301, payload := .poseidon { rowStart := 2414, rowEnd := 3014, inputColumns := [350653, 351247, 351248, 351249, 351250, 351251, 351252, 351253], firstAllocatedColumn := 351254 } },
   { rowStart := 354301, rowEnd := 354370, payload := .canonicalU64 351846 351854 },
   { rowStart := 354370, rowEnd := 354474, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 351854 350649) },
   { rowStart := 354474, rowEnd := 354543, payload := .canonicalU64 351847 352012 },
   { rowStart := 354543, rowEnd := 354647, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 352012 352011) },
   { rowStart := 354647, rowEnd := 354716, payload := .canonicalU64 351848 352170 },
   { rowStart := 354716, rowEnd := 354820, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 352170 352169) },
   { rowStart := 354820, rowEnd := 354889, payload := .canonicalU64 351849 352328 },
   { rowStart := 354889, rowEnd := 354993, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 352328 352327) },
   { rowStart := 354993, rowEnd := 354997, payload := .ordinary [⟨[(352486, 1), (0, 18446744069414584319)], [(0, 1)], []⟩,
      ⟨[(352487, 1), (0, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(352488, 1), (0, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(352489, 1), (0, 18446744069414584320)], [(0, 1)], []⟩] },
   { rowStart := 354997, rowEnd := 355597, payload := .poseidon { rowStart := 3710, rowEnd := 4310, inputColumns := [352486, 352487, 352488, 352489, 351850, 351851, 351852, 351853], firstAllocatedColumn := 352490 } },
   { rowStart := 355597, rowEnd := 355666, payload := .canonicalU64 353082 353090 },
   { rowStart := 355666, rowEnd := 355770, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 353090 352485) },
   { rowStart := 355770, rowEnd := 355839, payload := .canonicalU64 353083 353248 },
   { rowStart := 355839, rowEnd := 355943, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 353248 353247) },
   { rowStart := 355943, rowEnd := 356012, payload := .canonicalU64 353084 353406 },
   { rowStart := 356012, rowEnd := 356116, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 353406 353405) },
   { rowStart := 356116, rowEnd := 356185, payload := .canonicalU64 353085 353564 },
   { rowStart := 356185, rowEnd := 356289, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 353564 353563) },
   { rowStart := 356289, rowEnd := 356293, payload := .ordinary [⟨[(353722, 1), (0, 18446744069414584319)], [(0, 1)], []⟩,
      ⟨[(353723, 1), (0, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(353724, 1), (0, 18446744069414584319)], [(0, 1)], []⟩,
      ⟨[(353725, 1), (0, 18446744069414584320)], [(0, 1)], []⟩] },
   { rowStart := 356293, rowEnd := 356893, payload := .poseidon { rowStart := 5006, rowEnd := 5606, inputColumns := [353722, 353723, 353724, 353725, 353086, 353087, 353088, 353089], firstAllocatedColumn := 353726 } },
   { rowStart := 356893, rowEnd := 356962, payload := .canonicalU64 354318 354326 },
   { rowStart := 356962, rowEnd := 357066, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 354326 353721) },
   { rowStart := 357066, rowEnd := 357135, payload := .canonicalU64 354319 354484 },
   { rowStart := 357135, rowEnd := 357239, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 354484 354483) },
   { rowStart := 357239, rowEnd := 357308, payload := .canonicalU64 354320 354642 },
   { rowStart := 357308, rowEnd := 357412, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 354642 354641) },
   { rowStart := 357412, rowEnd := 357481, payload := .canonicalU64 354321 354800 },
   { rowStart := 357481, rowEnd := 357585, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 354800 354799) },
   { rowStart := 357585, rowEnd := 357589, payload := .ordinary [⟨[(354958, 1), (0, 18446744069414584319)], [(0, 1)], []⟩,
      ⟨[(354959, 1), (0, 18446744069414584320)], [(0, 1)], []⟩,
      ⟨[(354960, 1), (0, 18446744069414584318)], [(0, 1)], []⟩,
      ⟨[(354961, 1), (0, 18446744069414584320)], [(0, 1)], []⟩] },
   { rowStart := 357589, rowEnd := 358189, payload := .poseidon { rowStart := 6302, rowEnd := 6902, inputColumns := [354958, 354959, 354960, 354961, 354322, 354323, 354324, 354325], firstAllocatedColumn := 354962 } },
   { rowStart := 358189, rowEnd := 358258, payload := .canonicalU64 355554 355562 },
   { rowStart := 358258, rowEnd := 358362, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 355562 354957) },
   { rowStart := 358362, rowEnd := 358431, payload := .canonicalU64 355555 355720 },
   { rowStart := 358431, rowEnd := 358535, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 355720 355719) },
   { rowStart := 358535, rowEnd := 358604, payload := .canonicalU64 355556 355878 },
   { rowStart := 358604, rowEnd := 358708, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 355878 355877) },
   { rowStart := 358708, rowEnd := 358777, payload := .canonicalU64 355557 356036 },
   { rowStart := 358777, rowEnd := 358881, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows 356036 356035) },
   { rowStart := 358881, rowEnd := 361480, payload := .ordinary (AlphabetSamplingResidualTemplate.tailRows [351854, 352012, 352170, 352328, 353090, 353248, 353406, 353564, 354326, 354484, 354642, 354800, 355562, 355720, 355878, 356036] 356194) }]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated
