import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPriorLinkArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-! Generated sponge certificate for the exact recursive prior-link owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkPoseidonHashes

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288

def traces : List Trace :=
[
  { inputColumns := [867868, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 867935, 867936, 868003, 868004, 868071, 868072, 10848, 10849, 10850, 10851, 10856, 10857, 10858, 10859], zeroColumn := 868073, zeroRow := 218, rounds := [
      { kind := .absorb [867868, 10834, 10835, 10836], stateBeforeColumns := [868073, 868073, 868073, 868073, 868073, 868073, 868073, 868073], permutationInputColumns := [868074, 868075, 868076, 868077, 868073, 868073, 868073, 868073], permutationOutputColumns := [868670, 868671, 868672, 868673, 868674, 868675, 868676, 868677], definingRows := [219, 220, 221, 222], call := { rowStart := 223, rowEnd := 823, inputColumns := [868074, 868075, 868076, 868077, 868073, 868073, 868073, 868073], firstAllocatedColumn := 868078 } }
    , { kind := .absorb [10837, 10838, 10839, 10840], stateBeforeColumns := [868670, 868671, 868672, 868673, 868674, 868675, 868676, 868677], permutationInputColumns := [868678, 868679, 868680, 868681, 868674, 868675, 868676, 868677], permutationOutputColumns := [869274, 869275, 869276, 869277, 869278, 869279, 869280, 869281], definingRows := [823, 824, 825, 826], call := { rowStart := 827, rowEnd := 1427, inputColumns := [868678, 868679, 868680, 868681, 868674, 868675, 868676, 868677], firstAllocatedColumn := 868682 } }
    , { kind := .absorb [10841, 867935, 867936, 868003], stateBeforeColumns := [869274, 869275, 869276, 869277, 869278, 869279, 869280, 869281], permutationInputColumns := [869282, 869283, 869284, 869285, 869278, 869279, 869280, 869281], permutationOutputColumns := [869878, 869879, 869880, 869881, 869882, 869883, 869884, 869885], definingRows := [1427, 1428, 1429, 1430], call := { rowStart := 1431, rowEnd := 2031, inputColumns := [869282, 869283, 869284, 869285, 869278, 869279, 869280, 869281], firstAllocatedColumn := 869286 } }
    , { kind := .absorb [868004, 868071, 868072, 10848], stateBeforeColumns := [869878, 869879, 869880, 869881, 869882, 869883, 869884, 869885], permutationInputColumns := [869886, 869887, 869888, 869889, 869882, 869883, 869884, 869885], permutationOutputColumns := [870482, 870483, 870484, 870485, 870486, 870487, 870488, 870489], definingRows := [2031, 2032, 2033, 2034], call := { rowStart := 2035, rowEnd := 2635, inputColumns := [869886, 869887, 869888, 869889, 869882, 869883, 869884, 869885], firstAllocatedColumn := 869890 } }
    , { kind := .absorb [10849, 10850, 10851, 10856], stateBeforeColumns := [870482, 870483, 870484, 870485, 870486, 870487, 870488, 870489], permutationInputColumns := [870490, 870491, 870492, 870493, 870486, 870487, 870488, 870489], permutationOutputColumns := [871086, 871087, 871088, 871089, 871090, 871091, 871092, 871093], definingRows := [2635, 2636, 2637, 2638], call := { rowStart := 2639, rowEnd := 3239, inputColumns := [870490, 870491, 870492, 870493, 870486, 870487, 870488, 870489], firstAllocatedColumn := 870494 } }
    , { kind := .absorb [10857, 10858, 10859], stateBeforeColumns := [871086, 871087, 871088, 871089, 871090, 871091, 871092, 871093], permutationInputColumns := [871094, 871095, 871096, 871089, 871090, 871091, 871092, 871093], permutationOutputColumns := [871689, 871690, 871691, 871692, 871693, 871694, 871695, 871696], definingRows := [3239, 3240, 3241], call := { rowStart := 3242, rowEnd := 3842, inputColumns := [871094, 871095, 871096, 871089, 871090, 871091, 871092, 871093], firstAllocatedColumn := 871097 } }
    , { kind := .pad, stateBeforeColumns := [871689, 871690, 871691, 871692, 871693, 871694, 871695, 871696], permutationInputColumns := [871697, 871690, 871691, 871692, 871693, 871694, 871695, 871696], permutationOutputColumns := [872290, 872291, 872292, 872293, 872294, 872295, 872296, 872297], definingRows := [3842], call := { rowStart := 3843, rowEnd := 4443, inputColumns := [871697, 871690, 871691, 871692, 871693, 871694, 871695, 871696], firstAllocatedColumn := 871698 } }
    ], outputColumns := [872290, 872291, 872292, 872293] }
]

theorem traces_accepted :
traces.all (fun trace => decide (trace.Valid FPrimeFullHistoryPriorLink.rows)) = true := by
native_decide

theorem traces_valid :
∀ trace ∈ traces, trace.Valid FPrimeFullHistoryPriorLink.rows := by
intro trace member
exact of_decide_eq_true ((List.all_eq_true.mp traces_accepted) trace member)

def priorXOutTrace : Trace := traces[0]!

theorem priorXOutTrace_output :
priorXOutTrace.outputColumns = FPrimeFullHistoryPriorLink.digestColumns := by
native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkPoseidonHashes
