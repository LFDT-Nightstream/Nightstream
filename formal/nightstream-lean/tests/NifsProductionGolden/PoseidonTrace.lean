import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden
import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import Nightstream.Implementation.Rust.NifsProductionGolden.Poseidon2Trace
import Nightstream.Implementation.Rust.PiCcsExecution.Checker

/-! Executable certificate for the first production transcript permutation. -/

set_option autoImplicit false

namespace NightstreamTests.NifsProductionGolden.PoseidonTrace

open Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.Rust.NifsProductionGolden
open Nightstream.Implementation.Rust.NifsProductionGolden.Poseidon2Trace
open Nightstream.Implementation.Rust.PiCcsExecution

def firstPermutationInput : Values :=
  (absorbFields (receipt.piCcsStatement.publicFields.take 2)
    (initialTranscript receipt.piCcsStatement)).lanes

def firstTrace : RawPermutationTrace :=
  receipt.poseidonPermutationTraces.getD 0 { states := [] }

theorem firstTraceChecks :
    check Poseidon2CanonicalConstants.selected firstPermutationInput firstTrace = true := by
  native_decide

example : output firstTrace =
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference.referencePermutation
      Poseidon2CanonicalConstants.selected firstPermutationInput :=
  output_eq_reference _ _ _ firstTraceChecks

end NightstreamTests.NifsProductionGolden.PoseidonTrace
