import Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden
import Nightstream.Implementation.Rust.NifsProductionGolden

/-! Executable checks and soundness regression for the production NIFS golden. -/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace NightstreamTests.NifsProductionGolden

open Nightstream.Implementation.R1CS.Artifacts.NifsProductionGolden
open Nightstream.Implementation.Rust.NifsProductionGolden

theorem receiptShapeChecks : receiptShapeCheck receipt = true := by native_decide
theorem crossPhaseChecks : ExecutionChecker.crossPhaseCheck receipt = true := by
  native_decide
theorem piCcsChecks : PiCcsChecker.checkReceipt receipt = true := by native_decide
theorem piRlcChecks : PiRlcChecker.checkReceipt receipt = true := by native_decide
theorem piDecChecks : PiDecChecker.checkReceipt receipt = true := by native_decide

theorem executionChecks : ExecutionChecker.checkReceipt receipt = true := by
  native_decide

example : ExecutionChecker.PaperExecution.Accepts receipt :=
  ExecutionChecker.checkReceipt_sound receipt executionChecks

end NightstreamTests.NifsProductionGolden
