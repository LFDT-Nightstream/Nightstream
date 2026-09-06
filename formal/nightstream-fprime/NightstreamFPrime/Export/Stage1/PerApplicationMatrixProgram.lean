import NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
import NightstreamFPrime.Export.Stage1.NextPreimageMatrixProgram
import NightstreamFPrime.Export.Stage1.PerApplicationProductionPlan
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram
import NightstreamFPrime.Export.Stage1.PiDECMatrixProgram
import NightstreamFPrime.Export.Stage1.PiRLCMatrixProgram
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixProgram
import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgram
import NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram
import NightstreamFPrime.Export.Stage1.PilotPoseidonMatrixProgram
import NightstreamFPrime.Export.Stage1.PinMatrixPrograms
import NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram

/-!
Owns the exact compact 14-matrix row program for one Lean-authored
application. Its thirteen children use the same order and geometry projections
as `PerApplicationProductionPlan.canonical`.

This module selects no package bytes, verification key, or Rust consumer.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgram

open NightstreamFPrime.Export.MatrixProgram

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program

def applicationGeometry (application : ApplicationProgram) :=
  PerApplicationProductionPlan.applicationGeometry application

def samplerGeometry (application : ApplicationProgram) :=
  PerApplicationProductionPlan.samplerGeometry application

def piDecGeometry (application : ApplicationProgram) :=
  PerApplicationProductionPlan.piDecGeometry application

def poseidonGeometry (application : ApplicationProgram) :=
  DirectPiDECPrefixPlan.poseidonGeometry (piDecGeometry application)

def pilotGeometry (application : ApplicationProgram) :=
  DirectPrefixPlan.pilotGeometry (poseidonGeometry application)

def piCcsOrdinaryGeometry (application : ApplicationProgram) :=
  DirectPiDECPrefixPlan.piCcsOrdinaryGeometry (piDecGeometry application)

def pilotOrdinaryGeometry (application : ApplicationProgram) :=
  DirectPiDECPrefixPlan.pilotOrdinaryGeometry (piDecGeometry application)

def piRlcGeometry (application : ApplicationProgram) :=
  DirectPrefixPlan.prefixGeometry (poseidonGeometry application)

def runningGeometry (application : ApplicationProgram) :=
  DirectPiDECPrefixPlan.runningGeometry (piDecGeometry application)

def pilotPoseidonProgram (application : ApplicationProgram) : Program :=
  PilotPoseidonMatrixProgram.matrixProgram (pilotGeometry application)

def piCcsPoseidonProgram (application : ApplicationProgram) : Program :=
  PiCCSPoseidonMatrixProgram.matrixProgram (piCcsOrdinaryGeometry application)

def piCcsOrdinaryProgram (application : ApplicationProgram) : Program :=
  PiCCSOrdinaryMatrixProgram.matrixProgram
    (piCcsOrdinaryGeometry application)

def pilotOrdinaryProgram (application : ApplicationProgram) : Program :=
  PilotOrdinaryMatrixProgram.matrixProgram (pilotOrdinaryGeometry application)

def pilotDigestBindingProgram (application : ApplicationProgram) : Program :=
  PinMatrixPrograms.pilotDigestBindingProgram (piDecGeometry application)

def piCcsEndpointProgram (application : ApplicationProgram) : Program :=
  PinMatrixPrograms.piCcsEndpointProgram (piDecGeometry application)

def samplerPoseidonProgram (application : ApplicationProgram) : Program :=
  PiRLCSamplerPoseidonMatrixProgram.matrixProgram
    (poseidonGeometry application)

def samplerOrdinaryProgram (application : ApplicationProgram) : Program :=
  PiRLCSamplerOrdinaryMatrixProgram.matrixProgram
    (samplerGeometry application)

def piRlcProgram (application : ApplicationProgram) : Program :=
  PiRLCMatrixProgram.matrixProgram (piRlcGeometry application)

def piDecProgram (application : ApplicationProgram) : Program :=
  PiDECMatrixProgram.matrixProgram (piDecGeometry application)

def runningTransitionProgram (application : ApplicationProgram) : Program :=
  RunningTransitionMatrixProgram.matrixProgram (runningGeometry application)

def applicationProgram (application : ApplicationProgram) : Program :=
  ApplicationMatrixProgram.matrixProgram (applicationGeometry application)

def nextPreimageProgram (application : ApplicationProgram) : Program :=
  NextPreimageMatrixProgram.matrixProgram
    (piCcsOrdinaryGeometry application)

def recursivePublicOutputProgram (application : ApplicationProgram) : Program :=
  PinMatrixPrograms.recursivePublicOutputProgram
    (applicationGeometry application)

/-- Compact row program selected by one semantic block opcode. -/
def blockProgram (application : ApplicationProgram) :
    PerApplicationProductionPlan.BlockKind → Program
  | .pilotPoseidon => pilotPoseidonProgram application
  | .piCcsPoseidon => piCcsPoseidonProgram application
  | .piCcsOrdinary => piCcsOrdinaryProgram application
  | .pilotOrdinary => pilotOrdinaryProgram application
  | .pilotDigestBinding => pilotDigestBindingProgram application
  | .piCcsEndpoint => piCcsEndpointProgram application
  | .samplerPoseidon => samplerPoseidonProgram application
  | .samplerOrdinary => samplerOrdinaryProgram application
  | .piRlc => piRlcProgram application
  | .piDec => piDecProgram application
  | .runningTransition => runningTransitionProgram application
  | .application => applicationProgram application
  | .nextPreimage => nextPreimageProgram application
  | .recursivePublicOutput => recursivePublicOutputProgram application

theorem blockProgram_rowCount (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (kind : PerApplicationProductionPlan.BlockKind) :
    (blockProgram application kind).rowCount =
      (kind.plan application fits).rowCount := by
  rw [PerApplicationProductionPlan.BlockKind.plan_rowCount]
  cases kind <;>
    simp [blockProgram, PerApplicationProductionPlan.BlockKind.rowCount,
      pilotPoseidonProgram, piCcsPoseidonProgram, piCcsOrdinaryProgram,
      pilotOrdinaryProgram, pilotDigestBindingProgram, piCcsEndpointProgram,
      samplerPoseidonProgram, samplerOrdinaryProgram, piRlcProgram,
      piDecProgram, runningTransitionProgram, applicationProgram,
      nextPreimageProgram, recursivePublicOutputProgram]

/-- Interpret the same ordered tree as a compact matrix program. -/
def compileMatrix (application : ApplicationProgram) :
    PerApplicationProductionPlan.Program → Program
  | .leaf kind => blockProgram application kind
  | .append left right =>
      (compileMatrix application left).append (compileMatrix application right)

def piCcsPoseidonPrefixProgram (application : ApplicationProgram) : Program :=
  (pilotPoseidonProgram application).append (piCcsPoseidonProgram application)

def piCcsCoreProgram (application : ApplicationProgram) : Program :=
  (piCcsPoseidonPrefixProgram application).append
    (piCcsOrdinaryProgram application)

def pilotOrdinaryPrefixProgram (application : ApplicationProgram) : Program :=
  (piCcsCoreProgram application).append (pilotOrdinaryProgram application)

def pilotBindingPrefixProgram (application : ApplicationProgram) : Program :=
  (pilotOrdinaryPrefixProgram application).append
    (pilotDigestBindingProgram application)

def piCcsCompleteProgram (application : ApplicationProgram) : Program :=
  (pilotBindingPrefixProgram application).append
    (piCcsEndpointProgram application)

def samplerPrefixProgram (application : ApplicationProgram) : Program :=
  (piCcsCompleteProgram application).append
    (samplerPoseidonProgram application)

def samplerCompleteProgram (application : ApplicationProgram) : Program :=
  (samplerPrefixProgram application).append
    (samplerOrdinaryProgram application)

def piRlcCompleteProgram (application : ApplicationProgram) : Program :=
  (samplerCompleteProgram application).append (piRlcProgram application)

def piDecCompleteProgram (application : ApplicationProgram) : Program :=
  (piRlcCompleteProgram application).append (piDecProgram application)

def runningCompleteProgram (application : ApplicationProgram) : Program :=
  (piDecCompleteProgram application).append
    (runningTransitionProgram application)

/-- Running prefix followed by the selected application rows. -/
def applicationCompleteProgram (application : ApplicationProgram) : Program :=
  (runningCompleteProgram application).append (applicationProgram application)

def throughNextPreimageProgram (application : ApplicationProgram) : Program :=
  (applicationCompleteProgram application).append
    (nextPreimageProgram application)

/-- The one compact matrix program for the selected application. -/
def matrixProgram (application : ApplicationProgram) : Program :=
  (throughNextPreimageProgram application).append
    (recursivePublicOutputProgram application)

@[simp] theorem compileMatrix_canonical (application : ApplicationProgram) :
    compileMatrix application PerApplicationProductionPlan.canonical =
      matrixProgram application := by
  rfl

/-- Named child programs paired with the canonical semantic opcode order. -/
def children (application : ApplicationProgram) :
    List (PerApplicationProductionPlan.BlockKind × Program) :=
  [(.pilotPoseidon, pilotPoseidonProgram application),
    (.piCcsPoseidon, piCcsPoseidonProgram application),
    (.piCcsOrdinary, piCcsOrdinaryProgram application),
    (.pilotOrdinary, pilotOrdinaryProgram application),
    (.pilotDigestBinding, pilotDigestBindingProgram application),
    (.piCcsEndpoint, piCcsEndpointProgram application),
    (.samplerPoseidon, samplerPoseidonProgram application),
    (.samplerOrdinary, samplerOrdinaryProgram application),
    (.piRlc, piRlcProgram application),
    (.piDec, piDecProgram application),
    (.runningTransition, runningTransitionProgram application),
    (.application, applicationProgram application),
    (.nextPreimage, nextPreimageProgram application),
    (.recursivePublicOutput, recursivePublicOutputProgram application)]

@[simp] theorem children_kinds (application : ApplicationProgram) :
    (children application).map Prod.fst =
      PerApplicationProductionPlan.canonicalKinds := by
  rfl

theorem matrixProgram_blocks (application : ApplicationProgram) :
    (matrixProgram application).blocks =
      (children application).flatMap fun child => child.2.blocks := by
  simp [matrixProgram, throughNextPreimageProgram, applicationCompleteProgram,
    runningCompleteProgram,
    piDecCompleteProgram,
    piRlcCompleteProgram, samplerCompleteProgram, samplerPrefixProgram,
    piCcsCompleteProgram, pilotBindingPrefixProgram,
    pilotOrdinaryPrefixProgram, piCcsCoreProgram,
    piCcsPoseidonPrefixProgram, children, Program.append]

@[simp] theorem matrixProgram_rowCount (application : ApplicationProgram) :
    (matrixProgram application).rowCount =
      6369850 + (PerApplicationPackage.applicationPlan application).rowCount +
        9 := by
  simp [matrixProgram, throughNextPreimageProgram, applicationCompleteProgram,
    runningCompleteProgram,
    piDecCompleteProgram,
    piRlcCompleteProgram, samplerCompleteProgram, samplerPrefixProgram,
    piCcsCompleteProgram, pilotBindingPrefixProgram,
    pilotOrdinaryPrefixProgram, piCcsCoreProgram,
    piCcsPoseidonPrefixProgram, pilotPoseidonProgram, piCcsPoseidonProgram,
    piCcsOrdinaryProgram, pilotOrdinaryProgram, pilotDigestBindingProgram,
    piCcsEndpointProgram, samplerPoseidonProgram, samplerOrdinaryProgram,
    piRlcProgram, piDecProgram, runningTransitionProgram, applicationProgram,
    nextPreimageProgram, recursivePublicOutputProgram]

theorem matrixProgram_rowCount_eq_structuralPlan
    (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application) :
    (matrixProgram application).rowCount =
      (PerApplicationFixedPoint.structuralPlan application fits).rowCount := by
  rw [matrixProgram_rowCount,
    PerApplicationFixedPoint.structuralPlan_rowCount]

end NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgram
