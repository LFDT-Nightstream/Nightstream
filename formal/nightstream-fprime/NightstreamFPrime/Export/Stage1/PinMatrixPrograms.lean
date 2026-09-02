import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.DirectPiDECPrefixPlan
import NightstreamFPrime.Export.Stage1.RecursivePublicOutputPlan

/-!
Owns the three small explicit pin blocks in the canonical Stage 1 14-matrix
program: pilot digest custody, PiCCS transcript endpoint custody, and the
recursive public-output binding.

The blocks carry final sparse forms. They do not select package order or
claim final package integration.
-/

namespace NightstreamFPrime.Export.Stage1.PinMatrixPrograms

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout.Stage1

abbrev Program := Lifecycle.Stage1.Application.Program

def pilotDigestBinding
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :
    Pin.Block :=
  Pin.Block.ofSemantic (PilotDigestBindingPlan.interface
    (DirectPiDECPrefixPlan.pilotOrdinaryGeometry geometry))

@[simp] theorem pilotDigestBinding_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :
    (pilotDigestBinding geometry).rowCount = 8 := by
  simp [pilotDigestBinding, PilotDigestBindingPlan.rowCount,
    PilotDigestBindingPlan.chainCount, PilotDigestBindingPlan.laneCount]

theorem pilotDigestBinding_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (row : Fin PilotDigestBindingPlan.rowCount) :
    (pilotDigestBinding geometry).row? logicalWidth row.val =
      some (Layout.ProductionRelation.PinFamilyPlan.forms
        (PilotDigestBindingPlan.interface
          (DirectPiDECPrefixPlan.pilotOrdinaryGeometry geometry)) row) := by
  exact Pin.Block.row?_ofSemantic _ row

def pilotDigestBindingProgram
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.pin (pilotDigestBinding geometry)]

@[simp] theorem pilotDigestBindingProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :
    (pilotDigestBindingProgram geometry).rowCount = 8 := by
  rw [show pilotDigestBindingProgram geometry = MatrixProgram.Program.mk
      [.pin (pilotDigestBinding geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact pilotDigestBinding_rowCount geometry

theorem pilotDigestBindingProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option Layout.R1CS.Row)
    (row : Fin PilotDigestBindingPlan.rowCount) :
    (pilotDigestBindingProgram geometry).row?
        logicalWidth sourceRow row.val =
      some (Layout.ProductionRelation.PinFamilyPlan.forms
        (PilotDigestBindingPlan.interface
          (DirectPiDECPrefixPlan.pilotOrdinaryGeometry geometry)) row
        ).meaningfulForm := by
  have bound : row.val <
      (MatrixProgram.Block.pin (pilotDigestBinding geometry)).rowCount := by
    change row.val < (pilotDigestBinding geometry).rowCount
    rw [pilotDigestBinding_rowCount]
    simpa [PilotDigestBindingPlan.rowCount_eq] using row.isLt
  rw [show pilotDigestBindingProgram geometry = MatrixProgram.Program.mk
      [.pin (pilotDigestBinding geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos bound]
  change (do
    let forms ← (pilotDigestBinding geometry).row? logicalWidth row.val
    pure forms.meaningfulForm) = _
  rw [pilotDigestBinding_row? geometry row]
  rfl

def piCcsEndpoint
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :
    Pin.Block :=
  Pin.Block.ofSemantic (PiCCSTranscriptEndpointPlan.interface
    (DirectPiDECPrefixPlan.poseidonGeometry geometry)
    (DirectPiDECPrefixPlan.piCcsOrdinaryGeometry geometry))

@[simp] theorem piCcsEndpoint_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :
    (piCcsEndpoint geometry).rowCount = 32 := by
  simp [piCcsEndpoint, PiCCSTranscriptEndpointPlan.rowCount,
    PiCCSTranscriptEndpointPlan.familyCount,
    PiCCSTranscriptEndpointPlan.laneCount, Spec.Poseidon2.width]

theorem piCcsEndpoint_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (row : Fin PiCCSTranscriptEndpointPlan.rowCount) :
    (piCcsEndpoint geometry).row? logicalWidth row.val =
      some (Layout.ProductionRelation.PinFamilyPlan.forms
        (PiCCSTranscriptEndpointPlan.interface
          (DirectPiDECPrefixPlan.poseidonGeometry geometry)
          (DirectPiDECPrefixPlan.piCcsOrdinaryGeometry geometry)) row) := by
  exact Pin.Block.row?_ofSemantic _ row

def piCcsEndpointProgram
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.pin (piCcsEndpoint geometry)]

@[simp] theorem piCcsEndpointProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth) :
    (piCcsEndpointProgram geometry).rowCount = 32 := by
  rw [show piCcsEndpointProgram geometry = MatrixProgram.Program.mk
      [.pin (piCcsEndpoint geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact piCcsEndpoint_rowCount geometry

theorem piCcsEndpointProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option Layout.R1CS.Row)
    (row : Fin PiCCSTranscriptEndpointPlan.rowCount) :
    (piCcsEndpointProgram geometry).row?
        logicalWidth sourceRow row.val =
      some (Layout.ProductionRelation.PinFamilyPlan.forms
        (PiCCSTranscriptEndpointPlan.interface
          (DirectPiDECPrefixPlan.poseidonGeometry geometry)
          (DirectPiDECPrefixPlan.piCcsOrdinaryGeometry geometry)) row
        ).meaningfulForm := by
  have bound : row.val <
      (MatrixProgram.Block.pin (piCcsEndpoint geometry)).rowCount := by
    change row.val < (piCcsEndpoint geometry).rowCount
    rw [piCcsEndpoint_rowCount]
    simpa [PiCCSTranscriptEndpointPlan.rowCount_eq] using row.isLt
  rw [show piCcsEndpointProgram geometry = MatrixProgram.Program.mk
      [.pin (piCcsEndpoint geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos bound]
  change (do
    let forms ← (piCcsEndpoint geometry).row? logicalWidth row.val
    pure forms.meaningfulForm) = _
  rw [piCcsEndpoint_row? geometry row]
  rfl

theorem pilotDigestBindingProgram_plan_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option Layout.R1CS.Row)
    (row : Fin (DirectPiDECPrefixPlan.pilotBindingPlan geometry).rowCount) :
    (pilotDigestBindingProgram geometry).row?
        logicalWidth sourceRow row.val =
      some ((DirectPiDECPrefixPlan.pilotBindingPlan geometry).forms row) := by
  simpa [DirectPiDECPrefixPlan.pilotBindingPlan,
    PilotDigestBindingPlan.plan, Layout.ProductionRelation.PinFamilyPlan.plan]
    using pilotDigestBindingProgram_row? geometry sourceRow row

theorem piCcsEndpointProgram_plan_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiDECRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option Layout.R1CS.Row)
    (row : Fin (DirectPiDECPrefixPlan.piCcsEndpointPlan geometry).rowCount) :
    (piCcsEndpointProgram geometry).row?
        logicalWidth sourceRow row.val =
      some ((DirectPiDECPrefixPlan.piCcsEndpointPlan geometry).forms row) := by
  simpa [DirectPiDECPrefixPlan.piCcsEndpointPlan,
    PiCCSTranscriptEndpointPlan.plan,
    Layout.ProductionRelation.PinFamilyPlan.plan] using
      piCcsEndpointProgram_row? geometry sourceRow row

def recursivePublicOutput
    {program : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry program logicalWidth) :
    Pin.Block :=
  Pin.Block.ofSemantic (RecursivePublicOutputPlan.interface geometry)

@[simp] theorem recursivePublicOutput_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry program logicalWidth) :
    (recursivePublicOutput geometry).rowCount = 4 := by
  simp [recursivePublicOutput, RecursivePublicOutputPlan.rowCount]

theorem recursivePublicOutput_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry program logicalWidth)
    (row : Fin RecursivePublicOutputPlan.rowCount) :
    (recursivePublicOutput geometry).row? logicalWidth row.val =
      some (Layout.ProductionRelation.PinFamilyPlan.forms
        (RecursivePublicOutputPlan.interface geometry) row) := by
  exact Pin.Block.row?_ofSemantic _ row

def recursivePublicOutputProgram
    {program : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program where
  blocks := [.pin (recursivePublicOutput geometry)]

@[simp] theorem recursivePublicOutputProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry program logicalWidth) :
    (recursivePublicOutputProgram geometry).rowCount = 4 := by
  rw [show recursivePublicOutputProgram geometry = MatrixProgram.Program.mk
      [.pin (recursivePublicOutput geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact recursivePublicOutput_rowCount geometry

theorem recursivePublicOutputProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option Layout.R1CS.Row)
    (row : Fin RecursivePublicOutputPlan.rowCount) :
    (recursivePublicOutputProgram geometry).row?
        logicalWidth sourceRow row.val =
      some (Layout.ProductionRelation.PinFamilyPlan.forms
        (RecursivePublicOutputPlan.interface geometry) row).meaningfulForm := by
  have bound : row.val <
      (MatrixProgram.Block.pin (recursivePublicOutput geometry)).rowCount := by
    change row.val < (recursivePublicOutput geometry).rowCount
    rw [recursivePublicOutput_rowCount]
    simpa [RecursivePublicOutputPlan.rowCount_eq] using row.isLt
  rw [show recursivePublicOutputProgram geometry = MatrixProgram.Program.mk
      [.pin (recursivePublicOutput geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos bound]
  change (do
    let forms ← (recursivePublicOutput geometry).row? logicalWidth row.val
    pure forms.meaningfulForm) = _
  rw [recursivePublicOutput_row? geometry row]
  rfl

theorem recursivePublicOutputProgram_plan_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option Layout.R1CS.Row)
    (row : Fin (RecursivePublicOutputPlan.plan geometry).rowCount) :
    (recursivePublicOutputProgram geometry).row?
        logicalWidth sourceRow row.val =
      some ((RecursivePublicOutputPlan.plan geometry).forms row) := by
  simpa [RecursivePublicOutputPlan.plan,
    Layout.ProductionRelation.PinFamilyPlan.plan] using
      recursivePublicOutputProgram_row? geometry sourceRow row

end NightstreamFPrime.Export.Stage1.PinMatrixPrograms
