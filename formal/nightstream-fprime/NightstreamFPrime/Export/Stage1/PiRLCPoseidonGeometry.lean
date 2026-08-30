import NightstreamFPrime.Export.Stage1.PiRLCRetainedGeometry
import NightstreamFPrime.Export.Stage1.PoseidonInputRetainedBlock

/-!
Owns the append-only retained geometry for the two pilot Poseidon2 preimages.
The existing PiRLC retained prefix does not move. Later Poseidon2 input blocks
may extend this geometry again.

This module does not select later PiCCS or PiRLC permutation inputs.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCPoseidonGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.sourceWidth program

def priorInputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PoseidonInputRetainedBlock.priorBlock.lift
    (PiRLCRetainedGeometry.poseidonSourceFits program)

def outputInputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  PoseidonInputRetainedBlock.outputBlock.lift
    (PiRLCRetainedGeometry.poseidonSourceFits program)

def priorInputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.prefixLogicalWidth program

def outputInputStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  priorInputStart program + (priorInputBlock program).coordinateCount

def pilotLogicalWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  outputInputStart program + (outputInputBlock program).coordinateCount

@[simp] theorem pilotLogicalWidth_eq
    (program : Lifecycle.Stage1.Application.Program) :
    pilotLogicalWidth program = 184304620 := by
  unfold pilotLogicalWidth outputInputStart priorInputStart
    priorInputBlock outputInputBlock
  simp

theorem pilotLogicalWidth_le_cube
    (program : Lifecycle.Stage1.Application.Program) :
    pilotLogicalWidth program ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [pilotLogicalWidth_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

structure Geometry (program : Lifecycle.Stage1.Application.Program)
    (logicalWidth : Nat) : Prop where
  pilotFits : pilotLogicalWidth program ≤ logicalWidth

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth where
  prefixFits := by
    apply Nat.le_trans _ geometry.pilotFits
    unfold pilotLogicalWidth outputInputStart priorInputStart
    omega

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Fin logicalWidth :=
  PiRLCRetainedGeometry.oneColumn (prefixGeometry geometry)

def priorInputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    priorInputStart program + (priorInputBlock program).coordinateCount ≤
      logicalWidth := by
  apply Nat.le_trans _ geometry.pilotFits
  unfold pilotLogicalWidth outputInputStart
  omega

def outputInputFits {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    outputInputStart program + (outputInputBlock program).coordinateCount ≤
      logicalWidth :=
  geometry.pilotFits

end NightstreamFPrime.Export.Stage1.PiRLCPoseidonGeometry
