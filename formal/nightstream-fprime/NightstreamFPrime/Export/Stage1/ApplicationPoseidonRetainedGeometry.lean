import NightstreamFPrime.Export.Stage1.ApplicationPoseidonRetainedBlock
import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryGeometry
import NightstreamFPrime.Layout.Stage1.Poseidon2HashChainCompact

/-! Place the checked application's message and S-box values after the unchanged prefix. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationPoseidonRetainedGeometry

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open ApplicationPoseidonRetainedBlock

abbrev inputStart := ApplicationOrdinaryGeometry.inputStart
abbrev outputStart := ApplicationOrdinaryGeometry.outputStart
abbrev witnessStart := ApplicationOrdinaryGeometry.witnessStart
abbrev localStart := ApplicationOrdinaryGeometry.localStart

def completeLogicalWidth (application : Stage1.Application.Program)
    (certificate : Certificate application) : Nat :=
  localStart application + (block application certificate).coordinateCount

theorem completeLogicalWidth_eq (application : Stage1.Application.Program)
    (certificate : Certificate application) :
    completeLogicalWidth application certificate = 149292999 := by
  rw [completeLogicalWidth, block_coordinateCount]
  unfold localStart ApplicationOrdinaryGeometry.localStart
    ApplicationOrdinaryGeometry.witnessStart
  rw [PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth_eq]
  change 149282257 + application.witnessWordCount * 41 + 10578 = _
  rw [certificate.wordCount]
  rfl

structure Geometry (application : Stage1.Application.Program)
    (certificate : Certificate application) (columns : Nat) : Prop where
  completeFits : completeLogicalWidth application certificate ≤ columns

def prefixGeometry {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) :
    PiRLCSamplerOrdinaryRetainedGeometry.Geometry application columns where
  completeFits := by
    have bound := geometry.completeFits
    rw [completeLogicalWidth_eq] at bound
    rw [PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth_eq]
    omega

def pilotGeometry {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) :
    PiRLCPoseidonGeometry.Geometry application columns where
  pilotFits := by
    have bound := geometry.completeFits
    rw [completeLogicalWidth_eq] at bound
    rw [PiRLCPoseidonGeometry.pilotLogicalWidth_eq]
    omega

def oneColumn {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) : Fin columns :=
  PiRLCSamplerOrdinaryRetainedGeometry.oneColumn (prefixGeometry geometry)

def inputFits {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) :
    inputStart application + (ApplicationRetainedBlocks.inputBlock application).coordinateCount ≤ columns := by
  have pilot := PiRLCPoseidonGeometry.priorInputFits (pilotGeometry geometry)
  have full : (PiRLCPoseidonGeometry.priorInputBlock application).coordinateCount = 2025113 := by
    simp [PiRLCPoseidonGeometry.priorInputBlock]
  rw [full] at pilot
  change PiRLCPoseidonGeometry.priorInputStart application + 35 * 41 + 4 * 41 ≤ columns
  omega

def outputFits {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) :
    outputStart application + (ApplicationRetainedBlocks.outputBlock application).coordinateCount ≤ columns := by
  have pilot := PiRLCPoseidonGeometry.outputInputFits (pilotGeometry geometry)
  have full : (PiRLCPoseidonGeometry.outputInputBlock application).coordinateCount = 2025113 := by
    simp [PiRLCPoseidonGeometry.outputInputBlock]
  rw [full] at pilot
  change PiRLCPoseidonGeometry.outputInputStart application + 35 * 41 + 4 * 41 ≤ columns
  omega

def witnessFits {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) :
    witnessStart application + (ApplicationRetainedBlocks.witnessBlock application).coordinateCount ≤ columns := by
  apply Nat.le_trans _ geometry.completeFits
  unfold completeLogicalWidth localStart ApplicationOrdinaryGeometry.localStart witnessStart
  omega

def localFits {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) :
    localStart application + (block application certificate).coordinateCount ≤ columns :=
  geometry.completeFits

def interface {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) :
    Layout.Stage1.Poseidon2HashChainCompact.Interface columns where
  oneColumn := oneColumn geometry
  priorState := (ApplicationRetainedBlocks.inputBlock application).form
    (inputStart application) (inputFits geometry)
  message := fun lane => (ApplicationRetainedBlocks.witnessBlock application).form
    (witnessStart application) (witnessFits geometry)
    ⟨lane.val, by have := lane.isLt; change lane.val < application.witnessWordCount; rw [certificate.wordCount]; exact this⟩
  digest := (ApplicationRetainedBlocks.outputBlock application).form
    (outputStart application) (outputFits geometry)
  sbox := fun invocation row => (block application certificate).form
    (localStart application) (localFits geometry) (Fin.encodeProd (invocation, row))

def plan {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) : ProductionRelation.Plan columns :=
  Layout.Stage1.Poseidon2HashChainCompact.plan (interface geometry)

theorem plan_rowCount {application : Stage1.Application.Program}
    {certificate : Certificate application} {columns : Nat}
    (geometry : Geometry application certificate columns) : (plan geometry).rowCount = 262 :=
  Layout.Stage1.Poseidon2HashChainCompact.plan_rowCount _

end NightstreamFPrime.Export.Stage1.ApplicationPoseidonRetainedGeometry
