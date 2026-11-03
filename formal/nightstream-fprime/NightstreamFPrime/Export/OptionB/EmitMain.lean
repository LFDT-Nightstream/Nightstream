import NightstreamFPrime.Export.OptionB.FinishRelation
import NightstreamFPrime.Export.OptionB.Parity
import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
import NightstreamFPrime.Export.Stage1.NebulaMemoryV1Package
import NightstreamFPrime.Export.Stage1.NebulaMemoryN1Package

/-! The option-B emitter (CW-5.5), first part: the row census of the light and the refresh step
circuits for today's fixture application (`Poseidon2HashChainV1.program`). For each circuit it
prints the constraints, the Poseidon2 blocks and ring products that `Pieces.split` recognizes,
whether the blocks pack (`CircuitRelation.packable`), and the rows and columns of the circuit's
relation (`CircuitRelation.relationRows` and 150 S-box rows per packed block), and the counts
for each top-level child, largest first. It then writes both packages (`StepPackage`) and the
parity file that Rust's expansion test reads (`Parity`). -/

namespace NightstreamFPrime.Export.OptionB.EmitMain

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Layout

/-- The census width; the step circuits' rows do not depend on it. -/
def width : Nat := 2 ^ 24

theorem publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth width := by
  decide

theorem cubeFits : Phi81CarrierLayout.carrierWidth width ≤ 2 ^ cubeVariables := by
  decide

/-- The relation argument of the step circuits; their rows do not read its matrices. -/
def relation : ProductionKey.LogicalRelation width publicFits :=
  ⟨fun _ _ _ => 0, cubeFits⟩

def program : Stage1.Application.Program := Stage1.Poseidon2HashChainV1.program

/-- The application that `--program` selects and the stem of its package files: today's fixture
by default, or a Nebula memory program (`NebulaMemoryV1Package`, `NebulaMemoryN1Package`). -/
def selected (arguments : List String) : IO (Stage1.Application.Program × String) :=
  match arguments.dropWhile (· != "--program") with
  | _ :: "nebula-v1" :: _ => pure (Export.Stage1.NebulaMemoryV1Package.application, "nebula-v1-")
  | _ :: "nebula-n1" :: _ => pure (Export.Stage1.NebulaMemoryN1Package.application, "nebula-n1-")
  | _ :: name :: _ => throw (IO.userError s!"unknown program {name}")
  | _ => pure (program, "")

/-- Prints a line and flushes it, so a capped run keeps its timings. -/
def say (line : String) : IO Unit := do
  IO.println line
  (← IO.getStdout).flush

/-- The ring products of the ring-product pieces: one per cell. -/
def ringProducts (pieces : List Pieces.Piece) : Nat :=
  pieces.foldl (fun total piece => match piece with
    | .ringProduct cells .. => total + cells
    | _ => total) 0

def census (name : String) (ops : List Op) (offset : Nat) (digestColumn : Fin 4 → Nat) :
    IO (List Pieces.Piece) := do
  let start ← IO.monoMsNow
  let canonical := PilotData.canonicalConstraints ()
  let constraints := flatConstraints ops ++ LogicalPlan.decodeConstraints digestColumn
  let firstFresh := offset + localLength ops
  IO.println s!"{name}: {constraints.length} constraints, {localLength ops} locals from {offset}"
  let pieces := Pieces.split canonical constraints
  let blocks := (CircuitRelation.blocksOf pieces).length
  let products := ringProducts pieces
  let generic := Pieces.genericConstraints pieces
  let genericRows := (Stage1.Rows.lowerConstraintsTR generic firstFresh).rows.length
  IO.println s!"{name}: {blocks} Poseidon2 blocks, {products} ring products, \
    {generic.length} other constraints, {genericRows} other rows ({(← IO.monoMsNow) - start} ms)"
  let packed := CircuitRelation.packable CircuitRelation.publicColumns pieces
  IO.println s!"{name}: blocks pack: {packed} ({(← IO.monoMsNow) - start} ms)"
  let rows := CircuitRelation.relationRows CircuitRelation.publicColumns pieces firstFresh width
  let sboxRows := if packed then 150 * blocks else 0
  let lastColumn := if packed then CircuitRelation.sboxStart pieces firstFresh + 150 * blocks
    else LogicalPlan.rowsMax rows + 1
  IO.println s!"{name}: {rows.length} R1CS rows + {sboxRows} S-box rows = \
    {rows.length + sboxRows} logical rows; largest R1CS column {LogicalPlan.rowsMax rows}, \
    columns below {lastColumn} ({(← IO.monoMsNow) - start} ms)"
  return pieces

/-- The application's region (`StepPackage.Package.application`): its witness word count, its
first local, its local count, its pieces and its fresh columns. The application is the circuit's first child; the
emitter checks that its own package pieces are the package's first pieces. -/
def applicationRegion (name : String) (words : Nat) (ops : List Op) (offset firstFresh : Nat)
    (package : List StepPackage.PackagePiece) : IO (Nat × Nat × Nat × Nat × Nat) := do
  let some application := ops.head? | throw (IO.userError s!"{name}: no application child")
  match application with
  | .subcircuit child =>
      unless child.name == s!"{name}.application" do
        throw (IO.userError s!"{name}: the first child is {child.name}")
  | _ => throw (IO.userError s!"{name}: the first operation is not the application")
  let own := Pieces.split (PilotData.canonicalConstraints ()) application.flatConstraints
  let render := fun (piece : StepPackage.PackagePiece) =>
    (StepPackage.PackagePiece.format.encode piece).render
  unless (StepPackage.packagePieces own firstFresh).map render ==
      (package.take own.length).map render do
    throw (IO.userError s!"{name}: the application's pieces are not the first pieces")
  pure (words, offset, localLength [application], own.length,
    StepPackage.freshAfter own firstFresh - firstFresh)

/-- Writes the circuit's package (`StepPackage.packageOf`); the blocks must pack and start at or
after the shared words (`FinishRelation.checks_finish` reads the words below them). The finish
package has no application: its region is the witness words alone, and its fresh columns start
after the refresh step's locals (`firstFresh`), whose layout its gadgets keep. -/
def emitPackage (name : String) (circuit : Nat) (ops : List Op) (offset : Nat)
    (pieces : List Pieces.Piece) (firstFresh : Option Nat := none)
    (program : Stage1.Application.Program := program) (stem : String := "") :
    IO StepPackage.Package := do
  let start ← IO.monoMsNow
  unless CircuitRelation.packable CircuitRelation.publicColumns pieces do
    throw (IO.userError s!"{name}: the Poseidon2 blocks do not pack")
  unless CircuitRelation.startsFrom offset pieces do
    throw (IO.userError s!"{name}: a Poseidon2 block starts below the shared end")
  let firstFresh := firstFresh.getD (offset + localLength ops)
  let package := StepPackage.packageOf circuit CircuitRelation.publicColumns pieces
    firstFresh width (witnesses ops) (program.witnessWordCount, offset, 0, 0, 0)
  let region ← if circuit < 2 then
      applicationRegion name program.witnessWordCount ops offset firstFresh package.pieces
    else pure package.application
  let package := { package with application := region }
  let path := s!"artifacts/nightstream-fprime-option-b-{stem}{name}-v1.json"
  IO.FS.writeFile path ((StepPackage.Package.format.encode package).render ++ "\n")
  let slotColumns := package.slots.foldl (fun total (_, count, _) => total + count) 0
  let digitColumns := package.slots.foldl (fun total (_, count, kind) =>
    total + count * (SlotLayout.kindOf kind).width) 0
  say s!"{name}: wrote {path}: {package.pieces.length} pieces, \
    {package.witness.length} witness batches, {package.slots.length} slot runs, \
    {slotColumns} slotted columns, {digitColumns} digit columns, application \
    {package.application} ({(← IO.monoMsNow) - start} ms)"
  return package

/-- One line per top-level child: its name, Poseidon2 blocks, ring products, other constraints
and their R1CS rows, largest first. -/
def childCensus (name : String) (ops : List Op) : IO Unit := do
  let canonical := PilotData.canonicalConstraints ()
  let lines := ops.map fun op =>
    let label := match op with
      | .subcircuit child => child.name
      | .witness _ => "witness"
      | .assertZero _ => "assertion"
    let pieces := Pieces.split canonical op.flatConstraints
    let permutations := (pieces.filter fun | .permutation .. => true | _ => false).length
    let generic := Pieces.genericConstraints pieces
    (label, permutations, ringProducts pieces, generic.length,
      (Stage1.Rows.lowerConstraintsTR generic 0).rows.length)
  let sorted := lines.toArray.qsort (fun left right => left.2.2.2.2 > right.2.2.2.2)
  for (label, permutations, products, constraints, rows) in sorted do
    IO.println s!"{name} child {label}: {permutations} blocks, {products} ring products, \
      {constraints} other constraints, {rows} other rows"

/-- Writes the finish package (`FinishRelation`): the refresh step's layout without the
application and the next hashes, with fresh columns after the refresh step's locals. -/
def emitFinish : IO Unit := do
  let start ← IO.monoMsNow
  let base := StepRelation.base
  let words := program.witnessWordCount
  let refreshOffset := RefreshAssembler.sharedEnd base words
  let interface := RefreshAssembler.interface (logicalWidth := width) (publicFits := publicFits)
    base words
  let refreshOps := Circuit.ops (OptionB.RefreshStepLayout.main program relation 1 interface)
    refreshOffset
  let finishOps := Circuit.ops (OptionB.FinishStep.main program relation 1 interface) refreshOffset
  let _ ← emitPackage "finish" 2 finishOps refreshOffset
    (Pieces.split (PilotData.canonicalConstraints ())
      (FinishRelation.constraints (logicalWidth := width) (publicFits := publicFits) program
        relation))
    (some (refreshOffset + localLength refreshOps))
  say s!"finish package written ({(← IO.monoMsNow) - start} ms)"

/-- Writes both step packages of the `--program` application (`selected`), and for today's
fixture the parity file; with `--finish`, only the finish package (one Lean run renders
at most two packages under the 1,500 s cap); with `--census`, prints the row census first; with
`--identity`, also computes their one identity (`StepPackage.identityFast`), which the spec
field arithmetic makes slow: Rust computes it natively. -/
def main (arguments : List String) : IO Unit := do
  if arguments.contains "--finish" then
    emitFinish
    return
  let withCensus := arguments.contains "--census"
  let (program, stem) ← selected arguments
  let start ← IO.monoMsNow
  let base := StepRelation.base
  let words := program.witnessWordCount
  let lightOffset := LightAssembler.sharedEnd base words
  say s!"start ({(← IO.monoMsNow) - start} ms)"
  let lightOps := Circuit.ops (OptionB.LightStepLayout.main program relation 0
      (LightAssembler.interface (logicalWidth := width) (publicFits := publicFits) base words))
      lightOffset
  let lightPieces ← if withCensus then do
      let pieces ← census "light" lightOps lightOffset (StepRelation.lightDigestColumn program)
      childCensus "light" lightOps
      pure pieces
    else pure (Pieces.split (PilotData.canonicalConstraints ())
      (flatConstraints lightOps ++
        LogicalPlan.decodeConstraints (StepRelation.lightDigestColumn program)))
  let light ← emitPackage "light" 0 lightOps lightOffset lightPieces (program := program)
    (stem := stem)
  let refreshOffset := RefreshAssembler.sharedEnd base words
  say s!"light done ({(← IO.monoMsNow) - start} ms)"
  let refreshOps := Circuit.ops (OptionB.RefreshStepLayout.main program relation 1
      (RefreshAssembler.interface (logicalWidth := width) (publicFits := publicFits) base words))
      refreshOffset
  let refreshPieces ← if withCensus then do
      let pieces ← census "refresh" refreshOps refreshOffset
        (StepRelation.refreshDigestColumn program)
      childCensus "refresh" refreshOps
      pure pieces
    else pure (Pieces.split (PilotData.canonicalConstraints ())
      (flatConstraints refreshOps ++
        LogicalPlan.decodeConstraints (StepRelation.refreshDigestColumn program)))
  let refresh ← emitPackage "refresh" 1 refreshOps refreshOffset refreshPieces
    (program := program) (stem := stem)
  say s!"packages written ({(← IO.monoMsNow) - start} ms)"
  unless stem.isEmpty do
    say s!"emitter total {(← IO.monoMsNow) - start} ms"
    return
  let parity := Parity.parity width (lightPieces, light.firstFresh, lightOffset, witnesses lightOps)
    (refreshPieces, refresh.firstFresh, refreshOffset, witnesses refreshOps)
  let parityPath := "artifacts/nightstream-fprime-option-b-parity-v1.json"
  IO.FS.writeFile parityPath (parity.render ++ "\n")
  match parity with
  | .array [_, lightSummary, refreshSummary, _, probeIdentity] =>
      say s!"wrote {parityPath}: light {lightSummary.render}, refresh {refreshSummary.render}, \
        probe identity {probeIdentity.render}"
  | _ => pure ()
  say s!"emitter total {(← IO.monoMsNow) - start} ms"
  if arguments.contains "--identity" then
    let identity := StepPackage.identityFast light refresh
    say s!"identity {identity.map (·.val)} ({(← IO.monoMsNow) - start} ms)"

end NightstreamFPrime.Export.OptionB.EmitMain

def main (arguments : List String) : IO Unit :=
  NightstreamFPrime.Export.OptionB.EmitMain.main arguments
