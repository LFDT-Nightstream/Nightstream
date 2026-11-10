import NightstreamFPrime.Export.OptionB.FinishRelation
import NightstreamFPrime.Export.OptionB.Parity
import NightstreamFPrime.Export.OptionB.StepCompleteness
import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
import NightstreamFPrime.Export.Stage1.NebulaMemoryV1Package
import NightstreamFPrime.Export.Stage1.NebulaMemoryN1Package
import NightstreamFPrime.Export.Stage1.ApplicationPackage
import NightstreamFPrime.Export.Stage1.NebulaMemoryTerminal
import NightstreamFPrime.Lifecycle.Nebula.SecondPlan
import NightstreamFPrime.Lifecycle.Nebula.ThirdPlan
import NightstreamFPrime.Export.Stage1.DirectLowNormFootprint
import NightstreamFPrime.Layout.OptionB.LightAssemblerCompleteness
import NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness
import NightstreamFPrime.Layout.OptionB.StepCircuits
import NightstreamFPrime.Layout.PiDEC.v1_2.Ownership
import NightstreamFPrime.Layout.ProductionRelation.MultiplicationFamilyPlan
import NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport
import NightstreamFPrime.Lifecycle.Nebula.ProgramSoundness
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.CcsTerminal
import NightstreamFPrime.Lifecycle.Stage1.Terminal
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Nebula

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

/-- The source width: every source column of a step package lies below it. -/
def width : Nat := 2 ^ 24

/-- The step circuits' relation argument, at the width that both circuit keys share. -/
abbrev relation := StepRelation.keyRelation

def program : Stage1.Application.Program := Stage1.Poseidon2HashChainV1.program

/-- The application that `--program` selects and the stem of its package files: today's fixture
by default, or a Nebula memory program (`NebulaMemoryV1Package`, `NebulaMemoryN1Package`). -/
def selected (arguments : List String) : IO (Stage1.Application.Program × String) :=
  match arguments.dropWhile (· != "--program") with
  | _ :: "nebula-v1" :: _ => pure (Export.Stage1.NebulaMemoryV1Package.application, "nebula-v1-")
  | _ :: "nebula-n1" :: _ => pure (Export.Stage1.NebulaMemoryN1Package.application, "nebula-n1-")
  | _ :: "nebula-r1" :: _ => pure (Export.Stage1.NebulaMemoryPackage.application
      ⟨Nebula.ThirdPlan.geometry, Nebula.ThirdPlan.valid, Nebula.ThirdPlan.secure⟩ Nebula.ThirdPlan.app,
      "nebula-r1-")
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

def census (name : String) (ops : List Op) (offset : Nat) (digestColumn : Fin 4 → Nat)
    (pieces : List Pieces.Piece) : IO Unit := do
  let start ← IO.monoMsNow
  let constraints := flatConstraints ops ++ LogicalPlan.decodeConstraints digestColumn
  let firstFresh := offset + localLength ops
  IO.println s!"{name}: {constraints.length} constraints, {localLength ops} locals from {offset}"
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

/-- The completeness check of a step package (`PieceRowsCompleteness.pieceBelow`), and the number
of pieces that read a fresh column. -/
def completenessChecks (pieces : List Pieces.Piece) (firstFresh : Nat) : Bool × Nat :=
  let scope := (pieces.filter fun piece => !PieceRowsCompleteness.pieceBelow firstFresh piece).length
  (scope == 0, scope)

/-- The completeness checks of the finish package (`FinishCompleteness.FinishComplete`): every
piece reads only columns below the first fresh column, and the shared words, which end at
`offset`, lie below it. For a failure, also the pieces that read a fresh column. -/
def finishChecks (pieces : List Pieces.Piece) (offset firstFresh : Nat) : Bool × Nat :=
  if pieces.all (PieceRowsCompleteness.pieceBelow firstFresh) && decide (offset ≤ firstFresh) then
    (true, 0)
  else
    (false, (pieces.filter fun piece => !PieceRowsCompleteness.pieceBelow firstFresh piece).length)

/-- Writes the circuit's package (`StepPackage.packageOf`); the blocks must pack and start at or
after the shared words (`FinishRelation.checks_finish` reads the words below them). The finish
package has no application: its region is the witness words alone, and its fresh columns start
after the refresh step's locals (`firstFresh`), whose layout its gadgets keep. Before it writes, it
runs the package's completeness checks (`completenessChecks` for a step package, `finishChecks`
for the finish package) and checks that no row of the circuit reads a public column. -/
def emitPackage (name : String) (circuit : Nat) (ops : List Op) (offset : Nat)
    (pieces : List Pieces.Piece) (firstFresh : Option Nat := none)
    (program : Stage1.Application.Program := program) (stem : String := "")
    (write : Bool := true) :
    IO StepPackage.Package := do
  let start ← IO.monoMsNow
  unless CircuitRelation.packable CircuitRelation.publicColumns pieces do
    throw (IO.userError s!"{name}: the Poseidon2 blocks do not pack")
  unless CircuitRelation.startsFrom offset pieces do
    throw (IO.userError s!"{name}: a Poseidon2 block starts below the shared end")
  let firstFresh := firstFresh.getD (offset + localLength ops)
  unless CircuitRelation.publicColumns ≤ firstFresh do
    throw (IO.userError s!"{name}: the first fresh column is inside the public input")
  let free := StepCompleteness.publicFree (flatConstraints ops)
  if circuit < 2 then
    let (complete, scope) := completenessChecks pieces firstFresh
    say s!"{name}: completeness checks pass: {complete}, rows read no public column: {free} \
      ({(← IO.monoMsNow) - start} ms)"
    unless complete && free do
      throw (IO.userError s!"{name}: the completeness checks fail: {scope} pieces read a fresh \
        column, rows read no public column: {free}")
  else
    let (complete, scope) := finishChecks pieces offset firstFresh
    say s!"{name}: completeness checks pass: {complete}, rows read no public column: {free} \
      ({(← IO.monoMsNow) - start} ms)"
    unless complete && free do
      throw (IO.userError s!"{name}: the completeness checks fail: {scope} pieces read a fresh \
        column, shared words end at {offset}, first fresh column {firstFresh}")
  unless write do
    return StepPackage.packageOf circuit CircuitRelation.publicColumns pieces firstFresh width
      (witnesses ops) (program.witnessWordCount, offset, 0, 0, 0)
  let package := StepPackage.packageOf circuit CircuitRelation.publicColumns pieces
    firstFresh width (witnesses ops) (program.witnessWordCount, offset, 0, 0, 0)
  let region ← if circuit < 2 then
      applicationRegion name program.witnessWordCount ops offset firstFresh package.pieces
    else pure package.application
  let package := { package with application := region }
  let logical := ProductionAssignment.logicalWidth (SlotLayout.runSlots width package.slots)
  unless logical ≤ StepRelation.keyWidth do
    throw (IO.userError s!"{name}: logical width {logical} exceeds the key width")
  let path := s!"artifacts/nightstream-fprime-option-b-{stem}{name}-v1.json"
  IO.FS.writeFile path ((StepPackage.Package.format.encode package).render ++ "\n")
  let slotColumns := package.slots.foldl (fun total (_, count, _) => total + count) 0
  let digitColumns := package.slots.foldl (fun total (_, count, kind) =>
    total + count * (SlotLayout.kindOf kind).width) 0
  say s!"{name}: wrote {path}: {package.pieces.length} pieces, \
    {package.witness.length} witness batches, {package.slots.length} slot runs, \
    {slotColumns} slotted columns, {digitColumns} digit columns, logical width {logical}, application \
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
application and the next hashes, with fresh columns after the refresh step's locals. Without
`write`, it only runs the completeness checks. -/
def emitFinish (write : Bool) : IO Unit := do
  let start ← IO.monoMsNow
  let base := StepRelation.base
  let words := program.witnessWordCount
  let refreshOffset := RefreshAssembler.sharedEnd base words
  let interface := RefreshAssembler.interface (logicalWidth := StepRelation.keyWidth)
    (publicFits := StepRelation.keyPublicFits) base words
  let refreshOps := Circuit.ops (OptionB.RefreshStepLayout.main program relation 1 interface)
    refreshOffset
  let finishOps := Circuit.ops (OptionB.FinishStep.main program relation 1 interface) refreshOffset
  let _ ← emitPackage "finish" 2 finishOps refreshOffset
    (FinishRelation.pieces (logicalWidth := StepRelation.keyWidth)
      (publicFits := StepRelation.keyPublicFits) program relation)
    (some (refreshOffset + localLength refreshOps)) (write := write)
  say s!"finish {if write then "package written" else "checks done"} \
    ({(← IO.monoMsNow) - start} ms)"

/-- FNV-1a over the little-endian bytes of `word`. -/
def fnvWord (hash word : UInt64) : UInt64 := Id.run do
  let mut hash := hash
  for k in [0:8] do
    hash := (hash ^^^ ((word >>> (8 * k).toUInt64) &&& 0xff)) * 0x100000001b3
  return hash

/-- One side of a row in canonical form: its constant, then its terms by variable, with equal
variables merged and zero coefficients dropped. -/
def sideWords (toVariable : Nat → Except String Nat) (side : R1CS.LinearCombination) :
    Except String (List UInt64) := do
  let mut merged : Std.HashMap Nat F := {}
  for (column, coefficient) in side.terms do
    let index ← toVariable column
    merged := merged.insert index (merged.getD index 0 + coefficient)
  let terms := (merged.toList.filter (·.2 ≠ 0)).mergeSort (·.1 ≤ ·.1)
  return [side.constant.val.toUInt64, terms.length.toUInt64] ++
    terms.flatMap fun (index, coefficient) => [index.toUInt64, coefficient.val.toUInt64]

/-- The rows of `program`'s application, and their digest in the Rust application's variable
order (`ApplicationCircuit`): the input state, the witness words, the output state, then the
locals and the lowering's fresh columns. `tests/nebula/parity.rs` requires the Rust-built
memory circuit to have the same digest. -/
def applicationDigest (program : Stage1.Application.Program) : Except String (Nat × UInt64) := do
  let columns := Export.Stage1.ApplicationPackage.productionColumns program
  let privateStart := Layout.Stage1.ApplicationInputs.localStart program
  let words := program.witnessWordCount
  let constraints := Export.Stage1.ApplicationPackage.constraints program columns privateStart
  let rows := (Export.Stage1.Rows.lowerConstraintsTR constraints
    (Export.Stage1.ApplicationPackage.r1csFreshStart program columns privateStart)).rows
  let mut ports : Std.HashMap Nat Nat := {}
  for lane in List.finRange 4 do
    ports := ports.insert (Layout.Stage1.ApplicationInputs.inputColumn lane) lane.val
    ports := ports.insert (Layout.Stage1.ApplicationInputs.outputColumn lane) (4 + words + lane.val)
  for word in List.finRange words do
    ports := ports.insert (Layout.Stage1.ApplicationInputs.witnessColumn word) (4 + word.val)
  let toVariable (column : Nat) : Except String Nat :=
    match ports[column]? with
    | some index => pure index
    | none => if privateStart ≤ column then pure (8 + words + (column - privateStart))
      else throw s!"column {column} is no application variable"
  let mut hash := fnvWord 0xcbf29ce484222325 rows.length.toUInt64
  for row in rows do
    for side in [row.a, row.b, row.c] do
      for word in ← sideWords toVariable side do
        hash := fnvWord hash word
  return (rows.length, hash)

/-- A memory package's named assertion row ranges as `[[name, first, end], ...]`, counted from
the application's first row. The Rust conformance suite reads them. -/
def nebulaRows (ranges : List (String × ℕ × ℕ)) : String :=
  "[" ++ ",".intercalate (ranges.map fun (name, first, stop) => s!"[\"{name}\",{first},{stop}]") ++ "]\n"

/-- `D_init` of `plan` (SPEC.md section 7.4, the start carry's `D_mem`) as `{"d_init":[...]}`.
`ofWords_initRootWords`: the words are `(context plan).initRoot`. Rust tests compare their own
value with it. -/
def nebulaStart (plan : Spec.Nebula.Plan) : String :=
  "{\"d_init\":[" ++ ",".intercalate ((Lifecycle.Nebula.initRootWords plan).map (toString ·.val)) ++ "]}\n"

/-- The memory fixtures that the Rust tests read: the named row ranges, `D_init` and the
SPEC.md Algorithm 10 terminal-check cases (`NebulaMemoryTerminal.fixture`). -/
def nebulaFixture : List String → Option String
  | ["--application-rows", "nebula-v1"] => some (nebulaRows Stage1.NebulaMemoryV1Package.namedRowRanges)
  | ["--application-rows", "nebula-n1"] => some (nebulaRows Stage1.NebulaMemoryN1Package.namedRowRanges)
  | ["--application-start", "nebula-v1"] => some (nebulaStart Lifecycle.Nebula.FirstPlan.plan)
  | ["--application-start", "nebula-v1-other"] => some (nebulaStart Lifecycle.Nebula.FirstPlan.otherPlan)
  | ["--application-start", "nebula-n1"] => some (nebulaStart Lifecycle.Nebula.SecondPlan.plan)
  | ["--application-terminal", "nebula-v1"] =>
      some (Stage1.NebulaMemoryTerminal.fixture Lifecycle.Nebula.FirstPlan.plan)
  | ["--application-terminal", "nebula-n1"] =>
      some (Stage1.NebulaMemoryTerminal.fixture Lifecycle.Nebula.SecondPlan.plan)
  | _ => none

/-- Writes both step packages of the `--program` application (`selected`), and for today's
fixture the parity file; with `--finish`, only the finish package (one Lean run renders
at most two packages under the 1,500 s cap); with `--checks`, only runs the completeness checks of
both step packages, or with `--finish` of the finish package, and writes nothing; with
`--census`, prints the row census first; with `--identity`, also computes their one identity
(`StepPackage.identityFast`), which the spec field arithmetic makes slow: Rust computes it
natively. -/
def main (arguments : List String) : IO Unit := do
  if arguments.contains "--finish" then
    emitFinish (!arguments.contains "--checks")
    return
  if let [mode, name, path] := arguments then
    if let some text := nebulaFixture [mode, name] then
      IO.FS.writeFile path text
      say s!"{mode} {name}: {path}"
      return
  if let ["--application-digest", name, path] := arguments then
    let (program, _) ← selected ["--program", name]
    let (rows, digest) ← IO.ofExcept (applicationDigest program)
    IO.FS.writeFile path s!"\{\"rows\": {rows}, \"digest\": {digest}}\n"
    say s!"{name}: {rows} application rows, digest {digest}"
    return
  let withCensus := arguments.contains "--census"
  let write := !arguments.contains "--checks"
  let (program, stem) ← selected arguments
  let start ← IO.monoMsNow
  let base := StepRelation.base
  let words := program.witnessWordCount
  let lightOffset := LightAssembler.sharedEnd base words
  say s!"start ({(← IO.monoMsNow) - start} ms)"
  let lightOps := StepCompleteness.lightOps program relation
  let lightPieces := StepCompleteness.piecesOf lightOps (StepRelation.lightDigestColumn program)
  if withCensus then
    census "light" lightOps lightOffset (StepRelation.lightDigestColumn program) lightPieces
    childCensus "light" lightOps
  let light ← emitPackage "light" 0 lightOps lightOffset lightPieces (program := program)
    (stem := stem) (write := write)
  let refreshOffset := RefreshAssembler.sharedEnd base words
  say s!"light done ({(← IO.monoMsNow) - start} ms)"
  let refreshOps := StepCompleteness.refreshOps program relation
  let refreshPieces := StepCompleteness.piecesOf refreshOps
    (StepRelation.refreshDigestColumn program)
  if withCensus then
    census "refresh" refreshOps refreshOffset (StepRelation.refreshDigestColumn program)
      refreshPieces
    childCensus "refresh" refreshOps
  let refresh ← emitPackage "refresh" 1 refreshOps refreshOffset refreshPieces
    (program := program) (stem := stem) (write := write)
  unless write do
    say s!"checks done ({(← IO.monoMsNow) - start} ms)"
    return
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
