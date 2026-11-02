import NightstreamFPrime.Export.OptionB.StepRelation
import NightstreamFPrime.Export.OptionB.StepPackage
import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

/-! The option-B emitter (CW-5.5), first part: the row census of the light and the refresh step
circuits for today's fixture application (`Poseidon2HashChainV1.program`). For each circuit it
prints the constraints, the Poseidon2 blocks and ring products that `Pieces.split` recognizes,
whether the blocks pack (`CircuitRelation.packable`), and the rows and columns of the circuit's
relation (`CircuitRelation.relationRows` and 150 S-box rows per packed block), and the counts
for each top-level child, largest first. It then writes both packages (`StepPackage`) and their
`artifacts/`. -/

namespace NightstreamFPrime.Export.OptionB.EmitMain

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic

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

/-- Writes the circuit's package (`StepPackage.packageOf`); the blocks must pack. -/
def emitPackage (name : String) (circuit : Nat) (ops : List Op) (offset : Nat)
    (pieces : List Pieces.Piece) : IO StepPackage.Package := do
  let start ← IO.monoMsNow
  unless CircuitRelation.packable CircuitRelation.publicColumns pieces do
    throw (IO.userError s!"{name}: the Poseidon2 blocks do not pack")
  let package := StepPackage.packageOf circuit CircuitRelation.publicColumns pieces
    (offset + localLength ops) width (witnesses ops)
  let path := s!"artifacts/nightstream-fprime-option-b-{name}-v1.json"
  IO.FS.writeFile path ((StepPackage.Package.format.encode package).render ++ "\n")
  say s!"{name}: wrote {path}: {package.pieces.length} pieces, \
    {package.witness.length} witness batches ({(← IO.monoMsNow) - start} ms)"
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

/-- Writes both packages; with `--census`, prints the row census first; with `--identity`, also
computes their one identity (`StepPackage.identityFast`), which the spec field arithmetic makes
slow: Rust computes it natively. -/
def main (arguments : List String) : IO Unit := do
  let withCensus := arguments.contains "--census"
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
  let light ← emitPackage "light" 0 lightOps lightOffset lightPieces
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
  say s!"packages written; emitter total {(← IO.monoMsNow) - start} ms"
  if arguments.contains "--identity" then
    let identity := StepPackage.identityFast light refresh
    say s!"identity {identity.map (·.val)} ({(← IO.monoMsNow) - start} ms)"

end NightstreamFPrime.Export.OptionB.EmitMain

def main (arguments : List String) : IO Unit :=
  NightstreamFPrime.Export.OptionB.EmitMain.main arguments
