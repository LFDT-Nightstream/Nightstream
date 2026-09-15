import NightstreamFPrime.Export.SignedUnitSourceInput
import NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
import NightstreamFPrime.Export.Stage1.PiCCSFirstRound
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages
import NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages
import NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache
import NightstreamFPrime.Export.Stage1.PiCCSCachedSelector
import NightstreamFPrime.Export.Stage1.PiCCSNormCache
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache

/-!
Compute a first-round pair range from original witnesses and public claims.
Rust round messages and output evaluations are not inputs. A partial range
is a measurement/accumulation result, not a complete first round. The runner
aggregates linear carried reads before the existing numeric matrix interpreter.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiCCSFirstRoundReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NightstreamFPrime.Export.Codec

private abbrev Sources := Fin productionShape.sourceCount →
  Phi81Relation.Assignment PiCCSSourceImages.shape

private def layout := Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
  PiCCSSourceImages.shape.carrierWidth
  (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
    Poseidon2HashChainV1Package.fits).cubeFits

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok value => pure value
  | .error error => throw (IO.userError error)

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private def extensionValue (value : K) : Value :=
  .array [.atom value.c0.val, .atom value.c1.val]

private def sources (masks : Array (Array (Nat × Nat))) : Sources :=
  fun source column => SignedUnitSourceInput.scalar
    (masks[column.val / ringDegree]?.getD #[])
    ⟨source.val, source.isLt⟩ ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩

private def images (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (padBasis matrixBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (powers : Nat → K)
    (assignments : Sources) (vertex : BooleanVertex cubeVariables) :
    IO (ProtocolPolynomial.OutputMessage K productionShape × K × K) := do
  let some result := PiCCSAggregatedImages.endpoint? program sourceRow layout assignments
      padBasis matrixBasis blocks powers vertex
    | throw (IO.userError "matrix row failed to load")
  return result

private structure CachedInvocation where
  firstRow : Nat
  fresh : Vector Spec.ProductionRelation.RowSemantics.PortValues 94
  carried : Vector (Vector K Spec.ProductionRelation.matrixCount) 94

private def invocationCache? (program : MatrixProgram.Program)
    (assignments : Sources) (matrixBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (index : Nat) : Option CachedInvocation := Id.run do
  let mut start := 0
  for block in program.blocks do
    if index < start + block.rowCount then
      match block with
      | .poseidon poseidon =>
          let some (interface, row) := PiDECPoseidonNumericBlock.loadRow? poseidon
              PiCCSSourceImages.logicalWidth (index - start) | return none
          return some {
            firstRow := index - row.val
            fresh := PiDECPoseidonNumericRows.stored
              (PiCCSSourceImages.plainRead (assignments (freshSourceIndex ⟨0, by decide⟩))) interface
            carried := PiCCSCarriedReadCache.invocation matrixBasis blocks interface }
      | _ => return none
    start := start + block.rowCount
  return none

private def cachedImages (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (padBasis matrixBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (powers : Nat → K)
    (assignments : Sources) (cached : Option CachedInvocation)
    (vertex : BooleanVertex cubeVariables) :
    IO (Option CachedInvocation × (ProtocolPolynomial.OutputMessage K productionShape × K × K)) := do
  let index := NumericBooleanDomain.index vertex
  let reusable := cached.filter fun value =>
    value.firstRow ≤ index && index < value.firstRow + 94
  let ready := reusable.orElse fun _ => invocationCache? program assignments matrixBasis blocks index
  if let some value := ready then
    if within : value.firstRow ≤ index ∧ index < value.firstRow + 94 then
      let row : Fin 94 := ⟨index - value.firstRow, by omega⟩
      let fresh := Vector.ofFn fun port => ((value.fresh.get row).get port)
      return (ready, PiCCSAggregatedImages.fromRows layout assignments padBasis blocks powers
        vertex fresh (value.carried.get row))
  return (none, ← images program sourceRow padBasis matrixBasis blocks powers assignments vertex)

private def replay (publicPath sourcePath outputPath : System.FilePath)
    (first finish : Nat) : IO UInt32 := do
  unless first < finish && finish ≤ 2 ^ (cubeVariables - 1) do
    throw (IO.userError "invalid first-round pair range")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let statementInput ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre statementInput)
  report [("event", .str "public_coins_ready"),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let sourceStarted ← IO.monoNanosNow
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let assignments := sources masks
  report [("event", .str "original_sources_ready"), ("records", Lean.toJson records),
    ("blocks", Lean.toJson masks.size),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - sourceStarted))]
  let preparationStarted ← IO.monoNanosNow
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let verifierInput := PiCCSPublicReplay.verifierInput statementInput
  let normTable := PiCCSNormCache.prepare power
  let alphaWeights := PiCCSTensorWeights.prepare extensionOps coins.alpha.coordinates.tail
  let priorWeights := PiCCSTensorWeights.prepare extensionOps verifierInput.priorPoint.coordinates.tail
  let basis ← IO.wait (Task.spawn fun _ => PiCCSAggregatedImages.prepare tables power)
  let blocks := PiCCSAggregatedImages.combinedBlock power assignments
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  let cache ← IO.wait (Task.spawn fun _ =>
    PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
  report [("event", .str "matrix_basis_ready"),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - preparationStarted))]
  let mut total : FixedPolynomial K 9 := FixedPolynomial.zero extensionOps.toOps 9
  let mut cached : Option CachedInvocation := none
  for index in [first:finish] do
    if within : index < 2 ^ 27 then
      let suffix := NumericBooleanDomain.vertex 27 ⟨index, within⟩
      let imageStarted ← IO.monoNanosNow
      let (next, low) ← cachedImages program (fun row => cache[row]?) basis.1 basis.2 blocks power
        assignments cached
        (PiCCSFirstRound.endpointVertex (by decide) false suffix)
      cached := next
      let (next, high) ← cachedImages program (fun row => cache[row]?) basis.1 basis.2 blocks power
        assignments cached
        (PiCCSFirstRound.endpointVertex (by decide) true suffix)
      cached := next
      let imageNs := (← IO.monoNanosNow) - imageStarted
      let kernelStarted ← IO.monoNanosNow
      let constructed ← IO.wait (Task.spawn fun _ =>
        PiCCSFirstRoundPair.pairPolynomialWithNorm extensionOps verifierInput power
          (PiCCSCachedSelector.equalitySelector extensionOps suffix coins.alpha alphaWeights)
          (PiCCSCachedSelector.equalitySelector extensionOps suffix verifierInput.priorPoint priorWeights)
          low.1 high.1 low.2.1 high.2.1 low.2.2 high.2.2
          (PiCCSNormCache.sourceNorm normTable power low.1 high.1))
      let term : FixedPolynomial K 9 :=
        PiCCSPublicReplay.degree_eq statementInput ▸ constructed
      total := FixedPolynomial.add extensionOps.toOps total term
      report [("event", .str "pair_complete"), ("pair", Lean.toJson index),
        ("image_ns", Lean.toJson imageNs),
        ("kernel_ns", Lean.toJson ((← IO.monoNanosNow) - kernelStarted))]
    else throw (IO.userError "pair exceeds the selected 27-bit suffix domain")
  let value := Value.array [.atom 1, .atom first, .atom finish,
    .array (total.coefficients.map extensionValue)]
  IO.FS.writeFile outputPath (value.render ++ "\n")
  report [("event", .str "first_round_range_complete"),
    ("first", Lean.toJson first), ("end", Lean.toJson finish),
    ("complete_round", Lean.toJson (first == 0 && finish == 2 ^ 27)),
    ("coefficients", Lean.toJson total.coefficients.length),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

end NightstreamFPrime.Export.PiCCSFirstRoundReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | [publicPath, sourcePath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replay
            publicPath sourcePath outputPath first finish
      | _, _ => throw (IO.userError "pair bounds must be natural numbers")
  | _ =>
      IO.eprintln "usage: replayPiCCSFirstRound <public-input> <original-sources> <new-output> <first-pair> <end-pair>"
      return 2
