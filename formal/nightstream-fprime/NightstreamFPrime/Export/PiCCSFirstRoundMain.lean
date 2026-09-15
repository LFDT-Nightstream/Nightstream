import NightstreamFPrime.Export.SignedUnitSourceInput
import NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
import NightstreamFPrime.Export.Stage1.PiCCSFirstRound
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache

/-!
Compute a first-round pair range from original witnesses and public claims.
Rust round messages and output evaluations are not inputs. A partial range
is a measurement/accumulation result, not a complete first round. The runner
uses prepared basis reads and the existing numeric matrix interpreter.
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
    (tables : FixedArray (FixedArray (ProductionRelation.SparseForm ringDegree) ringDegree) ringDegree)
    (assignments : Sources) (vertex : BooleanVertex cubeVariables) :
    IO (ProtocolPolynomial.OutputMessage K productionShape) := do
  let some result := PiCCSSourceImages.images? program sourceRow tables layout assignments vertex
    | throw (IO.userError "matrix row failed to load")
  return result

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
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  let cache ← IO.wait (Task.spawn fun _ =>
    PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
  report [("event", .str "matrix_basis_ready"),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - preparationStarted))]
  let mut total : FixedPolynomial K 9 := FixedPolynomial.zero extensionOps.toOps 9
  for index in [first:finish] do
    if within : index < 2 ^ 27 then
      let suffix := NumericBooleanDomain.vertex 27 ⟨index, within⟩
      let imageStarted ← IO.monoNanosNow
      let low ← images program (fun row => cache[row]?) tables assignments
        (PiCCSFirstRound.endpointVertex (by decide) false suffix)
      let high ← images program (fun row => cache[row]?) tables assignments
        (PiCCSFirstRound.endpointVertex (by decide) true suffix)
      let imageNs := (← IO.monoNanosNow) - imageStarted
      let kernelStarted ← IO.monoNanosNow
      let constructed ← IO.wait (Task.spawn fun _ =>
        PiCCSFirstRoundPair.pairPolynomialWithPowers extensionOps
          (PiCCSPublicReplay.verifierInput statementInput) power
          (PiCCSFirstRound.equalitySelector extensionOps suffix coins.alpha)
          (PiCCSFirstRound.equalitySelector extensionOps suffix
            (PiCCSPublicReplay.verifierInput statementInput).priorPoint) low high)
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
