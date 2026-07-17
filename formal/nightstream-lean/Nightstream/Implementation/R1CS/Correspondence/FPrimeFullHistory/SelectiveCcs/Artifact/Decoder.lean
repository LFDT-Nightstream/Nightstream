import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsSchema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Schema

/-!
Contract: fail-closed decoder from artifact-owned selective-CCS wire data to
the current normalized compact matrix bundle.

Owns: schema-version and thirteen-port gates, canonical Goldilocks decoding,
seed-byte validation, exact Rust matrix-tag handling, attachment of a
caller-supplied sampler execution bound, and the final `ProductionValid` gate.

Does not own: generated values, a production profile, sampler fuel selection,
the selective polynomial, Rust conformance, matrix-action refinement, or row
removal.

Emits constraints: no.

Authority boundary: successful decoding returns a proof-carrying
`Bundle.ProductionValid`; generated data never supplies that proof. Sampler
fuel is an external execution certificate and is absent from the wire schema.

| Decode branch | Rejection condition | Normalized result |
|---|---|---|
| CSC | dimension mismatch or noncanonical field word | `Schema.CscPayload` |
| seeded block | malformed 32-byte seed | `Schema.SeededBlock` with supplied fuel |
| matrix tag | identity or empty compact variant | canonical `Schema.CompactMatrix` |
| bundle | wrong version, wrong port count, or failed production validity | `ProductionBundle` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Schema

/-- The only wire schema interpreted by this decoder. -/
def supportedSchemaVersion : Nat := 1

/-- Reject modular aliases instead of reducing untrusted artifact words. -/
def decodeField (value : Nat) : Option F :=
  if canonical : value < goldilocksModulus then
    some ⟨value, canonical⟩
  else
    none

private def SeedBytes.Valid (seed : List Nat) : Prop :=
  seed.length = 32 ∧ ∀ byte ∈ seed, byte < 256

private instance (seed : List Nat) : Decidable (SeedBytes.Valid seed) := by
  unfold SeedBytes.Valid
  infer_instance

private def decodeSeed (seed : List Nat) : Option (List Nat) :=
  if SeedBytes.Valid seed then some seed else none

private def decodeSeedRow
    (seeds : List (List Nat)) : Option (List (List Nat)) :=
  seeds.mapM decodeSeed

/-- Preserve exact CSC arrays after checking the matrix-local dimensions and
canonical field representatives. Structural CSC validity is checked by the
final production-validity gate. -/
def decodeCsc (rows columns : Nat) (payload : RawCsc) : Option CscPayload :=
  if payload.rows = rows ∧ payload.columns = columns then do
    let vals ← payload.vals.mapM decodeField
    pure
      { colPtr := payload.colPtr
        rowIdx := payload.rowIdx
        vals }
  else
    none

/-- Attach a verifier-side execution bound to validated Rust seed bytes. The
bound is not read from, or written back into, artifact data. -/
def decodeSeededBlock
    (rejectionFuel : Nat) (block : RawSeededBlock) : Option SeededBlock := do
  let seedsByOutput ← block.chunkSeedsByRow.mapM decodeSeedRow
  pure
    { rowStart := block.rowStart
      wordStarts := block.wordStarts
      wordWidth := block.wordWidth
      kappa := block.kappa
      messageColumns := block.messageCols
      schedule :=
        { chunkSize := block.chunkSize
          seedsByOutput
          rejectionFuel }
      transformedColumns := block.superneoTransformedColumns }

def decodeGeometricRun
    (run : RawGeometricRun) : Option GeometricRowRun := do
  let initial ← decodeField run.initial
  let ratio ← decodeField run.ratio
  pure
    { row := run.row
      columnStart := run.columnStart
      length := run.length
      initial
      ratio }

/-- Preserve and check the Rust enum tag before erasing it into the normalized
three-component matrix. Identity has no representation in this selective
schema, and an empty compact tag is rejected rather than collapsed to CSC. -/
def decodeMatrix (rejectionFuel rows columns : Nat) :
    RawMatrix → Option CompactMatrix
  | .identity _ => none
  | .csc payload => do
      let csc ← decodeCsc rows columns payload
      pure { csc, seededBlocks := [], geometricRuns := [] }
  | .cscWithSeededPhi81 payload blocks geometricRuns =>
      if blocks ≠ [] ∨ geometricRuns ≠ [] then do
        let csc ← decodeCsc rows columns payload
        let seededBlocks ← blocks.mapM (decodeSeededBlock rejectionFuel)
        let geometricRuns ← geometricRuns.mapM decodeGeometricRun
        pure { csc, seededBlocks, geometricRuns }
      else
        none

/-- Decode raw values without manufacturing validity. Consumers that require
the proof-carrying boundary use `decodeProductionBundle`. -/
def decodeBundle (rejectionFuel : Nat) (raw : RawBundle) : Option Bundle :=
  if raw.schemaVersion = supportedSchemaVersion ∧
      raw.matrices.length = portCount then do
    let matrices ← raw.matrices.mapM
      (decodeMatrix rejectionFuel raw.rows raw.columns)
    pure
      { rows := raw.rows
        columns := raw.columns
        matrices }
  else
    none

/-- Successful output of the complete fail-closed decoder. -/
structure ProductionBundle where
  bundle : Bundle
  productionValid : bundle.ProductionValid

/-- Decode and independently check every normalized production invariant. -/
def decodeProductionBundle
    (rejectionFuel : Nat) (raw : RawBundle) : Option ProductionBundle := do
  let bundle ← decodeBundle rejectionFuel raw
  if valid : bundle.ProductionValid then
    pure ⟨bundle, valid⟩
  else
    none

/-- Existing total interpreters need only the weaker validity component. -/
def ProductionBundle.validated (decoded : ProductionBundle) : ValidatedBundle :=
  { raw := decoded.bundle
    valid := decoded.productionValid.valid }

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder
