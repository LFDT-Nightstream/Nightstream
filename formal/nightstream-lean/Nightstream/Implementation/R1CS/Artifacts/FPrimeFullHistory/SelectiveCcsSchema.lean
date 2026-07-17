/-!
Artifact-owned wire schema for a compact selective-CCS matrix bundle.

Owns: literal Rust matrix variants, raw CSC arrays, raw seeded-Phi81
metadata, raw geometric runs, and the enclosing untyped bundle.

Does not own: field semantics, sampler execution fuel, validity proofs, the
thirteen-port polynomial, correspondence, or production artifact values.

Emits constraints: no.

Authority boundary: these structures are inert data. In particular, `Nat`
coefficients and seed bytes remain untrusted until a handwritten decoder
checks their canonical ranges and assigns mathematical meaning.

| Wire branch | Rust source | Preserved data |
|---|---|---|
| `RawCsc` | `CscMat` | dimensions, column pointers, row indices, values |
| `RawSeededBlock` | `SeededPhi81LinearBlock` | geometry, chunking, seeds, transform flag |
| `RawGeometricRun` | `GeometricRowRun` | row interval and geometric coefficients |
| `RawMatrix` | `CcsMatrix` | exact enum tag and payload |
| `RawBundle` | selective CCS structure | schema version, dimensions, ordered matrices |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Wire

/-- Literal CSC storage. Dimensions are retained per matrix so a decoder can
reject disagreement with the enclosing structure instead of repairing it. -/
structure RawCsc where
  rows : Nat
  columns : Nat
  colPtr : List Nat
  rowIdx : List Nat
  vals : List Nat
deriving DecidableEq, Repr

/-- Literal compact seeded-Phi81 metadata. `chunkSeedsByRow` is Rust's nested
32-byte seed carrier represented with rejectable `Nat` bytes. Execution fuel
is intentionally absent because it is a Lean decoder certificate, not Rust
artifact data. -/
structure RawSeededBlock where
  rowStart : Nat
  wordStarts : List Nat
  wordWidth : Nat
  kappa : Nat
  messageCols : Nat
  chunkSize : Nat
  chunkSeedsByRow : List (List (List Nat))
  superneoTransformedColumns : Bool
deriving DecidableEq, Repr

/-- Literal compact geometric row payload with untrusted field words. -/
structure RawGeometricRun where
  row : Nat
  columnStart : Nat
  length : Nat
  initial : Nat
  ratio : Nat
deriving DecidableEq, Repr

/-- Exact Rust matrix variant. A decoder may reject variants outside the
supported selective profile, but the wire layer never silently normalizes a
tag. -/
inductive RawMatrix where
  | identity (dimension : Nat)
  | csc (payload : RawCsc)
  | cscWithSeededPhi81
      (payload : RawCsc)
      (blocks : List RawSeededBlock)
      (geometricRuns : List RawGeometricRun)
deriving DecidableEq, Repr

/-- Untyped fixed-artifact envelope. Matrix arity remains a list property so
the handwritten decoder can reject any count other than the semantic arity. -/
structure RawBundle where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  matrices : List RawMatrix
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Wire
