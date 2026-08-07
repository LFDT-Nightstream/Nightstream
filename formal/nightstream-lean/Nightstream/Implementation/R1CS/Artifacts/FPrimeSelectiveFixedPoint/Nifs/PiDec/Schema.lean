/-!
Proof-free wire schema for the bounded active strict-`PiDEC` source artifact.

Owns: raw Rust-exported commitment, claim, and outer-layout records.

Does not own: record validity, semantic decoding, compiler rows, assignment
satisfaction, protocol acceptance, or row removal.

Emits constraints: no.

| Record | Exported data | Authority before checking |
|---|---|---|
| `RawCommitment` | dimensions and commitment field columns | untrusted data |
| `RawClaim` | active paper carrier plus implementation sidecars | untrusted data |
| `RawLayout` | source interval layout and canonical-X traces | untrusted data |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec

structure RawCommitment where
  dCol : Nat
  kappaCol : Nat
  dataCols : List Nat
deriving DecidableEq, Repr

structure RawClaim where
  commitment : RawCommitment
  xActiveCols : List Nat
  xRows : Nat
  xWidth : Nat
  xRowsCol : Nat
  xWidthCol : Nat
  mIn : Nat
  mInCol : Nat
  yRingCols : List (List Nat)
  ctCols : List (Nat × Nat)
  rCols : List (Nat × Nat)
  foldDigestCols : List Nat
deriving DecidableEq, Repr

structure RawLayout where
  schemaVersion : Nat
  radix : Nat
  ringDimension : Nat
  extensionLimbs : Nat
  firstAllocatedColumn : Nat
  parent : RawClaim
  children : List RawClaim
  xSignTraces : List (Nat × Nat)
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec
