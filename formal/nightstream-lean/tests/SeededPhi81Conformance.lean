import Nightstream.Implementation.R1CS.Artifacts.SeededPhi81.Generated.SeededPhi81ConformanceArtifact

/-!
Production-class Rust/Lean seeded-Phi81 conformance (campaign bar 3).

Each block below carries the exact seeds and geometry of one production code
path of the frozen campaign profile, and `expectedRows*` are the exact rows of
Rust `SeededPhi81LinearBlock::for_each_term`. `Block.rows` re-derives every
coefficient through the independent Lean sampler (ChaCha8 stream, rejection
replacement, Phi81 rotation, bit-column mapping, zero elision), so each
equality pins the two samplers on that path with exact data.
-/

namespace NightstreamTests.SeededPhi81Conformance

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SeededPhi81ConformanceArtifact

set_option maxRecDepth 65536

/-- Width-41 words, four message columns, uneven two-chunk schedule. -/
example : blockMultiChunk.Valid := by native_decide

example : blockMultiChunk.rows = expectedRowsMultiChunk := by native_decide

/-- Two seeded output rows (kappa 2) with independent per-chunk seeds. -/
example : blockTwoOutputs.Valid := by native_decide

example : blockTwoOutputs.rows = expectedRowsTwoOutputs := by native_decide

/-- The first 54-word draw rejects one word; the accepted vector must take
the exact replacement that Rust takes from the stream tail. -/
example : blockRejection.Valid := by native_decide

example : blockRejection.rows = expectedRowsRejection := by native_decide

end NightstreamTests.SeededPhi81Conformance
