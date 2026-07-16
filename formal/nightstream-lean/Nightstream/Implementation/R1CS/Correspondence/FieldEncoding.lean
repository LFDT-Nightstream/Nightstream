import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutManifest
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutWidthFloor
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.SourceCensus
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.PackedSourceCensus
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.OrdinaryPlacement
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.InactiveNoninterference
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.FreshAssignmentPacking
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Phi81ColumnLayoutRefinement
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.DerivedBorrow
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.OrdinaryPrivateField

/-!
Owns: the public import root and shallow ownership map for ordinary private
Goldilocks field encoding.

Does not own: a generated fixed-F-prime branch artifact, Rust materialization,
selector composition, fresh CCS commitment authority, or production row
deletion.

Emits constraints: no. These modules specify semantics, candidate refinement,
and the artifact interface consumed by a future production materializer.

Authority boundary: the complete encoded vector must be the same assignment
committed and norm-checked by the outer fresh CCS relation. Decoded source
residues are semantic inputs only; candidate row removals require the
verifier-owned `b = 2` norm and concrete Rust/artifact refinement in addition
to the model theorems exported here.

| Child path | Mathematical obligation | Emits constraints? | Main owner |
|---|---|---|---|
| `CenteredTernary` | Exact 41-coordinate parser, encoder, width floor, and representation choice | no | encoding semantics |
| `NormDischarged` | Centered alphabet follows from verifier-owned `b = 2` norm | no | semantic row discharge |
| `DerivedNegative` | Reconstruct negative indicators from centered digits | no | candidate refinement |
| `Refinement.DerivedBorrow` | Substitute reconstructed indicators into borrow equations with checked degree | no | candidate row schedule |
| `Refinement.OrdinaryPrivateField` | Exact 41-coordinate materializer, safe logical lowering, and same-index fresh-CCS authority contract | no | ordinary-private refinement boundary |
| `LinearCompiler` | Transport arbitrary source rows through a proof-carrying 41-coordinate layout | no | generic compiler theorem |
| `LayoutManifest` | Fail-closed source/encoded/CE ownership partitions for one generated branch | no | artifact interface |
| `LayoutWidthFloor` | Reconcile exact owner-run lengths and derive generated-census-conditioned `eligible × 41` width floors | no | cost lower-bound theorem |
| `SourceCensus` | Exact source-only role census and prospective per-field-41 budget test for separate base/recursive artifacts | no | source accounting schema |
| `PackedSourceCensus` | Stream packed generated runs into cursor-derived source segments and prove one successful check instantiates `SourceCensusArtifact` | no | artifact decoder and generic soundness |
| `OrdinaryPlacement` | Derive every ordinary 41-coordinate word start and source-phase end from checked source roles plus fixed allocation widths | no | placement-only Rust conformance boundary |
| `InactiveNoninterference` | Selector, selected-equation, and authority-output invariance under branch-relative inactive changes | no | model-level support theorem |
| `FreshAssignmentPacking` | Identity-ordered outer assignment, `D = 54` ring packing, zero padding, and public prefix | no | fresh CCS assignment ABI |
| `Phi81ColumnLayoutRefinement` | Cellwise equality between executable assignment packing and the paper-derived partial 54-lane layout | no | model-level packing correspondence |
-/
