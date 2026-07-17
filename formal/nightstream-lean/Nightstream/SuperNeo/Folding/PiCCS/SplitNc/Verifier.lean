import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types.SourceProjection
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTypes
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductAlignment
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

/-!
Independent verifier-facing layer for production-shaped Split-NC `Pi_CCS`.

Owns: the facade and export boundary for the narrow public carrier,
polynomial, fixed-width SumCheck, sequential transcript, and output-authority
children. The children named below own their mathematics; this file emits no
rows.

Does not own: equations implemented by its children, hidden semantic sources,
complete FE/NC composition, Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

| Child stage | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.verify.types` | public input and raw output values | no | `Verifier.Types` |
| `nifs.pi_ccs.verify.source_projection` | erase independent sources into the raw public verifier carrier | no | `Verifier.Types.SourceProjection` |
| `nifs.pi_ccs.product` | fix the concrete public input/output carrier types | no | `Verifier.ProductTypes` |
| `nifs.pi_ccs.product.partition` | preserve fresh/running source partitions across semantic and public products | no | `Verifier.ProductAlignment` |
| `nifs.pi_ccs.input` | bind every public source field to the independent semantic source and derive genuine input membership | no | `Verifier.InputAuthority` |
| `nifs.pi_ccs.verify.polynomial` | source-derived FE/NC polynomial hierarchy | no | `Verifier.Polynomial` |
| `nifs.pi_ccs.verify.sumcheck` | exact-width finite claimed-chain replay | no | `Verifier.SumCheck` |
| `nifs.pi_ccs.verify.transcript.fe` | mixed-width FE messages are replayed once with a direct row/lane state handoff | no | `Verifier.Transcript.Fe` |
| `nifs.pi_ccs.verify.transcript.nc` | exact-count messages are absorbed before verifier-derived NC challenges | no | `Verifier.Transcript.Nc` |
| `nifs.pi_ccs.verify.output_authority` | output-derived NC terminal is semantic only under explicit `yZcol` source binding | no | `Verifier.OutputAuthority.Nc` |
| `nifs.pi_ccs.output` | canonical CE materialization from source data, input openings, FE point, and bound `yRing` | no | `Verifier.OutputProduct` |
| `nifs.pi_ccs.verify.fe` | canonical mixed-width FE phase evaluator and deterministic soundness composition | no | `Verifier.Fe` |
| `nifs.pi_ccs.fe.prover` | honest mixed-width messages constructed before their own transcript challenges | no | `Verifier.Fe.HonestProver` |
| `nifs.pi_ccs.verify.nc` | canonical message-only NC phase evaluator and deterministic soundness composition | no | `Verifier.Nc` |
| `nifs.pi_ccs.nc.prover` | honest messages constructed before their own transcript challenges | no | `Verifier.Nc.HonestProver` |
| `nifs.pi_ccs.nc.block_lane.prover` | honest exact-count block-then-lane messages constructed before their challenges | no | `Verifier.Nc.BlockLane.HonestProver` |
| `nifs.pi_ccs.verify.protocol` | exact paper-obligation statement and sequential FE/NC soundness composition | no | `Verifier.Protocol` |
| `nifs.pi_ccs.transcript.authority` | one statement-bound schedule derives shared FE/NC coins, machines, and output handoff | no | `Verifier.Protocol.TranscriptAuthority` |
| `nifs.pi_ccs.transcript.block_lane` | one lane dimension and challenge record drive canonical FE plus block×lane NC | no | `Verifier.Protocol.TranscriptAuthority.BlockLane` |
| `nifs.pi_ccs.verify.protocol.block_lane` | statement-derived shared coins, exact FE-to-NC flow, output binding, and composed soundness | no | `Verifier.Protocol.BlockLane` |
| `nifs.pi_ccs.prover.block_lane` | honest canonical FE-to-block×lane-NC construction and source-derived output | no | `Verifier.Protocol.BlockLane.HonestProver` |
| `nifs.pi_ccs.handoff.block_lane` | refine canonical acceptance plus explicit authority into the complete CE product | no | `Verifier.Protocol.BlockLane.OutputRefinement` |
| `nifs.pi_ccs.prover` | honest FE→NC construction and source-bound output at derived points | no | `Verifier.Protocol.HonestProver` |
| `nifs.pi_ccs.handoff` | refine accepted, input-bound, output-bound protocol executions to the complete canonical CE product | no | `Verifier.Protocol.OutputRefinement` |
-/
