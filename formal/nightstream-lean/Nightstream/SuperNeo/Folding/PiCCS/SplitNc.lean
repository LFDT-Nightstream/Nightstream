import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-!
Independent Phi81 SplitNc semantics for SuperNeo `Pi_CCS`.

This component keeps the paper's square one-joint model intact and states the
production-motivated two-domain semantic obligations separately. It derives
fresh completion, carried assignment authority, coefficient matrices, FE
truth, and NC truth from explicit mathematical sources. It does not accept an
existing verifier or circuit as the definition of correctness.

Open obligations are recorded in child contract headers and
`specs/fpr-nifs-bridge.md`; this facade exports no editable diagnostic status
list because changing such a list cannot discharge an obligation.

Owns: the curated imports and ownership map for independent Phi81 Split-NC
semantics.

Does not own: Fiat--Shamir execution, Rust/R1CS refinement, cost accounting,
or row-removal authority.

Emits constraints: no.

| Child | Stable ownership | Excluded boundary |
|---|---|---|
| `Parameters` | semantic shape and row/carrier product domains | protocol acceptance |
| `Sources` | independent matrices, assignments, and derived coefficient images | verifier messages |
| `Semantics` | uncompressed FE and full-carrier NC truth | Fiat--Shamir and implementation |
| `Necessity` | countermodels for obligations that cannot be weakened | production bug claims |
| `Verifier` | typed public carrier and semantic verifier phases | Rust/R1CS refinement and row removal |
-/
