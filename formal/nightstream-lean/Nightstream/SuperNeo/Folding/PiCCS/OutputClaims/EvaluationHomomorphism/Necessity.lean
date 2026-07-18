import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Necessity.FlatColumnAction
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Necessity.ScalarBinding

/-!
Necessity ledger for packed `Pi_CCS` output transport.

Assurance tier: model-level.

Owns: countermodels showing that the flat carrier cannot replace the typed
block×lane action and that a source-only scalar check cannot replace a
separately justified equality to the canonical `Pi_RLC` parent projection.

Does not own: production openings, transcript timing, probability bounds,
Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

| Child | Removed obligation | Kernel-checked consequence |
|---|---|---|
| `FlatColumnAction` | typed block×lane action | the flat projection fails to commute with the ring action |
| `ScalarBinding` | right-hand parent scalar binding | a forged source scalar accepts outside mixing and bad-root events |
-/
