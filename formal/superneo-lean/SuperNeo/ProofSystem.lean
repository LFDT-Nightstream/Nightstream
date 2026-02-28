import SuperNeo.ProofSystem.ConstraintSystem
import SuperNeo.ProofSystem.Sumcheck
import SuperNeo.ProofSystem.Security
import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Folding
import SuperNeo.ProofSystem.Protocol

/-!
Paper-facing proof-system facade.

Intended import for protocol users who want clear, protocol-native surfaces:
- `ConstraintSystem` (CCS/CE relations)
- `Sumcheck`
- `Types`
- `Folding` (Pi_CCS, Pi_RLC, Pi_DEC)
- `Protocol`
-/
