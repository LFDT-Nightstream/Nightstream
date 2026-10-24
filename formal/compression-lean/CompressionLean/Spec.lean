import CompressionLean.Spec.Relation
import CompressionLean.Spec.Replay
import CompressionLean.Spec.Messages
import CompressionLean.Spec.Openings
import CompressionLean.Spec.Checks
import CompressionLean.Spec.Parent
import CompressionLean.Spec.Program
import CompressionLean.Spec.Bound

/-!
`Spec/` owns the vocabulary of the claims: every definition that a claim of
`Claims.lean` names and that no model or port folder below owns, and the
definitions that those read. It holds definitions only; the proofs about
them are in the folders above (`Bridge/`, `Final/`, `Checks/`, `Sim/`).

- `Relation`, `Replay`, `Messages`, `Openings`: the replay's relation,
  statement and proof as the layer-1 model reads them, and the WHIR and
  Merkle assumptions.
- `Checks`: the facts that the claims take about Rust's files, and the checks
  that decide them.
- `Parent`: the claim of a final proof in F′'s terms, and the setup and runs
  of F′'s key.
- `Program`, `Bound`: the production final program, what layer 1 reads from
  its witness, and the event that the production bound bounds.

`CompressionLeanTests/Vocabulary.lean` fails the build when a claim names a
constant of a proof folder.
-/
