import NightstreamFPrime.Spec.Compression.Akita
import NightstreamFPrime.Spec.Compression.Akita.Binding
import NightstreamFPrime.Spec.Compression.Akita.Challenge
import NightstreamFPrime.Spec.Compression.Akita.Commit
import NightstreamFPrime.Spec.Compression.Akita.Euclid
import NightstreamFPrime.Spec.Compression.Akita.Fold
import NightstreamFPrime.Spec.Compression.Akita.FoldCheck
import NightstreamFPrime.Spec.Compression.Akita.FoldSumCheck
import NightstreamFPrime.Spec.Compression.Akita.Layout
import NightstreamFPrime.Spec.Compression.Akita.LiftReduction
import NightstreamFPrime.Spec.Compression.Akita.Opening
import NightstreamFPrime.Spec.Compression.Akita.Prover
import NightstreamFPrime.Spec.Compression.Akita.Range
import NightstreamFPrime.Spec.Compression.Akita.RingCheck
import NightstreamFPrime.Spec.Compression.Akita.RowBatch
import NightstreamFPrime.Spec.Compression.Akita.Shapes
import NightstreamFPrime.Spec.Compression.Akita.Terminal
import NightstreamFPrime.Spec.Compression.Akita.Trace
import NightstreamFPrime.Spec.Compression.Akita.TraceFold
import NightstreamFPrime.Spec.Compression.Game.Law
import NightstreamFPrime.Spec.Compression.Game.SumCheck
import NightstreamFPrime.Spec.Compression.Lib.Combination
import NightstreamFPrime.Spec.Compression.Lib.LogUp
import NightstreamFPrime.Spec.Compression.Algebra.Batch
import NightstreamFPrime.Spec.Compression.Algebra.Key
import NightstreamFPrime.Spec.Compression.Verifier.Complete
import NightstreamFPrime.Spec.Compression.Production
import NightstreamFPrime.Spec.Compression.Verifier.Pass
import NightstreamFPrime.Spec.Compression.Claim.Slot
import NightstreamFPrime.Spec.Compression.Fresh
import NightstreamFPrime.Spec.Compression.Rust.Soundness
import NightstreamFPrime.Spec.Compression.Rust.Transcript

/-! The soundness of the compression argument over Akita (Rust `ns-compress`, SPEC.md section 6):
a public-coin game with bad sets and its state-restoration bound (`Game`), and the lemmas of its
steps (`Lib`), the exact algebraic identities of the argument (`Algebra`), and the verifier as a game with its
soundness bound and its completeness for the honest prover (`Verifier`), and the production
instance of the argument (`Production`), with the ring equations of a claim as the opening
and the evaluations of `Spec.CE.Holds` (`Claim`), and Π_CCS on one source as a game (`Fresh`),
and Akita as a polynomial commitment scheme with the premises of `[AKITA]` (`Akita`), with the
Akita commitment (`Akita/Commitment`, `Akita/Commit`), the shapes of the key group of the schedule
files (`Akita/Shapes`) and the binding of the commitment of the compression key from Module-SIS
(`Akita/Binding`), and the embedding of the challenge subring (`Akita/Embed`), the norms of the fold
challenges (`Akita/Challenge`) and the opening path of a packing fold (`Akita/Opening`), and the
relation, the weak opening, the weak binding and the extraction of one packing fold
(`Akita/Fold`), and its honest prover and the layout of its scalar opening (`Akita/Prover`), and
the Euclidean norms and the Euclidean route of the weak binding (`Akita/Euclid`), the norm checks of
the next witness (`Akita/Range`), the checks of the ring relations (`Akita/RingCheck`), the packing
map and the trace identity for `k = 2` (`Akita/Trace`), one fold with the evaluation-trace method
(`Akita/TraceFold`) and the evaluation-trace terminal (`Akita/Terminal`), and the row batching and
the quotient-free reduction (`Akita/RowBatch`), the quotient-lift reduction
(`Akita/LiftReduction`), the sum-checks of one fold (`Akita/FoldSumCheck`), the checks of one fold
as a game with its error (`Akita/FoldCheck`) and the link from these checks to the branch relation
(`Akita/Layout`). -/
