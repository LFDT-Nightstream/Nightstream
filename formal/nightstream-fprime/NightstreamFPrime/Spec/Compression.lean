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

/-! The soundness of the compression argument over Akita (Rust `neo-compress`, SPEC.md section 6):
a public-coin game with bad sets and its state-restoration bound (`Game`), and the lemmas of its
steps (`Lib`), the exact algebraic identities of the argument (`Algebra`), and the verifier as a game with its
soundness bound and its completeness at the level of the claims (`Verifier`), and the production
instance of the argument (`Production`), with the ring equations of a claim as the opening
and the evaluations of `Spec.CE.Holds` (`Claim`). -/
