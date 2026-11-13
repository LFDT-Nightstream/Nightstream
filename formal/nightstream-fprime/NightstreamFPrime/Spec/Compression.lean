import NightstreamFPrime.Spec.Compression.Game.Law
import NightstreamFPrime.Spec.Compression.Game.SumCheck
import NightstreamFPrime.Spec.Compression.Lib.Combination
import NightstreamFPrime.Spec.Compression.Lib.LogUp
import NightstreamFPrime.Spec.Compression.Algebra.Batch
import NightstreamFPrime.Spec.Compression.Algebra.Key
import NightstreamFPrime.Spec.Compression.Verifier.Complete

/-! The soundness of the compression argument over Akita (Rust `neo-compress`, SPEC.md section 6):
a public-coin game with bad sets and its state-restoration bound (`Game`), and the lemmas of its
steps (`Lib`), the exact algebraic identities of the argument (`Algebra`), and the verifier as a game with its
soundness bound and its completeness at the level of the claims (`Verifier`). -/
