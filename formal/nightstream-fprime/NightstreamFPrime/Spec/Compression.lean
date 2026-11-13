import NightstreamFPrime.Spec.Compression.Game.Law
import NightstreamFPrime.Spec.Compression.Game.SumCheck
import NightstreamFPrime.Spec.Compression.Lib.Combination
import NightstreamFPrime.Spec.Compression.Lib.LogUp

/-! The soundness of the compression argument over Akita (Rust `neo-compress`, SPEC.md section 6):
a public-coin game with bad sets and its state-restoration bound (`Game`), and the lemmas of its
steps (`Lib`). -/
