import CompressionLean.Checks.Fact

/-!
Owns the proof that a key that fails no check of `keyFailure` has `KeyFacts`
(`keyFacts_of_check`); both are in `Spec/Checks.lean`. The production key is
read at run time, so `lowerCheck layer1` and `terminalCheck` run the check on
the key that they read.
-/

namespace CompressionLean.Checks

open CompressionLean.Spec
open CompressionLean.Circuit (Layer1Input)

/-- A key that fails no check has `KeyFacts`. -/
theorem keyFacts_of_check {key : Layer1Input} (pass : keyFailure key = none) : KeyFacts key := by
  obtain ⟨run, row, slot, root, short, k1, k0, p1, p1count, p0, rowFits, matricesFit, bits, publicFits, sizes, high,
    checked, kappa, blocks, point, matrices, norm, shape, -⟩ := all_of_firstFailure pass
  exact ⟨⟨run, row, slot, root, short, k1, k0, p1, p1count, p0⟩,
    ⟨rowFits, matricesFit, bits, publicFits, sizes, high, checked⟩, ⟨kappa, blocks, point, matrices, norm⟩, shape⟩

end CompressionLean.Checks
