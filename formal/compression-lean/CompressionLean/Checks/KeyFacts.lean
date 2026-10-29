import CompressionLean.Checks.Fact

namespace CompressionLean.Checks

open CompressionLean.Spec
open CompressionLean.Circuit (Layer1Key)

theorem keyFailure_sound {key : Layer1Key} (pass : keyFailure key = none) : KeyFacts key := by
  obtain ⟨run, row, slot, root, short, k1, k0, p1, p1count, p0, rowFits, matricesFit, bits, publicFits, sizes, high,
    checked, kappa, blocks, point, matrices, norm, shape, -⟩ := all_of_firstFailure pass
  exact ⟨⟨run, row, slot, root, short, k1, k0, p1, p1count, p0⟩,
    ⟨rowFits, matricesFit, bits, publicFits, sizes, high, checked⟩, ⟨kappa, blocks, point, matrices, norm⟩, shape⟩

end CompressionLean.Checks
