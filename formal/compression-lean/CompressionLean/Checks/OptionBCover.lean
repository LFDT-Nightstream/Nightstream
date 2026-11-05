import CompressionLean.Checks.Fact
import CompressionLean.Spec.OptionBCover

namespace CompressionLean.Checks.OptionB

open CompressionLean.Spec CompressionLean.Spec.OptionB
open NightstreamFPrime.Export.OptionB.StepPackage (Package)
open CompressionLean.Circuit (Layer1Key)

/-- A pass of `coverCheck` gives `Spec.OptionB.Covers`. -/
theorem coverFailure_sound {p : Package} {key : Layer1Key} (pass : coverFailure p key = none) : Covers p key := by
  obtain ⟨normCovers, blocksCover, -⟩ := all_of_firstFailure pass
  exact ⟨normCovers, blocksCover⟩

end CompressionLean.Checks.OptionB
