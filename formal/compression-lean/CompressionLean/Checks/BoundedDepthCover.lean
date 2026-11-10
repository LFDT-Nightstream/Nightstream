import CompressionLean.Checks.Fact
import CompressionLean.Spec.BoundedDepthCover

namespace CompressionLean.Checks.BoundedDepth

open CompressionLean.Spec CompressionLean.Spec.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.StepPackage (Package)
open CompressionLean.Circuit (Layer1Key)

/-- A pass of `coverCheck` gives `Spec.BoundedDepth.Covers`. -/
theorem coverFailure_sound {p : Package} {key : Layer1Key} (pass : coverFailure p key = none) : Covers p key := by
  obtain ⟨normCovers, blocksCover, -⟩ := all_of_firstFailure pass
  exact ⟨normCovers, blocksCover⟩

end CompressionLean.Checks.BoundedDepth
