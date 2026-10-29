import CompressionLean.Spec.Checks

namespace CompressionLean.Checks

open CompressionLean.Spec
open NightstreamFPrime.Export.Package

theorem templatesNamed_sound {package : CircuitPackage} (named : templatesNamed package = true) :
    package.CompactTemplatesValid := fun invocation member =>
  Option.isSome_iff_exists.mp (List.all_eq_true.mp named invocation member)

end CompressionLean.Checks
