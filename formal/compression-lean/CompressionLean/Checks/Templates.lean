import CompressionLean.Spec.Checks

/-!
Owns the proof that a terminal package that passes the template check
(`templatesNamed`, `Spec/Checks.lean`) has F′'s `CompactTemplatesValid`
(`templatesNamed_sound`). `terminalCheck` evaluates the check on
the terminal package.
-/

namespace CompressionLean.Checks

open CompressionLean.Spec
open NightstreamFPrime.Export.Package

/-- A package whose invocations all name a template has `CompactTemplatesValid`. -/
theorem templatesNamed_sound {package : CircuitPackage} (named : templatesNamed package = true) :
    package.CompactTemplatesValid := fun invocation member =>
  Option.isSome_iff_exists.mp (List.all_eq_true.mp named invocation member)

end CompressionLean.Checks
