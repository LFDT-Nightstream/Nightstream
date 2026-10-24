import NightstreamFPrime.Export.PackageExpansion

/-!
Owns the template check of a terminal package (`templatesNamed`): every
compact row invocation names a template. A pass gives F′'s
`CompactTemplatesValid` (`compactTemplatesValid_of_named`). `terminalCheck`
evaluates it on the terminal package.
-/

namespace CompressionLean.Checks

open NightstreamFPrime.Export.Package

/-- Every compact row invocation names a template (`CompactTemplatesValid`),
decided. -/
def templatesNamed (package : CircuitPackage) : Bool :=
  package.compactRowInvocations.all fun invocation => (package.compactRowTemplates[invocation.templateIndex]?).isSome

/-- A package whose invocations all name a template has `CompactTemplatesValid`. -/
theorem compactTemplatesValid_of_named {package : CircuitPackage} (named : templatesNamed package = true) :
    package.CompactTemplatesValid := fun invocation member =>
  Option.isSome_iff_exists.mp (List.all_eq_true.mp named invocation member)

end CompressionLean.Checks
