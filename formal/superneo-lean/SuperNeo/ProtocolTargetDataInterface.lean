import SuperNeo.ProtocolTargetData

/-!
Contract interface for `SuperNeo.ProtocolTargetData`.

Spec: `specs/ProtocolTargetData.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
-/

namespace SuperNeo

namespace ProtocolTargetDataInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProtocolTargetData"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String :=
  ["§7.3-§7.5 protocol target / arithmetic obligations"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "ProtocolTargetData"
  , "ProtocolTargetData.invDeltaInvertible"
  , "ProtocolTargetData.protocolTargetProp"
  , "ProtocolTargetData.assumptions"
  , "ProtocolTargetData.ofPaperCarrierDiff"
  , "ProtocolTargetData.ofBasisKernelAssumption"
  , "ProtocolTargetData.ofBasisKernelCheck"
  , "ProtocolTargetData.ofNativePaperCarrierDiff"
  , "protocolTargetProp_of_data"
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- [Role: Theorem-Target] Explicit protocol-side owner for Section 7.5 target data. -/
abbrev ProtocolTargetData := SuperNeo.ProtocolTargetData

/-- [Role: Theorem-Target] Recover invertibility of `ctx.invDelta` from the carried paper-facing data. -/
theorem ProtocolTargetData_invDeltaInvertible
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  invertibleRq ctx.invDelta :=
  SuperNeo.ProtocolTargetData.invDeltaInvertible h

/-- [Role: Theorem-Target] Recover the compact protocol-target proposition. -/
theorem ProtocolTargetData_protocolTargetProp
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  protocolTargetProp ctx :=
  SuperNeo.ProtocolTargetData.protocolTargetProp h

/-- [Role: Theorem-Target] Recover the legacy compatibility assumption bundle. -/
def ProtocolTargetData_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  ProtocolTargetAssumptions ctx :=
  SuperNeo.ProtocolTargetData.assumptions h

/-- [Role: Theorem-Target] Build protocol-side target data on the active paper-carrier-difference route. -/
abbrev ProtocolTargetData_ofPaperCarrierDiff
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolTargetData.ofPaperCarrierDiff (ctx := ctx)

/-- [Role: Theorem-Target] Build protocol-side target data from the finite basis-kernel characterization of Theorem 3. -/
abbrev ProtocolTargetData_ofBasisKernelAssumption
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolTargetData.ofBasisKernelAssumption (ctx := ctx)

/-- [Role: Theorem-Target] Build protocol-side target data from the executable finite basis-kernel checker. -/
abbrev ProtocolTargetData_ofBasisKernelCheck
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolTargetData.ofBasisKernelCheck (ctx := ctx)

/-- [Role: Theorem-Target] Build protocol-side target data on the active native-bar paper-carrier-difference route. -/
abbrev ProtocolTargetData_ofNativePaperCarrierDiff
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolTargetData.ofNativePaperCarrierDiff (ctx := ctx)

/-- [Role: Theorem-Target] Derive the compact protocol target from protocol-side target data. -/
theorem protocolTargetProp_of_data
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  protocolTargetProp ctx :=
  SuperNeo.protocolTargetProp_of_data h

end ProtocolTargetDataInterface

end SuperNeo
