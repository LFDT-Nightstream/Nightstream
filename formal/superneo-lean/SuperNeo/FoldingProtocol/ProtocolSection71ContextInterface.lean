import SuperNeo.FoldingProtocol.ProtocolSection71Context

/-!
Contract interface for `SuperNeo.ProtocolSection71Context`.

Spec: `specs/ProtocolSection71Context.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
-/

namespace SuperNeo

namespace ProtocolSection71ContextInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProtocolSection71Context"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String :=
  ["§7.1 Structure / CCS / CE", "§7.2 Global Reduction Parameters"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "ProtocolSection71Context"
  , "ProtocolSection71Context.ccsRelation"
  , "ProtocolSection71Context.ceRelation"
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/--
[Role: Theorem-Target] One compact protocol target together with one
specialized paper-faithful Section 7.1 theorem instance.
-/
abbrev ProtocolSection71Context := SuperNeo.ProtocolSection71Context

/-- [Role: Theorem-Target] Recover compact `ccsRelation` from one Section 7.1 context. -/
theorem ProtocolSection71Context_ccsRelation
  (h : ProtocolSection71Context) :
  SuperNeo.ccsRelation h.target :=
  SuperNeo.ProtocolSection71Context.ccsRelation h

/-- [Role: Theorem-Target] Recover compact `ceRelation` from one Section 7.1 context. -/
theorem ProtocolSection71Context_ceRelation
  (h : ProtocolSection71Context) :
  SuperNeo.ceRelation h.target :=
  SuperNeo.ProtocolSection71Context.ceRelation h

end ProtocolSection71ContextInterface

end SuperNeo
