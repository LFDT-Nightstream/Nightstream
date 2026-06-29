import DirectCcsFPrime.Audit.RedTeam.DirectParentOnlyProductionPrivateDecNoSwapAudit

/-!
Typed interface for production private DEC no-swap audit projections.

Spec: `specs/Audit/RedTeam/DirectParentOnlyProductionPrivateDecNoSwapAudit.spec.md`
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionPrivateDecNoSwapAuditInterface

abbrev ProductionContext :=
  DirectParentOnlyProductionPrivateDecNoSwapAudit.ProductionContext

abbrev NonAggregateFacts :=
  @DirectParentOnlyProductionPrivateDecNoSwapAudit.NonAggregateFacts

abbrev PointwiseRequirements :=
  @DirectParentOnlyProductionPrivateDecNoSwapAudit.PointwiseRequirements

abbrev auditOfFacts :=
  @DirectParentOnlyProductionPrivateDecNoSwapAudit.auditOfFacts

abbrev auditOfEndToEnd :=
  @DirectParentOnlyProductionPrivateDecNoSwapAudit.auditOfEndToEnd

end DirectParentOnlyProductionPrivateDecNoSwapAuditInterface

end DirectCcsFPrime
