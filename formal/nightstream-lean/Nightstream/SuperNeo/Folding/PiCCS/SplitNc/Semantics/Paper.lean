import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-!
Generalized Section 7.3 relation obligations for production-shaped `Pi_CCS`
sources.

Protocol: SuperNeo `Pi_CCS`.
Phase: independent statement before FE/NC arithmetization or transcript replay.
Constraint family: fresh CCS, all-source strict norm, and carried evaluations.

Owns: the three mathematical relation obligations stated by Section 7.3,
directly over the sole source family.

Does not own: either Split-NC polynomial, the paper's single displayed `Q`,
SumCheck, Fiat--Shamir, verifier acceptance, Rust, R1CS, or costs.

Emits constraints: no.

Authority boundary: every obligation is derived from `Sources.Data`. The
paper's displayed verifier additionally assumes a square relation with
`M_1 = I` and reuses its first matrix evaluation for the norm check. The active
thirteen-role relation has no identity role, so Split-NC instead derives NC
directly from the authoritative assignment. `Holds` names the same underlying
CCS/norm/carried-evaluation relation obligations; it is not a theorem that the
two verifier message flows are identical. This file does not import the
Split-NC verifier and cannot receive a semantic callback, residual table,
challenge, transcript, or accepted certificate.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.paper.fresh_ccs` | every fresh assignment satisfies the explicit CCS structure | independent specification | `FreshCcsHolds` |
| `nifs.pi_ccs.paper.all_source_norm` | every complete-carrier coordinate of every source has strict norm `< 2` | independent specification | `AllSourceNormsHold` |
| `nifs.pi_ccs.paper.carried_evaluations` | every running coefficient claim equals its derived matrix-image evaluation | independent specification | `CarriedEvaluationsHold` |
| `nifs.pi_ccs.paper.statement` | exact conjunction in paper order | independent specification | `Holds` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Every fresh source satisfies the explicit CCS relation. -/
def FreshCcsHolds
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  data.freshBatch.AllConstraintsSatisfied ConcreteCarrier.baseOps

/-- Every authoritative coordinate of every fresh or running source has the
paper's strict `b = 2` norm. -/
def AllSourceNormsHold
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  ∀ source column,
    centeredMagnitude (data.assignment source column) < 2

/-- Every running source satisfies every claimed prior-point coefficient
evaluation. -/
def CarriedEvaluationsHold
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  CarriedEvaluationResidual.AllClaimsHold
    ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
    data.carriedData

/-- Exact Section 7.3 obligation set, ordered as fresh CCS, all-source norm,
then carried evaluations. -/
def Holds
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  FreshCcsHolds data ∧
    AllSourceNormsHold data ∧
    CarriedEvaluationsHold data

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper
