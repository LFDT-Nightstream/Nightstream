import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

/-!
Focused compile-time regressions for the exact Split-NC `Pi_CCS` to CE-product
handoff.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.handoff.acceptance` | transcript acceptance and output authority remain separate premises | transcript digest treated as output authority |
| `nifs.pi_ccs.handoff.opening` | paper truth, input authority, and `yRing` authority yield the concrete CE product | delayed `yZcol` incorrectly made a CE premise |
| `nifs.pi_ccs.handoff.soundness` | accepted execution yields CE membership or a named FE/NC bad event | unconditional soundness or an unnamed escape hatch |
| `nifs.pi_ccs.handoff.completeness` | honest paper-valid sources construct an accepted physical certificate and CE product | certificate-shaped restatement with no honest message construction |
-/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

#check ProductHolds
#check materializedOutputsHold_of_yRingBound
#check accepted_and_outputBound_implies_outputsHold_or_badEvent
#check complete_of_paperObligations
