import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Schedule

/-!
Focused regressions for the terminal NC owner tree.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.owner` | exact 81-piece address space | missing or appended owner pieces |
| `nifs.pi_ccs.nc_sumcheck` | named phases cover every index exactly | unclassified physical constraints |
| `nifs.pi_ccs.nc_sumcheck.payloads` | 48 Poseidon and 33 ordinary pieces have the expected owners | phase/payload drift |
| `nifs.pi_ccs.nc_sumcheck.round.algebra` | ordinary tails expose 30 equations plus an optional next pin | hidden mixed-family accounting |
-/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc

#check Rows.ownerPieces_length
#check Rows.pieceAt_mem
#check Schedule.prologuePin_payload
#check Schedule.prologueCall_payload
#check Schedule.firstMessageCall_payload
#check Schedule.firstSqueezePin_payload
#check Schedule.firstSqueezeCall_payload
#check Schedule.firstAlgebra_payload
#check Schedule.laterMessageCall_payload
#check Schedule.laterSqueezePin_payload
#check Schedule.laterSqueezeCall_payload
#check Schedule.laterAlgebra_payload
#check Schedule.firstAlgebra_row_formula
#check Schedule.laterAlgebra_row_formula
#check Schedule.phaseIndices_eq_ownerRange
#check Schedule.familyCounts
