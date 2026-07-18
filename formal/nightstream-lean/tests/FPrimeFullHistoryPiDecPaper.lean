import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec

/-! External checks for the public paper Π_DEC refinement. -/

namespace tests.FPrimeFullHistoryPiDecPaper

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec

#check strictAccepted_refines_paperPublicAccepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.strictAccepted_refines_paperPublicAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms strictAccepted_refines_paperPublicAccepted

end tests.FPrimeFullHistoryPiDecPaper
