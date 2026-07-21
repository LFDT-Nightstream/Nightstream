import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec.ExactPaperBridge

/-! Focused interface regression for the conditional exact-paper bridge. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec

#check decodedCommitment
#check decodedParent
#check decodedOutput
#check PaperPremises
#check PaperPremises.matrixCount_eq_three
#check strictAccepted_typedCommitmentEquation
#check strictAccepted_decodedEvaluationsEquation
#check strictAccepted_refines_outputAccepted

example
    {dimensions : Dimensions}
    {key : PiRLCAlgebra.Commitment.Key dimensions.shape
      productionProfile.commitmentWidth}
    {assignment : Nat -> Nat}
    (premises : PaperPremises dimensions key assignment) :
    dimensions.shape.matrixCount = 3 :=
  premises.matrixCount_eq_three

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge
