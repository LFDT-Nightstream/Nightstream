import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction

/-!
Focused interface regression for the operational paper `Pi_RLC` weak
reduction and its generic coordinate-forking boundary.
-/

namespace tests.PiRLCPaperWeakReduction

open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision
open Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction

#check ChallengeVector
#check Oracle
#check ForkSample
#check AcceptedCoordinateFork
#check correctedLoss
#check Theorem10Contract

#check CoordinateForkCollisionReceipt
#check CoordinateForkCollisionReceipt.toRelaxedBindingCollision
#check coordinate_differingExtractions_imply_collisionReceipt
#check samePhi_differingExtractions_imply_relaxedBindingCollision

#check Adversary
#check response
#check verifies
#check acceptedFork_to_completeFork
#check acceptedFork_extracts_correctedAmbient
#check PairedWitnessDisagreement
#check PairedForkCollisionReceipt
#check pairedDisagreement_implies_collisionReceipt
#check ForkingContract
#check OperationalGame
#check weakGame
#check paperWeak

end tests.PiRLCPaperWeakReduction
