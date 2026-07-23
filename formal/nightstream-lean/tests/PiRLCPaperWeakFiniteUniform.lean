import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform

/-!
Focused interface regression for the finite-uniform operational paper
`Pi_RLC` weak reduction.
-/

namespace tests.PiRLCPaperWeakFiniteUniform

open Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform

#check VerifierData
#check Extractor
#check uniformChallengeExperiment
#check forkSampleExperiment
#check ExtractorExpectedPolynomialTime
#check ExpectedQueriesAtMost
#check theorem10Contract
#check pairedForkExperiment
#check RelaxedBindingSecurity
#check operationalGame
#check weakGame
#check paperWeak
#check @paperWeak

end tests.PiRLCPaperWeakFiniteUniform
