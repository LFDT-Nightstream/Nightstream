import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
Owns the sample and the events of the actual reverse-history result of one
application.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaHistoryProbability

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open HyperNovaHistory
variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

abbrev Sample := Statement × Envelope application × List (SourceResult application)

def Accepted (sample : Sample application) : Prop :=
  PerApplicationTerminal.Holds application fits setup sample.1 sample.2.1

def AdviceReturned (sample : Sample application) : Prop :=
  ∃ advice unused,
    run application fits sample.1 sample.2.1 sample.2.2 = .ok (advice, unused) ∧
    advice.length = sample.1.iteration ∧
    advice.foldl application.step sample.1.z0 = sample.1.zi

end NightstreamFPrime.Export.Stage1.HyperNovaHistoryProbability
