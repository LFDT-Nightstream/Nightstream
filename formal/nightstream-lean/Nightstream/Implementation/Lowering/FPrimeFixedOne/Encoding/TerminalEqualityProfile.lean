import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Footprints

/-!
Contract: a finite canonical encoding class for applications whose two
HyperNova terminal relations are equality of authoritative coordinate
encodings.

This is an application profile, not a claim that every HyperNova application
uses equality.  The Boolean checkers remain the frozen executable checkers;
the two `check_*_exact` fields prove their coordinate interpretation.

Owns: exact cross-codec widths, equality-row footprints, field/inverse laws,
and checker-to-coordinate equations for the running and fresh relations.

Does not own: a deployment selection, NIFS verification, a final fold,
caller-supplied relation validity, Rust, or generated rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.Goldilocks

/-- Selected terminal equality encoding for one application. -/
structure TerminalEqualityProfile (parameters : Parameters)
    extends DirectProfile parameters where
  runningWidthsEqual :
    codecs.runningWitness.width = codecs.running.width
  freshWidthsEqual :
    codecs.freshWitness.width = codecs.fresh.width
  runningFootprint :
    parameters.footprints.runningCheck =
      equalityFootprint codecs.running.width
  freshFootprint :
    parameters.footprints.freshCheck =
      equalityFootprint codecs.fresh.width
  runningCheck_exact :
    ∀ key running witness,
      parameters.terminalChecks.runningCheck
          Vocabulary.Step.selected key running witness =
        decide
          (codecs.running.encode running =
            codecs.runningWitness.encode witness)
  freshCheck_exact :
    ∀ key fresh witness,
      parameters.terminalChecks.freshCheck
          Vocabulary.Step.selected key fresh witness =
        decide
          (codecs.fresh.encode fresh =
            codecs.freshWitness.encode witness)

namespace TerminalEqualityProfile

def family
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters) :
    Family (typeSystem parameters) :=
  profile.toDirectProfile.family parameters

end TerminalEqualityProfile

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
