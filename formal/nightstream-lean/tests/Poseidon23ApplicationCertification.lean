import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification

/-!
Focused model-level regressions for the profile-indexed Phase 3/4
application certificate.

These tests fix the four-call surface, reject hash swapping and
terminal-call omission/duplication, and expose the exact codec and cost
boundaries.  They do not select a deployment application or claim Rust
conformance.
-/

set_option autoImplicit false

namespace NightstreamTests.Poseidon23ApplicationCertification

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification

theorem hash_order_is_not_interchangeable :
    calls ≠ [.hashNext, .hashPrior, .runningCheck, .freshCheck] := by
  decide

theorem running_cannot_replace_fresh :
    calls ≠ [.hashPrior, .hashNext, .runningCheck, .runningCheck] := by
  decide

theorem fresh_cannot_replace_running :
    calls ≠ [.hashPrior, .hashNext, .freshCheck, .freshCheck] := by
  decide

theorem neither_terminal_check_can_be_omitted :
    calls ≠ [.hashPrior, .hashNext, .runningCheck] ∧
      calls ≠ [.hashPrior, .hashNext, .freshCheck] := by
  decide

/-- A successfully decoded running value cannot arise from a malformed
coordinate width.  This is inherited from the selected semantic codec rather
than supplied by a row artifact. -/
theorem decoded_running_has_exact_width
    {parameters : Parameters}
    (profile : Poseidon23ApplicationProfile parameters)
    {coordinates :
      List Nightstream.SuperNeo.Concrete.F}
    {running : parameters.Running}
    (decoded :
      profile.codecs.running.decode coordinates = some running) :
    coordinates.length = profile.codecs.running.width :=
  profile.codecs.running.length_eq_width_of_decode decoded

/-- The corresponding malformed-width rejection boundary is independent for
fresh inputs. -/
theorem decoded_fresh_has_exact_width
    {parameters : Parameters}
    (profile : Poseidon23ApplicationProfile parameters)
    {coordinates :
      List Nightstream.SuperNeo.Concrete.F}
    {fresh : parameters.Fresh}
    (decoded :
      profile.codecs.fresh.decode coordinates = some fresh) :
    coordinates.length = profile.codecs.fresh.width :=
  profile.codecs.fresh.length_eq_width_of_decode decoded

theorem phase34_certificate_surface
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    calls = [.hashPrior, .hashNext, .runningCheck, .freshCheck] ∧
      (callOutputs parameters Call.hashPrior).map
            (fun port => port.layout.owners) ≠
        (callOutputs parameters Call.hashNext).map
            (fun port => port.layout.owners) ∧
      Call.runningCheck ≠ Call.freshCheck :=
  ⟨calls_exact, hash_outputs_distinct parameters profile,
    terminal_calls_distinct⟩

#check ApplicationCertification.poseidon23
#check ApplicationCertification.call_multiplicities
#check ApplicationCertification.hashPriorCost_exact
#check ApplicationCertification.hashNextCost_exact
#check ApplicationCertification.runningCheckCost_exact
#check ApplicationCertification.freshCheckCost_exact
#check ApplicationCertification.phase34Cost_exact

end NightstreamTests.Poseidon23ApplicationCertification
