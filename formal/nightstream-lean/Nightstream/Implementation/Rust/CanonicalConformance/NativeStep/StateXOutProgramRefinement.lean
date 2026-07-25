import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Generated.StateXOutProgram
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement

/-!
Contract: artifact-checked refinement of the Rust-emitted `state_x_out`
preimage schedules.

Owns exact equality for all four optional-lane variants, exact field costs,
universal expansion to the independently defined XOut preimage, and
coordinate alignment of a computed XOut with the generated plain public-link
program. It does not evaluate Poseidon2 or formalize compiled Rust semantics.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open StateXOutProgram

namespace GeneratedProgram

def select (semanticPresent nebulaPresent : Bool) : Program :=
  match semanticPresent, nebulaPresent with
  | false, false => Generated.StateXOutProgram.statelessPlain
  | false, true => Generated.StateXOutProgram.statelessNebula
  | true, false => Generated.StateXOutProgram.statefulPlain
  | true, true => Generated.StateXOutProgram.statefulNebula

end GeneratedProgram

theorem generated_eq_canonical
    (semanticPresent nebulaPresent : Bool) :
    GeneratedProgram.select semanticPresent nebulaPresent =
      canonical semanticPresent nebulaPresent := by
  cases semanticPresent <;> cases nebulaPresent <;>
    decide

theorem generated_starts_with_exact_domain
    (semanticPresent nebulaPresent : Bool) :
    (GeneratedProgram.select semanticPresent nebulaPresent).head? =
      some (.domain 0x4e460002) := by
  cases semanticPresent <;> cases nebulaPresent <;>
    decide

theorem generated_statelessPlain_cost :
    cost (GeneratedProgram.select false false) = 23 := by
  rw [generated_eq_canonical]
  exact statelessPlain_cost

theorem generated_statelessNebula_cost :
    cost (GeneratedProgram.select false true) = 28 := by
  rw [generated_eq_canonical]
  exact statelessNebula_cost

theorem generated_statefulPlain_cost :
    cost (GeneratedProgram.select true false) = 27 := by
  rw [generated_eq_canonical]
  exact statefulPlain_cost

theorem generated_statefulNebula_cost :
    cost (GeneratedProgram.select true true) = 32 := by
  rw [generated_eq_canonical]
  exact statefulNebula_cost

/-- Every Rust-emitted variant expands to the exact independent field
preimage selected by the typed optional coordinates. -/
theorem generated_execute_eq_encodeStateXOutPreimage
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    execute
        (GeneratedProgram.select
          preimage.semanticState.isSome preimage.nebula.isSome)
        table preimage =
      encodeStateXOutPreimage table preimage := by
  rw [generated_eq_canonical]
  simpa [forPreimage] using execute_forPreimage table preimage

/-- Coordinate alignment at the outgoing Construction-2 boundary: after XOut
is computed, the Rust-emitted plain public-link program accepts its exact
canonical one-plus-256-bit carrier. -/
theorem generated_publicLink_accepts_computedXOut
    {Params Structure HeaderDigest Running Fresh Nebula NebulaDigestValue :
      Type}
    (semantics :
      XOut.Semantics Params Structure HeaderDigest
        Nightstream.Implementation.Encoding.FPrime.Digest
        Nebula NebulaDigestValue)
    (mode : XOut.Mode)
    (context :
      XOut.Context Params Structure HeaderDigest
        Nightstream.Implementation.Encoding.FPrime.Digest)
    (state :
      State Nightstream.Implementation.Encoding.FPrime.Digest
        Running Fresh Nebula) :
    CanonicalPublicInputLinkProgram.run
        CanonicalPublicInputLinkProgramRefinement.generatedPlain
        (XOut.compute semantics mode context state)
        CanonicalPlainCarrierLink.carrierWidth
        (CanonicalPlainCarrierLink.encodeRawClaim
          (XOut.compute semantics mode context state)) = true := by
  rw [
    CanonicalPublicInputLinkProgramRefinement.generated_run_eq_sourceCheck,
    CanonicalPlainCarrierSource.sourceCheck_eq_true_iff
  ]

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement
