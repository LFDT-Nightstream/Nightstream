import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.FeRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.NcRefinement

/-!
Typed semantic-transcript refinement for the canonical exact `Pi_CCS`
FE-to-NC sub-schedule.

Assurance tier: executable implementation refinement.

Owns: exact-list preservation through the FE/NC typed certificate adapters
and composition of both typed semantic derives with the single canonical
exact schedule.

Does not own: FE/NC polynomial truth, derivation of the FE initial claim or
pre-SumCheck coins, SumCheck security, native/gadget prologue refinement,
Rust/R1CS conformance, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: these theorems consume only `Exact.Carrier` projections.
No loose raw certificate, challenge vector, or independently supplied phase
state enters either semantic transcript derive.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.exact.refinement.fe.encoding` | FE typed row/lane serialization equals the exact raw boundary list | derived | `encode_feRounds_eq_concreteRounds` |
| `nifs.pi_ccs.exact.refinement.nc.encoding` | NC typed serialization equals the exact raw boundary list | derived | `encode_ncRounds_eq_concreteRounds` |
| `nifs.pi_ccs.exact.refinement.fe.derive` | typed FE challenges and successor equal the FE fields of the joint schedule | derived | `feDerive_refines_run` |
| `nifs.pi_ccs.exact.refinement.nc.derive` | typed NC challenges and successor equal the NC fields of the joint schedule, starting from FE's successor | derived | `ncDerive_refines_run` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement

open Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- FE row/lane serialization through the typed checker adapter is exactly
the FE list emitted by the exact carrier codec. -/
@[simp] theorem encode_feRounds_eq_concreteRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    (encode expectedFeInitial carrier).feRounds =
      Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteRounds
        carrier.toFeCertificate := by
  rw [encode_feRounds_values]
  rw [
    Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteRounds_eq_row_then_lane]
  simp only [
    Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteRowRounds,
    Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteLaneRounds,
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.rowRawRounds,
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.laneRawRounds,
    Carrier.toFeCertificate,
    ExactRoundProjection.ofFn_toFunction,
    List.map_map]
  congr 1

/-- NC serialization through the typed transcript adapter is exactly the NC
list emitted by the exact carrier codec. -/
@[simp] theorem encode_ncRounds_eq_concreteRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    (encode expectedFeInitial carrier).ncRounds =
      Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.concreteRounds
        carrier.toNcCertificate := by
  rw [encode_ncRounds_values]
  simp only [
    Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.concreteRounds,
    Carrier.toNcCertificate,
    ExactRoundProjection.ofFn_toFunction]
  rfl

/-- The independent typed FE derive is exactly the FE projection of the
canonical joint schedule. -/
theorem feDerive_refines_run
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    ((Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive
          (Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.machine
            input.expectedFeInitial)
          input.initialState input.carrier.toFeCertificate).challengePoint.coordinates,
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive
          (Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.machine
            input.expectedFeInitial)
          input.initialState input.carrier.toFeCertificate).finalState) =
      ((Schedule.run input).feChallenges.map toK,
        (Schedule.run input).afterFe) := by
  calc
    _ =
        let concrete :=
          Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runFe
            input.initialState (Schedule.rawMessages input)
        (concrete.2.map toK, concrete.1) :=
      Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.derive_refines_runFe
        input.initialState input.expectedFeInitial input.carrier.toFeCertificate
        (Schedule.rawMessages input) (encode_feInitial _ _)
        (encode_feRounds_eq_concreteRounds _ _)
    _ = _ := by
      exact
        (congrArg
          (fun result => (result.2.map toK, result.1))
          (Schedule.run_feJoint input)).symm

/-- The independent typed NC derive is exactly the NC projection of the
canonical joint schedule and starts from the FE successor owned by that same
schedule. -/
theorem ncDerive_refines_run
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    ((Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.derive
          Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine
          (Schedule.run input).afterFe
          input.carrier.toNcCertificate).challengePoint.coordinates,
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.derive
          Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine
          (Schedule.run input).afterFe
          input.carrier.toNcCertificate).finalState) =
      ((Schedule.run input).ncChallenges.map toK,
        (Schedule.run input).afterNc) := by
  calc
    _ =
        let concrete :=
          Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc
            (Schedule.run input).afterFe (Schedule.rawMessages input)
        (concrete.2.map toK, concrete.1) :=
      Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.derive_refines_runNc
        (Schedule.run input).afterFe input.carrier.toNcCertificate
        (Schedule.rawMessages input)
        (encode_ncRounds_eq_concreteRounds _ _)
    _ = _ := by
      exact
        (congrArg
          (fun result => (result.2.map toK, result.1))
          (Schedule.run_ncJoint input)).symm

end Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement
