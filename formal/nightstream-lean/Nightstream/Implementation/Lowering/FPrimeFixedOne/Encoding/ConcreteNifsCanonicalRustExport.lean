import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRustManifestJson

/-!
Exact JSON export for one proof-carrying canonical F-prime deployment.

Owns: the direct composition from
`ConcreteNifsCanonicalCertification.Deployment` to the schema-versioned
proof-free JSON string consumed by Rust.

Does not own: deployment selection, file I/O, witness generation, a Rust
implementation, or equality with the current Rust circuit.

Emits constraints: no. It serializes the exact certified Step and Terminal
programs.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRustExport

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private abbrev TranscriptState := Poseidon2Duplex.State

section

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    TerminalRelations
      (Key dimensions TranscriptState verifierRows)
      (Running dimensions verifierRows)
      RunningWitness
      (Fresh dimensions verifierRows)
      FreshWitness 1)
variable
  (terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations)
variable (widths : Widths) (footprints : Footprints)

/-- Deterministic schema-v1 JSON for the exact deployment manifest.

The application recipe remains the deployment's proof-carrying Step field.
No Rust value or measured cost enters the result. -/
noncomputable def render
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) : String :=
  ConcreteNifsRustManifestJson.render
    (ConcreteNifsCanonicalCertification.manifest
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment)

theorem render_exact
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    render setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment =
      ConcreteNifsRustManifestJson.render
        (ConcreteNifsCanonicalCertification.manifest
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment) :=
  rfl

end

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRustExport
