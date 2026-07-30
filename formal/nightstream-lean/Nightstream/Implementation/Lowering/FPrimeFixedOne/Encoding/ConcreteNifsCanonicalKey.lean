import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalSerialization
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine
import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants

/-!
Contract: construct the selected fixed-one ConcretePhi81 verifier key from
Lean-owned protocol choices and setup-owned relation data.

The relation matrices and the Ajtai verifier key are setup inputs.  The
source arity, 270-coordinate public carrier, Split-NC domains, Poseidon2
constants, transcript serialization, transcript schedule, sampler machine,
and challenge-set size are fixed here.

Owns: the concrete `SelectedKey`, its domain/profile proofs, and the exact
selected schedule and sampler equations.

Does not own: an application step, terminal relations, a prover message,
verifier acceptance, physical R1CS rows, Rust, or generated artifacts.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalSerialization
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev SelectedShape (dimensions : Dimensions) :=
  ConcreteNifsPlain270Profile.Shape dimensions

/-- Setup-owned data for one production relation.

`domainCovers` and `rowNonempty` are shape facts.  They cannot be derived from
the weak legacy `Dimensions` record because that record intentionally permits
arbitrary relation sizes.  They do not carry verifier acceptance or a proof
equation. -/
structure RelationSetup
    (dimensions : Dimensions)
    (verifierRows : Nat) where
  verifierKey :
    VerifierKey
      (SelectedShape dimensions) publicRingColumns (publicFits dimensions)
      verifierRows
  system :
    Phi81Relation.Structure
      (RelationShape
        (SelectedShape dimensions) publicRingColumns (publicFits dimensions))
  domainCovers :
    PiCcsDomains.production.nc.Covers (SelectedShape dimensions)
  rowNonempty : 0 < (SelectedShape dimensions).rowVariables

/-- The fixed-active source partition is definitionally aligned with the
production semantic shape. -/
def sourceAlignment (dimensions : Dimensions) :
    SourceAlignment
      (SelectedShape dimensions) productionGlobalParams FixedActive.arity where
  freshCount_eq := rfl
  runningCount_eq := rfl

/-- The selected FE profile uses the production row domain and the exact
six-variable lane cube. -/
def feProfile
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    Polynomial.Fe.SupportedProfile
      (SelectedShape dimensions) PiCcsDomains.production.fe where
  row_nonempty := setup.rowNonempty
  fresh_nonempty := by
    change 0 < 1
    decide
  lane_variables := rfl

/-- The selected transcript serialization is the Lean-owned dynamic
statement/output encoding. -/
noncomputable def selectedSerialization
    (dimensions : Dimensions)
    (verifierRows : Nat) :=
  ConcreteNifsCanonicalSerialization.serialization
    (SelectedShape dimensions) publicRingColumns verifierRows
      (publicFits dimensions)

/-- The sole selected fixed-one NIFS key.

No field is copied from Rust.  Relation matrices and the Ajtai key remain
setup inputs because SuperNeo setup selects them. -/
noncomputable def selected
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    Key dimensions TranscriptState verifierRows where
  template := {
    covers := setup.domainCovers
    key := setup.verifierKey
    alignment := sourceAlignment dimensions
    piCcsSchedule :=
      KSplitNcPoseidonSchedule.schedule
        (domains := PiCcsDomains.production)
        Poseidon2CanonicalConstants.selected
        (selectedSerialization dimensions verifierRows)
    piRlcMachine :=
      PiRlcCanonicalMachine.machine
        Poseidon2CanonicalConstants.selected
    profile := feProfile setup
    challengeSetSize :=
      Nightstream.Implementation.R1CS.goldilocksP
  }
  system := setup.system

@[simp] theorem selected_system
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (selected setup).system = setup.system := by
  rfl

@[simp] theorem selected_constraintPolynomial
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (selected setup).system.constraintPolynomial =
      setup.system.constraintPolynomial := by
  rfl

@[simp] theorem selected_schedule
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (selected setup).template.piCcsSchedule =
      KSplitNcPoseidonSchedule.schedule
        (domains := PiCcsDomains.production)
        Poseidon2CanonicalConstants.selected
        (selectedSerialization dimensions verifierRows) := by
  rfl

@[simp] theorem selected_samplerMachine
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (selected setup).template.piRlcMachine =
      PiRlcCanonicalMachine.machine
        Poseidon2CanonicalConstants.selected := by
  rfl

@[simp] theorem selected_challengeSetSize
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (selected setup).template.challengeSetSize =
      Nightstream.Implementation.R1CS.goldilocksP := by
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
