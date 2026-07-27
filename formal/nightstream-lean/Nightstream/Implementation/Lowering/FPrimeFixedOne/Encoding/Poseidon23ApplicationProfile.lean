import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalEqualityProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23RecipeSemantics
import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge

/-!
Contract: finite application-certificate class for the two fixed-one binding
hashes and the two independent terminal relations.

HyperNova Construction 2 leaves the application state and relation encoding
to setup.  This profile therefore owns explicit finite coordinate projections
from the four authoritative hash operands.  It does not accept a hash result,
an acceptance proposition, or a physical outcome from the caller.

Owns: the exact 23-coordinate Poseidon2 preimage projection, the complete
alignment comparison (including presence), prior/next semantic alignment,
the optional five-coordinate result, and the selected hash footprint.

Does not own: a deployment selection, Rust, generated rows, collision
resistance, Fiat--Shamir, `step`, or `nifsVerify`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace Poseidon23Hash

/-- The physical source vector is the normalized iteration coordinate,
followed by `z0`, current state, and running encodings in call order. -/
def sourceWidth
    {parameters : Parameters}
    (codecs : DataCodecs parameters) : Nat :=
  1 + codecs.state.width + codecs.state.width + codecs.running.width

/-- A bounded projection cannot read an absent source coordinate. -/
structure CoordinatePlan
    (sourceWidth alignmentWidth : Nat) where
  preimage : Fin 23 -> Fin sourceWidth
  alignmentLeft : Fin alignmentWidth -> Fin sourceWidth
  alignmentRight : Fin alignmentWidth -> Fin sourceWidth

def select
    {sourceWidth targetWidth : Nat}
    (source : List Field)
    (projection : Fin targetWidth -> Fin sourceWidth) :
    List Field :=
  List.ofFn fun index => source.getD (projection index).val 0

@[simp] theorem select_length
    {sourceWidth targetWidth : Nat}
    (source : List Field)
    (projection : Fin targetWidth -> Fin sourceWidth) :
    (select source projection).length = targetWidth := by
  simp [select]

/-- Prior and next hashes share one physical projection.  The only semantic
difference is the normalized first coordinate. -/
def normalizedIteration
    (next : Bool)
    (iteration : Nat) : Field :=
  let coordinate := (boundedNatCodec.encode iteration).getD 0 0
  if next then coordinate + 1 else coordinate

def sourceCoordinates
    {parameters : Parameters}
    (codecs : DataCodecs parameters)
    (next : Bool)
    (iteration : Nat)
    (z0 current : parameters.State)
    (running : parameters.Running) :
    List Field :=
  normalizedIteration next iteration ::
    (codecs.state.encode z0 ++
      (codecs.state.encode current ++ codecs.running.encode running))

@[simp] theorem sourceCoordinates_length
    {parameters : Parameters}
    (codecs : DataCodecs parameters)
    (next : Bool)
    (iteration : Nat)
    (z0 current : parameters.State)
    (running : parameters.Running) :
    (sourceCoordinates codecs next iteration z0 current running).length =
      sourceWidth codecs := by
  simp [sourceCoordinates, sourceWidth, codecs.state.encode_length,
    codecs.running.encode_length]
  omega

/-- Pure four-coordinate result of the exact selected fixed-23 core. -/
def digestCoordinates (preimage : List Field) : List Field :=
  List.ofFn fun lane : Fin 4 =>
    residue
      (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.dataChunks
          (fun index => (preimage.getD index.val 0).val))
        lane)

@[simp] theorem digestCoordinates_length (preimage : List Field) :
    (digestCoordinates preimage).length = 4 := by
  simp [digestCoordinates]

/-- The complete total hash result.  Failed presence/alignment has the unique
all-zero optional encoding; success has tag one and four digest lanes. -/
def resultCoordinates
    {sourceWidth alignmentWidth : Nat}
    (plan : CoordinatePlan sourceWidth alignmentWidth)
    (source : List Field) : List Field :=
  if select source plan.alignmentLeft =
      select source plan.alignmentRight
  then 1 :: digestCoordinates (select source plan.preimage)
  else [0, 0, 0, 0, 0]

@[simp] theorem resultCoordinates_length
    {sourceWidth alignmentWidth : Nat}
    (plan : CoordinatePlan sourceWidth alignmentWidth)
    (source : List Field) :
    (resultCoordinates plan source).length = 5 := by
  unfold resultCoordinates
  split <;> simp [digestCoordinates]

def footprint (alignmentWidth : Nat) : CallFootprint where
  recurringRows :=
    (2 * alignmentWidth + alignmentWidth.pred + 1) + 2502
  temporaries :=
    [ auxiliaryLayout 1
    , auxiliaryLayout 23
    , auxiliaryLayout alignmentWidth
    , auxiliaryLayout alignmentWidth
    , auxiliaryLayout alignmentWidth.pred
    , auxiliaryLayout 1
    , auxiliaryLayout 1
    , auxiliaryLayout 4
    , auxiliaryLayout 2464
    ]

end Poseidon23Hash

/-- Complete application-selected class for canonical Phases 3 and 4.

The two semantic equations bind the finite projections to the unchanged
frozen machine hash.  They are serialization/refinement facts, not supplied
acceptance conclusions. -/
structure Poseidon23ApplicationProfile (parameters : Parameters)
    extends TerminalEqualityProfile parameters where
  alignmentWidth : Nat
  hashPlan :
    Poseidon23Hash.CoordinatePlan
      (Poseidon23Hash.sourceWidth codecs) alignmentWidth
  digestWidth : codecs.digest.width = 5
  /-- Every semantic digest is inside the selected codec's decoding domain.
  This is a codec-domain fact, not a hash-security assumption. -/
  digestAdmissible : ∀ digest, codecs.digest.Admissible digest
  hashFootprint :
    parameters.footprints.hash =
      Poseidon23Hash.footprint alignmentWidth
  hashPrior_exact :
    ∀ iteration z0 current running,
      codecs.digest.encode
          (parameters.machine.hash {
            verifierKeys := parameters.setup.verifierKeys
            iteration := iteration
            z0 := z0
            current := current
            running := fun _ => running
            pc := 1
          }) =
        Poseidon23Hash.resultCoordinates hashPlan
          (Poseidon23Hash.sourceCoordinates codecs false
            iteration z0 current running)
  hashNext_exact :
    ∀ iteration z0 current running,
      codecs.digest.encode
          (parameters.machine.hash {
            verifierKeys := parameters.setup.verifierKeys
            iteration := iteration + 1
            z0 := z0
            current := current
            running := fun _ => running
            pc := 1
          }) =
        Poseidon23Hash.resultCoordinates hashPlan
          (Poseidon23Hash.sourceCoordinates codecs true
            iteration z0 current running)

namespace Poseidon23ApplicationProfile

def family
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    Family (typeSystem parameters) :=
  profile.toTerminalEqualityProfile.family parameters

end Poseidon23ApplicationProfile

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
