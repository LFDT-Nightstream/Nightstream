import Mathlib.Data.Fintype.Basic
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Codecs
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.StepRecipeCore
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23ApplicationProfile

/-!
Contract: typed fixed-one application data for the 42-times-6 WASM
integration fixture.

Assurance tier: model-level.

Owns: the benchmark machine, canonical NIFS carrier codecs, independent
terminal equality relations, derived widths, and every non-NIFS footprint.

Does not own: a production WASM compiler, one Ajtai setup, the NIFS footprint,
a complete deployment, a recursive fixed point, Rust, or generated artifacts.
The NIFS footprint remains an explicit argument until it is derived from the
complete selected verifier rows.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
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

private abbrev TranscriptState := Poseidon2Duplex.State

local instance stateDecidableEq : DecidableEq State :=
  Fintype.decidablePiFintype

local instance encodedDecidableEq : DecidableEq Encoded :=
  Fintype.decidablePiFintype

abbrev BenchmarkKey
    (dimensions : Dimensions) (verifierRows : Nat) :=
  ConcreteNifsPlain270Profile.Key dimensions TranscriptState verifierRows

abbrev BenchmarkRunning
    (dimensions : Dimensions) (verifierRows : Nat) :=
  ConcreteNifsPlain270Profile.Running dimensions verifierRows

abbrev BenchmarkFresh
    (dimensions : Dimensions) (verifierRows : Nat) :=
  ConcreteNifsPlain270Profile.Fresh dimensions verifierRows

abbrev BenchmarkProof
    (dimensions : Dimensions) (verifierRows : Nat) :=
  ConcreteNifsPlain270Profile.Proof dimensions TranscriptState verifierRows

private abbrev SelectedShape (dimensions : Dimensions) :=
  ConcreteNifsPlain270Profile.Shape dimensions

noncomputable def runningCodec
    (dimensions : Dimensions) (verifierRows : Nat) :
    Codec (BenchmarkRunning dimensions verifierRows) :=
  ConcreteNifsCanonicalRunningCodec.runningCodec
    (SelectedShape dimensions) publicRingColumns verifierRows
      (publicFits dimensions)

noncomputable def freshCodec
    (dimensions : Dimensions) (verifierRows : Nat) :
    Codec (BenchmarkFresh dimensions verifierRows) :=
  ConcreteNifsCanonicalRunningCodec.freshCodec
    (SelectedShape dimensions) publicRingColumns verifierRows
      (publicFits dimensions)

noncomputable def proofCodec
    {dimensions : Dimensions} {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    Codec (BenchmarkProof dimensions verifierRows) :=
  ConcreteNifsCanonicalProofCodec.proofCodec
    (SelectedShape dimensions) setup.system.constraintPolynomial 0
      publicRingColumns verifierRows (publicFits dimensions)

/-- Terminal relations compare the exact canonical NIFS carrier coordinates.
No semantic equality instance for the large carrier is needed. -/
noncomputable def terminalRelations
    (dimensions : Dimensions) (verifierRows : Nat) :
    TerminalRelations
      (BenchmarkKey dimensions verifierRows)
      (BenchmarkRunning dimensions verifierRows)
      (BenchmarkRunning dimensions verifierRows)
      (BenchmarkFresh dimensions verifierRows)
      (BenchmarkFresh dimensions verifierRows) 1 where
  runningHolds := fun _ _ value witness =>
    (runningCodec dimensions verifierRows).encode value =
      (runningCodec dimensions verifierRows).encode witness
  freshHolds := fun _ _ value witness =>
    (freshCodec dimensions verifierRows).encode value =
      (freshCodec dimensions verifierRows).encode witness

noncomputable def terminalChecks
    (dimensions : Dimensions) (verifierRows : Nat) :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      (terminalRelations dimensions verifierRows) where
  runningCheck := fun _ _ value witness =>
    decide
      ((runningCodec dimensions verifierRows).encode value =
        (runningCodec dimensions verifierRows).encode witness)
  freshCheck := fun _ _ value witness =>
    decide
      ((freshCodec dimensions verifierRows).encode value =
        (freshCodec dimensions verifierRows).encode witness)
  runningCheck_iff := by
    intro slot key value witness
    simp [terminalRelations]
  freshCheck_iff := by
    intro slot key value witness
    simp [terminalRelations]

/-- First five complete public-input coordinates of the selected fresh
instance. This is the benchmark link value; it is not a digest authority. -/
def freshPublic
    {dimensions : Dimensions} {verifierRows : Nat}
    (fresh : BenchmarkFresh dimensions verifierRows) : Encoded :=
  fun coordinate =>
    fresh.publicInput
      ⟨coordinate.val, by
        have coordinateLt : coordinate.val < 5 := coordinate.isLt
        change coordinate.val < ringDegree * publicRingColumns
        simp only [ringDegree, publicRingColumns]
        omega⟩

def encodeInstance (digest : Digest) : Encoded :=
  digest

/-- The 23 selected preimage slots cycle through the first fifteen source
coordinates. Slot zero therefore carries the normalized iteration, and no
slot can read outside the source vector. -/
def hashPreimageCoordinate
    {parameters : Parameters}
    (codecs : DataCodecs parameters)
    (sourceAtLeast15 : 15 ≤ Poseidon23Hash.sourceWidth codecs)
    (slot : Fin 23) :
    Fin (Poseidon23Hash.sourceWidth codecs) :=
  ⟨slot.val % 15, by
    have reduced : slot.val % 15 < 15 := Nat.mod_lt _ (by decide)
    exact Nat.lt_of_lt_of_le reduced sourceAtLeast15⟩

def hashPlan
    {parameters : Parameters}
    (codecs : DataCodecs parameters)
    (sourceAtLeast15 : 15 ≤ Poseidon23Hash.sourceWidth codecs) :
    Poseidon23Hash.CoordinatePlan
      (Poseidon23Hash.sourceWidth codecs) 0 where
  preimage := hashPreimageCoordinate codecs sourceAtLeast15
  alignmentLeft := Fin.elim0
  alignmentRight := Fin.elim0

@[simp] theorem hashPreimageCoordinate_zero
    {parameters : Parameters}
    (codecs : DataCodecs parameters)
    (sourceAtLeast15 : 15 ≤ Poseidon23Hash.sourceWidth codecs) :
    (hashPreimageCoordinate codecs sourceAtLeast15
      ⟨0, by decide⟩).val = 0 := by
  simp [hashPreimageCoordinate]

/-- Closed benchmark plan before a dependent `DataCodecs` value exists.
Its source width is definitionally the width that `dataCodecs` later selects. -/
def benchmarkHashPlan
    {dimensions : Dimensions} {verifierRows : Nat} :
    Poseidon23Hash.CoordinatePlan
      (1 + stateCodec.width + stateCodec.width +
        (runningCodec dimensions verifierRows).width) 0 where
  preimage := fun slot =>
    ⟨slot.val % 15, by
      have reduced : slot.val % 15 < 15 := Nat.mod_lt _ (by decide)
      have stateWidth : stateCodec.width = 7 := stateCodec_width
      omega⟩
  alignmentLeft := Fin.elim0
  alignmentRight := Fin.elim0

@[simp] theorem benchmarkHashPlan_slot_zero
    {dimensions : Dimensions} {verifierRows : Nat} :
    (benchmarkHashPlan
      (dimensions := dimensions) (verifierRows := verifierRows)
      ).preimage ⟨0, by decide⟩ = ⟨0, by
        have stateWidth : stateCodec.width = 7 := stateCodec_width
        omega⟩ := by
  rfl

/-- Machine-level source coordinates use the actual iteration value. The
profile later proves that `iteration + 1` is the same field coordinate as the
next-mode normalized source. -/
noncomputable def hashSource
    {dimensions : Dimensions} {verifierRows : Nat}
    (iteration : Nat)
    (z0 current : State)
    (running : BenchmarkRunning dimensions verifierRows) :
    List Field :=
  (boundedNatCodec.encode iteration).getD 0 0 ::
    (stateCodec.encode z0 ++
      (stateCodec.encode current ++
        (runningCodec dimensions verifierRows).encode running))

noncomputable def hashResult
    {dimensions : Dimensions} {verifierRows : Nat}
    (plan :
      Poseidon23Hash.CoordinatePlan
        (1 + stateCodec.width + stateCodec.width +
          (runningCodec dimensions verifierRows).width) 0)
    (iteration : Nat)
    (z0 current : State)
    (running : BenchmarkRunning dimensions verifierRows) :
    Digest :=
  digestOfCoordinates
    (Poseidon23Hash.resultCoordinates plan
      (hashSource iteration z0 current running))

/-- Benchmark machine at the application boundary. It remains generic in the
selected NIFS setup key. -/
noncomputable def machine
    {dimensions : Dimensions} {verifierRows : Nat}
    (plan :
      Poseidon23Hash.CoordinatePlan
        (1 + stateCodec.width + stateCodec.width +
          (runningCodec dimensions verifierRows).width) 0) :
    Machine
      (BenchmarkKey dimensions verifierRows)
      Digest State Witness
      (BenchmarkRunning dimensions verifierRows)
      (BenchmarkFresh dimensions verifierRows)
      Encoded 1 where
  control := fun _ _ => ⟨0, by decide⟩
  step := fun _ state witness => WasmBenchmark42x6.step state witness
  freshPublic := WasmBenchmark42x6.freshPublic
  encodeInstance := WasmBenchmark42x6.encodeInstance
  hash := fun input =>
    hashResult plan input.iteration input.z0 input.current
      (input.running ⟨0, by decide⟩)

noncomputable def dataCodecs
    {dimensions : Dimensions} {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows)
    (defaultRunning : BenchmarkRunning dimensions verifierRows)
    (plan :
      Poseidon23Hash.CoordinatePlan
        (1 + stateCodec.width + stateCodec.width +
          (runningCodec dimensions verifierRows).width) 0)
    (widths : Widths) (footprints : Footprints) :
    DataCodecs
      (ConcreteNifsPlain270Profile.selected dimensions
        (fun _ => ConcreteNifsCanonicalKey.selected setup)
        defaultRunning (machine plan)
        (terminalRelations dimensions verifierRows)
        (terminalChecks dimensions verifierRows) widths footprints) where
  field := fieldCodec
  digest := digestCodec
  state := stateCodec
  witness := witnessCodec
  running := runningCodec dimensions verifierRows
  fresh := freshCodec dimensions verifierRows
  nifsProof := proofCodec setup
  encoded := encodedCodec
  runningWitness := runningCodec dimensions verifierRows
  freshWitness := freshCodec dimensions verifierRows

noncomputable def widths
    {dimensions : Dimensions} {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) : Widths where
  iteration := 1
  state := stateCodec.width
  witness := witnessCodec.width
  running := (runningCodec dimensions verifierRows).width
  fresh := (freshCodec dimensions verifierRows).width
  nifsProof := (proofCodec setup).width
  digest := digestCodec.width
  encoded := encodedCodec.width
  runningWitness := (runningCodec dimensions verifierRows).width
  freshWitness := (freshCodec dimensions verifierRows).width
  bit := 1

noncomputable def footprints
    {dimensions : Dimensions} {verifierRows : Nat}
    (nifs : CallFootprint) : Footprints where
  iterationZero := zeroFootprint
  stateEqual := equalityFootprint stateCodec.width
  step := stepFootprint
  hash := Poseidon23Hash.footprint 0
  freshPublic := affineFootprint encodedCodec.width
  encodeInstance := affineFootprint encodedCodec.width
  encodedEqual := equalityFootprint encodedCodec.width
  nifsVerify := nifs
  runningCheck := equalityFootprint
    (runningCodec dimensions verifierRows).width
  freshCheck := equalityFootprint
    (freshCodec dimensions verifierRows).width

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
