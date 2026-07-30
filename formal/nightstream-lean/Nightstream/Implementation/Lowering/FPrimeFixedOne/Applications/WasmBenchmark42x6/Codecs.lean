import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Semantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
import Nightstream.Implementation.Lowering.Goldilocks.CodecRecovery

/-!
Contract: exact field-coordinate codecs for the 42-times-6 WASM integration
fixture.

Assurance tier: model-level.

Owns: application state, empty witness, five-coordinate digest, and
five-coordinate encoded-value codecs. It also reuses the selected NIFS
running and fresh codecs as the two terminal witness codecs.

Does not own: hash semantics, terminal relations, physical rows, a deployment,
general WASM data, Rust, or artifacts.

Every application-owned codec is exact-width recoverable. Thus, M4 decoding
cannot fail because a field string has the declared width but no application
value.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Five field coordinates: a presence coordinate followed by four Poseidon2
digest lanes. -/
abbrev Digest := Fin 5 -> Field

/-- The F-prime link encoding has the same exact five-coordinate carrier. -/
abbrev Encoded := Fin 5 -> Field

noncomputable def stateCodec : Codec State :=
  Codec.finFunction 7 fieldCodec

noncomputable def witnessCodec : Codec Witness :=
  Codec.finFunction 0 fieldCodec

noncomputable def digestCodec : Codec Digest :=
  Codec.finFunction 5 fieldCodec

noncomputable def encodedCodec : Codec Encoded :=
  Codec.finFunction 5 fieldCodec

@[simp] theorem stateCodec_width :
    stateCodec.width = 7 := by
  rfl

@[simp] theorem witnessCodec_width :
    witnessCodec.width = 0 := by
  rfl

@[simp] theorem digestCodec_width :
    digestCodec.width = 5 := by
  rfl

@[simp] theorem encodedCodec_width :
    encodedCodec.width = 5 := by
  rfl

theorem stateCodec_admissible (state : State) :
    stateCodec.Admissible state := by
  intro coordinate
  trivial

theorem witnessCodec_admissible (witness : Witness) :
    witnessCodec.Admissible witness := by
  intro coordinate
  exact Fin.elim0 coordinate

theorem digestCodec_admissible (digest : Digest) :
    digestCodec.Admissible digest := by
  intro coordinate
  trivial

theorem encodedCodec_admissible (encoded : Encoded) :
    encodedCodec.Admissible encoded := by
  intro coordinate
  trivial

theorem stateCodec_exactWidthRecoverable :
    stateCodec.ExactWidthRecoverable :=
  Codec.finFunction_exactWidthRecoverable
    fieldCodec Codec.fieldCodec_exactWidthRecoverable 7

theorem witnessCodec_exactWidthRecoverable :
    witnessCodec.ExactWidthRecoverable :=
  Codec.finFunction_exactWidthRecoverable
    fieldCodec Codec.fieldCodec_exactWidthRecoverable 0

theorem digestCodec_exactWidthRecoverable :
    digestCodec.ExactWidthRecoverable :=
  Codec.finFunction_exactWidthRecoverable
    fieldCodec Codec.fieldCodec_exactWidthRecoverable 5

theorem encodedCodec_exactWidthRecoverable :
    encodedCodec.ExactWidthRecoverable :=
  Codec.finFunction_exactWidthRecoverable
    fieldCodec Codec.fieldCodec_exactWidthRecoverable 5

/-- One authoritative state coordinate in the exact seven-coordinate codec. -/
noncomputable def stateView (coordinate : Fin 7) :
    FView stateCodec (fun state => state coordinate) where
  index := ⟨coordinate.val, by
    change coordinate.val < 7
    exact coordinate.isLt⟩
  encodeValue := by
    intro state
    have selected :=
      Codec.encodeFin_getD fieldCodec 7 state coordinate
        ⟨0, by simp [fieldCodec]⟩ 0
    change
      (Codec.encodeFin fieldCodec 7 state).getD coordinate.val 0 =
        state coordinate
    simpa [fieldCodec] using selected

/-- Interpret a five-coordinate result list without adding a digest
authority. Missing coordinates map to zero; the hash recipe separately proves
that its result list has length five. -/
def digestOfCoordinates (coordinates : List Field) : Digest :=
  fun index => coordinates.getD index.val 0

theorem digestOfCoordinates_encode
    (coordinates : List Field)
    (lengthExact : coordinates.length = 5) :
    digestCodec.encode (digestOfCoordinates coordinates) = coordinates := by
  have coordinatesWidth : coordinates.length = digestCodec.width := by
    simpa using lengthExact
  rcases digestCodec_exactWidthRecoverable
      coordinates coordinatesWidth with
    ⟨digest, admissible, encoded⟩
  have digestExact : digest = digestOfCoordinates coordinates := by
    funext index
    have selected :=
      congrArg (fun values : List Field =>
        values.getD index.val 0) encoded
    have encodedCoordinate :=
      Codec.encodeFin_getD fieldCodec 5 digest index
        ⟨0, by simp [fieldCodec]⟩ 0
    change
      (Codec.encodeFin fieldCodec 5 digest).getD index.val 0 =
        coordinates.getD index.val 0 at selected
    have digestCoordinate :
        (Codec.encodeFin fieldCodec 5 digest).getD index.val 0 =
          digest index := by
      simpa [fieldCodec] using encodedCoordinate
    exact digestCoordinate.symm.trans selected
  rw [← digestExact]
  exact encoded

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
