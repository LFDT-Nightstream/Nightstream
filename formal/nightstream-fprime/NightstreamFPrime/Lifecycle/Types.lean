import NightstreamFPrime.Spec

/-!
Owns the concrete Stage 1 instantiation of HyperNova Construction 2 over the
SuperNeo NIFS for the fixed Nightstream Goldilocks profile. Every type
parameter of `Spec.HyperNova.Construction2.Paper` is fixed here; no `Prop`
field is left for a caller to supply.

Stage 1 is uniform IVC: one augmented function, so `slotCount = 1` and the
program counter is the constant `1`. The running product is the `CE(b)^k`
vector with `k = 16`; the fresh batch is one CCS instance; NIFS.V is
Π_CCS → Π_RLC → Π_DEC in one message.
-/

namespace NightstreamFPrime.Lifecycle

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def slotCount : Nat := 1

def functionIndex : Fin slotCount := ⟨0, by decide⟩

def cubeVariables : Nat := 28

def productionShape : Shape :=
  Phi81MatrixSource.phi81Shape cubeVariables
    productionProfile.freshSources productionProfile.runningSources
    productionProfile.ccsMatrices

theorem productionShape_sourceCount :
    productionShape.sourceCount = productionProfile.piRlcInputs := by
  decide

abbrev Digest := List F
abbrev AppState := List F
abbrev AppWitness := List F

abbrev KeyDigest := List F

abbrev TranscriptState := Poseidon2.State

inductive Phase where
  | priorStateHash
  | piCcsSumcheck
  | piCcsFinal
  | piRlc
  | piDec
  | application
  | outputHash
  | terminal
deriving Repr, DecidableEq

end NightstreamFPrime.Lifecycle
