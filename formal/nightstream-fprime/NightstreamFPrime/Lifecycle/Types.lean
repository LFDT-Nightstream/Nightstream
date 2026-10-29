import NightstreamFPrime.Spec

/-! HyperNova Construction 2. -/

namespace NightstreamFPrime.Lifecycle

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def slotCount : Nat := 1

def functionIndex : Fin slotCount := ⟨0, by decide⟩

def cubeVariables : Nat := 28

@[implicit_reducible]
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
