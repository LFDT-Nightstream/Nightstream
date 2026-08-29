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

/-- Number of augmented functions: Stage 1 is uniform IVC. -/
def slotCount : Nat := 1

/-- The one augmented-function index. -/
def functionIndex : Fin slotCount := ⟨0, by decide⟩

/-- Paper-joint shape for the production profile: the F′ CCS relation has a
`2^28` row cube, one fresh source, 16 running sources, 14 matrices, and
54 coefficient lanes. The cube exponent is also the PiCCS round count. -/
def cubeVariables : Nat := 28

def productionShape : Shape :=
  Phi81MatrixSource.phi81Shape cubeVariables
    productionProfile.freshSources productionProfile.runningSources
    productionProfile.ccsMatrices

theorem productionShape_sourceCount :
    productionShape.sourceCount = productionProfile.piRlcInputs := by
  decide

/-- Digests, application state, and witnesses are Goldilocks field vectors. -/
abbrev Digest := List F
abbrev AppState := List F
abbrev AppWitness := List F

/-- A verifier key is identified inside the state hash by its Poseidon2
digest; the full key is verifier-owned data, never prover-carried. -/
abbrev KeyDigest := List F

/-- Transcript state is the Poseidon2 sponge state. -/
abbrev TranscriptState := Poseidon2.State

/-- The Stage 1 phase order. Each constructor names one slot of the relation
that a gadget builder must refine; the order is the verifier's data
dependency order. Stage 2 inserts its memory phase between `application`
and `outputHash` and its terminal memory acceptance after `terminal`. -/
inductive Phase where
  | priorStateHash     -- u.x = H(vk, i, z0, zi, U, pc); base-case pin
  | piCcsSumcheck      -- log m rounds, transcript-bound
  | piCcsFinal         -- terminal identity incl. norm product over K+k
  | piRlc              -- ρ from the strong set; 17-input combination
  | piDec              -- split_b recomputation, 16 children recombination
  | application        -- z_{i+1} = F(z_i, ω_i); pc_{i+1} = 1
  | outputHash         -- x = H(vk, i+1, z0, z_{i+1}, U_{i+1}, pc_{i+1})
  | terminal           -- decider: open the final running product
deriving Repr, DecidableEq

end NightstreamFPrime.Lifecycle
