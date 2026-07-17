import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Challenges
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Transport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial

/-!
Typed semantic coin projection from the production-shaped `Pi_CCS`
Poseidon2 challenge phase.

Assurance tier: executable implementation refinement.

Owns: alignment of semantic row/column/lane dimensions with the concrete
challenge schedule; lossless extension-carrier transport; construction of
the FE and NC coin records; and the shared `betaA`/`gamma` authority theorem.

Does not own: the binding prefix that supplies the incoming state, polynomial
truth, SumCheck messages, Fiat--Shamir probability, native/gadget/R1CS
refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: callers supply only the incoming transcript state and
verifier-owned dimensions. Every coin is projected from one execution of
`Challenges.run`; no challenge list, dimension proof, or semantic coin record
can be supplied independently.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.transcript.coins.shape` | concrete `ellD/ellN/ellM` equal semantic lane/row/column dimensions | verifier-owned | `shapeWithDegree` |
| `nifs.pi_ccs.transcript.coins.fe` | `alpha`, `betaA`, `betaR`, and `gamma` come from one concrete replay | computed | `feCoins` |
| `nifs.pi_ccs.transcript.coins.nc` | `betaM`, `betaA`, and `gamma` come from that same replay | computed | `ncCoins` |
| `nifs.pi_ccs.transcript.coins.shared` | FE and NC use identical `betaA` and `gamma` values | direct dataflow | `ncCoins_betaA_eq_feCoins_betaA`, `ncCoins_gamma_eq_feCoins_gamma` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Coins

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- Concrete transcript dimensions aligned with the independent Split-NC
semantic domains. The degree bound is carried for the later complete
SumCheck schedule; challenge derivation itself does not inspect it. -/
def shapeWithDegree
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) : Challenges.Shape where
  ellD := domain.laneVariables
  ellN := shape.rowVariables
  ellM := domain.columnVariables
  degreeBound := degreeBound

/-- The sole concrete pre-SumCheck challenge execution at aligned semantic
dimensions. -/
def run
    (initial : State)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) : Challenges.Output :=
  Challenges.run initial (shapeWithDegree shape domain degreeBound)

/-- FE semantic coins projected from one concrete Poseidon2 execution. -/
def feCoins
    (initial : State)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) : Polynomial.Fe.Coins shape domain :=
  let output := run initial shape domain degreeBound
  {
    alpha := {
      coordinates := output.alpha.map toK
      dimension := by
        rw [List.length_map]
        exact Challenges.run_alpha_length initial
          (shapeWithDegree shape domain degreeBound)
    }
    betaA := {
      coordinates := output.betaA.map toK
      dimension := by
        rw [List.length_map]
        exact Challenges.run_betaA_length initial
          (shapeWithDegree shape domain degreeBound)
    }
    betaR := {
      coordinates := output.betaR.map toK
      dimension := by
        rw [List.length_map]
        exact Challenges.run_betaR_length initial
          (shapeWithDegree shape domain degreeBound)
    }
    gamma := toK output.gamma
  }

/-- NC semantic coins projected from the same concrete Poseidon2 execution. -/
def ncCoins
    (initial : State)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) : Polynomial.Nc.Mixing.Coins domain :=
  let output := run initial shape domain degreeBound
  {
    betaM := {
      coordinates := output.betaM.map toK
      dimension := by
        rw [List.length_map]
        exact Challenges.run_betaM_length initial
          (shapeWithDegree shape domain degreeBound)
    }
    betaA := {
      coordinates := output.betaA.map toK
      dimension := by
        rw [List.length_map]
        exact Challenges.run_betaA_length initial
          (shapeWithDegree shape domain degreeBound)
    }
    gamma := toK output.gamma
  }

/-- FE and NC cannot diverge on the shared lane/Ajtai challenge. -/
theorem ncCoins_betaA_eq_feCoins_betaA
    (initial : State)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) :
    (ncCoins initial shape domain degreeBound).betaA =
      (feCoins initial shape domain degreeBound).betaA := by
  rfl

/-- FE and NC cannot diverge on the shared gamma challenge. -/
theorem ncCoins_gamma_eq_feCoins_gamma
    (initial : State)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) :
    (ncCoins initial shape domain degreeBound).gamma =
      (feCoins initial shape domain degreeBound).gamma := by
  rfl

/-- The semantic coin projection and the complete schedule use the same
post-challenge successor state. -/
theorem run_state
    (initial : State)
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (degreeBound : Nat) :
    (run initial shape domain degreeBound).state =
      (Challenges.run initial
        (shapeWithDegree shape domain degreeBound)).state := by
  rfl

end Nightstream.Implementation.R1CS.PiCcsTranscript.Coins
