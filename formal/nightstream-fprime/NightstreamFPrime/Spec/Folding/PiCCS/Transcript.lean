import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiatShamir

/-!
Paper authority: SuperNeo v1.2, Section 7.3, Step 1; Fiat–Shamir transform.
Obligation: Derive every coordinate of `α`, then `γ`, from the verifier-owned
transcript state before any SumCheck message.

Inputs:
- the transcript oracle;
- the state after the complete public statement absorption.

Outputs:
- `α` in canonical cube-coordinate order;
- `γ`;
- the state after all 28 labelled squeezes.

Parent coverage:
- the pre-SumCheck prefix of `PiCCS.Coverage.transcript`.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.Transcript

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uContext uField uState

def deriveFromState
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (initial : State) : FiatShamir.PreSumcheck Field State shape :=
  let alphaResult := FiatShamir.squeezeMany oracle initial
    (FiatShamir.alphaLabels shape)
  let gammaResult := oracle.squeeze alphaResult.2 .gamma
  {
    alpha := {
      coordinates := alphaResult.1
      dimension := by
        rw [FiatShamir.squeezeMany_values_length oracle,
          FiatShamir.alphaLabels_length]
    }
    gamma := gammaResult.1
    state := gammaResult.2
  }

structure PreSumcheckHolds
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (initial : State)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (finalState : State) : Prop where
  alpha_eq : alpha = (deriveFromState oracle initial).alpha
  gamma_eq : gamma = (deriveFromState oracle initial).gamma
  finalState_eq : finalState = (deriveFromState oracle initial).state

theorem deriveFromState_initialState
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (context : Context) :
    deriveFromState oracle (oracle.initialState context) =
      FiatShamir.derivePreSumcheck oracle context := by
  rfl

theorem derivePreSumcheck_holds
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (context : Context) :
    PreSumcheckHolds oracle (oracle.initialState context)
      (FiatShamir.derivePreSumcheck oracle context).alpha
      (FiatShamir.derivePreSumcheck oracle context).gamma
      (FiatShamir.derivePreSumcheck oracle context).state := by
  rw [← deriveFromState_initialState oracle context]
  exact ⟨rfl, rfl, rfl⟩

theorem derive_alpha_eq_preSumcheck
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (context : Context)
    (certificate : FiatShamir.Certificate Field shape) :
    (FiatShamir.derive oracle context certificate).alpha =
      (FiatShamir.derivePreSumcheck oracle context).alpha := by
  rfl

theorem derive_gamma_eq_preSumcheck
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (context : Context)
    (certificate : FiatShamir.Certificate Field shape) :
    (FiatShamir.derive oracle context certificate).gamma =
      (FiatShamir.derivePreSumcheck oracle context).gamma := by
  rfl

structure RoundsHolds
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (rounds : Fin shape.cubeVariables -> SumCheck.Finite.Message Field)
    (initial : State)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (finalState : State) : Prop where
  roundPoint_eq : roundPoint.coordinates =
    (FiatShamir.deriveRoundsFrom oracle rounds initial
      (canonicalFinIndices shape.cubeVariables)).1
  finalState_eq : finalState =
    (FiatShamir.deriveRoundsFrom oracle rounds initial
      (canonicalFinIndices shape.cubeVariables)).2

theorem derive_rounds_holds
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape)
    (context : Context)
    (certificate : FiatShamir.Certificate Field shape) :
    RoundsHolds oracle certificate.rounds
      (FiatShamir.derivePreSumcheck oracle context).state
      (FiatShamir.derive oracle context certificate).roundPoint
      (FiatShamir.derive oracle context certificate).finalState := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Spec.Folding.PiCCS.Transcript
