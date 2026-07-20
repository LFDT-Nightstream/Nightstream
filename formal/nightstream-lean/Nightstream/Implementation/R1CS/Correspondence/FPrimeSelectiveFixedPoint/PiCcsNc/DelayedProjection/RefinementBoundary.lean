import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction

/-!
Concrete refinement boundary for the full-witness delayed `y_zcol` checker.

Assurance tier: model-level until generated rows and the Rust verifier
discharge the failures below.

Owns: decomposition of the remaining semantic-input premise into the exact
public polynomial input and public source-product bindings; decomposition of
the child-opening premise; explicit verifier-key continuity; and recursive
and terminal theorems which derive all positive premises before invoking the
full-witness delayed soundness chain.

Does not own: generated source-product rows, commitment extraction, the
terminal opening verifier, Rust dataflow, primitive binding probability,
`y_ring`, costs, or row removal.

Emits constraints: none; correspondence theorem only.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.refinement.polynomial_input` | bind the public FE input to the authoritative source projection | checked boundary |
| `f_prime.pi_ccs_nc.delayed.refinement.source_product` | bind every source structure, commitment, public input, point, evaluation, and stage | checked / commitment boundary |
| `f_prime.pi_ccs_nc.delayed.refinement.children` | obtain canonical child openings | extraction / binding boundary |
| `f_prime.pi_ccs_nc.delayed.refinement.key` | preserve the verifier-owned commitment key across the delayed edge | checked setup boundary |
| `f_prime.pi_ccs_nc.delayed.refinement.recursive` | successful adjacent checks imply the previous semantic fold or one exact failure | derived |
| `f_prime.pi_ccs_nc.delayed.refinement.terminal` | successful terminal checking implies the final semantic fold or one exact failure | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RefinementBoundary

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open PackedWitness
open PackedWitnessProduction

universe uState uEncoding uDigest

variable
  {shape : SemanticShape}
  {State : Type uState}
  {Encoding : Type uEncoding}
  {Digest : Type uDigest}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Every physical refinement still outside the adjacent full-witness
checker, split along the existing independent semantic interfaces.  In
particular, no constructor says merely that an output is unbound. -/
inductive RecursiveRefinementFailure
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape) : Prop where
  | previousPolynomialInput
      (failure : ¬ PublicInputBound previousContext.materialize
        (decodedData previousTemplate previousWitnesses)) :
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses
  | previousSourceProductBinding
      (failure : ¬ InputBound previousContext.materialize
        (decodedData previousTemplate previousWitnesses)) :
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses
  | previousChildOpening
      (failure : ¬ ChildOpenings previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        previousCertificate) :
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses
  | nextPolynomialInput
      (failure : ¬ PublicInputBound nextContext.materialize
        (decodedData nextTemplate nextWitnesses)) :
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses
  | nextSourceProductBinding
      (failure : ¬ InputBound nextContext.materialize
        (decodedData nextTemplate nextWitnesses)) :
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses
  | verifierKeyContinuity
      (failure : ¬ nextContext.materialize.key =
        previousContext.materialize.key) :
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses

/-- Two adjacent full-witness executions derive production acceptance, both
state bindings, both semantic inputs, child authority, and key continuity,
or expose the first exact refinement boundary that failed.  The successful
branch retains the complete `SemanticFold.Holds` relation, including the
delayed parent and child equalities. -/
theorem checkedPair_implies_previousSemanticFold_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked :
      PackedWitnessProduction.check previousContext previousTemplate
        previousWitnesses previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked :
      PackedWitnessProduction.check nextContext nextTemplate nextWitnesses
        nextCertificate = true)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true) :
    SemanticFold.Holds previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate ∨
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses := by
  classical
  by_cases previousPolynomial : PublicInputBound
      previousContext.materialize
      (decodedData previousTemplate previousWitnesses)
  · by_cases previousSources : InputBound previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
    · by_cases previousChildren : ChildOpenings previousContext.materialize
          (decodedData previousTemplate previousWitnesses)
          previousCertificate
      · by_cases nextPolynomial : PublicInputBound nextContext.materialize
            (decodedData nextTemplate nextWitnesses)
        · by_cases nextSources : InputBound nextContext.materialize
              (decodedData nextTemplate nextWitnesses)
          · by_cases sameKey : nextContext.materialize.key =
                previousContext.materialize.key
            · let previousInput : SemanticInput previousContext.materialize
                  (decodedData previousTemplate previousWitnesses) := {
                publicInput := previousPolynomial
                sources := previousSources
              }
              let nextInput : SemanticInput nextContext.materialize
                  (decodedData nextTemplate nextWitnesses) := {
                publicInput := nextPolynomial
                sources := nextSources
              }
              rcases
                  PackedWitnessProduction.checkedPair_of_stateChecks_implies_previousSemanticFold_or_badEvent
                    noZeroDivisors scheme stateDigest previousContext
                    previousTemplate previousWitnesses previousCertificate
                    previousInput previousChildren previousChecked
                    previousStateChecked nextContext nextTemplate nextWitnesses
                    nextCertificate nextInput nextChecked nextStateChecked
                    sameKey with semantic | yRing | bad
              · exact Or.inl semantic
              · exact Or.inr (Or.inl yRing)
              · exact Or.inr (Or.inr (Or.inl bad))
            · exact Or.inr (Or.inr (Or.inr
                (.verifierKeyContinuity sameKey)))
          · exact Or.inr (Or.inr (Or.inr
              (.nextSourceProductBinding nextSources)))
        · exact Or.inr (Or.inr (Or.inr
            (.nextPolynomialInput nextPolynomial)))
      · exact Or.inr (Or.inr (Or.inr
          (.previousChildOpening previousChildren)))
    · exact Or.inr (Or.inr (Or.inr
        (.previousSourceProductBinding previousSources)))
  · exact Or.inr (Or.inr (Or.inr
      (.previousPolynomialInput previousPolynomial)))

/-- Strong backward claims-level refinement step. The successful branch keeps
the predecessor packed equation together with its semantic fold, allowing the
terminal-derived value to propagate through every edge of a finite trace. -/
theorem messageCheckedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked : PackedWitnessProduction.messageCheck previousContext
      previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked : PackedWitnessProduction.messageCheck nextContext
      nextCertificate = true)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      nextContext.materialize.covers (decodedData nextTemplate nextWitnesses)
      (CombinedNc.ProductionPiCcs.ncPoint nextContext.materialize
        nextCertificate).block nextCertificate.piCcs.output)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true) :
    (Terminal.PackedYZcolBoundAtBlock previousContext.materialize.covers
          (decodedData previousTemplate previousWitnesses)
          (DelayedProduction.outgoingPending previousContext.materialize
            previousCertificate).oldBlock
          previousCertificate.piCcs.output ∧
        SemanticFold.Holds previousContext.materialize
          (decodedData previousTemplate previousWitnesses)
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate ∨
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses := by
  classical
  by_cases previousPolynomial : PublicInputBound previousContext.materialize
      (decodedData previousTemplate previousWitnesses)
  · by_cases previousSources : InputBound previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
    · by_cases previousChildren : ChildOpenings previousContext.materialize
          (decodedData previousTemplate previousWitnesses)
          previousCertificate
      · by_cases nextPolynomial : PublicInputBound nextContext.materialize
            (decodedData nextTemplate nextWitnesses)
        · by_cases nextSources : InputBound nextContext.materialize
              (decodedData nextTemplate nextWitnesses)
          · by_cases sameKey : nextContext.materialize.key =
                previousContext.materialize.key
            · let previousInput : SemanticInput previousContext.materialize
                  (decodedData previousTemplate previousWitnesses) := {
                publicInput := previousPolynomial
                sources := previousSources
              }
              let nextInput : SemanticInput nextContext.materialize
                  (decodedData nextTemplate nextWitnesses) := {
                publicInput := nextPolynomial
                sources := nextSources
              }
              rcases
                  PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPackedAndSemanticFold_or_badEvent
                    noZeroDivisors scheme stateDigest previousContext
                    previousTemplate previousWitnesses previousCertificate
                    previousInput previousChildren previousChecked
                    previousStateChecked nextContext nextTemplate nextWitnesses
                    nextCertificate nextInput nextChecked nextPacked
                    nextStateChecked sameKey with semantic | yRing | bad
              · exact Or.inl semantic
              · exact Or.inr (Or.inl yRing)
              · exact Or.inr (Or.inr (Or.inl bad))
            · exact Or.inr (Or.inr (Or.inr
                (.verifierKeyContinuity sameKey)))
          · exact Or.inr (Or.inr (Or.inr
              (.nextSourceProductBinding nextSources)))
        · exact Or.inr (Or.inr (Or.inr
            (.nextPolynomialInput nextPolynomial)))
      · exact Or.inr (Or.inr (Or.inr
          (.previousChildOpening previousChildren)))
    · exact Or.inr (Or.inr (Or.inr
        (.previousSourceProductBinding previousSources)))
  · exact Or.inr (Or.inr (Or.inr
      (.previousPolynomialInput previousPolynomial)))

/-- Backward claims-level refinement step. The positive successor packed
opening is generated by terminal closure or the following accepted edge; this
projection forgets only the induction value and retains the exact failure
partition. -/
theorem messageCheckedPair_of_nextPacked_implies_previousSemanticFold_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked : PackedWitnessProduction.messageCheck previousContext
      previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked : PackedWitnessProduction.messageCheck nextContext
      nextCertificate = true)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      nextContext.materialize.covers (decodedData nextTemplate nextWitnesses)
      (CombinedNc.ProductionPiCcs.ncPoint nextContext.materialize
        nextCertificate).block nextCertificate.piCcs.output)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true) :
    SemanticFold.Holds previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate ∨
      RecursiveRefinementFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses := by
  rcases
      messageCheckedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_namedFailure
        noZeroDivisors scheme stateDigest previousContext previousTemplate
        previousWitnesses previousCertificate previousChecked
        previousStateChecked nextContext nextTemplate nextWitnesses
        nextCertificate nextChecked nextPacked nextStateChecked with
    success | yRing | bad | refinementFailure
  · exact Or.inl success.2
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr refinementFailure))

/-- Exact physical refinements outside the final full-witness checker. -/
inductive TerminalRefinementFailure
    (context : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize) : Prop where
  | polynomialInput
      (failure : ¬ PublicInputBound context.materialize
        (decodedData template witnesses)) :
      TerminalRefinementFailure context template witnesses certificate
  | sourceProductBinding
      (failure : ¬ InputBound context.materialize
        (decodedData template witnesses)) :
      TerminalRefinementFailure context template witnesses certificate
  | childOpening
      (failure : ¬ ChildOpenings context.materialize
        (decodedData template witnesses) certificate) :
      TerminalRefinementFailure context template witnesses certificate

/-- The terminal path derives its complete semantic input and child-opening
premises before invoking delayed closure.  The verifier-owned raw terminal
opening check remains the explicit executable acceptance input. -/
theorem checkedTerminal_implies_semanticFold_or_namedFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (checked : PackedWitnessProduction.check context template witnesses
      certificate = true)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck context certificate
      terminalWitnesses = true) :
    SemanticFold.Holds context.materialize
        (decodedData template witnesses)
        (derive context.materialize certificate).piRlcOutput
        (outputChildren context.materialize certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (decodedData template witnesses) certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (decodedData template witnesses) certificate ∨
      TerminalRefinementFailure context template witnesses certificate := by
  classical
  by_cases polynomial : PublicInputBound context.materialize
      (decodedData template witnesses)
  · by_cases sources : InputBound context.materialize
        (decodedData template witnesses)
    · by_cases children : ChildOpenings context.materialize
          (decodedData template witnesses) certificate
      · let input : SemanticInput context.materialize
            (decodedData template witnesses) := {
          publicInput := polynomial
          sources := sources
        }
        rcases PackedWitnessProduction.checkedTerminal_implies_semanticFold_or_badEvent
            noZeroDivisors context template witnesses certificate input
            children checked terminalWitnesses terminal with semantic | yRing | bad
        · exact Or.inl semantic
        · exact Or.inr (Or.inl yRing)
        · exact Or.inr (Or.inr (Or.inl bad))
      · exact Or.inr (Or.inr (Or.inr (.childOpening children)))
    · exact Or.inr (Or.inr (Or.inr (.sourceProductBinding sources)))
  · exact Or.inr (Or.inr (Or.inr (.polynomialInput polynomial)))

/-- Claims-level terminal acceptance uses the same concrete refinement
partition, but the terminal packed opening is established before converting
the public message terminal to the raw-source NC relation.  Therefore this
theorem has no packed-output extraction failure outcome. -/
theorem messageCheckedTerminal_implies_semanticFold_or_namedFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (checked : PackedWitnessProduction.messageCheck context certificate = true)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck context certificate
      terminalWitnesses = true) :
    SemanticFold.Holds context.materialize
        (decodedData template witnesses)
        (derive context.materialize certificate).piRlcOutput
        (outputChildren context.materialize certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (decodedData template witnesses) certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (decodedData template witnesses) certificate ∨
      TerminalRefinementFailure context template witnesses certificate := by
  classical
  by_cases polynomial : PublicInputBound context.materialize
      (decodedData template witnesses)
  · by_cases sources : InputBound context.materialize
        (decodedData template witnesses)
    · by_cases children : ChildOpenings context.materialize
          (decodedData template witnesses) certificate
      · let input : SemanticInput context.materialize
            (decodedData template witnesses) := {
          publicInput := polynomial
          sources := sources
        }
        rcases
            PackedWitnessProduction.messageCheckedTerminal_implies_semanticFold_or_badEvent
              noZeroDivisors context template witnesses certificate input
              children checked terminalWitnesses terminal with
          semantic | yRing | bad
        · exact Or.inl semantic
        · exact Or.inr (Or.inl yRing)
        · exact Or.inr (Or.inr (Or.inl bad))
      · exact Or.inr (Or.inr (Or.inr (.childOpening children)))
    · exact Or.inr (Or.inr (Or.inr (.sourceProductBinding sources)))
  · exact Or.inr (Or.inr (Or.inr (.polynomialInput polynomial)))

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RefinementBoundary
