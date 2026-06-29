import DirectCcsFPrime.ProofSystem.PrivatePiDec.Security.GoldilocksChildTableAuthorization
import Mathlib.Tactic

/-!
Necessity checks for reduced-source DEC authorization.

The positive reduced-source theorem requires two non-negotiable conditions:

1. the reduced source must bind the parent residues uniquely enough that the
   same source cannot authorize different parents;
2. the next `Pi_CCS` input must be the same child table proved by the DEC
   verifier.

This module gives concrete one-column counterexamples showing why those
conditions are not optional.
-/

namespace DirectCcsFPrime

namespace ReducedSourceNecessity

open DecDigitUniqueness
open BinaryChildTableAuthorization
open GoldilocksChildTableAuthorization

private def zero14 : List Nat :=
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

private def oneThenZero13 : List Nat :=
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

private def zeroChildren : ColumnDigits 1 :=
  fun _ => zero14

private def oneChildren : ColumnDigits 1 :=
  fun _ => oneThenZero13

private def parentZero : Fin 1 → Nat :=
  fun _ => 0

private def parentOne : Fin 1 → Nat :=
  fun _ => 1

private def trivialSourceBindsParent (_ : Unit) (_ : Fin 1 → Nat) : Prop :=
  True

private def verifyGoldilocksTable
    (_source : Unit)
    (parent : Fin 1 → Nat)
    (children : ColumnDigits 1)
    (_proof : Unit) : Prop :=
  binaryColumnDigits children ∧
  fixedColumnLength 14 children ∧
  ∀ j,
    recomposeNatDigits (children j) % SuperNeo.Goldilocks.q =
    parent j % SuperNeo.Goldilocks.q

private theorem zero_children_verified :
    verifyGoldilocksTable () parentZero zeroChildren () := by
  constructor
  · intro j d hd
    fin_cases j
    simp [zeroChildren, zero14] at hd
    omega
  constructor
  · intro j
    fin_cases j
    rfl
  · intro j
    fin_cases j
    rfl

private theorem one_children_verified :
    verifyGoldilocksTable () parentOne oneChildren () := by
  constructor
  · intro j d hd
    fin_cases j
    simp [oneChildren, oneThenZero13] at hd
    omega
  constructor
  · intro j
    fin_cases j
    rfl
  · intro j
    fin_cases j
    rfl

private theorem zeroChildren_ne_oneChildren :
    zeroChildren ≠ oneChildren := by
  intro h
  have hCol := congrFun h ⟨0, by decide⟩
  simp [zeroChildren, oneChildren, zero14, oneThenZero13] at hCol

/--
If the same reduced source is allowed to bind different parent residues, then
a deterministic challenge over that source can be identical while the accepted
next child accumulator differs.

This is a concrete counterexample to using a source that omits the parent
residue data or binds it only self-consistently.
-/
theorem same_source_without_parent_uniqueness_can_authorize_different_inputs :
    let source : Unit := ()
    let challenge : Unit → Nat := fun _ => 0
    AcceptedGoldilocksChildTable
      trivialSourceBindsParent
      verifyGoldilocksTable
      source
      parentZero
      zeroChildren
      zeroChildren
      () ∧
    AcceptedGoldilocksChildTable
      trivialSourceBindsParent
      verifyGoldilocksTable
      source
      parentOne
      oneChildren
      oneChildren
      () ∧
    challenge source = challenge source ∧
    zeroChildren ≠ oneChildren := by
  simp only
  constructor
  · exact
      { sourceBound := trivial
        proofVerified := zero_children_verified
        wireIdentity := rfl }
  constructor
  · exact
      { sourceBound := trivial
        proofVerified := one_children_verified
        wireIdentity := rfl }
  exact ⟨trivial, zeroChildren_ne_oneChildren⟩

/--
If the next `Pi_CCS` input is not wired to the child table verified by the
proof, the prover can verify one table and feed another. This holds even for
the same source and the same parent residues.
-/
theorem missing_wire_identity_can_feed_different_next_input :
    verifyGoldilocksTable () parentZero zeroChildren () ∧
    zeroChildren ≠ oneChildren := by
  exact ⟨zero_children_verified, zeroChildren_ne_oneChildren⟩

end ReducedSourceNecessity

end DirectCcsFPrime
