import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.ProjectionPrefix

namespace tests.PiRlcProjectionPrefix

open SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-!
Regression and axiom-surface checks for the Pi_RLC projection-prefix model.

| Check | Mathematical property | Production claim |
|---|---|---|
| parent binding | Equal typed SIS digests fix parent `y_zcol` or expose a collision | Model-level only |
| complete prefix | Equal output and projection digests produce equal beta-prefix results | Model-level only |
| child timing | Distinct Pi_DEC child messages leave beta unchanged | Model-level diagnosis; Rust trace refinement open |
| empty byte string | A length header that fills the rate remains buffered when there are no limbs | Exact native helper edge case |
| axiom report | Exported theorems use no project axioms | Lean kernel report |
-/

example (core : Poseidon2Core) (state : SpongeState) :
    (appendPackedBytes core
      { state := state, absorbed := ⟨3, by decide⟩ } []).absorbed.val = 4 := by
  rfl

example
    (sisDigest : List SuperNeo.F -> Digest4)
    (left right : ProjectionBindingMaterial)
    (hDigest :
      sisDigest (serializeProjectionBinding left) =
        sisDigest (serializeProjectionBinding right)) :
    left.yZcol = right.yZcol ∨
      TypedSisBindingCollision sisDigest left right :=
  parentYZcol_eq_or_typedSisCollision sisDigest left right hDigest

example
    (core : Poseidon2Core)
    (sisDigest : List SuperNeo.F -> Digest4)
    (initial : SpongeCursor)
    (left right : BetaPrefixInput)
    (hOutputs : left.piCcsOutputDigest = right.piCcsOutputDigest)
    (hProjection :
      sisDigest (serializeProjectionBinding left.projection) =
        sisDigest (serializeProjectionBinding right.projection)) :
    betaPrefixResult core sisDigest initial left =
        betaPrefixResult core sisDigest initial right ∧
      (left.projection.yZcol = right.projection.yZcol ∨
        TypedSisBindingCollision sisDigest left.projection
          right.projection) :=
  betaPrefix_fixes_parentYZcol_or_collision core sisDigest initial
    left right hOutputs hProjection

example
    {Children : Type}
    (core : Poseidon2Core)
    (sisDigest : List SuperNeo.F -> Digest4)
    (initial : SpongeCursor)
    (input : BetaPrefixInput)
    (left right : Children)
    (different : left ≠ right) :
    left ≠ right ∧
      betaWithChildren core sisDigest initial input left =
        betaWithChildren core sisDigest initial input right :=
  piDecChildren_not_fixed_by_betaPrefix
    core sisDigest initial input left right different

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.parentYZcol_eq_or_typedSisCollision' depends on axioms: [propext] -/
#guard_msgs in
#print axioms parentYZcol_eq_or_typedSisCollision

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.betaPrefix_fixes_parentYZcol_or_collision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms betaPrefix_fixes_parentYZcol_or_collision

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.piDecChildren_not_fixed_by_betaPrefix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms piDecChildren_not_fixed_by_betaPrefix

end tests.PiRlcProjectionPrefix
