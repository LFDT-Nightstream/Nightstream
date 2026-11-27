import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinNote

/-! The work bits of the stages of the opening of `key20.aks` in the model (Akita, ePrint
2026/1983, revision of 2026-11-23, (146); `Tier.bits`, `Tier.foldBits`, `Root.rootBits`,
`Terminal.foldBits`; the code: `jolt_transcript::grind_bits_for_loss`, `FoldChallengeWork`).

- `bits20` lists the work bits of each stage of `Key20.Tower.accepts` in the order of the
  stages: the stage 0 of the batching root, then for each level 0 to 5 its fold stage and the
  stages of its checks (with the tensor reduction of the next level), then the fold stage of the
  terminal. `stages20` is its count: 473 stages.
- `tower_bits20`: for each tower data `o`, the work bits of the model are `bits20`. The plan of
  the code has one run with the bits of each such stage, in the same order. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Terminal)

/-- The work bits of the checks of a level of the Euclidean route with the product stages of
arity 2 and 4 (the levels 4 and 5) and the tensor reduction of the next level. -/
def squaredBits20 (vars : ℕ) : List ℕ :=
  [7, 5, 3] ++ List.replicate vars 2 ++ [1] ++ List.replicate vars 3 ++ [3, 1] ++
    List.replicate vars 3 ++ [1, 1] ++ List.replicate vars 2 ++ 1 :: List.replicate (vars - 1) 2

/-- The work bits of each stage of the opening of `key20.aks` in the model, in the order of the
stages: the batching root, the fold stage and the checks of each level 0 to 5, and the fold stage
of the terminal. -/
def bits20 : List (List ℕ) :=
  [[2],
    6 :: [11, 5, 3] ++ List.replicate 22 3 ++ [1, 1] ++ List.replicate 22 2,
    5 :: [11, 5, 3] ++ List.replicate 21 3 ++ [1, 1] ++ List.replicate 21 2 ++
      1 :: List.replicate 20 2,
    4 :: [8, 5, 3] ++ List.replicate 19 3 ++ [2] ++ List.replicate 19 3 ++ [1, 1] ++
      List.replicate 19 2 ++ 1 :: List.replicate 18 2,
    4 :: [8, 5, 3] ++ List.replicate 18 3 ++ [2, 1] ++ List.replicate 18 3 ++ [1, 1, 1] ++
      List.replicate 18 2 ++ 1 :: List.replicate 17 2,
    3 :: squaredBits20 17,
    3 :: squaredBits20 17,
    [3]]

/-- The count of the stages of the opening of `key20.aks`: 473. -/
theorem stages20 : (bits20.map List.length).sum = 473 := by decide

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Wire Coord : Type} {total : ℕ} (o : Key20.Tower E Wire Coord total)

local notation "code₁" =>
  (CodeSlot.open Key20.open64 Key20.open64_fits :
    CodeSlot E → Option (Rq Key20.level₁.opening.carrierDegree))

local notation "code₂" =>
  (CodeSlot.open Key20.open128 Key20.open128_fits :
    CodeSlot E → Option (Rq Key20.level₂.shape.innerDegree))

local notation "code₃" =>
  (CodeSlot.kept128 : CodeSlot E → Option (Rq Key20.level₃.shape.innerDegree))

local notation "code₄" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₄.shape.innerDegree))

local notation "code₅" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₅.shape.innerDegree))

local notation "zeroth" =>
  Extract.liftBatch o.β o.key o.openKey o.claim Key20.rootVars (And.left Key20.root_sized) 8
    Key20.quotientDigits Key20.root_room Key20.rootBatch (And.left (And.right Key20.root_sized))
    o.block (Key20.rootDecode E) o.read Key20.rootMass (0 + 1)

local notation "first" =>
  Tier.tensored (Extract.liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ Key20.level₁.vars
    (And.left Key20.level₁_sized) Key20.quotientDigits (And.left (And.right Key20.level₁_sized))
    Key20.level₁.batch (And.left (And.right (And.right Key20.level₁_sized))) o.block₁ code₁ o.read
    51 (Tier.finish zeroth)) Key20.level₁.tail

local notation "second" =>
  Tier.tensored (Extract.committedTrace (fold := Key20.level₂.fold) o.key₂ o.openKey₂ o.claim₂
    Key20.level₂.vars (And.left Key20.sized₂) Key20.level₂.batch (And.left (And.right Key20.sized₂))
    o.block₂ code₂ o.read 31 (Tier.finish first)) Key20.level₂.tail

local notation "third" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₃.fold) o.key₃ o.openKey₃ o.claim₃
    Key20.level₃.vars (And.left Key20.sized₃) Key20.level₃.batch
    (And.left (And.right Key20.sized₃)) o.block₃ code₃ o.read 31 Key20.norm₃.operator
    Key20.norm₃.limit (Tier.finish second)) Key20.level₃.tail

local notation "fourth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₄.fold) o.key₄ o.openKey₄ o.claim₄
    Key20.level₄.vars (And.left Key20.sized₄) Key20.level₄.batch
    (And.left (And.right Key20.sized₄)) o.block₄ code₄ o.read 53 Key20.norm₄.operator
    Key20.norm₄.limit (Tier.finish third)) Key20.level₄.tail

local notation "fifth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₅.fold) o.key₅ o.openKey₅ o.claim₅
    Key20.level₅.vars (And.left Key20.sized₅) Key20.level₅.batch
    (And.left (And.right Key20.sized₅)) o.block₅ code₅ o.read 53 Key20.norm₅.operator
    Key20.norm₅.limit (Tier.finish fourth)) Key20.level₅.tail

/-- The work bits of the model for the tower data `o` are `bits20`: `Root.rootBits 3` of the
batching root, `Tier.foldBits` and `Tier.bits` of the levels 0 to 5 (the levels of
`Key20.Tower.accepts`), and `Terminal.foldBits` of the terminal. -/
theorem tower_bits20 :
    [[Extract.Root.rootBits 3], (zeroth).foldBits :: (zeroth).bits,
      (first).foldBits :: (first).bits, (second).foldBits :: (second).bits,
      (third).foldBits :: (third).bits, (fourth).foldBits :: (fourth).bits,
      (fifth).foldBits :: (fifth).bits, [Terminal.foldBits o.terminal]] = bits20 := by
  simp only [Tier.tensored, Extract.committedTraceSq, Extract.committedTrace, Extract.liftFold,
    Extract.liftBatch, Terminal.foldBits]
  decide

end NightstreamFPrime.Spec.Compression.Akita.Replay
