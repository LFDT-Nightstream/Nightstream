import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink

/-! A trace level of the opening suffix of Akita on the Euclidean route (ePrint 2026/1983, revision
of 2026-11-23, Section 6.2 with (83), Lemma 10.23 with (177), Corollaries 10.26, 10.32 and 10.35
(ii), and Lemma 10.34 with (184)).

- `traceSq` is the opening method of a trace fold on the Euclidean route. Its exact branch
  relation adds the bound `‖G_{b_z,m_A}·ẑ‖_2² ≤ S_max` of the response, its weak opening adds the
  Euclidean certificates `(κ̄_mul,2, β̄_2²) = (2Γ, 4·S_max)`, and a witness of `R_bind` for `A` has
  `‖z‖_2² ≤ 64·Γ²·S_max`. `traceSq_sound` is Akita Corollary 10.32 with Corollary 10.26 for it.
- `Tier.collideSq` is `Coll_ℓ` for two weak openings of a level on the Euclidean route.
- `traceOn` and `Route.collide` are the method and `Coll_ℓ` of a trace level on a route.
- `Tier.suffix_push_traceSq` is `Tier.suffix_push_trace` for a next level with the method
  `traceSq`, and `Tier.suffix_push_route` is the step for a next level on a route. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Method

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [CommRing E]
  [Module Zq E]

/-- The method of a fold with evaluation trace for one claim on the Euclidean route (Akita
Sections 5.3, 6.2, 7.1 and 7.2). The exact branch relation is `Witness.TraceSatisfies` with the
bound `‖G_{b_z,m_A}·ẑ‖_2² ≤ S_max` (`limit`) of (83), which the direct norm check certifies
(Lemma 10.34, (184)). The weak opening is `Weak.TraceOpens` with `κ̄_1 = 2ω` and
`β̄_∞ = Δ^cert_f`, and the Euclidean certificates `κ̄_mul,2 = 2Γ` and `β̄_2² = 4·S_max` of
Corollary 10.26, for the operator bound `Γ` (`operator`) of the fold challenges. A witness of
`R_bind` for `A` has the radius `64·Γ²·S_max` of Corollary 10.35 (ii). -/
noncomputable def traceSq (key : Key S) (openKey : OpeningKey O) (weight operator limit : ℕ) :
    Method (Fin S.blocks) (fun _ => Rq S.innerDegree) (TraceClaim O E)
      (LevelBreakSq key openKey (64 * operator ^ 2 * limit)) (Witness O fold) (Weak O) where
  Satisfies w claim challenge := w.TraceSatisfies (claim.statement key openKey) challenge ∧
    vecNormSq (gadget fold.responseBase w.response) ≤ limit
  Opens ω claim := ω.TraceOpens (2 * weight) (fold.sourceBound S) (claim.statement key openKey)
      (S.decode ω.source) ∧ ω.toCertified.Euclidean (2 * operator) (4 * limit)
  merge claim first second w :=
    match traceForkCollision (claim.statement key openKey)
        ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩
        w.toCarried with
    | some solution => .inr (some (.inr solution))
    | none => .inl (traceExtract
        ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ w.toCarried)
  broken solution := solution.isSome

/-- Akita Corollary 10.32 with Corollary 10.26 for the method `traceSq`
(`traceExtract_opens_or_collision` and `traceExtract_euclidean`). The challenge family has units
as differences (Akita Definition 9.2, E1), members of the norm `ω` and members with
`‖c‖_mul,2 ≤ Γ`, and `b_z ≥ 2`. -/
theorem traceSq_sound {key : Key S} {openKey : OpeningKey O} {weight peak operator limit : ℕ}
    {family : Set (Rq S.innerDegree)} (units : UnitDifferences family)
    (sparse : ∀ challenge ∈ family, Sparse weight peak challenge)
    (bounded : ∀ challenge ∈ family, MulBound operator challenge)
    (base : 2 ≤ fold.responseBase) :
    (traceSq (fold := fold) (E := E) key openKey weight operator limit).Sound fun _ => family := by
  intro claim first second w challenge firstMember secondMember different agree firstSatisfies
    secondSatisfies good
  have accepts : TraceFork.Accepts family
      (⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ :
        TraceFork O fold) (claim.statement key openKey) :=
    ⟨firstMember, secondMember, different, agree, fun i => (firstSatisfies i).1,
      fun i => (secondSatisfies i).1⟩
  have result := traceExtract_opens_or_collision units sparse base accepts
    (Witness.TraceSatisfies.carried good.1)
  have euclid := traceExtract_euclidean (digits := w.toCarried) units accepts bounded
    (fun i => (firstSatisfies i).2) (fun i => (secondSatisfies i).2)
  unfold traceSq
  rcases collision : traceForkCollision (claim.statement key openKey)
      (⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ :
        TraceFork O fold) w.toCarried with _ | solution
  · rw [collision] at result
    simp only [collision]
    exact ⟨result.resolve_left (by simp), euclid⟩
  · simp only [collision]
    rfl

/-- `Coll_ℓ` for two weak openings of a level on the Euclidean route: the solution of
`tableCollisionSq` for their certified decommitments. -/
noncomputable def Tier.collideSq (key : Key S) (openKey : OpeningKey O) (bound : ℕ)
    (opening other : Option (Weak O)) : LevelBreakSq key openKey bound :=
  match opening, other with
  | some ω, some ω' =>
      (tableCollisionSq key bound ω.toCertified ω'.toCertified).map (Sum.map id Sum.inl)
  | _, _ => none

/-- The method of a trace level on a route: `trace` or `traceSq`. -/
noncomputable def traceOn (key : Key S) (openKey : OpeningKey O) (weight : ℕ) :
    (route : Route) → Method (Fin S.blocks) (fun _ => Rq S.innerDegree) (TraceClaim O E)
      (route.Break key openKey weight (fold.sourceBound S)) (Witness O fold) (Weak O)
  | .coefficient => trace key openKey weight
  | .euclidean operator limit => traceSq key openKey weight operator limit

/-- `Coll_ℓ` for two weak openings of a trace level on a route: `Tier.collide` or
`Tier.collideSq`. -/
noncomputable def Route.collide (key : Key S) (openKey : OpeningKey O) (weight sourceBound : ℕ) :
    (route : Route) → Option (Weak O) → Option (Weak O) →
      route.Break key openKey weight sourceBound
  | .coefficient => Tier.collide key openKey (2 * (2 * weight) * sourceBound)
  | .euclidean operator limit => Tier.collideSq key openKey (64 * operator ^ 2 * limit)

end Method

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

section Receiver

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {S' : Shape} {O' : OpeningShape S'}
  {fold' : Descriptor} {Ring Stmt Brk Wire Coord Slot Result' Source' : Type} [Zero Ring]
  {key' : Key S'} {openKey' : OpeningKey O'} {weight' : ℕ}
  {T : Tier (Fin S.blocks) (fun _ => Ring) Stmt Brk (Witness O fold) (Weak O) Wire Coord Slot
    (Option (List E × E × E)) Table}
  {total : ℕ} [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type} {m : ℕ} {accept : Prefix Wire (Option Coord → Slot) total → Prop}
  {tail : Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) Inner
    BreakT CandT
    KeyT (Option (Weak O'))}
  {error : ℝ} {loops tree : ℕ}

omit [Fintype E] [DecidableEq E] in
/-- `Tier.suffix_push_trace` for a next level `N` with the method `traceSq` (Akita
Proposition 10.44 for one more level, on the Euclidean route). The final claim of the level is
true for the table of a weak opening of `N` by its scalar row (101), which is Akita
Lemma 10.42 (ii). Two weak openings of `N` with different tables after one stage input of the
checkpoint are two certified decommitments of one payload with the Euclidean certificates
`(2Γ, 4·S_max)`, and `tableCollisionSq_isSome` gives a solution of Module-SIS for a matrix of `N`
with the radius `64·Γ²·S_max` for `A` (Akita Corollary 10.35 (ii)). The other hypotheses are those
of `Tier.suffix_push_trace`. -/
theorem Tier.suffix_push_traceSq {operator' limit' : ℕ}
    {N : Tier (Fin S'.blocks) (fun _ => Rq S'.innerDegree) (TraceClaim O' E)
      (LevelBreakSq key' openKey' (64 * operator' ^ 2 * limit')) (Witness O' fold') (Weak O')
      Wire Coord Slot Result' Source'}
    (valid : T.Valid) (traced : N.method = traceSq key' openKey' weight' operator' limit')
    (next : N.start = T.finish) (quarter : 4 * m = S'.innerDegree) (commit : Wire → S'.Payload)
    (fits : T.finish ≤ total)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T.passes table (some (ρ, θ, v)))
    (linked : ∀ point, accept point → T.TraceLinked N quarter commit point)
    (embed : LevelBreakSq key' openKey' (64 * operator' ^ 2 * limit') → BreakT)
    (embedded : ∀ solution, solution.isSome → tail.breaks (embed solution))
    (later : Tier.Suffix accept N.start N.Opens tail error loops tree) :
    Tier.Suffix accept T.start T.Opens
      (T.over accept tail
        (fun inner =>
          flatten (E := E) (unpackTable S'.innerDegree m (tableOf (tail.witness inner))))
        (fun inner other => embed (Tier.collideSq key' openKey' (64 * operator' ^ 2 * limit')
          (tail.witness inner) (tail.witness other))))
      (T.levelError + error) (S.blocks + 1 + loops)
      (2 * (S.blocks + 1) * tree) := by
  have opened : ∀ point ω, accept point → N.Opens point (some ω) →
      (ω.TraceOpens (2 * weight') (fold'.sourceBound S')
        ((N.claim (point.wire N.start)).statement key' openKey') (S'.decode ω.source) ∧
        ω.toCertified.Euclidean (2 * operator') (4 * limit')) := by
    rintro point ω _ ⟨_, same, opens⟩
    obtain rfl := Option.some.inj same
    rw [traced] at opens
    exact opens
  refine Tier.suffix_before (Fintype.card_fin _) valid (next ▸ later) fits
    (fun opening => flatten (E := E) (unpackTable S'.innerDegree m (tableOf opening)))
    (fun opening other => embed (Tier.collideSq key' openKey' (64 * operator' ^ 2 * limit')
      opening other))
    (fun point accepted => (linked point accepted).2.1) ?_ ?_
  · rintro point _ accepted ⟨ω, rfl, opens⟩
    obtain ⟨long, decodes, block, position, pack, θ, v, openPayload, ended, blocks, length,
      width, claimed⟩ := linked _ accepted
    have left := (opened point ω accepted ⟨ω, rfl, opens⟩).1
    rw [claimed] at left
    refine ⟨long, decodes, _, ended_of_bare T.read (uniform _ _ _ _) ended, passes _ _ _ _ ?_⟩
    rw [packed_flatten]
    exact (traceOpens_evaluation left blocks length width).symm
  · rintro point other _ _ accepted acceptedOther ⟨ω, rfl, opens⟩ ⟨ω', rfl, opens'⟩ same different
    have sameWire : other.wire (T.start + 1) = point.wire (T.start + 1) := by
      rw [← Prefix.wire_take le_rfl other, ← same, Prefix.wire_take le_rfl]
    obtain ⟨left, euclid⟩ := opened point ω accepted ⟨ω, rfl, opens⟩
    obtain ⟨right, euclid'⟩ := opened other ω' acceptedOther ⟨ω', rfl, opens'⟩
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, claimed⟩ := linked _ accepted
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, claimed'⟩ := linked _ acceptedOther
    rw [claimed] at left
    rw [claimed', sameWire] at right
    refine embedded _ ?_
    show ((tableCollisionSq key' (64 * operator' ^ 2 * limit') ω.toCertified
      ω'.toCertified).map (Sum.map id Sum.inl)).isSome
    rw [Option.isSome_map]
    have found := tableCollisionSq_isSome left.certified right.certified euclid euclid'
      fun equal => different (congrArg
        (fun table => flatten (E := E) (unpackTable S'.innerDegree m table)) equal)
    rwa [show 4 * (2 * operator') ^ 2 * (4 * limit') = 64 * operator' ^ 2 * limit' by ring]
      at found

omit [Fintype E] [DecidableEq E] in
/-- The step of Akita Proposition 10.44 for a next trace level `N` on a route: the method of `N`
is `traceOn` of the route, and `Coll_ℓ` of `N` is `Route.collide`. It is
`Tier.suffix_push_trace` on the coefficient route and `Tier.suffix_push_traceSq` on the Euclidean
route, with the same hypotheses. -/
theorem Tier.suffix_push_route {route : Route}
    {N : Tier (Fin S'.blocks) (fun _ => Rq S'.innerDegree) (TraceClaim O' E)
      (route.Break key' openKey' weight' (fold'.sourceBound S')) (Witness O' fold') (Weak O')
      Wire Coord Slot Result' Source'}
    (valid : T.Valid) (traced : N.method = traceOn key' openKey' weight' route)
    (next : N.start = T.finish) (quarter : 4 * m = S'.innerDegree) (commit : Wire → S'.Payload)
    (fits : T.finish ≤ total)
    (uniform : ∀ wire challenge table other,
      bare (T.check wire challenge table) = bare (T.check wire challenge other))
    (passes : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T.passes table (some (ρ, θ, v)))
    (linked : ∀ point, accept point → T.TraceLinked N quarter commit point)
    (embed : route.Break key' openKey' weight' (fold'.sourceBound S') → BreakT)
    (embedded : ∀ solution, N.method.broken solution → tail.breaks (embed solution))
    (later : Tier.Suffix accept N.start N.Opens tail error loops tree) :
    Tier.Suffix accept T.start T.Opens
      (T.over accept tail
        (fun inner =>
          flatten (E := E) (unpackTable S'.innerDegree m (tableOf (tail.witness inner))))
        (fun inner other => embed (route.collide key' openKey' weight' (fold'.sourceBound S')
          (tail.witness inner) (tail.witness other))))
      (T.levelError + error) (S.blocks + 1 + loops)
      (2 * (S.blocks + 1) * tree) := by
  cases route with
  | coefficient =>
      exact Tier.suffix_push_trace valid traced next quarter commit fits uniform passes linked
        embed (fun solution found => embedded solution (by rw [traced]; exact found)) later
  | euclidean operator limit =>
      exact Tier.suffix_push_traceSq valid traced next quarter commit fits uniform passes linked
        embed (fun solution found => embedded solution (by rw [traced]; exact found)) later

end Receiver

end NightstreamFPrime.Spec.Compression.Akita.Extract
