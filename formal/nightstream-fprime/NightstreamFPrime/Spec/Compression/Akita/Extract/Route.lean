import NightstreamFPrime.Spec.Compression.Akita.Terminal
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tier

/-! The two norm routes of a trace level and of the terminal of Akita (ePrint 2026/1983, revision
of 2026-11-23, Section 6, Lemma 10.23 with (176) and (177), Proposition 10.29 with (181) and
(182), and Corollary 10.35).

- `Route` is the norm certificate of a level: the coefficient route with `‖·‖_∞`, or the
  Euclidean route with `‖c‖_mul,2 ≤ Γ` for each fold challenge and `‖z‖_2² ≤ S_max` for each
  response.
- `Route.Challenges`, `Route.Short` and `Route.Kernel` are the conditions on the challenges, the
  bound of a clear response and the solution of Module-SIS for `A_term` of the terminal on a
  route. `Route.kernelOf_isSome` is Akita Proposition 10.29 with the radius (181) or (182).
- `tableCollisionSq` is the reduction of Akita Lemma 10.23 and Corollary 10.25 on the Euclidean
  route, and `LevelBreakSq` and `Route.Break` are the witnesses of `R_bind` for the matrices of a
  level on a route. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

section Collision

variable {S : Shape}

/-- The reduction of Akita Lemma 10.23 for two certified decommitments on the Euclidean route:
the solution of Lemma 4.3 for the path of `B`, or else the solution `z_A` for `A` with
`‖z_A‖_2² ≤ bound`. -/
noncomputable def tableCollisionSq (key : Key S) (bound : ℕ) (st st' : Certified S) :
    Option (ShortKernelSq key.inner bound ⊕ OuterBreak key) :=
  ((outerCollision key st.inner st'.inner st.chain st'.chain).map Sum.inr).or
    ((sourceCollisionSq key bound st st').map Sum.inl)

/-- Akita Corollary 10.25 on the Euclidean route: two certified decommitments of one payload to
different tables, with the Euclidean certificates `(κ̄_mul,2, β̄_2²)`, give a solution of
Module-SIS for `A` with `‖z_A‖_2² ≤ 4·κ̄_mul,2²·β̄_2²` (the second line of (177)), for `B` with the
bound `b⋆ − 1`, or for a compression map with the bound 1. -/
theorem tableCollisionSq_isSome {key : Key S}
    {mulBound sqBound factorBound sourceBound factorBound' sourceBound' : ℕ}
    {st st' : Certified S} {payload : S.Payload} {table table' : Table}
    (opens : st.Opens factorBound sourceBound key payload table)
    (opens' : st'.Opens factorBound' sourceBound' key payload table')
    (euclid : st.Euclidean mulBound sqBound) (euclid' : st'.Euclidean mulBound sqBound)
    (different : table ≠ table') :
    (tableCollisionSq key (4 * mulBound ^ 2 * sqBound) st st').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases digits : st.inner ≠ st'.inner ∨ st.chain ≠ st'.chain
  · exact Or.inl (outerCollision_isSome opens.source.rows opens'.source.rows
      opens.source.innerDigits opens'.source.innerDigits opens.source.chainDigits
      opens'.source.chainDigits digits)
  · refine Or.inr (sourceCollisionSq_isSome_of_bounds opens opens' euclid euclid'
      (not_not.mp fun differs => digits (Or.inl differs)) fun same => different ?_)
    rw [opens.source.decodes, opens'.source.decodes, same]

end Collision

namespace Extract

/-- The norm certificate of a level (Akita Section 6): the coefficient route, or the Euclidean
route with the operator bound `Γ` of the fold challenges and the cap `S_max` of `‖z‖_2²`. -/
inductive Route
  | coefficient
  | euclidean (operator limit : ℕ)

/-- The Euclidean route has a fold challenge family with the operator filter. -/
def Route.filtered : Route → Bool
  | .coefficient => false
  | .euclidean _ _ => true

section Terminal

variable {d : ℕ} [NeZero d]

/-- The condition on the fold challenges of the terminal on a route: `‖c‖_1 = κ_{1,term}` on the
coefficient route, and `‖c‖_mul,2 ≤ Γ_term` on the Euclidean route. -/
def Route.Challenges : Route → ℕ → Set (Rq d) → Prop
  | .coefficient, weight, family => ∃ peak, ∀ c ∈ family, Sparse weight peak c
  | .euclidean operator _, _, family => ∀ c ∈ family, MulBound operator c

/-- The bound of a clear response on a route: `‖z‖_∞ ≤ Z_∞` with `bound = Z_∞` on the
coefficient route, and `‖z‖_2² ≤ S_max` on the Euclidean route. -/
def Route.Short {ι : Type} [Fintype ι] : Route → ℕ → (ι → Rq d) → Prop
  | .coefficient, bound, z => vecNormInf z ≤ bound
  | .euclidean _ limit, _, z => vecNormSq z ≤ limit

variable {rows cols : Type} [Fintype cols]

/-- A solution of Module-SIS for `A_term` on a route: `‖z‖_∞ ≤ 8·κ_{1,term}·Z_{∞,term}` of Akita
(181), or `‖z‖_2² ≤ 64·Γ_term²·S_max,term` of Akita (182). -/
def Route.Kernel : Route → Matrix rows cols (Rq d) → ℕ → ℕ → Type
  | .coefficient, M, weight, bound => ShortKernel M (8 * weight * bound)
  | .euclidean operator limit, M, _, _ => ShortKernelSq M (64 * operator ^ 2 * limit)

/-- The solution `x − x'` of Module-SIS of the route, when `x` and `x'` collide. -/
noncomputable def Route.kernelOf : (route : Route) → (M : Matrix rows cols (Rq d)) →
    (weight bound : ℕ) → (cols → Rq d) → (cols → Rq d) → Option (route.Kernel M weight bound)
  | .coefficient, M, weight, bound, x, x' => ShortKernel.ofCollision? M (8 * weight * bound) x x'
  | .euclidean operator limit, M, _, _, x, x' =>
      ShortKernelSq.ofCollision? M (64 * operator ^ 2 * limit) x x'

end Terminal

variable {S : Shape} {E : Type} [CommRing E] [Module Zq E]

/-- Akita Proposition 10.29 on a route, with the radius (181) or (182). Two forks pass the
checks of two terminal statements with one matrix `A_term` and one state `(t_i)_i`, the
challenges satisfy `Route.Challenges`, and each clear response satisfies `Route.Short`. If the
extracted candidates for block `i` differ, `z_A` is a solution of Module-SIS for `A_term`. -/
theorem Route.kernelOf_isSome (route : Route) {weight bound : ℕ}
    {st st' : TerminalStatement S E} {fork fork' : TerminalFork S}
    {family : Set (Rq S.innerDegree)} {partials partials' : Fin S.blocks → Rq S.innerDegree}
    (units : UnitDifferences family) (challenges : route.Challenges weight family)
    (accepts : fork.Accepts family st partials) (accepts' : fork'.Accepts family st' partials')
    (matrix : st'.matrix = st.matrix) (state : st'.state = st.state)
    (short : ∀ i, route.Short bound (fork.first i).response ∧
      route.Short bound (fork.second i).response)
    (short' : ∀ i, route.Short bound (fork'.first i).response ∧
      route.Short bound (fork'.second i).response)
    {i : Fin S.blocks} (different : fork.source i ≠ fork'.source i) :
    (route.kernelOf st.matrix weight bound (fork'.difference i • fork.responseDifference i)
      (fork.difference i • fork'.responseDifference i)).isSome := by
  cases route with
  | coefficient =>
      obtain ⟨peak, sparse⟩ := challenges
      exact terminalCollision_isSome units sparse accepts accepts' matrix state short short'
        different
  | euclidean operator limit =>
      exact terminalCollisionSq_isSome units challenges accepts accepts' matrix state short short'
        different

/-- A witness of `R_bind` for the matrices of one level on the Euclidean route, or `none`: a
solution of Module-SIS for `A` with `‖z‖_2² ≤ bound`, for `B` or `D` with the bound `b⋆ − 1`, or
for a compression map with the bound 1 (Akita Table 3). -/
abbrev LevelBreakSq {O : OpeningShape S} (key : Key S) (openKey : OpeningKey O) (bound : ℕ) :
    Type :=
  Option (ShortKernelSq key.inner bound ⊕ (OuterBreak key ⊕ OpenBreak openKey))

/-- The witnesses of `R_bind` for the matrices of a trace level on a route: `LevelBreak` with
`η_A = 2·κ̄_1·Δ^cert_f` of Akita (176), or `LevelBreakSq` with `64·Γ²·S_max` of Corollary 10.35
(ii). -/
def Route.Break {O : OpeningShape S} (key : Key S) (openKey : OpeningKey O)
    (weight sourceBound : ℕ) : Route → Type
  | .coefficient => LevelBreak key openKey (2 * (2 * weight) * sourceBound)
  | .euclidean operator limit => LevelBreakSq key openKey (64 * operator ^ 2 * limit)

end Extract

end NightstreamFPrime.Spec.Compression.Akita
