import NightstreamFPrime.Spec.Checkpoint.Protocol
import Mathlib.Algebra.BigOperators.Intervals

/-! The checkpoint structure of Akita Definition 10.16 (ePrint 2026/1983, pp. 76–77), in the
restricted form of the proof path of Akita: each extraction stage is an indexed binary stage. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass)

/-- Definition 10.16 (i): the type of a stage. An indexed stage reads one coordinate of the answer
for each entry of `coordinates`, with `k = 2` at each coordinate. The segment of a checkpoint is
the checkpoint and the `span` plain stages after it. -/
inductive Kind (Coord : Type) where
  | indexed (coordinates : List Coord)
  | checkpoint (span : ℕ)
  | plain

/-- The data of Definition 10.16. The answer at a stage input is a tuple of independent uniform
slots, one for each coordinate. This is the model of one squeeze run: one long output at one
transcript prefix, of which coordinate `u` of the challenge is a fixed slot. It is not one oracle
output for each coordinate: the paper gives each coordinate `(ξ_r, u)` its own table entry, and
here one query to `ξ_r` returns all slots. `key level coordinate` decodes one slot to a challenge value `ρ_{r,u}`, or to `none`
for a value that makes the verifier reject. `weight` is `1/|S_{r,u}|` and `error` is `ε_r` of a
checkpoint or plain stage. `holds level` is the certificate set `Y_level`, `breaks` is the set `T`
of the witnesses of `R_bind`, `combine` is `Comb_r`, `candidate` is `Cand_ℓ`, `good` is `Good_ℓ`,
`collide` is `Coll_ℓ`, `refuted level ξ w` is `D(ξ; w) = 1` in the segment of the checkpoint
`level`, `bad` is `Bad_j(ξ; w)`, and `witness` is `Wit`. `point` is the transcript prefix that a
certificate records. `work level` is the set of the answers whose work-predicate bits are zero,
and `chance level` is `2^{−g_r}` (Remark 10.18, Proposition 10.3). -/
structure Structure (Point Coord Slot Proof Cert Break Cand Key Witness : Type) where
  protocol : Protocol Point (Coord → Slot) Proof
  kind : ℕ → Kind Coord
  key : ℕ → Coord → Slot → Option Key
  weight : ℕ → Coord → ℝ
  work : ℕ → Set (Coord → Slot)
  chance : ℕ → ℝ
  error : ℕ → ℝ
  leaf : Point → Cert
  point : Cert → Point
  holds : ℕ → Cert → Prop
  breaks : Break → Prop
  combine : ℕ → Cert → (Coord → Cert) → Cert ⊕ Break
  candidate : ℕ → Cert → Cand
  good : ℕ → Cert → Prop
  collide : ℕ → Cert → Cert → Break
  refuted : ℕ → Point → Cand → Prop
  bad : ℕ → Point → Cand → Set (Coord → Slot)
  witness : Cert → Witness
  relation : Point → Witness → Prop

variable {Point Coord Slot Proof Cert Break Cand Key Witness : Type}

namespace Structure

variable (structure_ : Structure Point Coord Slot Proof Cert Break Cand Key Witness)

/-- The conditions of Definition 10.16. Theorem 10.17 assumes items (iv)–(viii) for each auxiliary
environment in `Good`; here the environment is fixed and the conditions are hypotheses. -/
structure Lawful [Fintype Coord] [DecidableEq Coord] [Fintype Slot] : Prop where
  /-- Item (i): a segment is inside the protocol and its later stages are plain. -/
  segment : ∀ level span, structure_.kind level = .checkpoint span →
    level + span < structure_.protocol.stages ∧
      ∀ later, level < later → later ≤ level + span → structure_.kind later = .plain
  /-- The indexed-stage answer law (p. 76): a slot that decodes to no value makes the verifier
  reject. -/
  key_some : ∀ level coordinates coordinate point deeper,
    structure_.kind level = .indexed coordinates → coordinate ∈ coordinates →
    structure_.protocol.accept point → structure_.protocol.cut (level + 1) point = some deeper →
    (structure_.key level coordinate (structure_.protocol.last deeper coordinate)).isSome
  /-- The indexed-stage answer law (p. 76): a coordinate takes each value with probability at most
  `1/|S_{r,u}|`. -/
  key_mass : ∀ level coordinates coordinate value,
    structure_.kind level = .indexed coordinates → coordinate ∈ coordinates →
    mass {slot | structure_.key level coordinate slot = some value} ≤
      structure_.weight level coordinate
  weight_nonnegative : ∀ level coordinate, 0 ≤ structure_.weight level coordinate
  /-- Remark 10.18 at an indexed stage: the verifier requires the work bits to be zero. -/
  work_accept : ∀ level coordinates point deeper, structure_.kind level = .indexed coordinates →
    structure_.protocol.accept point → structure_.protocol.cut (level + 1) point = some deeper →
    structure_.protocol.last deeper ∈ structure_.work level
  /-- The work bits are not in the slot of a coordinate of an indexed stage. -/
  work_keep : ∀ level coordinates coordinate reply slot,
    structure_.kind level = .indexed coordinates → coordinate ∈ coordinates →
    (Function.update reply coordinate slot ∈ structure_.work level ↔
      reply ∈ structure_.work level)
  /-- The work bits are zero with probability at most `2^{−g_r}`. -/
  work_mass : ∀ level coordinates, structure_.kind level = .indexed coordinates →
    mass (structure_.work level) ≤ structure_.chance level
  /-- Item (ii): `Y_{m+1}` is the set of the accepting transcripts. -/
  leaf_point : ∀ point, structure_.point (structure_.leaf point) = point
  leaf_holds : ∀ point, structure_.protocol.accept point →
    structure_.holds structure_.protocol.stages (structure_.leaf point)
  /-- Item (iii). -/
  plain_holds : ∀ level certificate, structure_.kind level = .plain →
    structure_.holds (level + 1) certificate → structure_.holds level certificate
  /-- Item (iv): `Comb_r` on a reference certificate and one certificate for each coordinate. The
  certificates share the stage input, and the answer of the certificate of a coordinate differs
  from the reference answer in the slot of that coordinate only, with a different decoded value. -/
  combine_holds : ∀ level coordinates reference family anchor deeper,
    structure_.kind level = .indexed coordinates → structure_.holds (level + 1) reference →
    structure_.protocol.cut level (structure_.point reference) = some anchor →
    structure_.protocol.cut (level + 1) (structure_.point reference) = some deeper →
    (∀ coordinate ∈ coordinates, structure_.holds (level + 1) (family coordinate) ∧
      structure_.protocol.cut level (structure_.point (family coordinate)) = some anchor ∧
      ∃ other slot,
        structure_.protocol.cut (level + 1) (structure_.point (family coordinate)) = some other ∧
        structure_.protocol.last other =
          Function.update (structure_.protocol.last deeper) coordinate slot ∧
        structure_.key level coordinate slot ≠
          structure_.key level coordinate (structure_.protocol.last deeper coordinate)) →
    Sum.elim
      (fun certificate => structure_.holds level certificate ∧ ∀ earlier ≤ level,
        structure_.protocol.cut earlier (structure_.point certificate) =
          structure_.protocol.cut earlier (structure_.point reference))
      structure_.breaks (structure_.combine level reference family)
  /-- Item (v): `Y_ℓ = {y ∈ Y_{ℓ+1} : Good_ℓ(y) = 1}`. -/
  good_holds : ∀ level span certificate, structure_.kind level = .checkpoint span →
    structure_.holds (level + 1) certificate → structure_.good level certificate →
    structure_.holds level certificate
  /-- Item (vi). -/
  collide_breaks : ∀ level span certificate other anchor,
    structure_.kind level = .checkpoint span →
    structure_.holds (level + 1) certificate → structure_.holds (level + 1) other →
    structure_.protocol.cut level (structure_.point certificate) = some anchor →
    structure_.protocol.cut level (structure_.point other) = some anchor →
    structure_.candidate level certificate ≠ structure_.candidate level other →
    structure_.breaks (structure_.collide level certificate other)
  /-- Item (vii)(a). -/
  refuted_start : ∀ level span certificate anchor, structure_.kind level = .checkpoint span →
    structure_.holds (level + 1) certificate → ¬ structure_.good level certificate →
    structure_.protocol.cut level (structure_.point certificate) = some anchor →
    structure_.refuted level anchor (structure_.candidate level certificate)
  /-- Item (vii)(b). -/
  refuted_step : ∀ level span input candidate reply proof,
    structure_.kind level = .checkpoint span →
    level ≤ structure_.protocol.stage input → structure_.protocol.stage input ≤ level + span →
    structure_.refuted level input candidate → reply ∉ structure_.bad level input candidate →
    structure_.refuted level (structure_.protocol.step proof input reply) candidate
  /-- Item (vii)(c). -/
  refuted_end : ∀ level span certificate input, structure_.kind level = .checkpoint span →
    structure_.holds (level + 1) certificate →
    structure_.protocol.cut (level + span + 1) (structure_.point certificate) = some input →
    ¬ structure_.refuted level input (structure_.candidate level certificate)
  /-- Item (vii): a bad set has probability at most `ε_j`. -/
  bad_mass : ∀ level span input candidate, structure_.kind level = .checkpoint span →
    level ≤ structure_.protocol.stage input → structure_.protocol.stage input ≤ level + span →
    mass (structure_.bad level input candidate) ≤
      structure_.error (structure_.protocol.stage input)
  error_nonnegative : ∀ level, 0 ≤ structure_.error level
  /-- Item (viii). -/
  witness_holds : ∀ certificate statement, structure_.holds 0 certificate →
    structure_.protocol.cut 0 (structure_.point certificate) = some statement →
    structure_.relation statement (structure_.witness certificate)

/-- The stage error `ε_r`: `e_r = 2^{−g_r}·Σ_u 1/|S_{r,u}|` for an indexed stage. In the model of
one squeeze run, this is also the weight (159) of a stage input of stage `r`, because one query
reads all coordinates. The factor `2^{−g_r}` is that of Remark 10.18: the extractor keeps the work
bits when it forks a coordinate, so a fork fails only at an answer with zero work bits. -/
noncomputable def charge (level : ℕ) : ℝ :=
  if level < structure_.protocol.stages then
    match structure_.kind level with
    | .indexed coordinates =>
        structure_.chance level * (coordinates.map (structure_.weight level)).sum
    | _ => structure_.error level
  else 0

/-- The number of retry loops of the extractor at a stage. -/
def loops (level : ℕ) : ℕ :=
  match structure_.kind level with
  | .indexed coordinates => coordinates.length
  | .checkpoint _ => 1
  | .plain => 0

/-- The number of children of a node of the extractor: `b_r = 1 + |U_r|` at an indexed stage and
2 at a checkpoint. -/
def arity (level : ℕ) : ℕ :=
  match structure_.kind level with
  | .indexed coordinates => 1 + coordinates.length
  | .checkpoint _ => 2
  | .plain => 1

/-- The tree size of the stages from `level`. -/
def treeFrom (level : ℕ) : ℕ :=
  ∏ later ∈ Finset.Ico level structure_.protocol.stages, structure_.arity later

/-- The tree size `K̂ = 2^{|H|}·Π_{r∈X} b_r` of Definition 10.16. -/
def treeSize : ℕ := structure_.treeFrom 0

/-- The number of retry loops of the stages from `level`. -/
def loopsFrom (level : ℕ) : ℕ :=
  ∑ later ∈ Finset.Ico level structure_.protocol.stages, structure_.loops later

/-- `ε_ck = Σ_r ε_r`. -/
noncomputable def checkError : ℝ :=
  ∑ level ∈ Finset.range structure_.protocol.stages, structure_.charge level

theorem treeFrom_succ {level : ℕ} (inside : level < structure_.protocol.stages) :
    structure_.treeFrom level = structure_.arity level * structure_.treeFrom (level + 1) :=
  Finset.prod_eq_prod_Ico_succ_bot inside _

theorem loopsFrom_succ {level : ℕ} (inside : level < structure_.protocol.stages) :
    structure_.loopsFrom level = structure_.loops level + structure_.loopsFrom (level + 1) :=
  Finset.sum_eq_sum_Ico_succ_bot inside _

theorem charge_nonnegative [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
    (lawful : structure_.Lawful) (level : ℕ) : 0 ≤ structure_.charge level := by
  unfold charge
  split
  · split
    · rename_i coordinates indexed
      refine mul_nonneg ((RandomOracle.mass_nonnegative _).trans
        (lawful.work_mass level coordinates indexed)) (List.sum_nonneg fun value member => ?_)
      obtain ⟨coordinate, _, rfl⟩ := List.mem_map.mp member
      exact lawful.weight_nonnegative level coordinate
    · exact lawful.error_nonnegative level
  · exact le_rfl

end Structure

end NightstreamFPrime.Spec.Checkpoint
