import NightstreamFPrime.Spec.Compression.Rust.Placement

/-! The layout of Rust `Layout::new` at the production parameters. When `layoutNew` accepts the
claims of `Production.params`, its layout satisfies `WellFormed`, the key polynomial holds the
blocks of each claim, and each witness slice has at most `2^25` cells (`layoutNew_production`). So
`soundness` and `params_run_soundness` hold for that layout without these premises
(`layout_soundness`, `layout_run_soundness`). The one premise left is the check of Rust `layout`
(`crates/neo-compress/src/lib.rs`) that a claim has at most as many public cells as cells. -/

namespace NightstreamFPrime.Spec.Compression.Rust

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Compression.Verifier NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Game

/-- What `layoutNew` checks when it accepts: the schedule file of the key, the blocks of each
claim, and the schedule row that holds the cells. -/
theorem layoutNew_ok {claims : List Claim} {keyBlocks : ℕ} {l : Placed}
    (ok : layoutNew claims keyBlocks = .ok l) :
    ∃ schedule ∈ schedules, schedule.blocks = keyBlocks ∧ (∀ c ∈ claims, c.blocks ≤ keyBlocks) ∧
      ∃ row ∈ schedule.rows, rowFits claims row = true ∧ l = assemble claims schedule row := by
  unfold layoutNew at ok
  split at ok
  · cases ok
  rename_i schedule found
  split_ifs at ok with empty above none wide
  split at ok
  · cases ok
  rename_i row fits
  cases ok
  refine ⟨schedule, List.mem_of_find?_eq_some found, by simpa using List.find?_some found,
    fun c member => ?_, row, List.mem_of_find?_eq_some fits, List.find?_some fits, rfl⟩
  simp only [List.any_eq_true, decide_eq_true_eq, not_exists, not_and, not_lt] at above
  exact above c member

theorem fits_of {s : Slice} {cells n : ℕ} (aligned : 2 ^ s.vars ∣ s.offset)
    (inside : s.offset + 2 ^ s.vars ≤ cells) (cube : varsFor cells ≤ n) : s.Fits n := by
  have within : cells ≤ 2 ^ n :=
    (Nat.le_pow_clog one_lt_two cells).trans (Nat.pow_le_pow_right two_pos cube)
  refine ⟨aligned, inside.trans within, ?_⟩
  by_contra above
  have := Nat.pow_lt_pow_right one_lt_two (not_le.mp above)
  omega

theorem assemble_cell (claims : List Claim) (schedule : Schedule) (row : ℕ × ℕ) {i : ℕ}
    (below : i < claims.length) :
    (assemble claims schedule row).cells.getD i ⟨⟨0, 0⟩, 0, none⟩ = cellOf claims i := by
  simp [assemble, List.getD_eq_getElem?_getD, below]

section Production

variable (key : ℕ → ℕ → ℕ → F) (matrices : Fin 3 → Matrices F) (depths : Fin 2 → ℕ)

/-- The claims of `params key matrices depths`, for Rust `Layout::new`. -/
abbrev claimsOf : List Claim := paramsClaims (fun c => (matrices c).columns / ringDegree) depths

theorem claimsOf_getD (c : Fin 3) : (claimsOf matrices depths).getD c default =
    ⟨(matrices c).columns / ringDegree, if (c : ℕ) < 2 then some (bound depths c) else none⟩ := by
  fin_cases c <;> rfl

/-- When `layoutNew` accepts the claims of `params`, and each claim has at least as many cells as
the 320 public cells, the layout is well formed, the key holds each claim, and each witness slice
has at most `2^25` cells. -/
theorem layoutNew_production {keyBlocks : ℕ} {l : Placed}
    (ok : layoutNew (claimsOf matrices depths) keyBlocks = .ok l)
    (cells_fit : ∀ c, 320 ≤ ringDegree * ((matrices c).columns / ringDegree)) :
    WellFormed (params key matrices depths l.production) ∧
      (∀ c : Fin 3, (l.production.witness c).vars - 6 ≤ l.production.keyBlockVars) ∧
      ∀ c : Fin 3, (l.production.witness c).vars ≤ 25 := by
  obtain ⟨schedule, member, keyBlocks_eq, fit, row, -, fits, rfl⟩ := layoutNew_ok ok
  simp only [rowFits, Bool.and_eq_true, decide_eq_true_eq] at fits
  obtain ⟨cube, -⟩ := fits
  have keyVars : 11 ≤ schedule.keyVars ∧ schedule.keyVars ≤ 30 := by
    simp only [schedules, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> decide
  have below : ∀ c : Fin 3, (c : ℕ) < (claimsOf matrices depths).length := fun c => c.isLt
  have cell := fun c : Fin 3 => assemble_cell (claimsOf matrices depths) schedule row (below c)
  have vars : ∀ c : Fin 3, (cellOf (claimsOf matrices depths) c).witness.vars =
      varsFor ((matrices c).columns / ringDegree) + 6 := fun c => by
    rw [(cellOf_witness (below c)).1, witnessVarsOf, claimsOf_getD]
    rfl
  have fitAt : ∀ c : Fin 3, (matrices c).columns / ringDegree ≤ 2 ^ (schedule.keyVars - 11) :=
    fun c => by
      have member : (claimsOf matrices depths).getD c default ∈ claimsOf matrices depths := by
        rw [List.getD_eq_getElem _ _ (below c)]
        exact List.getElem_mem _
      have := fit _ member
      rw [claimsOf_getD, ← keyBlocks_eq] at this
      exact this
  have keyFit : ∀ c : Fin 3, varsFor ((matrices c).columns / ringDegree) ≤ schedule.keyVars - 11 :=
    fun c => Nat.clog_le_of_le_pow (fitAt c)
  have span_cube : varsFor (tableSlices (claimsOf matrices depths)).2 ≤ row.1 :=
    (Nat.clog_mono_right 2 ((tables_le_others _).trans (others_le_counts _))).trans cube
  have table_of : ∀ c : Fin 3, decide ((c : ℕ) < 2) = true →
      ((claimsOf matrices depths).getD c default).table = some (bound depths c) := fun c lt => by
    rw [claimsOf_getD]
    simp [of_decide_eq_true lt]
  refine ⟨⟨fun c => ?_, fun c lt => ?_, fun c => ?_, fun c => ?_, fun c lt => ?_, fun c lt => ?_,
    span_cube⟩, fun c => ?_, fun c => ?_⟩
  · change ((assemble _ schedule row).cells.getD c ⟨⟨0, 0⟩, 0, none⟩).witness.Fits row.1
    rw [cell c]
    obtain ⟨_, aligned, inside, _⟩ := cellOf_witness (below c)
    exact fits_of aligned (inside.trans (others_le_counts _)) cube
  · obtain ⟨s, found, -, aligned, inside⟩ := cellOf_table (below c) (table_of c lt)
    change ((((assemble _ schedule row).cells.getD c ⟨⟨0, 0⟩, 0, none⟩).table.map
      Prod.snd).getD ⟨0, 0⟩).Fits row.1
    rw [cell c, found]
    exact fits_of aligned inside cube
  · change 2 ^ 6 * ((matrices c).columns / ringDegree) ≤
      2 ^ ((assemble _ schedule row).cells.getD c ⟨⟨0, 0⟩, 0, none⟩).witness.vars
    rw [cell c, vars c, pow_add, mul_comm]
    exact Nat.mul_le_mul_right _ (Nat.le_pow_clog one_lt_two _)
  · exact cells_fit c
  · obtain ⟨s, found, size, -, -⟩ := cellOf_table (below c) (table_of c lt)
    change 2 * bound depths c + 1 ≤ 2 ^ ((((assemble _ schedule row).cells.getD c
      ⟨⟨0, 0⟩, 0, none⟩).table.map Prod.snd).getD ⟨0, 0⟩).vars
    rw [cell c, found]
    simp only [Option.map_some, Option.getD_some, size]
    exact Nat.le_pow_clog one_lt_two _
  · change ((assemble _ schedule row).cells.getD c ⟨⟨0, 0⟩, 0, none⟩).witness.offset +
        2 ^ ((assemble _ schedule row).cells.getD c ⟨⟨0, 0⟩, 0, none⟩).witness.vars ≤
        2 ^ varsFor (tableSlices _).2 ∧
      ((assemble _ schedule row).cells.getD c ⟨⟨0, 0⟩, 0, none⟩).witness.vars ≤
        varsFor (tableSlices _).2
    rw [cell c]
    obtain ⟨_, aligned, _, span⟩ := cellOf_witness (below c)
    have fits := fits_of aligned (span (by rw [table_of c lt]; rfl)) le_rfl
    exact ⟨fits.2.1, fits.2.2⟩
  · change ((assemble _ schedule row).cells.getD c ⟨⟨0, 0⟩, 0, none⟩).witness.vars - 6 ≤
      schedule.keyVars - laneVars - rowVars
    rw [cell c, vars c]
    have := keyFit c
    simp only [laneVars, rowVars]
    omega
  · change ((assemble _ schedule row).cells.getD c ⟨⟨0, 0⟩, 0, none⟩).witness.vars ≤ 25
    rw [cell c, vars c]
    have := keyFit c
    omega

variable (rel : ℕ → Factored F) (which : Fin 3 → ℕ) (zeros : ℕ)

/-- `params_soundness` for the layout of Rust `Layout::new`: `WellFormed`, the key bound and the
slot bound follow from `layoutNew`. -/
theorem layout_soundness {keyBlocks : ℕ} {l : Placed}
    (ok : layoutNew (claimsOf (fun c => (rel (which c)).matrices) depths) keyBlocks = .ok l)
    (cells_fit : ∀ c, 320 ≤ ringDegree * ((rel (which c)).matrices.columns / ringDegree))
    (s : Strategy (Option (Option (Statement (params key (fun c => (rel (which c)).matrices)
      depths l.production) × (ℕ → F))))) (q : ℕ) :
    prob s (restore (game (verifierData key rel which depths l.production zeros)) q [])
        (fun r => ¬ ∀ a, r = some a → ∀ o, a = some o → Relation _ o.1 o.2) ≤
      q * error (verifierData key rel which depths l.production zeros) := by
  obtain ⟨wf, key_fits, small⟩ := layoutNew_production key _ depths ok cells_fit
  exact params_soundness key rel which depths l.production zeros wf (fun c _ => small c) key_fits
    s q

/-- `params_run_soundness` for the layout of Rust `Layout::new`: one interactive run accepts claims
without `Relation` with probability at most the exact Rust sum of that layout. -/
theorem layout_run_soundness {keyBlocks : ℕ} {l : Placed}
    (ok : layoutNew (claimsOf (fun c => (rel (which c)).matrices) depths) keyBlocks = .ok l)
    (cells_fit : ∀ c, 320 ≤ ringDegree * ((rel (which c)).matrices.columns / ringDegree))
    (s : Strategy (Option (Statement (params key (fun c => (rel (which c)).matrices) depths
      l.production) × (ℕ → F)))) :
    prob s (game (verifierData key rel which depths l.production zeros))
        (fun r => ¬ ∀ o, r = some o → Relation _ o.1 o.2) ≤
      (runError (productionCounts l.production depths) : ℝ) := by
  obtain ⟨wf, key_fits, small⟩ := layoutNew_production key _ depths ok cells_fit
  exact params_run_soundness key rel which depths l.production zeros wf (fun c _ => small c)
    key_fits s

end Production

end NightstreamFPrime.Spec.Compression.Rust
