import CompressionLean.Shrink.BlockPart

namespace CompressionLean.Shrink

open CompressionLean.Layer1

def splitEq {K : Type} [CommRing K] (point : List K) (index : ℕ) : K :=
  eqAt (point.take (point.length / 2)) (index % 2 ^ (point.length / 2)) *
    eqAt (point.drop (point.length / 2)) (index / 2 ^ (point.length / 2))

theorem splitEq_eq {K : Type} [CommRing K] (point : List K) (index : ℕ) : splitEq point index = eqAt point index := by
  unfold splitEq
  rw [← eqAt_split point _ _ _ (Nat.mod_lt _ (pow_pos (by norm_num) _)), Nat.mod_add_div]

section Glue

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

variable (T : Template F) (s : Counts) (rx ry : List K) (ρ : K)

def sideValue (terms : List (Wire × F)) : K :=
  (terms.map fun t => splitEq ry (Layout.column T s t.1) * ι t.2).sum

def productsValue : ℕ → List (List (Wire × F) × List (Wire × F)) → K
  | _, [] => 0
  | i, (a, b) :: rest =>
      ρ ^ (1 + 2 * i) * sideValue T s ry a + ρ ^ (2 + 2 * i) * sideValue T s ry b + productsValue (i + 1) rest

def rowCallValue (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F)) : K :=
  ρ ^ 7 * sideValue T s ry linear + productsValue T s ry ρ 0 products

open Classical in
noncomputable def formValue (form : Form F) : K :=
  form.terms.foldl (fun sum t => sum + splitEq ry (Layout.column T s t.1) * ι t.2)
    (if form.constant = 0 then 0 else splitEq ry (Layout.column T s (.pub 0)) * ι form.constant)

noncomputable def r1csSideValue (columns : List (Form F)) (terms : List (ℕ × F)) : K :=
  (terms.map fun t => formValue T s ry (columns.getD t.1 ⟨[], 0⟩) * ι t.2).sum

noncomputable def r1csRowValue (columns : List (Form F)) (row : R1csRow F) : K :=
  ρ ^ 1 * r1csSideValue T s ry columns row.a + ρ ^ 2 * r1csSideValue T s ry columns row.b +
    ρ ^ 7 * r1csSideValue T s ry columns row.c

noncomputable def r1csValue (columns : List (Form F)) : ℕ → List (R1csRow F) → K
  | _, [] => 0
  | g, row :: rest =>
      splitEq rx (Layout.glueRow T s g) * r1csRowValue T s ry ρ columns row + r1csValue columns (g + 1) rest

noncomputable def glueTotal : ℕ → List (Call F) → K
  | _, [] => 0
  | g, .row products linear _ :: rest =>
      splitEq rx (Layout.glueRow T s g) * rowCallValue T s ry ρ products linear + glueTotal (g + 1) rest
  | g, .r1cs rows columns :: rest => r1csValue T s rx ry ρ columns g rows + glueTotal (g + rows.length) rest

section Lemmas

variable {T : Template F} {s : Counts} {rx ry : List K} {ρ : K}

omit [Field F] [Algebra F K] in
theorem foldl_add_eq {α : Type} (f : α → K) : ∀ (l : List α) (init : K),
    l.foldl (fun sum t => sum + f t) init = init + (l.map f).sum
  | [], init => by simp
  | a :: l, init => by
      rw [List.foldl_cons, foldl_add_eq f l, List.map_cons, List.sum_cons]
      ring

theorem sideValue_append (a b : List (Wire × F)) :
    sideValue T s ry (a ++ b) = sideValue T s ry a + sideValue T s ry b := by
  simp [sideValue]

theorem sideValue_expand (columns : List (Form F)) :
    ∀ terms : List (ℕ × F), sideValue T s ry (expand terms columns) = r1csSideValue T s ry columns terms
  | [] => by simp [expand, sideValue, r1csSideValue]
  | t :: rest => by
      have restEq := sideValue_expand columns rest
      unfold expand at restEq ⊢
      rw [List.flatMap_cons, sideValue_append, restEq, sideValue_append]
      unfold r1csSideValue formValue
      rw [List.map_cons, List.sum_cons, foldl_add_eq]
      generalize columns.getD t.1 ⟨[], 0⟩ = form
      congr 1
      have terms : sideValue T s ry (form.terms.map fun u => (u.1, u.2 * t.2)) =
          (form.terms.map fun u => splitEq ry (Layout.column T s u.1) * ι u.2).sum * ι t.2 := by
        unfold sideValue
        rw [← List.sum_map_mul_right, List.map_map]
        congr 1
        apply List.map_congr_left
        intro u _
        simp only [Function.comp, map_mul]
        ring
      rw [terms]
      by_cases zero : form.constant = 0
      · simp [zero, sideValue]
      · simp only [zero, ne_eq, not_false_eq_true, ite_true, ite_false, sideValue, List.map_cons, List.map_nil,
          List.sum_cons, List.sum_nil, map_mul]
        ring

theorem matrixValue_append (a b : List (Entry F)) :
    matrixValue (a ++ b) rx ry ρ = matrixValue a rx ry ρ + matrixValue b rx ry ρ := by
  simp [matrixValue]

omit [Field F] in
theorem glueEntries_append (a b : List (GlueRow F)) :
    ∀ g, Layout.glueEntries T s g (a ++ b) = Layout.glueEntries T s g a ++ Layout.glueEntries T s (g + a.length) b := by
  induction a with
  | nil => intro g; simp [Layout.glueEntries]
  | cons row rest ih =>
      intro g
      simp only [List.cons_append, Layout.glueEntries, ih (g + 1), List.append_assoc, List.length_cons]
      congr 3
      omega

omit [Field F] [Algebra F K] in
theorem sum_map_flatMap {α β : Type} (f : α → List β) (h : β → K) :
    ∀ l : List α, ((l.flatMap f).map h).sum = (l.map fun a => ((f a).map h).sum).sum
  | [] => rfl
  | a :: l => by rw [List.flatMap_cons, List.map_append, List.sum_append, sum_map_flatMap f h l]; rfl

theorem matrixValue_rowEntries (g : ℕ) (row : GlueRow F) :
    matrixValue (Layout.rowEntries T s g row) rx ry ρ =
      eqAt rx (Layout.glueRow T s g) * ∑ side : Fin 7, ρ ^ ((side : ℕ) + 1) * sideValue T s ry (row.side side) := by
  unfold matrixValue Layout.rowEntries
  rw [sum_map_flatMap, Fin.sum_univ_def, ← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro side _
  simp only [List.map_map, sideValue, ← List.sum_map_mul_left, splitEq_eq]
  congr 1
  apply List.map_congr_left
  intro t _
  simp only [Function.comp_apply, Fin.val_succ]
  ring

theorem sideValue_nil : sideValue T s ry ([] : List (Wire × F)) = 0 := rfl

theorem rowCall_sides (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
    (three : products.length ≤ 3) :
    ∑ side : Fin 7, ρ ^ ((side : ℕ) + 1) * sideValue T s ry ((GlueRow.ofCall products linear).side side) =
      rowCallValue T s ry ρ products linear := by
  rw [Fin.sum_univ_seven]
  match products, three with
  | [], _ =>
      simp [GlueRow.ofCall, rowCallValue, productsValue, sideValue_nil]
  | [(a0, b0)], _ =>
      simp [GlueRow.ofCall, rowCallValue, productsValue, sideValue_nil]
      ring
  | [(a0, b0), (a1, b1)], _ =>
      simp [GlueRow.ofCall, rowCallValue, productsValue, sideValue_nil]
      ring
  | [(a0, b0), (a1, b1), (a2, b2)], _ =>
      simp [GlueRow.ofCall, rowCallValue, productsValue]
      ring
  | _ :: _ :: _ :: _ :: _, four => simp at four

theorem r1csValue_eq (columns : List (Form F)) :
    ∀ (g : ℕ) (rows : List (R1csRow F)), r1csValue T s rx ry ρ columns g rows =
      matrixValue (Layout.glueEntries T s g (rows.map fun r =>
        GlueRow.ofCall [(expand r.a columns, expand r.b columns)] (expand r.c columns))) rx ry ρ
  | _, [] => by simp [r1csValue, Layout.glueEntries, matrixValue]
  | g, row :: rest => by
      rw [List.map_cons, Layout.glueEntries, matrixValue_append, matrixValue_rowEntries,
        rowCall_sides _ _ (by simp), ← r1csValue_eq columns (g + 1) rest, r1csValue, splitEq_eq]
      congr 1
      simp only [rowCallValue, productsValue, r1csRowValue, sideValue_expand]
      ring

theorem glueTotal_eq : ∀ (g : ℕ) (calls : List (Call F)),
    glueTotal T s rx ry ρ g calls = matrixValue (Layout.glueEntries T s g (calls.flatMap Call.rows)) rx ry ρ
  | _, [] => by simp [glueTotal, Layout.glueEntries, matrixValue]
  | g, .row products linear three :: rest => by
      rw [List.flatMap_cons, Call.rows, List.singleton_append, Layout.glueEntries, matrixValue_append,
        matrixValue_rowEntries, rowCall_sides _ _ three, ← glueTotal_eq (g + 1) rest, glueTotal, splitEq_eq]
  | g, .r1cs rows columns :: rest => by
      rw [List.flatMap_cons, Call.rows, glueEntries_append, matrixValue_append, ← r1csValue_eq, List.length_map,
        ← glueTotal_eq (g + rows.length) rest, glueTotal]

end Lemmas

end Glue

section Matrices

variable {F K : Type} [Field F] [Field K] [Algebra F K] [DecidableEq K]

noncomputable def verifierMatrices (T : Template F) (s : Counts) (run : Run F) (rx ry : List K) (ρ : K) : K :=
  glueTotal T s rx ry ρ 0 run.calls + blockPart T s rx ry ρ

theorem verifierMatrices_eq (T : Template F) (s : Counts) (run : Run F) (rx ry : List K) (ρ : K)
    (wf : T.WellFormed) (rxLength : rx.length = Layout.rowVars T s)
    (ryLength : ry.length = Layout.cellVars T s + 1) :
    verifierMatrices T s run rx ry ρ = matrixValue (run.ccs T s).entries rx ry ρ := by
  unfold verifierMatrices Run.ccs Run.rows
  rw [matrixValue_append, glueTotal_eq, blockPart_eq T s rx ry ρ wf rxLength ryLength, add_comm]

end Matrices

end CompressionLean.Shrink
