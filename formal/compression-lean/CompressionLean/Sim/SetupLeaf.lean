import CompressionLean.Sim.Merkle
import CompressionLean.Ports.Circuit.Honest
import CompressionLean.Bridge.Setup

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext foldVars generator twoAdicGenerator)
open CompressionLean.Bridge (two_groups)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

theorem monomial_add (l : List Ext) (u k : ℕ) : Replay.monomial l (u + 2 ^ l.length * k) = Replay.monomial l u := by
  induction l generalizing u with
  | nil => rfl
  | cons a l ih =>
      rw [List.length_cons, pow_succ, mul_comm (2 ^ l.length) 2, mul_assoc]
      simp only [Replay.monomial]
      rw [show (u + 2 * (2 ^ l.length * k)) / 2 = u / 2 + 2 ^ l.length * k by omega, ih,
        show (u + 2 * (2 ^ l.length * k)) % 2 = u % 2 by omega]

theorem monomial_append (l : List Ext) (x : Ext) (u : ℕ) :
    Replay.monomial (l ++ [x]) u = Replay.monomial l u * (if u / 2 ^ l.length % 2 = 1 then x else 1) := by
  induction l generalizing u with
  | nil =>
      simp only [List.nil_append, Replay.monomial, List.length_nil, pow_zero, Nat.div_one]
      rw [ext_mul_one, ext_one_mul]
  | cons a l ih =>
      have e : u / 2 ^ (l.length + 1) = u / 2 / 2 ^ l.length := by rw [pow_succ', Nat.div_div_eq_div_mul]
      simp only [List.cons_append, Replay.monomial, List.length_cons, ih, e]
      rw [ext_mul_assoc]

theorem monomials_succ (l : List Ext) (x : Ext) :
    (Array.range (2 ^ l.length)).map (Replay.monomial l) ++
        (Array.range (2 ^ l.length)).map (fun u => Replay.monomial l u * x) =
      (Array.range (2 ^ (l.length + 1))).map (Replay.monomial (l ++ [x])) := by
  refine Array.ext (by simp [pow_succ]; omega) fun i h₁ h₂ => ?_
  rw [Array.getElem_map, Array.getElem_range, monomial_append]
  by_cases lt : i < 2 ^ l.length
  · rw [Array.getElem_append_left (by simpa using lt), Array.getElem_map, Array.getElem_range,
      Nat.div_eq_of_lt lt, ite_eq_right (by omega), ext_mul_one]
  · have hi : i < 2 ^ l.length + 2 ^ l.length := by simpa using h₁
    rw [Array.getElem_append_right (by simpa using lt), Array.getElem_map, Array.getElem_range]
    simp only [Array.size_map, Array.size_range]
    rw [show i / 2 ^ l.length = 1 from Nat.div_eq_of_lt_le (by omega) (by omega), ite_eq_left rfl]
    conv_rhs => rw [show i = i - 2 ^ l.length + 2 ^ l.length * 1 by omega, monomial_add]

theorem add_scale_zero (x w : Ext) : x + w.scale default = x :=
  coords_injective (F := Gl) (by
    rw [coords_add, coords_scale]
    funext k
    simp [show (default : Word) = 0 from rfl])

theorem pathOk_of {root : Array Word} {index : ℕ} {leaf : Replay.Leaf}
    (h : ∀ k < 4, (Replay.merkleRoot (Replay.hashLeaf leaf.values) index leaf.path)[k]! = root[k]!) :
    Replay.pathOk root index leaf = true := by
  unfold Replay.pathOk
  rw [List.all_eq_true]
  intro lane hl
  rw [h lane (List.mem_range.mp hl)]
  exact beq_self_eq_true (root[lane]!).word

theorem setupCheck_ok {root : Array Word} {vars index : ℕ} {leaf : Replay.Leaf} {a b alpha : Array Ext}
    (path : Replay.pathOk root index leaf = true) :
    Replay.setupCheck root vars index leaf #[a, b] alpha = .ok
      #[Replay.sumRange (2 ^ foldVars) fun v => Replay.leafWeight alpha
          (Replay.powerFromBits (twoAdicGenerator (vars + 1)).inv index (vars + 1 - foldVars) * generator.inv)
          (twoAdicGenerator foldVars) v *
            Replay.sumRange a.size fun k => a[k]!.scale leaf.values[(0 + k) * 2 ^ foldVars + v]!,
        Replay.sumRange (2 ^ foldVars) fun v => Replay.leafWeight alpha
          (Replay.powerFromBits (twoAdicGenerator (vars + 1)).inv index (vars + 1 - foldVars) * generator.inv)
          (twoAdicGenerator foldVars) v *
            Replay.sumRange b.size fun k => b[k]!.scale leaf.values[(a.size + k) * 2 ^ foldVars + v]!] := by
  unfold Replay.setupCheck
  rw [path]
  dsimp only
  rw [two_groups]
  rfl

instance setupCheck_appends (root : DigestVar) (vars : ℕ) (bits : Array Var) (leaf : LeafView)
    (groups : Array (Array ExtVar)) (alpha : Array ExtVar) :
    Appends (Circuit.setupCheck root vars bits leaf groups alpha) := by
  unfold Circuit.setupCheck; appends

section LeafCheck

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

private theorem monomials_ensures {alpha : Array ExtVar} {galpha : Array Ext} {one : ExtVar} (hone : IsExt c n one 1)
    (ha : IsExts c n alpha galpha) :
    Ensures c n (forIn alpha #[one] fun a (monomials : Array ExtVar) => do
        let scaled ← forIn monomials (#[] : Array ExtVar) fun m (scaled : Array ExtVar) => do
          let x ← Build.extMul m a
          Pure.pure (ForInStep.yield (scaled.push x))
        Pure.pure (ForInStep.yield (monomials ++ scaled)))
      fun res m => IsExts c m res ((Array.range (2 ^ galpha.size)).map (Replay.monomial galpha.toList)) := by
  rw [← Array.forIn_toList]
  refine (Ensures.loop (fun j (acc : Array ExtVar) m =>
      IsExts c m acc ((Array.range (2 ^ j)).map (Replay.monomial (galpha.toList.take j)))) ?_
    fun j hj acc m _ hacc => ?_).post fun _ _ q => ?_
  · rw [show Array.range (2 ^ 0) = #[0] by decide, Array.map_singleton, List.take_zero]
    exact ⟨rfl, fun i hi => match i, hi with | 0, _ => hone⟩
  · have hj' : j < alpha.size := by simpa using hj
    rw [show alpha.toList[j] = alpha[j]! by simp [getElem!_pos alpha j hj'], ← Array.forIn_toList]
    refine (Ensures.loop (fun i (sc : Array ExtVar) m' =>
        IsExts c m' sc ((Array.range i).map fun u => Replay.monomial (galpha.toList.take j) u * galpha[j]!))
      (IsExts.range_zero _) fun i hi sc m' _ hsc => ?_).bind fun sc m'' _ hsc => Ensures.pure ⟨rfl, ?_⟩
    · have hi' : i < acc.size := by simpa using hi
      rw [show acc.toList[i] = acc[i]! by simp [getElem!_pos acc i hi']]
      refine (extMul_ensures (hacc.get_map (by have := hacc.1; simp at this; omega)).mono
        (ha.get (by rw [← ha.1]; exact hj')).mono).bind fun x _ _ hx => Ensures.pure ⟨rfl, ?_⟩
      rw [map_range_succ]
      exact hsc.mono.push hx
    · have size := ha.1
      have accSize := hacc.1
      simp only [Array.size_map, Array.size_range] at accSize
      have e := (hacc.mono (m := m'')).append hsc
      rw [Array.length_toList, accSize] at e
      have lj : j < galpha.toList.length := by simp; omega
      have len : (galpha.toList.take j).length = j := by simp; omega
      have succ := monomials_succ (galpha.toList.take j) galpha[j]!
      rw [len] at succ
      rw [succ] at e
      rwa [List.take_add_one, List.getElem?_eq_getElem lj, Option.toList_some, Array.getElem_toList,
        ← getElem!_pos]
  · rwa [Array.length_toList, ha.1, List.take_of_length_le (by simp)] at q

private theorem coefficients_ensures {monomials : Array ExtVar} {k : ℕ} {M : ℕ → Ext} {scale y : Var}
    {g0 gy : Word} (hm : IsExts c n monomials ((Array.range k).map M)) (hs : IsWord c n scale g0)
    (hy : IsWord c n y gy) :
    Ensures c n (forIn monomials (scale, (#[] : Array ExtVar)) fun monomial (acc : Var × Array ExtVar) => do
        let coefficient ← Build.extScale monomial acc.1
        let next ← Build.mul acc.1 y
        Pure.pure (ForInStep.yield (next, acc.2.push coefficient)))
      fun res m => IsExts c m res.2 ((Array.range k).map fun u => (M u).scale (g0 * gy.pow u)) := by
  rw [← Array.forIn_toList]
  refine (Ensures.loop (fun j (acc : Var × Array ExtVar) m => IsWord c m acc.1 (g0 * gy.pow j) ∧
      IsExts c m acc.2 ((Array.range j).map fun u => (M u).scale (g0 * gy.pow u))) ⟨?_, IsExts.range_zero _⟩
    fun j hj acc m _ hacc => ?_).post fun _ _ q => ?_
  · have e : g0 * gy.pow 0 = g0 := Word.eq_of_toF (by rw [Word.toF_mul, Word.toF_pow, pow_zero, mul_one])
    rw [e]
    exact hs
  · obtain ⟨hscale, hcoefs⟩ := hacc
    have hj' : j < monomials.size := by simpa using hj
    rw [show monomials.toList[j] = monomials[j]! by simp [getElem!_pos monomials j hj']]
    refine (extScale_ensures (hm.get_map (by have := hm.1; simp at this; omega)).mono hscale).bind
      fun coefficient _ _ hcoefficient => ?_
    refine (mul_ensures hscale.mono hy.mono).bind fun next _ _ hnext => Ensures.pure ⟨rfl, ?_, ?_⟩
    · have e : g0 * gy.pow j * gy = g0 * gy.pow (j + 1) := Word.eq_of_toF (by
        rw [Word.toF_mul, Word.toF_mul, Word.toF_mul, Word.toF_pow, Word.toF_pow, pow_succ, mul_assoc])
      rw [← e]
      exact hnext
    · rw [map_range_succ]
      exact hcoefs.mono.push hcoefficient.mono
  · have size : monomials.size = k := by simpa using hm.1
    rw [Array.length_toList, size] at q
    exact q.2

private theorem weights_ensures {coefficients : Array ExtVar} {k : ℕ} {C : ℕ → Ext} (omega : Word)
    (hc : IsExts c n coefficients ((Array.range k).map C)) :
    Ensures c n (forIn [:2 ^ foldVars] (#[] : Array ExtVar) fun v (weight : Array ExtVar) => do
        let zero ← Algebra.extZero
        let total ← forIn coefficients.zipIdx zero fun x (total : ExtVar) => match x with
          | (coefficient, u) => do
            let term ← Algebra.extScaleConstant coefficient (omega.pow ((2 ^ foldVars - u) * v % 2 ^ foldVars))
            let total ← Build.extAdd total term
            Pure.pure (ForInStep.yield total)
        Pure.pure (ForInStep.yield (weight.push total)))
      fun res m => IsExts c m res ((Array.range (2 ^ foldVars)).map fun v => Replay.sumRange k
        fun u => (C u).scale (omega.pow ((2 ^ foldVars - u) * v % 2 ^ foldVars))) := by
  refine (Ensures.loopRange (fun j (acc : Array ExtVar) m => IsExts c m acc ((Array.range j).map fun v =>
      Replay.sumRange k fun u => (C u).scale (omega.pow ((2 ^ foldVars - u) * v % 2 ^ foldVars))))
    (IsExts.range_zero _) fun j _ acc m _ hacc => ?_).post fun _ _ q => by simpa using q
  rw [Nat.zero_add]
  refine extZero_ensures.bind fun zero _ _ hzero => ?_
  rw [← Array.forIn_toList]
  refine (Ensures.loop (fun u (total : ExtVar) m' =>
      IsExt c m' total (Replay.sumRange u fun u => (C u).scale (omega.pow ((2 ^ foldVars - u) * j % 2 ^ foldVars))))
    hzero fun u hu total m' _ htotal => ?_).bind fun total _ _ htotal => Ensures.pure ⟨rfl, ?_⟩
  · have hu' : u < coefficients.size := by simpa using hu
    rw [show coefficients.zipIdx.toList[u] = (coefficients[u]!, u) by simp [getElem!_pos coefficients u hu']]
    dsimp only
    refine (extScaleConstant_ensures _ (hc.get_map (by have := hc.1; simp at this; omega)).mono).bind
      fun term _ _ hterm => ?_
    exact (extAdd_ensures htotal.mono hterm).bind fun total' _ _ htotal' =>
      Ensures.pure ⟨rfl, by rw [sumRange_succ]; exact htotal'⟩
  · have size : coefficients.size = k := by simpa using hc.1
    rw [Array.length_toList, Array.size_zipIdx, size] at htotal
    rw [map_range_succ]
    exact hacc.mono.push htotal

private theorem groupStep_ensures {values : Array Var} {gvalues : Array Word} {weight ws : Array ExtVar}
    {gweight : ℕ → Ext} {gws gfolded : Array Ext} {acc : ℕ × Array ExtVar} (hv : IsWords c n values gvalues)
    (hweight : IsExts c n weight ((Array.range (2 ^ foldVars)).map gweight)) (hws : IsExts c n ws gws)
    (hf : IsExts c n acc.2 gfolded) :
    Ensures c n (do
        let zero ← Algebra.extZero
        let s ← forIn ws (acc.1, Array.replicate (2 ^ foldVars) zero) fun w (s : ℕ × Array ExtVar) => do
          let combined ← forIn (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)).zipIdx s.2
            fun x (combined : Array ExtVar) => match x with
            | (value, v) => do
              let term ← Build.extScale w value
              let sum ← Build.extAdd combined[v]! term
              Pure.pure (ForInStep.yield (combined.set! v sum))
          Pure.pure (ForInStep.yield (s.1 + 1, combined))
        let f ← Algebra.dot weight s.2
        Pure.pure (ForInStep.yield (s.1, acc.2.push f)))
      fun r m => r = .yield r.value ∧ r.value.1 = acc.1 + ws.size ∧
        IsExts c m r.value.2 (gfolded.push (Replay.sumRange (2 ^ foldVars) fun v => gweight v *
          Replay.sumRange gws.size fun k => gws[k]!.scale gvalues[(acc.1 + k) * 2 ^ foldVars + v]!)) := by
  obtain ⟨S, hS⟩ : ∃ S : ℕ → ℕ → Ext, S = fun k v =>
      Replay.sumRange k fun k' => gws[k']!.scale gvalues[(acc.1 + k') * 2 ^ foldVars + v]! := ⟨_, rfl⟩
  refine extZero_ensures.bind fun zero _ _ hzero => ?_
  rw [← Array.forIn_toList]
  refine (Ensures.loop (fun k (s : ℕ × Array ExtVar) m => s.1 = acc.1 + k ∧
      IsExts c m s.2 ((Array.range (2 ^ foldVars)).map (S k))) ?_
    fun k hk s m _ hs => ?_).bind fun s _ _ hs => ?_
  · exact ⟨rfl, IsExts.replicate_map hzero fun v _ => by rw [hS]; simp [Replay.sumRange]⟩
  · obtain ⟨hs1, hs2⟩ := hs
    have hk' : k < ws.size := by simpa using hk
    rw [show ws.toList[k] = ws[k]! by simp [getElem!_pos ws k hk'], ← Array.forIn_toList]
    refine (Ensures.loop (fun j (comb : Array ExtVar) m' => IsExts c m' comb ((Array.range (2 ^ foldVars)).map
        fun v => if v < j then S k v + gws[k]!.scale gvalues[2 ^ foldVars * s.1 + v]! else S k v))
      (hs2.congr_map fun v _ => by simp) fun j hj comb m' _ hcomb => ?_).bind
      fun comb _ _ hcomb => Ensures.pure ⟨rfl, by simp only [ForInStep.value]; omega, ?_⟩
    · have hj' : j < (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)).size := by
        simpa only [Array.length_toList, Array.size_zipIdx] using hj
      rw [show (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)).zipIdx.toList[j] =
        ((values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars))[j]!, j) by
        simp [getElem!_pos (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)) j hj']]
      dsimp only
      have small : j < 2 ^ foldVars := by simp at hj'; omega
      have inside : 2 ^ foldVars * s.1 + j < values.size := by simp at hj'; omega
      have hvalue : IsWord c n (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars))[j]!
          gvalues[2 ^ foldVars * s.1 + j]! := by
        rw [getElem!_pos (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)) j hj',
          Array.getElem_extract, ← getElem!_pos]
        exact hv.get (by rw [← hv.1]; exact inside)
      refine (extScale_ensures (gs := gvalues[2 ^ foldVars * s.1 + j]!)
        (hws.get (by rw [← hws.1]; exact hk')).mono hvalue.mono).bind fun term _ _ hterm => ?_
      have hold : IsExt c m' comb[j]! (S k j) := by
        have e := hcomb.get_map small
        rwa [ite_eq_right (lt_irrefl j)] at e
      refine (extAdd_ensures (ga := S k j) hold.mono hterm).bind fun sum m₅ _ hsum => Ensures.pure ⟨rfl, ?_⟩
      have size : comb.size = 2 ^ foldVars := by simpa using hcomb.1
      simp only [ForInStep.value]
      refine ⟨by simp [size], fun v hv' => ?_⟩
      rw [Array.size_set!, size] at hv'
      rw [getElem!_set! _ _ _ _ (by rw [size]; exact hv'), getElem!_map_range _ _ hv']
      by_cases e : j = v
      · subst e
        rw [ite_eq_left rfl, ite_eq_left (Nat.lt_succ_self j)]
        exact hsum
      · rw [ite_eq_right e]
        have old := (hcomb.get_map hv').mono (m := m₅)
        by_cases lt : v < j
        · rw [ite_eq_left lt, ite_eq_left (by omega)] at *; assumption
        · rw [ite_eq_right lt, ite_eq_right (by omega)] at *; assumption
    · refine hcomb.congr_map fun v _ => ?_
      simp only [hS, sumRange_succ]
      split_ifs with lt
      · rw [hs1, show 2 ^ foldVars * (acc.1 + k) = (acc.1 + k) * 2 ^ foldVars from Nat.mul_comm _ _]
      · have size := hv.1
        simp only [Array.length_toList, Array.size_zipIdx, Array.size_extract] at lt
        have far : ¬ 2 ^ foldVars * s.1 + v < gvalues.size := by rw [← size]; omega
        rw [show (acc.1 + k) * 2 ^ foldVars + v = 2 ^ foldVars * s.1 + v by
          rw [hs1, Nat.mul_comm], getElem!_neg gvalues _ far, add_scale_zero]
  · obtain ⟨hs1, hs2⟩ := hs
    refine (dot_ensures hweight.mono hs2).bind fun f _ _ hf' =>
      Ensures.pure ⟨rfl, by simp only [ForInStep.value]; rw [hs1, Array.length_toList], ?_⟩
    have size := hws.1
    refine hf.mono.push (hf'.congr ?_)
    rw [Replay.dot, Array.size_map, Array.size_map, Array.size_range, Nat.min_self]
    refine congrArg coords (sumRange_congr fun v hv => ?_)
    rw [getElem!_map_range _ _ hv, getElem!_map_range _ _ hv, show ws.toList.length = gws.size by simp [size], hS]

private theorem rootEqual_ensures {computed root : DigestVar} {gcomputed groot : Array Word}
    (hc : IsDigest c n computed gcomputed) (hr : IsDigest c n root groot) :
    Ensures c n (forIn (List.finRange 4) PUnit.unit fun lane _ => do
        Build.assertEqual (computed lane) (root lane)
        Pure.pure (ForInStep.yield PUnit.unit))
      fun _ _ => ∀ k < 4, gcomputed[k]! = groot[k]! := by
  refine (Ensures.loop (fun j _ _ => ∀ k < j, gcomputed[k]! = groot[k]!) (fun _ h => absurd h (by omega))
    fun k hk _ m _ hprev => ?_).post fun _ _ h => by simpa using h
  have k4 : k < 4 := by simpa using hk
  rw [List.getElem_finRange]
  have hck := hc.2 ⟨k, k4⟩
  have hrk := hr.2 ⟨k, k4⟩
  rw [Fin.getElem!_fin] at hck hrk
  refine (assertEqual_ensures (ga := gcomputed[k]!) (gb := groot[k]!) hck.mono hrk.mono).bind fun _ _ _ heq =>
    Ensures.pure ⟨rfl, fun j hj => ?_⟩
  rcases Nat.lt_succ_iff_lt_or_eq.mp hj with lt | rfl
  · exact hprev j lt
  · exact heq

private theorem groups_ensures {values : Array Var} {gvalues : Array Word} {weight w₀ w₁ : Array ExtVar}
    {gweight : ℕ → Ext} {gw₀ gw₁ : Array Ext} (hv : IsWords c n values gvalues)
    (hweight : IsExts c n weight ((Array.range (2 ^ foldVars)).map gweight)) (hw₀ : IsExts c n w₀ gw₀)
    (hw₁ : IsExts c n w₁ gw₁) :
    Ensures c n (forIn #[w₀, w₁] ((0 : ℕ), (#[] : Array ExtVar)) fun ws (acc : ℕ × Array ExtVar) => do
        let zero ← Algebra.extZero
        let s ← forIn ws (acc.1, Array.replicate (2 ^ foldVars) zero) fun w (s : ℕ × Array ExtVar) => do
          let combined ← forIn (values.extract (2 ^ foldVars * s.1) (2 ^ foldVars * s.1 + 2 ^ foldVars)).zipIdx s.2
            fun x (combined : Array ExtVar) => match x with
            | (value, v) => do
              let term ← Build.extScale w value
              let sum ← Build.extAdd combined[v]! term
              Pure.pure (ForInStep.yield (combined.set! v sum))
          Pure.pure (ForInStep.yield (s.1 + 1, combined))
        let f ← Algebra.dot weight s.2
        Pure.pure (ForInStep.yield (s.1, acc.2.push f)))
      fun r m => IsExts c m r.2
        #[Replay.sumRange (2 ^ foldVars) fun v => gweight v *
            Replay.sumRange gw₀.size fun k => gw₀[k]!.scale gvalues[(0 + k) * 2 ^ foldVars + v]!,
          Replay.sumRange (2 ^ foldVars) fun v => gweight v *
            Replay.sumRange gw₁.size fun k => gw₁[k]!.scale gvalues[(gw₀.size + k) * 2 ^ foldVars + v]!] := by
  rw [← Array.forIn_toList, show #[w₀, w₁].toList = [w₀, w₁] from rfl, List.forIn_cons]
  refine (groupStep_ensures (acc := (0, #[])) (gfolded := #[]) hv hweight hw₀ IsExts.empty).bind
    fun r _ _ hr => ?_
  obtain ⟨hyield, hoffset, hfolded⟩ := hr
  rcases r with b | b
  · exact absurd hyield (by simp)
  dsimp only
  rw [List.forIn_cons]
  refine (groupStep_ensures (acc := b) (gfolded := #[].push _) hv.mono hweight.mono hw₁.mono hfolded).bind
    fun r _ _ hr => ?_
  obtain ⟨hyield', -, hfolded'⟩ := hr
  rcases r with b' | b'
  · exact absurd hyield' (by simp)
  dsimp only
  rw [List.forIn_nil]
  refine Ensures.pure ?_
  have offset : b.1 = gw₀.size := by
    simp only [ForInStep.value, Nat.zero_add] at hoffset
    rw [hoffset, hw₀.1]
  simp only [ForInStep.value] at hfolded'
  rw [offset] at hfolded'
  exact hfolded'

theorem setupCheck_ensures {root : DigestVar} {groot : Array Word} {vars index : ℕ} {bits : Array Var}
    {leaf : LeafView} {g : Replay.Leaf} {w₀ w₁ alpha : Array ExtVar} {gw₀ gw₁ galpha : Array Ext}
    (hp : Permutes c.perm) (hroot : IsDigest c n root groot) (hbsize : bits.size = vars + 1 - foldVars)
    (hbits : ∀ i < bits.size, IsWord c n bits[i]! (Word.ofNat (index / 2 ^ i % 2)))
    (hvalues : IsWords c n leaf.values g.values) (hpsize : leaf.path.size = g.path.size)
    (hpvars : leaf.path.size = vars + 1 - foldVars)
    (hpath : ∀ k < leaf.path.size, IsDigest c n leaf.path[k]! g.path[k]!)
    (hw₀ : IsExts c n w₀ gw₀) (hw₁ : IsExts c n w₁ gw₁) (halpha : IsExts c n alpha galpha)
    (hsize : galpha.size = foldVars) :
    Ensures c n (Circuit.setupCheck root vars bits leaf #[w₀, w₁] alpha) fun res m => ∃ vals,
      Replay.setupCheck groot vars index g #[gw₀, gw₁] galpha = .ok vals ∧ IsExts c m res vals := by
  obtain ⟨Y, hY⟩ : ∃ Y, Y = Replay.powerFromBits (twoAdicGenerator (vars + 1)).inv index (vars + 1 - foldVars) *
      generator.inv := ⟨_, rfl⟩
  unfold Circuit.setupCheck
  refine (hashLeaf_ensures hp hvalues).bind fun digest _ _ hdigest => ?_
  refine (merkleRoot_ensures (index := index) (gpath := g.path) hp hdigest (fun i hi => (hbits i hi).mono)
    (fun k hk => (hpath k hk).mono) (by rw [hbsize, hpvars]) hpsize).bind fun computed _ _ hcomputed => ?_
  refine (rootEqual_ensures hcomputed hroot.mono).bind fun _ _ _ hequal => ?_
  have path := pathOk_of hequal
  refine (powerFromBits_ensures (twoAdicGenerator (vars + 1)).inv index fun i hi => (hbits i hi).mono).bind
    fun rotation _ _ hrotation => ?_
  refine (scale_ensures generator.inv hrotation).bind fun y _ _ hy => ?_
  rw [hbsize, ← hY] at hy
  refine extOne_ensures.bind fun one _ _ hone => ?_
  refine (monomials_ensures hone halpha.mono).bind fun monomials _ _ hmonomials => ?_
  refine (const_ensures (Word.ofNat (2 ^ foldVars)).inv).bind fun scale _ _ hscale => ?_
  refine (coefficients_ensures hmonomials.mono hscale hy.mono).bind fun coefs _ _ hcoefs => ?_
  refine (weights_ensures (twoAdicGenerator foldVars) hcoefs).bind fun weight _ _ hweight => ?_
  rw [hsize] at hweight
  refine (groups_ensures (gweight := Replay.leafWeight galpha Y (twoAdicGenerator foldVars)) hvalues.mono hweight
    hw₀.mono hw₁.mono).bind fun s _ _ hs => Ensures.pure ⟨_, setupCheck_ok path, ?_⟩
  rw [← hY]
  exact hs

end LeafCheck

end CompressionLean.Sim
