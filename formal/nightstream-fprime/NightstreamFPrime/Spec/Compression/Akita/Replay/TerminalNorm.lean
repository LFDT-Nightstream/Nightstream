import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalSpec

/-! The norm check and the decoder of `z` of the terminal (Akita Section 8.2, "Response bounds";
`checked_centered_l2_sq`, `decode_terminal_z_golomb_payload`) against the specification.

- `decode_values`: a decoded payload has `count` values, each of `i16`.
- `centeredNormSq_eq`: for such values, the squared norm of the code is `‖z‖_2²` of the response
  of the specification (`vecNormSq`), so the check `‖z‖_2² ≤ S_max` of the code is the bound
  `Route.Short` of the Euclidean route.
- `responseOf` is the response that the decoder gives to a wire message, with the response 0 for
  a payload that does not decode: the field `Terminal.response` of `Extract/TerminalSuffix.lean`
  for the concrete instance, as a DEFINITION. `responseOf_decoded` states that it is `zSource`
  of the decoded values; a payload that does not decode is rejected by `terminalRun`
  (`terminalRun_decode`), which is the premise `Terminal.Valid.checked` needs for it. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

namespace Golomb

theorem one_range {P : Params} {bytes : ByteArray} {pos : ℕ} {n : ℤ} {next : ℕ}
    (found : one P bytes pos = some (n, next)) : -32768 ≤ n ∧ n ≤ 32767 := by
  unfold one at found
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at found
  obtain ⟨⟨quotient, after⟩, _, remainder, _, value, _, rest⟩ := found
  split at rest
  · cases rest
  · split at rest
    · cases rest
    · rename_i inside
      cases rest
      omega

theorem many_values {P : Params} {bytes : ByteArray} :
    ∀ (count pos : ℕ) (out values : Array ℤ) (next : ℕ),
      many P bytes count pos out = some (values, next) →
        values.size = out.size + count ∧
          ∀ n ∈ values, n ∈ out ∨ (-32768 ≤ n ∧ n ≤ 32767)
  | 0, pos, out, values, next, found => by
      simp only [many, Option.some.injEq, Prod.mk.injEq] at found
      obtain ⟨rfl, rfl⟩ := found
      exact ⟨rfl, fun n member => .inl member⟩
  | count + 1, pos, out, values, next, found => by
      simp only [many, Option.bind_eq_bind, Option.bind_eq_some_iff] at found
      obtain ⟨⟨n, after⟩, first, rest⟩ := found
      obtain ⟨size, each⟩ := many_values count after (out.push n) values next rest
      refine ⟨by rw [size, Array.size_push]; omega, fun m member => ?_⟩
      rcases each m member with old | inside
      · rcases Array.mem_push.mp old with old | new
        · exact .inl old
        · exact .inr (new ▸ one_range first)
      · exact .inr inside

/-- A decoded payload has `count` values, each of `i16`. -/
theorem decode_values {P : Params} {bytes : ByteArray} {z : Array ℤ}
    (decoded : decode P bytes = some z) :
    z.size = P.count ∧ ∀ n ∈ z, -32768 ≤ n ∧ n ≤ 32767 := by
  unfold decode at decoded
  split at decoded
  · cases decoded
  · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at decoded
    obtain ⟨⟨values, pos⟩, found, rest⟩ := decoded
    split at rest
    · cases rest
      obtain ⟨size, each⟩ := many_values P.count 0 _ _ pos found
      refine ⟨by simpa using size, fun n member => ?_⟩
      rcases each n member with old | inside
      · simp at old
      · exact inside
    · cases rest

end Golomb

theorem foldl_sum (f : ℤ → ℕ) : ∀ (l : List ℤ) (start : ℕ),
    l.foldl (fun sum n => sum + f n) start =
      start + ∑ i ∈ Finset.range l.length, f (l.getD i 0)
  | [], start => by simp
  | x :: l, start => by
      rw [List.foldl_cons, foldl_sum f l, List.length_cons, Finset.sum_range_succ']
      simp only [List.getD_cons_succ, List.getD_cons_zero]
      ring

theorem magnitude_intCast {n : ℤ} (small : n.natAbs ≤ 32768) : magnitude (n : Zq) = n.natAbs := by
  rw [magnitude, (ZMod.valMinAbs_spec _ n).mpr ⟨rfl, ?_⟩]
  have bound : (32768 * 2 : ℤ) < fieldModulus := by decide
  constructor <;> omega

variable (S : Shape)

/-- The squared norm of the code is `‖z‖_2²` of the response of the specification, for `M·δ·d`
values of `i16`. -/
theorem centeredNormSq_eq {z : Array ℤ}
    (size : z.size = S.blockLength * S.sourceDigits * S.innerDegree)
    (range : ∀ n ∈ z, -32768 ≤ n ∧ n ≤ 32767) :
    centeredNormSq z = vecNormSq (zSource S z) := by
  have bounded : ∀ i, (z.toList.getD i 0).natAbs ≤ 32768 := fun i => by
    rw [List.getD_eq_getElem?_getD]
    cases found : z.toList[i]? with
    | none => simp
    | some n =>
        have member : n ∈ z := Array.mem_toList_iff.mp (List.mem_of_getElem? found)
        have inside := range n member
        simp only [Option.getD_some]
        omega
  let g : ℕ → ℕ := fun c => normSq (Poly.toRq S.innerDegree (zRing S z c))
  have inner : ∀ x : Fin S.blockLength, ∑ u : Fin S.sourceDigits, normSq (zSource S z (x, u)) =
      ∑ u ∈ Finset.range S.sourceDigits, g (x * S.sourceDigits + u) := fun x =>
    Fin.sum_univ_eq_sum_range (fun u => g (x * S.sourceDigits + u)) S.sourceDigits
  have column : ∀ c, g c = ∑ k ∈ Finset.range S.innerDegree,
      (z.toList.getD (c * S.innerDegree + k) 0).natAbs ^ 2 := fun c => by
    refine Finset.sum_congr rfl fun k member => ?_
    rw [Poly.coeff_toRq _ (Finset.mem_range.mp member), zRing,
      Poly.get_ofFn _ (Finset.mem_range.mp member), Fq.toZq_ofInt]
    have entry : z.getD (c * S.innerDegree + k) 0 = z.toList.getD (c * S.innerDegree + k) 0 := by
      simp [List.getD_eq_getElem?_getD]
    rw [entry, magnitude_intCast (bounded _)]
  rw [vecNormSq, Fintype.sum_prod_type, Finset.sum_congr rfl fun x _ => inner x,
    Fin.sum_univ_eq_sum_range (fun x => ∑ u ∈ Finset.range S.sourceDigits,
      g (x * S.sourceDigits + u)) S.blockLength, sum_range_mul g,
    Finset.sum_congr rfl fun c _ => column c,
    sum_range_mul (fun i => (z.toList.getD i 0).natAbs ^ 2), centeredNormSq, ← Array.foldl_toList,
    foldl_sum, Array.length_toList, size, zero_add]

/-- The response of a wire message of the terminal: `zSource` of the decoded payload, and 0 when
it does not decode (a DEFINITION of `Terminal.response` for the concrete instance). -/
noncomputable def responseOf (P : Golomb.Params) (payload : ByteArray) : S.Source :=
  match Golomb.decode P payload with
  | some z => zSource S z
  | none => 0

theorem responseOf_decoded {P : Golomb.Params} {payload : ByteArray} {z : Array ℤ}
    (decoded : Golomb.decode P payload = some z) : responseOf S P payload = zSource S z := by
  rw [responseOf, decoded]

/-- The norm check of the code on a payload that decodes to `M·δ·d` values is the check
`‖z‖_2² ≤ S_max` of the response (`Route.Short` of the Euclidean route). -/
theorem decoded_normSq {P : Golomb.Params} {payload : ByteArray} {z : Array ℤ}
    (decoded : Golomb.decode P payload = some z)
    (count : P.count = S.blockLength * S.sourceDigits * S.innerDegree) (cap : ℕ) :
    centeredNormSq z ≤ cap ↔ vecNormSq (responseOf S P payload) ≤ cap := by
  obtain ⟨size, range⟩ := Golomb.decode_values decoded
  rw [responseOf_decoded S decoded, centeredNormSq_eq S (size.trans count) range]

end NightstreamFPrime.Spec.Compression.Akita.Replay
