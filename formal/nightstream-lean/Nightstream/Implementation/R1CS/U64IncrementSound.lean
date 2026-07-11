import Nightstream.Implementation.R1CS.U64IncrementArtifact

/-!
Contract: artifact-level soundness of the no-wrap u64 increment gadget.

Claim: every canonical-residue assignment satisfying the exact 255 rows
exported from `alloc_u64_bits` followed by `enforce_u64_increment`
represents an integer increment by exactly one. In particular, no satisfying
assignment can wrap `u64::MAX` to zero. Property `CIR-U64INC`.

The generated rows remain data. `artifactRows_eq` checks that they are
exactly the rows described by the small, reviewable constructors below, and
the theorem is quantified over every assignment satisfying those rows.
-/

set_option maxRecDepth 16384

namespace Nightstream.Implementation.R1CS

open U64Increment

/-- Integer value of the 64 little-endian input-bit columns. -/
def incrementInputValue (z : Nat → Nat) : Nat :=
  (List.range 64).foldl (fun acc i => acc + 2 ^ i * z (inputBitCol i)) 0

/-- Integer value of the 64 little-endian output-bit columns. -/
def incrementOutputValue (z : Nat → Nat) : Nat :=
  (List.range 64).foldl (fun acc i => acc + 2 ^ i * z (outputBitCol i)) 0

/-- One non-final full-adder row: `a + carryIn = out + 2 * carryOut`. -/
def incrementCarryRow (a carryIn out carryOut : Nat) : Row :=
  ⟨[(a, 1), (carryIn, 1), (out, goldilocksP - 1), (carryOut, goldilocksP - 2)],
   [(0, 1)],
   []⟩

/-- Final no-wrap row: the top-bit addition has no carry-out wire. -/
def incrementFinalRow : Row :=
  ⟨[(inputBitCol 63, 1), (carryCol 62, 1),
     (outputBitCol 63, goldilocksP - 1)],
   [(0, 1)],
   []⟩

def incrementEquationRow (i : Nat) : Row :=
  incrementCarryRow
    (inputBitCol i)
    (if i = 0 then 0 else carryCol (i - 1))
    (outputBitCol i)
    (carryCol i)

def incrementInputBitRows : List Row :=
  (List.range 64).map (fun i => bitRow (inputBitCol i))

def incrementOutputBitRows : List Row :=
  (List.range 64).map (fun i => bitRow (outputBitCol i))

def incrementBodyRows : List Row :=
  (List.range 63).flatMap
    (fun i => [bitRow (carryCol i), incrementEquationRow i])

/-- Reviewable row program whose equality with the Rust export is checked. -/
def expectedIncrementRows : List Row :=
  incrementInputBitRows ++ incrementOutputBitRows ++
    incrementBodyRows ++ [incrementFinalRow]

/-- The checked Rust artifact is exactly the reviewable row program above. -/
theorem artifactRows_eq : rows = expectedIncrementRows := by
  decide

private theorem inputBitRow_mem {i : Nat} (hi : i < 64) :
    bitRow (inputBitCol i) ∈ expectedIncrementRows := by
  simp only [expectedIncrementRows, List.mem_append]
  exact Or.inl (Or.inl (Or.inl
    (List.mem_map.mpr ⟨i, List.mem_range.mpr hi, rfl⟩)))

private theorem outputBitRow_mem {i : Nat} (hi : i < 64) :
    bitRow (outputBitCol i) ∈ expectedIncrementRows := by
  simp only [expectedIncrementRows, List.mem_append]
  exact Or.inl (Or.inl (Or.inr
    (List.mem_map.mpr ⟨i, List.mem_range.mpr hi, rfl⟩)))

private theorem carryBitRow_mem {i : Nat} (hi : i < 63) :
    bitRow (carryCol i) ∈ expectedIncrementRows := by
  have hbody : bitRow (carryCol i) ∈ incrementBodyRows := by
    apply List.mem_flatMap.mpr
    exact ⟨i, List.mem_range.mpr hi, by simp⟩
  simp only [expectedIncrementRows, List.mem_append]
  exact Or.inl (Or.inr hbody)

private theorem equationRow_mem {i : Nat} (hi : i < 63) :
    incrementEquationRow i ∈ expectedIncrementRows := by
  have hbody : incrementEquationRow i ∈ incrementBodyRows := by
    apply List.mem_flatMap.mpr
    exact ⟨i, List.mem_range.mpr hi, by simp⟩
  simp only [expectedIncrementRows, List.mem_append]
  exact Or.inl (Or.inr hbody)

private theorem finalRow_mem :
    incrementFinalRow ∈ expectedIncrementRows := by
  simp [expectedIncrementRows]

/-- A satisfied non-final row is the intended integer carry equation once all
four participating wires are bits. -/
private theorem incrementCarryRow_sound
    {z : Nat → Nat} {a carryIn out carryOut : Nat}
    (ha : z a ≤ 1) (hcarryIn : z carryIn ≤ 1)
    (hout : z out ≤ 1) (hcarryOut : z carryOut ≤ 1)
    (hone : z 0 = 1)
    (h : RowHolds z (incrementCarryRow a carryIn out carryOut)) :
    z a + z carryIn = z out + 2 * z carryOut := by
  have ha' : z a = 0 ∨ z a = 1 := by omega
  have hcarryIn' : z carryIn = 0 ∨ z carryIn = 1 := by omega
  have hout' : z out = 0 ∨ z out = 1 := by omega
  have hcarryOut' : z carryOut = 0 ∨ z carryOut = 1 := by omega
  rcases ha' with ha' | ha' <;>
    rcases hcarryIn' with hcarryIn' | hcarryIn' <;>
    rcases hout' with hout' | hout' <;>
    rcases hcarryOut' with hcarryOut' | hcarryOut' <;>
    simp_all [RowHolds, incrementCarryRow, lcEval, goldilocksP]

/-- The final row enforces the top-bit equation with carry-out fixed to zero. -/
private theorem incrementFinalRow_sound
    {z : Nat → Nat}
    (hin : z (inputBitCol 63) ≤ 1)
    (hcarry : z (carryCol 62) ≤ 1)
    (hout : z (outputBitCol 63) ≤ 1)
    (hone : z 0 = 1)
    (h : RowHolds z incrementFinalRow) :
    z (inputBitCol 63) + z (carryCol 62) =
      z (outputBitCol 63) := by
  have hin' : z (inputBitCol 63) = 0 ∨ z (inputBitCol 63) = 1 := by omega
  have hcarry' : z (carryCol 62) = 0 ∨ z (carryCol 62) = 1 := by omega
  have hout' : z (outputBitCol 63) = 0 ∨ z (outputBitCol 63) = 1 := by omega
  rcases hin' with hin' | hin' <;>
    rcases hcarry' with hcarry' | hcarry' <;>
    rcases hout' with hout' | hout' <;>
    simp_all [RowHolds, incrementFinalRow, lcEval, goldilocksP]

/--
Artifact-level soundness of Rust's no-wrap u64 increment rows.
-/
theorem u64Increment_sound (hq : EuclidPrime goldilocksP)
    {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP)
    (hone : z 0 = 1)
    (hsat : Satisfies rows z) :
    incrementOutputValue z = incrementInputValue z + 1 := by
  rw [artifactRows_eq] at hsat
  have hinBound (i : Nat) (hi : i < 64) :
      z (inputBitCol i) ≤ 1 :=
    bitRow_le_one hq (hcanon _) hone
      (hsat _ (inputBitRow_mem hi))
  have houtBound (i : Nat) (hi : i < 64) :
      z (outputBitCol i) ≤ 1 :=
    bitRow_le_one hq (hcanon _) hone
      (hsat _ (outputBitRow_mem hi))
  have hcarryBound (i : Nat) (hi : i < 63) :
      z (carryCol i) ≤ 1 :=
    bitRow_le_one hq (hcanon _) hone
      (hsat _ (carryBitRow_mem hi))
  have hstep (i : Nat) (hi : i < 63) :
      z (inputBitCol i) +
          z (if i = 0 then 0 else carryCol (i - 1)) =
        z (outputBitCol i) + 2 * z (carryCol i) := by
    apply incrementCarryRow_sound
    · exact hinBound i (by omega)
    · by_cases hzero : i = 0
      · simp [hzero, hone]
      · simpa [hzero] using hcarryBound (i - 1) (by omega)
    · exact houtBound i (by omega)
    · exact hcarryBound i hi
    · exact hone
    · exact hsat _ (equationRow_mem hi)
  have hlast :
      z (inputBitCol 63) + z (carryCol 62) =
        z (outputBitCol 63) :=
    incrementFinalRow_sound
      (hinBound 63 (by omega))
      (hcarryBound 62 (by omega))
      (houtBound 63 (by omega))
      hone
      (hsat _ finalRow_mem)
  have h0 := hstep 0 (by omega)
  have h1 := hstep 1 (by omega)
  have h2 := hstep 2 (by omega)
  have h3 := hstep 3 (by omega)
  have h4 := hstep 4 (by omega)
  have h5 := hstep 5 (by omega)
  have h6 := hstep 6 (by omega)
  have h7 := hstep 7 (by omega)
  have h8 := hstep 8 (by omega)
  have h9 := hstep 9 (by omega)
  have h10 := hstep 10 (by omega)
  have h11 := hstep 11 (by omega)
  have h12 := hstep 12 (by omega)
  have h13 := hstep 13 (by omega)
  have h14 := hstep 14 (by omega)
  have h15 := hstep 15 (by omega)
  have h16 := hstep 16 (by omega)
  have h17 := hstep 17 (by omega)
  have h18 := hstep 18 (by omega)
  have h19 := hstep 19 (by omega)
  have h20 := hstep 20 (by omega)
  have h21 := hstep 21 (by omega)
  have h22 := hstep 22 (by omega)
  have h23 := hstep 23 (by omega)
  have h24 := hstep 24 (by omega)
  have h25 := hstep 25 (by omega)
  have h26 := hstep 26 (by omega)
  have h27 := hstep 27 (by omega)
  have h28 := hstep 28 (by omega)
  have h29 := hstep 29 (by omega)
  have h30 := hstep 30 (by omega)
  have h31 := hstep 31 (by omega)
  have h32 := hstep 32 (by omega)
  have h33 := hstep 33 (by omega)
  have h34 := hstep 34 (by omega)
  have h35 := hstep 35 (by omega)
  have h36 := hstep 36 (by omega)
  have h37 := hstep 37 (by omega)
  have h38 := hstep 38 (by omega)
  have h39 := hstep 39 (by omega)
  have h40 := hstep 40 (by omega)
  have h41 := hstep 41 (by omega)
  have h42 := hstep 42 (by omega)
  have h43 := hstep 43 (by omega)
  have h44 := hstep 44 (by omega)
  have h45 := hstep 45 (by omega)
  have h46 := hstep 46 (by omega)
  have h47 := hstep 47 (by omega)
  have h48 := hstep 48 (by omega)
  have h49 := hstep 49 (by omega)
  have h50 := hstep 50 (by omega)
  have h51 := hstep 51 (by omega)
  have h52 := hstep 52 (by omega)
  have h53 := hstep 53 (by omega)
  have h54 := hstep 54 (by omega)
  have h55 := hstep 55 (by omega)
  have h56 := hstep 56 (by omega)
  have h57 := hstep 57 (by omega)
  have h58 := hstep 58 (by omega)
  have h59 := hstep 59 (by omega)
  have h60 := hstep 60 (by omega)
  have h61 := hstep 61 (by omega)
  have h62 := hstep 62 (by omega)
  have hrange : List.range 64 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] := by
    decide
  simp [incrementInputValue, incrementOutputValue, inputBitCol,
    outputBitCol, carryCol, hrange] at *
  omega

end Nightstream.Implementation.R1CS
