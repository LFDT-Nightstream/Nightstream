import TwistShout.SumCheck

open MvPolynomial
open TwistShout

namespace tests

noncomputable section

example :
    hypercubeSum (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat) = 1 := by
  let x0 : Fin 0 → Rat := fun i => nomatch i
  calc
    hypercubeSum (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat) =
      hypercubeSum (K := Rat) (bindHead (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat) 0) +
        hypercubeSum (K := Rat) (bindHead (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat) 1) := by
          exact hypercubeSum_split_head (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat)
    _ =
      MvPolynomial.eval x0 (bindHead (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat) 0) +
        MvPolynomial.eval x0 (bindHead (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat) 1) := by
          rw [hypercubeSum_zero_eval (K := Rat)
              (bindHead (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat) 0) x0]
          rw [hypercubeSum_zero_eval (K := Rat)
              (bindHead (K := Rat) (X 0 : MvPolynomial (Fin 1) Rat) 1) x0]
    _ =
      MvPolynomial.eval (Fin.cases 0 x0) (X 0 : MvPolynomial (Fin 1) Rat) +
        MvPolynomial.eval (Fin.cases 1 x0) (X 0 : MvPolynomial (Fin 1) Rat) := by
          rw [eval_bindHead (K := Rat) (g := (X 0 : MvPolynomial (Fin 1) Rat)) (c := 0) (x := x0)]
          rw [eval_bindHead (K := Rat) (g := (X 0 : MvPolynomial (Fin 1) Rat)) (c := 1) (x := x0)]
    _ = 0 + 1 := by
          simp [x0]
    _ = 1 := by
          norm_num

def zeroPoly : MvPolynomial (Fin 2) Rat := 0

def zeroInst : SumCheckInstance (K := Rat) 2 :=
  { claimedSum := 0
    degreeBound := fun _ => 0 }

def zeroStmt : SumCheckStatement zeroInst :=
  { polynomial := zeroPoly
    hypercubeSumEqClaimed := by
      simp [zeroInst, zeroPoly, TwistShout.hypercubeSum]
    honestRoundDegreeLe := by
      intro challenges i
      fin_cases i
      · have hround0 :
            TwistShout.honestRoundPolys (K := Rat) zeroPoly challenges 0 = 0 := by
            simp [zeroPoly, TwistShout.honestRoundPolys, TwistShout.firstRoundPoly]
        simp [zeroInst, hround0]
      · have hround1 :
            TwistShout.honestRoundPolys (K := Rat) zeroPoly challenges 1 = 0 := by
            have hbind :
                TwistShout.bindHead (K := Rat) zeroPoly (challenges 0) = 0 := by
              simp [zeroPoly, TwistShout.bindHead]
            change TwistShout.honestRoundPolys (K := Rat) zeroPoly challenges (Fin.succ 0) = 0
            rw [TwistShout.honestRoundPolys_succ (K := Rat)
              (g := zeroPoly) (challenges := challenges) (i := 0)]
            rw [hbind]
            simp [TwistShout.honestRoundPolys, TwistShout.firstRoundPoly]
        simp [zeroInst, hround1] }

example (challenges : Fin 2 → Rat) :
    sumcheckVerifierAccepted zeroStmt
      (honestTranscript (K := Rat) zeroStmt.polynomial challenges) :=
  honestTranscript_verifierAccepted (K := Rat) zeroStmt challenges

#guard zeroInst.degreeBound 0 = 0

end

end tests
