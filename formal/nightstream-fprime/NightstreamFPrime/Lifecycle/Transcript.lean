import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Spec.Phi81StrongSet

/-!
Owns the Stage 1 Fiat–Shamir transcript over the Poseidon2 sponge: duplex
absorb and squeeze, the Π_CCS oracle (statement absorb, one absorb per
sum-check round, labelled `α`/`γ`/`r′` squeezes), absorption of the complete
Π_CCS output, and the Π_RLC challenge sampler into the strong set
`𝓒 = {coefficients in {−2,…,2}}` by first-accepted 3-bit chunks. The absorb
order is the paper's (SuperNeo B.1): every challenge is squeezed only after
the data it must depend on has been absorbed. All definitions are computable.
-/

namespace NightstreamFPrime.Lifecycle.Transcript

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle

abbrev State := Poseidon2.State

/-- Absorb a word list in `rate`-sized chunks, permuting after each chunk. -/
def absorb (s : State) (xs : List F) : State :=
  let chunks := (List.range ((xs.length + Poseidon2.rate - 1) / Poseidon2.rate)).map
    (fun c => (xs.drop (c * Poseidon2.rate)).take Poseidon2.rate)
  chunks.foldl Poseidon2.absorbBlock s

/-- Absorb a self-delimiting block: length prefix, then the words. -/
def absorbBlock (s : State) (xs : List F) : State := absorb s (block xs)

/-- Squeeze one field word (lane 0), then permute. -/
def squeezeF (s : State) : F × State := (s.getD 0 0, Poseidon2.permute s)

/-- Squeeze one extension element from two successive words. -/
def squeezeK (s : State) : K × State :=
  let (c0, s) := squeezeF s
  let (c1, s) := squeezeF s
  (⟨c0, c1⟩, s)

def squeezeKs : Nat → State → List K × State
  | 0, s => ([], s)
  | n + 1, s =>
    let (k, s) := squeezeK s
    let (ks, s) := squeezeKs n s
    (k :: ks, s)

/-! ## Π_CCS oracle -/

def initialState : State := Poseidon2.zeroState

/-- Domain tag absorbed before every protocol transcript. -/
def domainTag : List F :=
  "Nightstream/SuperNeo/NIFS/v1".toUTF8.toList.map fun b => Poseidon2.ofNat b.toNat

def serializeMessage (m : SumCheck.Finite.Message K) : List F :=
  m.coefficients.flatMap serializeK

/-- Absorb the verifier input of Π_CCS: prior point and claimed carried
coefficients (the constraint polynomial is key data, bound through the
verifier-key digest already absorbed with the statement). -/
def absorbVerifierInput (s : State)
    (input : ProtocolPolynomial.VerifierInput K productionShape) : State :=
  let s := absorbBlock s (serializePoint input.priorPoint)
  absorbBlock s ((List.finRange productionShape.coefficientCount).flatMap fun l =>
    (List.finRange productionShape.runningCount).flatMap fun i =>
      (List.finRange productionShape.matrixCount).flatMap fun j =>
        serializeK (input.claimedCoefficient ⟨i, j, l⟩))

/-- Label words keep `α`, `γ`, and round squeezes in distinct domains. -/
def labelWord : FiatShamir.ChallengeLabel productionShape → List F
  | .alpha c => [natWord 1, natWord c.val]
  | .gamma => [natWord 2]
  | .sumcheck r => [natWord 3, natWord r.val]

def piCcsOracle :
    ProtocolVerifier.Oracle K State productionShape where
  transcript :=
    { initialState := fun statement =>
        absorbVerifierInput statement.priorState statement.input
      absorbRound := fun s round m =>
        absorbBlock s (natWord round.val :: serializeMessage m)
      squeeze := fun s label => squeezeK (absorb s (labelWord label)) }
  absorbOutput := fun s out =>
    absorbBlock s
      (((List.finRange productionShape.freshCount).flatMap fun i =>
        (List.finRange productionShape.matrixCount).flatMap fun j =>
          serializeK (out.freshMatrixImage i j)) ++
      ((List.finRange productionShape.sourceCount).flatMap fun i =>
        serializeK (out.sourceAssignment i)) ++
      ((List.finRange productionShape.coefficientCount).flatMap fun l =>
        (List.finRange productionShape.runningCount).flatMap fun i =>
          (List.finRange productionShape.matrixCount).flatMap fun j =>
            serializeK (out.carriedImage ⟨i, j, l⟩)))

/-! ## Π_RLC challenge sampler -/

open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
  (Coefficient alphabetSize)

/-- The 21 three-bit chunks of one squeezed word (63 bits used). -/
def chunks (w : F) : List Nat :=
  (List.range 21).map fun i => (w.val >>> (3 * i)) &&& 7

/-- Accepted chunks of one word, in order, as alphabet symbols `{0,…,4}`
(embedded as `{−2,…,2}` by `Phi81StrongSet.embedScalar`). -/
def acceptedChunks (w : F) : List Coefficient :=
  (chunks w).filterMap fun c => if h : c < alphabetSize then some ⟨c, h⟩ else none

/-- Collect `need` accepted symbols from successive squeezes, at most `fuel`
squeezes; shortfall pads with the zero symbol (explicit sampling event). -/
def collect : Nat → Nat → State → List Coefficient → List Coefficient × State
  | 0, _, s, acc => (acc, s)
  | fuel + 1, need, s, acc =>
    if acc.length ≥ need then (acc, s) else
      let (w, s) := squeezeF s
      collect fuel need s (acc ++ acceptedChunks w)

/-- Maximum squeezes per ring challenge; shortfall below this bound is the
named sampling failure event. -/
def samplerFuel : Nat := 64

/-- The zero symbol of the alphabet (`0 ∈ {−2,…,2}`). -/
def zeroSymbol : Coefficient := ⟨2, by decide⟩

/-- One strong-set scalar: 54 symbols, then its ring embedding. -/
def sampleScalar (s : State) : NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.Scalar × State :=
  let (cs, s) := collect samplerFuel
    NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount
    s []
  (fun i => cs.getD i.val zeroSymbol, s)

@[irreducible] def sampleRingChallenge (s : State) : RingF × State :=
  let (scalar, s) := sampleScalar s
  (Phi81StrongSet.embedScalar scalar, s)

/-- Every sampled ring challenge is in the production strong set. -/
theorem sampleRingChallenge_member (s : State) :
    Phi81StrongSet.ProductionMember (sampleRingChallenge s).1 := by
  unfold sampleRingChallenge
  exact ⟨(sampleScalar s).1, rfl⟩

/-- One sampler step: domain-separate by index, sample, append. -/
def challengeStep (n : Nat) (acc : List RingF × State) : List RingF × State :=
  (acc.1 ++ [(sampleRingChallenge (absorb acc.2 [natWord 4, natWord n])).1],
   (sampleRingChallenge (absorb acc.2 [natWord 4, natWord n])).2)

/-- `ρ_1 … ρ_{K+k}` in exact `K + k` order, each squeezed after the previous,
each domain-separated by its index. -/
def piRlcChallengesWithState (s : State) : Nat → List RingF × State
  | 0 => ([], s)
  | n + 1 => challengeStep n (piRlcChallengesWithState s n)

theorem challengeStep_fst (n : Nat) (acc : List RingF × State) :
    (challengeStep n acc).1 =
      acc.1 ++ [(sampleRingChallenge (absorb acc.2 [natWord 4, natWord n])).1] := rfl

def piRlcChallenges (s : State) (count : Nat) : List RingF :=
  (piRlcChallengesWithState s count).1

theorem piRlcChallengesWithState_length (s : State) (count : Nat) :
    (piRlcChallengesWithState s count).1.length = count := by
  induction count with
  | zero => rfl
  | succ n ih =>
    rw [piRlcChallengesWithState, challengeStep_fst, List.length_append, ih]
    rfl

theorem piRlcChallenges_length (s : State) (count : Nat) :
    (piRlcChallenges s count).length = count :=
  piRlcChallengesWithState_length s count

theorem piRlcChallengesWithState_member (s : State) (count : Nat) :
    ∀ r ∈ (piRlcChallengesWithState s count).1, Phi81StrongSet.ProductionMember r := by
  induction count with
  | zero => intro r h; exact absurd h List.not_mem_nil
  | succ n ih =>
    intro r h
    rw [piRlcChallengesWithState, challengeStep_fst] at h
    rcases List.mem_append.mp h with h | h
    · exact ih r h
    · rw [List.mem_singleton] at h
      subst h
      exact sampleRingChallenge_member _

theorem piRlcChallenges_member (s : State) (count : Nat) :
    ∀ r ∈ piRlcChallenges s count, Phi81StrongSet.ProductionMember r :=
  piRlcChallengesWithState_member s count

end NightstreamFPrime.Lifecycle.Transcript
