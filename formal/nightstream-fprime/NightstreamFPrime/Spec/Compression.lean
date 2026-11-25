import NightstreamFPrime.Spec.Compression.Akita
import NightstreamFPrime.Spec.Compression.Akita.Batch
import NightstreamFPrime.Spec.Compression.Akita.BatchFlat
import NightstreamFPrime.Spec.Compression.Akita.Binding
import NightstreamFPrime.Spec.Compression.Akita.Challenge
import NightstreamFPrime.Spec.Compression.Akita.ChallengeFamily
import NightstreamFPrime.Spec.Compression.Akita.Commit
import NightstreamFPrime.Spec.Compression.Akita.Digest
import NightstreamFPrime.Spec.Compression.Akita.Euclid
import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchChain
import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchLevel
import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchRows
import NightstreamFPrime.Spec.Compression.Akita.Extract.BatchSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.Budget
import NightstreamFPrime.Spec.Compression.Akita.Extract.Chain
import NightstreamFPrime.Spec.Compression.Akita.Extract.Compose
import NightstreamFPrime.Spec.Compression.Akita.Extract.ComposeRoot
import NightstreamFPrime.Spec.Compression.Akita.Extract.Decoders
import NightstreamFPrime.Spec.Compression.Akita.Extract.Draw
import NightstreamFPrime.Spec.Compression.Akita.Extract.FlatSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.FreshGame
import NightstreamFPrime.Spec.Compression.Akita.Extract.Honest
import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.PairComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.PackingFold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Recursion
import NightstreamFPrime.Spec.Compression.Akita.Extract.Root
import NightstreamFPrime.Spec.Compression.Akita.Extract.Segment
import NightstreamFPrime.Spec.Compression.Akita.Extract.SplitGame
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged
import NightstreamFPrime.Spec.Compression.Akita.Extract.Suffix
import NightstreamFPrime.Spec.Compression.Akita.Extract.TerminalSuffix
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tier
import NightstreamFPrime.Spec.Compression.Akita.Extract.TierBound
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tiny
import NightstreamFPrime.Spec.Compression.Akita.Extract.TinyRoot
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tower
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceChain
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLevel
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.Transcript
import NightstreamFPrime.Spec.Compression.Akita.Extract.Uniform
import NightstreamFPrime.Spec.Compression.Akita.Extract.Wire
import NightstreamFPrime.Spec.Compression.Akita.FlatFold
import NightstreamFPrime.Spec.Compression.Akita.FlatWitness
import NightstreamFPrime.Spec.Compression.Akita.Fold
import NightstreamFPrime.Spec.Compression.Akita.FoldCheck
import NightstreamFPrime.Spec.Compression.Akita.FoldComplete
import NightstreamFPrime.Spec.Compression.Akita.FoldSumCheck
import NightstreamFPrime.Spec.Compression.Akita.KeptTail
import NightstreamFPrime.Spec.Compression.Akita.Key20
import NightstreamFPrime.Spec.Compression.Akita.Key20Challenge
import NightstreamFPrime.Spec.Compression.Akita.Key20Tower
import NightstreamFPrime.Spec.Compression.Akita.Key20Opening
import NightstreamFPrime.Spec.Compression.Akita.Layout
import NightstreamFPrime.Spec.Compression.Akita.LiftReduction
import NightstreamFPrime.Spec.Compression.Akita.LiftWitness
import NightstreamFPrime.Spec.Compression.Akita.Opening
import NightstreamFPrime.Spec.Compression.Akita.Prover
import NightstreamFPrime.Spec.Compression.Akita.Range
import NightstreamFPrime.Spec.Compression.Akita.RingCheck
import NightstreamFPrime.Spec.Compression.Akita.RowBatch
import NightstreamFPrime.Spec.Compression.Akita.Setup
import NightstreamFPrime.Spec.Compression.Akita.Shapes
import NightstreamFPrime.Spec.Compression.Akita.ShortUnit
import NightstreamFPrime.Spec.Compression.Akita.Tensor
import NightstreamFPrime.Spec.Compression.Akita.Terminal
import NightstreamFPrime.Spec.Compression.Akita.Trace
import NightstreamFPrime.Spec.Compression.Akita.TraceFold
import NightstreamFPrime.Spec.Compression.Akita.TracePack
import NightstreamFPrime.Spec.Compression.Game.Complete
import NightstreamFPrime.Spec.Compression.Game.Law
import NightstreamFPrime.Spec.Compression.Game.Path
import NightstreamFPrime.Spec.Compression.Game.SumCheck
import NightstreamFPrime.Spec.Compression.Lib.Combination
import NightstreamFPrime.Spec.Compression.Lib.LogUp
import NightstreamFPrime.Spec.Compression.Algebra.Batch
import NightstreamFPrime.Spec.Compression.Algebra.Key
import NightstreamFPrime.Spec.Compression.Verifier.Complete
import NightstreamFPrime.Spec.Compression.Production
import NightstreamFPrime.Spec.Compression.Verifier.Pass
import NightstreamFPrime.Spec.Compression.Verifier.Split
import NightstreamFPrime.Spec.Compression.Claim.Slot
import NightstreamFPrime.Spec.Compression.Fresh
import NightstreamFPrime.Spec.Compression.Rust.Soundness
import NightstreamFPrime.Spec.Compression.Rust.Transcript
import NightstreamFPrime.Spec.Compression.Akita.Extract.BackComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.CheckComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.ComposeComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.OffloadComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.Page
import NightstreamFPrime.Spec.Compression.Akita.Extract.Play
import NightstreamFPrime.Spec.Compression.Akita.Extract.Reject
import NightstreamFPrime.Spec.Compression.Akita.Extract.RootComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.TowerComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.ZeroFit
import NightstreamFPrime.Spec.Compression.Akita.Extract.GroupComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.ShortComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.OffLink
import NightstreamFPrime.Spec.Compression.Akita.Extract.OffComplete
import NightstreamFPrime.Spec.Compression.Akita.Key20Accept
import NightstreamFPrime.Spec.Compression.Akita.Key20Complete
import NightstreamFPrime.Spec.Compression.Akita.Key20Root
import NightstreamFPrime.Spec.Compression.Akita.Key30Complete
import NightstreamFPrime.Spec.Compression.Akita.Key30Root
import NightstreamFPrime.Spec.Compression.Akita.Key30Accept
import NightstreamFPrime.Spec.Compression.Akita.ResponseTail
import NightstreamFPrime.Spec.Compression.Akita.Extract.Nonce
import NightstreamFPrime.Spec.Compression.Akita.Extract.Present
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentDecode
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentChecks
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentPage
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentGiven
import NightstreamFPrime.Spec.Compression.Akita.FitChance
import NightstreamFPrime.Spec.Compression.Akita.TraceProver
import NightstreamFPrime.Spec.Compression.Akita.Replay.Plan
import NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix
import NightstreamFPrime.Spec.Compression.Akita.Replay.Readback
import NightstreamFPrime.Spec.Compression.Akita.Replay.Statement
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerSigns
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerStage
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerPool
import NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot
import NightstreamFPrime.Spec.Compression.Akita.Replay.FusedJoin
import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelPack
import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelIn
import NightstreamFPrime.Spec.Compression.Akita.Replay.RunLevel
import NightstreamFPrime.Spec.Compression.Akita.Replay.RunWhole
import NightstreamFPrime.Spec.Compression.Akita.Replay.RunProof
import NightstreamFPrime.Spec.Compression.Akita.Replay.RunStopForms
import NightstreamFPrime.Spec.Compression.Akita.Replay.RunRange
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalDrawn
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalGame
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalKey20
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalKey30
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeScript
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeErase
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeLayout20
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeKey20
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeLayout30
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeKey30
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeRange
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeStages
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeAccept
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkTrace
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkKey20
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkPack
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkRoot
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinNote
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBits
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinShape
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinPresent
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinAccept
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinLevels
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinDrawn
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseRuns
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseLevels
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverse
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinCount
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinSlots
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinRuns
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBlock
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinExists
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStage
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinGiven
import NightstreamFPrime.Spec.Compression.Akita.Replay.Causal
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBound
import NightstreamFPrime.Spec.Compression.Akita.Replay.Coupling
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerFair
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerFairField
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerFairFold
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageCoupling
import NightstreamFPrime.Spec.Compression.Akita.Replay.Grows
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Table
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Sum
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Match
import NightstreamFPrime.Spec.Compression.Akita.Replay.SumCheck
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalJoin
import NightstreamFPrime.Spec.Compression.Akita.Replay.Transcript
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightSpec
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightTraceSpec
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingSpec
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingRows
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingGroup
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingQuotient
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingLevel
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingJoin
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatch
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchReplay
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchLevel
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchFold
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchOffload

/-! The soundness of the compression argument over Akita (Rust `ns-compress`, SPEC.md section 6):
a public-coin game with bad sets and its state-restoration bound, and the honest prover of a
game (`Game`), and the lemmas of its
steps (`Lib`), the exact algebraic identities of the argument (`Algebra`), and the verifier as a game with its
soundness bound and its completeness for the honest prover (`Verifier`), and the production
instance of the argument (`Production`), with the ring equations of a claim as the opening
and the evaluations of `Spec.CE.Holds` (`Claim`), and Π_CCS on one source as a game (`Fresh`),
and Akita as a polynomial commitment scheme (`Akita`), with the
Akita commitment (`Akita/Commitment`, `Akita/Commit`), the shapes of the key group of the schedule
files (`Akita/Shapes`), the numbers of the levels of `key20.aks` (`Akita/Key20`) and the binding
of the commitment of the compression key from Module-SIS
(`Akita/Binding`), and the embedding of the challenge subring (`Akita/Embed`), the norms of the fold
challenges (`Akita/Challenge`) and the opening path of a packing fold (`Akita/Opening`), and the
relation, the weak opening, the weak binding and the extraction of one packing fold
(`Akita/Fold`), and its honest prover and the layout of its scalar opening (`Akita/Prover`), and
the Euclidean norms and the Euclidean route of the weak binding (`Akita/Euclid`), the norm checks of
the next witness (`Akita/Range`), the checks of the ring relations (`Akita/RingCheck`), the packing
map and the trace identity for `k = 2` (`Akita/Trace`), one fold with the evaluation-trace method
(`Akita/TraceFold`), the layout of its scalar row (`Akita/TracePack`), the tensor reduction for
`k = 2` (`Akita/Tensor`) and the evaluation-trace terminal (`Akita/Terminal`), and the row
batching and the quotient-free reduction (`Akita/RowBatch`), the quotient-lift reduction
(`Akita/LiftReduction`), the sum-checks of one fold (`Akita/FoldSumCheck`), the checks of one fold
as a game with its error (`Akita/FoldCheck`) and their completeness (`Akita/FoldComplete`), the
link from these checks to the branch relation
(`Akita/Layout`), the flat witness of one packing fold (`Akita/FlatWitness`) and the relation of
this fold on the flat witness (`Akita/FlatFold`), and the rewinding extraction of the packing folds
(`Akita/Extract`): a finite transcript protocol (`Transcript`), a game as the stages of a
transcript (`Staged`), the bounds of the draws of the checks `foldCheck` (`PackingFold`), and the
recursive opening suffix: the conditions of a suffix of the stages (`Suffix`), the verifier
without the witness
(`Uniform`), one level over the later stages (`Tier`, `TierBound`), the checks for a committed
flat witness (`FlatSource`), the composition of the levels (`Recursion`), the levels with the
checks `foldCheck` (`Chain`), the link to a receiver with evaluation trace (`TraceLink`), a
packing level over a trace level over the terminal (`TraceChain`), the Fiat–Shamir budget
(`Budget`), and the honest prover of two packing levels: its messages on a transcript
(`Honest`), the acceptance of its proofs (`PairComplete`), a finite type of wire messages
(`Wire`) and a small instance with numerals (`Tiny`), and a fold with opening groups: its flat
witness (`Akita/BatchFlat`), its relation on the flat witness (`BatchRows`), its checks for a
committed flat witness (`BatchSource`), its levels and the batching root over them (`BatchChain`)
and a small instance of this root with three groups (`TinyRoot`), and the quotient digits of the
ring rows in the flat witness (`Akita/LiftWitness`) with the levels of the quotient-lift route
(`LiftSource`), and the compression argument over the Akita opening: the verifier game in two
parts that end in the opening claims (`Verifier/Split`), the stages of these parts (`SplitGame`),
the two checkpoints of the argument before the opening (`Compose`), the fair decoders of its
challenges (`Decoders`), the opening with a batching root, a fold with three opening groups and a
last packing fold (`ComposeRoot`), and the opening with a batching root, a fold with opening
groups, a packing level, four trace levels and the terminal (`Tower`), with its instance for the
schedule `key20.aks` (`Akita/Key20Tower`), and the honest prover of this opening: the bounded
source of a trace level (`Akita/TraceProver`), its messages on a transcript (`Play`), the
completeness of the checks (`CheckComplete`), the trace levels (`TraceComplete`,
`TowerComplete`), the packing level (`LiftComplete`), the root fold and the verifier `towerLinks`
(`RootComplete`), a type of wire messages (`Page`) and the zero tables (`ZeroFit`), with its
instance for the schedule `key20.aks` (`Akita/Key20Complete`, `Akita/Key20Root`,
`Akita/Key20Accept`) and for the schedule `key30.aks` (`Akita/Key30Complete`, `Akita/Key30Root`,
`Akita/Key30Accept`), the sparse challenge families with their units and their decoders from
one slot (`Akita/ShortUnit`, `Akita/ChallengeFamily`, `Draw`, `Akita/Key20Challenge`), the
union bound of their filters over the frequencies (`Akita/KeptTail`), and the shared setup with its
reduction to ordinary Module-SIS (`Akita/Setup`) and the bad events of the hashed transcript
entries (`Akita/Digest`). -/
