import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Provenance.RewriteSteps

/-!
Bounded batch index for the exact production combined-NC rewrite stream.

Owns: a streaming partition of the 1,493 generated rewrite records into
adjacent equal-key batches, proof-free batch descriptors, bounded shard
certificates, exact batch/range distributions, and generic kernel facts for
coverage, offsets, and adjacency.

Does not own: rewrite/source semantic agreement, source-row satisfaction,
assignment authority, transcript order, commitment binding, costs outside
the covered source ranges, or permission to remove rows.

The executable boundary is deliberately the 24 generated rewrite shards.
No certificate below evaluates `Provenance.rewriteSteps`, or a `take`/`drop`
of that global list.  Each shard contains 64 proof-free records except the
21-record final shard.  Its explicit incoming carry contains at most 39
proof-free compact steps, so an executable scanner sees at most 103 records.

Assurance tier: artifact-checked after focused validation.
-/

/-!
Emits constraints: none; this module indexes existing compiler rewrite batches.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.rewrite_batch_index` | Match each generated rewrite batch to its source-definition interval. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBatchIndex

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated

/-! ## Proof-free scanner vocabulary -/

structure StepKey where
  rewriteId : Nat
  sourceRange : RawRowRange
  kind : RawRewriteKind
deriving DecidableEq, Repr, Inhabited

/-- Only the fields needed to form exact adjacent rewrite batches. -/
structure CompactStep where
  rewriteId : Nat
  sourceRange : RawRowRange
  kind : RawRewriteKind
  output : RawRewriteOutput
  previous : Option Nat
deriving DecidableEq, Repr, Inhabited

def CompactStep.key (step : CompactStep) : StepKey :=
  { rewriteId := step.rewriteId
    sourceRange := step.sourceRange
    kind := step.kind }

/-- Total projection used by the scanner.  The separate `rawShapeValid`
certificate proves that `headD` is the unique source range and therefore
that this projection neither guesses nor drops a range. -/
def CompactStep.ofRaw (raw : RawRewriteStep) : CompactStep :=
  { rewriteId := raw.rewriteId
    sourceRange := raw.sourceRows.headD default
    kind := raw.kind
    output := raw.output
    previous := raw.previous }

def RawShapeValid (raw : RawRewriteStep) : Prop :=
  raw.sourceRows = [(CompactStep.ofRaw raw).sourceRange]

instance (raw : RawRewriteStep) : Decidable (RawShapeValid raw) := by
  unfold RawShapeValid
  infer_instance

def rawShapeCheck (values : List RawRewriteStep) : Bool :=
  values.all fun raw => decide (RawShapeValid raw)

theorem rawShapeValid_of_check_true {values : List RawRewriteStep}
    (checked : rawShapeCheck values = true) {raw : RawRewriteStep}
    (member : raw ∈ values) : RawShapeValid raw := by
  apply of_decide_eq_true
  exact (List.all_eq_true.mp checked) raw member

/-- One auditable rewrite owner.  `stepOffset` is an offset in the exact
generated rewrite stream, not an emitted-row or source-row number. -/
structure BatchDescriptor where
  rewriteId : Nat
  sourceRange : RawRowRange
  stepOffset : Nat
  stepCount : Nat
  kind : RawRewriteKind
deriving DecidableEq, Repr, Inhabited

def BatchDescriptor.key (descriptor : BatchDescriptor) : StepKey :=
  { rewriteId := descriptor.rewriteId
    sourceRange := descriptor.sourceRange
    kind := descriptor.kind }

structure Batch where
  descriptor : BatchDescriptor
  steps : List CompactStep
deriving DecidableEq, Repr, Inhabited

def Batch.singleton (offset : Nat) (step : CompactStep) : Batch :=
  { descriptor :=
      { rewriteId := step.rewriteId
        sourceRange := step.sourceRange
        stepOffset := offset
        stepCount := 1
        kind := step.kind }
    steps := [step] }

def Batch.extend (batch : Batch) (step : CompactStep) : Batch :=
  { descriptor :=
      { batch.descriptor with
        stepCount := batch.descriptor.stepCount + 1 }
    steps := batch.steps ++ [step] }

def Batch.Homogeneous (batch : Batch) : Prop :=
  batch.steps ≠ [] ∧
    batch.descriptor.stepCount = batch.steps.length ∧
    (∀ step ∈ batch.steps,
      step.key = batch.descriptor.key)

instance (batch : Batch) : Decidable batch.Homogeneous := by
  unfold Batch.Homogeneous
  infer_instance

def homogeneousCheck (values : List Batch) : Bool :=
  values.all fun batch => decide batch.Homogeneous

def boundedCountCheck (values : List Batch) : Bool :=
  values.all fun batch => decide (batch.descriptor.stepCount ≤ 39)

theorem homogeneous_of_check_true {values : List Batch}
    (checked : homogeneousCheck values = true) {batch : Batch}
    (member : batch ∈ values) : batch.Homogeneous := by
  apply of_decide_eq_true
  exact (List.all_eq_true.mp checked) batch member

structure ScanState where
  closed : List Batch
  carry : Option Batch
  nextOffset : Nat
deriving DecidableEq, Repr, Inhabited

def push (state : ScanState) (step : CompactStep) : ScanState :=
  match state.carry with
  | none =>
      { state with
        carry := some (Batch.singleton state.nextOffset step)
        nextOffset := state.nextOffset + 1 }
  | some current =>
      if current.descriptor.key = step.key then
        { state with
          carry := some (Batch.extend current step)
          nextOffset := state.nextOffset + 1 }
      else
        { closed := state.closed ++ [current]
          carry := some (Batch.singleton state.nextOffset step)
          nextOffset := state.nextOffset + 1 }

def scan : ScanState → List CompactStep → ScanState
  | state, [] => state
  | state, step :: rest => scan (push state step) rest

def scanChunk (carry : Option Batch) (nextOffset : Nat)
    (steps : List CompactStep) : ScanState :=
  scan { closed := [], carry := carry, nextOffset := nextOffset } steps

def ScanState.allBatches (state : ScanState) : List Batch :=
  state.closed ++ state.carry.toList

def ScanState.ownedSteps (state : ScanState) : List CompactStep :=
  state.allBatches.flatMap Batch.steps

theorem ownedSteps_push (state : ScanState) (step : CompactStep) :
    (push state step).ownedSteps = state.ownedSteps ++ [step] := by
  rcases state with ⟨closed, carry, nextOffset⟩
  cases carry with
  | none =>
      simp [push, ScanState.ownedSteps, ScanState.allBatches,
        Batch.singleton]
  | some current =>
      by_cases same : current.descriptor.key = step.key
      · simp [push, same, ScanState.ownedSteps, ScanState.allBatches,
          Batch.extend, List.append_assoc]
      · simp [push, same, ScanState.ownedSteps, ScanState.allBatches,
          Batch.singleton, List.append_assoc]

/-- Generic kernel coverage: scanning appends every input compact step once,
in order, to the steps already owned by the state. -/
theorem ownedSteps_scan (state : ScanState) (steps : List CompactStep) :
    (scan state steps).ownedSteps = state.ownedSteps ++ steps := by
  induction steps generalizing state with
  | nil => simp [scan]
  | cons head tail inductionHypothesis =>
      rw [scan, inductionHypothesis, ownedSteps_push]
      simp [List.append_assoc]

theorem scanChunk_ownedSteps (carry : Option Batch) (nextOffset : Nat)
    (steps : List CompactStep) :
    (scanChunk carry nextOffset steps).ownedSteps =
      carry.toList.flatMap Batch.steps ++ steps := by
  simpa [scanChunk, ScanState.ownedSteps, ScanState.allBatches] using
    ownedSteps_scan
      { closed := [], carry := carry, nextOffset := nextOffset } steps

/-! ## Kernel relations checked by compact observations -/

inductive OffsetsFromTo : Nat → Nat → List Batch → Prop where
  | nil (start : Nat) : OffsetsFromTo start start []
  | cons {start stop : Nat} {batch : Batch} {rest : List Batch}
      (offset : batch.descriptor.stepOffset = start)
      (positive : 0 < batch.descriptor.stepCount)
      (tail : OffsetsFromTo
        (start + batch.descriptor.stepCount) stop rest) :
      OffsetsFromTo start stop (batch :: rest)

def offsetsCheck : Nat → List Batch → Option Nat
  | start, [] => some start
  | start, batch :: rest =>
      if batch.descriptor.stepOffset = start ∧
          0 < batch.descriptor.stepCount then
        offsetsCheck (start + batch.descriptor.stepCount) rest
      else
        none

theorem offsetsFromTo_of_check : ∀ {start stop batches},
    offsetsCheck start batches = some stop →
      OffsetsFromTo start stop batches := by
  intro start stop batches checked
  induction batches generalizing start with
  | nil =>
      simp only [offsetsCheck, Option.some.injEq] at checked
      subst stop
      exact .nil start
  | cons head tail inductionHypothesis =>
      simp only [offsetsCheck] at checked
      split at checked
      next valid =>
        exact .cons valid.1 valid.2 (inductionHypothesis checked)
      next invalid => contradiction

theorem OffsetsFromTo.append {start middle stop : Nat}
    {left right : List Batch}
    (leftOffsets : OffsetsFromTo start middle left)
    (rightOffsets : OffsetsFromTo middle stop right) :
    OffsetsFromTo start stop (left ++ right) := by
  induction leftOffsets with
  | nil => simpa using rightOffsets
  | cons offset positive rest inductionHypothesis =>
      exact .cons offset positive (inductionHypothesis rightOffsets)

def Batch.OwnsIndex (batch : Batch) (index : Nat) : Prop :=
  batch.descriptor.stepOffset ≤ index ∧
    index < batch.descriptor.stepOffset + batch.descriptor.stepCount

theorem OffsetsFromTo.member_offset_ge {start stop : Nat}
    {batches : List Batch} (offsets : OffsetsFromTo start stop batches)
    {batch : Batch} (member : batch ∈ batches) :
    start ≤ batch.descriptor.stepOffset := by
  induction offsets with
  | nil => contradiction
  | cons headOffset positive rest inductionHypothesis =>
      simp only [List.mem_cons] at member
      rcases member with equality | tailMember
      · subst batch
        omega
      · have later := inductionHypothesis tailMember
        omega

/-- Exact offset partitioning gives every stream index one and only one
batch owner.  This is uniqueness by stream position, not by structural
equality of raw records. -/
theorem OffsetsFromTo.existsUnique_owner {start stop index : Nat}
    {batches : List Batch} (offsets : OffsetsFromTo start stop batches)
    (lower : start ≤ index) (upper : index < stop) :
    ∃ batch,
      (batch ∈ batches ∧ batch.OwnsIndex index) ∧
      ∀ other,
        other ∈ batches ∧ other.OwnsIndex index → other = batch := by
  induction offsets generalizing index with
  | nil => omega
  | @cons start stop head tail headOffset positive rest inductionHypothesis =>
      by_cases inHead : index < start + head.descriptor.stepCount
      · refine ⟨head, ?_, ?_⟩
        · constructor
          · simp
          · simp [Batch.OwnsIndex, headOffset, lower, inHead]
        · intro other otherOwns
          have member := otherOwns.1
          simp only [List.mem_cons] at member
          rcases member with equality | tailMember
          · exact equality
          · have later := rest.member_offset_ge tailMember
            have earlier := otherOwns.2.1
            omega
      · have nextLower :
            start + head.descriptor.stepCount ≤ index := by omega
        rcases inductionHypothesis nextLower upper with
          ⟨owner, ownerFacts, unique⟩
        refine ⟨owner, ?_, ?_⟩
        · exact ⟨by simp [ownerFacts.1], ownerFacts.2⟩
        · intro other otherOwns
          have member := otherOwns.1
          simp only [List.mem_cons] at member
          rcases member with equality | tailMember
          · subst other
            exfalso
            have := otherOwns.2.2
            rw [headOffset] at this
            exact inHead this
          · exact unique other ⟨tailMember, otherOwns.2⟩

def KeysLinked (left right : StepKey) : Prop :=
  left ≠ right ∧ left.sourceRange.stop ≤ right.sourceRange.start

instance (left right : StepKey) : Decidable (KeysLinked left right) := by
  unfold KeysLinked
  infer_instance

def LinkedFrom : Option StepKey → List Batch → Prop
  | _, [] => True
  | previous, batch :: rest =>
      (match previous with
        | none => True
        | some key => KeysLinked key batch.descriptor.key) ∧
      LinkedFrom (some batch.descriptor.key) rest

def linkScan : Option StepKey → List Batch → Option (Option StepKey)
  | previous, [] => some previous
  | none, batch :: rest => linkScan (some batch.descriptor.key) rest
  | some previous, batch :: rest =>
      if KeysLinked previous batch.descriptor.key then
        linkScan (some batch.descriptor.key) rest
      else none

theorem linkedFrom_of_scan : ∀ {previous final batches},
    linkScan previous batches = some final →
      LinkedFrom previous batches := by
  intro previous final batches checked
  induction batches generalizing previous with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      cases previous with
      | none =>
          exact ⟨trivial, inductionHypothesis (by
            simpa [linkScan] using checked)⟩
      | some previous =>
          simp only [linkScan] at checked
          split at checked
          next linked =>
            exact ⟨linked, inductionHypothesis checked⟩
          next unlinked => contradiction

/-! ## Exact compact distributions -/

structure Distribution where
  batchCount : Nat
  count2 : Nat
  count9 : Nat
  count39 : Nat
  range5 : Nat
  range73 : Nat
  range78 : Nat
  range323 : Nat
  coveredSourceRows : Nat
deriving DecidableEq, Repr, Inhabited

def indicator (condition : Bool) : Nat := if condition then 1 else 0

def rangeLength (batch : Batch) : Nat :=
  batch.descriptor.sourceRange.stop - batch.descriptor.sourceRange.start

def distribution (batches : List Batch) : Distribution :=
  { batchCount := batches.length
    count2 := (batches.map fun batch =>
      indicator (batch.descriptor.stepCount == 2)).sum
    count9 := (batches.map fun batch =>
      indicator (batch.descriptor.stepCount == 9)).sum
    count39 := (batches.map fun batch =>
      indicator (batch.descriptor.stepCount == 39)).sum
    range5 := (batches.map fun batch =>
      indicator (rangeLength batch == 5)).sum
    range73 := (batches.map fun batch =>
      indicator (rangeLength batch == 73)).sum
    range78 := (batches.map fun batch =>
      indicator (rangeLength batch == 78)).sum
    range323 := (batches.map fun batch =>
      indicator (rangeLength batch == 323)).sum
    coveredSourceRows := (batches.map rangeLength).sum }

def Distribution.add (left right : Distribution) : Distribution :=
  { batchCount := left.batchCount + right.batchCount
    count2 := left.count2 + right.count2
    count9 := left.count9 + right.count9
    count39 := left.count39 + right.count39
    range5 := left.range5 + right.range5
    range73 := left.range73 + right.range73
    range78 := left.range78 + right.range78
    range323 := left.range323 + right.range323
    coveredSourceRows := left.coveredSourceRows + right.coveredSourceRows }

theorem distribution_append (left right : List Batch) :
    distribution (left ++ right) =
      Distribution.add (distribution left) (distribution right) := by
  simp [distribution, Distribution.add, List.map_append]

structure ChunkObservation where
  rawShapeValid : Bool
  carry : Option Batch
  nextOffset : Nat
  offsetsEnd : Option Nat
  linkEnd : Option (Option StepKey)
  homogeneous : Bool
  boundedCount : Bool
  distribution : Distribution
deriving DecidableEq, Repr, Inhabited

def observeChunk (raw : List RawRewriteStep) (incoming : Option Batch)
    (nextOffset closedStart : Nat) (previousKey : Option StepKey) :
    ChunkObservation :=
  let result := scanChunk incoming nextOffset (raw.map CompactStep.ofRaw)
  { rawShapeValid := rawShapeCheck raw
    carry := result.carry
    nextOffset := result.nextOffset
    offsetsEnd := offsetsCheck closedStart result.closed
    linkEnd := linkScan previousKey result.closed
    homogeneous := homogeneousCheck result.closed
    boundedCount := boundedCountCheck result.closed
    distribution := distribution result.closed }

def productKey (rewriteId start stop : Nat) : StepKey :=
  { rewriteId := rewriteId
    sourceRange := { start := start, stop := stop }
    kind := .productSum }

def carriedProduct (rewriteId start stop offset : Nat)
    (steps : List CompactStep) : Batch :=
  let key := productKey rewriteId start stop
  { descriptor :=
      { rewriteId := rewriteId
        sourceRange := key.sourceRange
        stepOffset := offset
        stepCount := steps.length
        kind := .productSum }
    steps := steps }

/-! ## Bounded generated-shard certificates -/

def exactDistribution (batchCount count2 count9 count39 range5 range73
    range78 range323 coveredSourceRows : Nat) : Distribution :=
  { batchCount := batchCount
    count2 := count2
    count9 := count9
    count39 := count39
    range5 := range5
    range73 := range73
    range78 := range78
    range323 := range323
    coveredSourceRows := coveredSourceRows }

def expectedObservation (carry : Batch) (nextOffset closedEnd : Nat)
    (lastKey : StepKey) (summary : Distribution) : ChunkObservation :=
  { rawShapeValid := true
    carry := some carry
    nextOffset := nextOffset
    offsetsEnd := some closedEnd
    linkEnd := some (some lastKey)
    homogeneous := true
    boundedCount := true
    distribution := summary }

def carry0 := carriedProduct 131278 3969685 3969690 62
  ((Provenance.RewriteSteps.Chunk0.values.drop 62).map CompactStep.ofRaw)
def carry1 := carriedProduct 131310 3977126 3977131 126
  ((Provenance.RewriteSteps.Chunk1.values.drop 62).map CompactStep.ofRaw)
def carry2 := carriedProduct 131342 3991782 3991787 190
  ((Provenance.RewriteSteps.Chunk2.values.drop 62).map CompactStep.ofRaw)
def carry3 := carriedProduct 131374 4006438 4006443 254
  ((Provenance.RewriteSteps.Chunk3.values.drop 62).map CompactStep.ofRaw)
def carry4 := carriedProduct 132121 4289033 4289038 318
  ((Provenance.RewriteSteps.Chunk4.values.drop 62).map CompactStep.ofRaw)
def carry5 := carriedProduct 132153 4289193 4289198 382
  ((Provenance.RewriteSteps.Chunk5.values.drop 62).map CompactStep.ofRaw)
def carry6 := carriedProduct 132185 4289353 4289358 446
  ((Provenance.RewriteSteps.Chunk6.values.drop 62).map CompactStep.ofRaw)
def carry7 := carriedProduct 132217 4289513 4289518 510
  ((Provenance.RewriteSteps.Chunk7.values.drop 62).map CompactStep.ofRaw)
def carry8 := carriedProduct 132242 4289638 4289961 560
  ((Provenance.RewriteSteps.Chunk8.values.drop 48).map CompactStep.ofRaw)
def carry9 := carriedProduct 132245 4289973 4290296 603
  ((Provenance.RewriteSteps.Chunk9.values.drop 27).map CompactStep.ofRaw)
def carry10 := carriedProduct 132251 4290643 4290966 689
  ((Provenance.RewriteSteps.Chunk10.values.drop 49).map CompactStep.ofRaw)
def carry11 := carriedProduct 132254 4290978 4291301 732
  ((Provenance.RewriteSteps.Chunk11.values.drop 28).map CompactStep.ofRaw)
def carry12 := carriedProduct 132260 4291648 4291971 818
  ((Provenance.RewriteSteps.Chunk12.values.drop 50).map CompactStep.ofRaw)
def carry13 := carriedProduct 132263 4291983 4292306 861
  ((Provenance.RewriteSteps.Chunk13.values.drop 29).map CompactStep.ofRaw)
def carry14 := carriedProduct 132269 4292653 4292976 947
  ((Provenance.RewriteSteps.Chunk14.values.drop 51).map CompactStep.ofRaw)
def carry15 := carriedProduct 132272 4292988 4293311 990
  ((Provenance.RewriteSteps.Chunk15.values.drop 30).map CompactStep.ofRaw)
def carry16 := carriedProduct 132278 4293658 4293981 1076
  ((Provenance.RewriteSteps.Chunk16.values.drop 52).map CompactStep.ofRaw)
def carry17 := carriedProduct 132281 4293993 4294316 1119
  ((Provenance.RewriteSteps.Chunk17.values.drop 31).map CompactStep.ofRaw)
def carry18 := carriedProduct 132292 4294690 4294695 1215
  ((Provenance.RewriteSteps.Chunk18.values.drop 63).map CompactStep.ofRaw)
def carry19 := carriedProduct 132320 4294923 4294928 1278
  ((Provenance.RewriteSteps.Chunk19.values.drop 62).map CompactStep.ofRaw)
def carry20 := carriedProduct 132352 4295117 4295122 1342
  ((Provenance.RewriteSteps.Chunk20.values.drop 62).map CompactStep.ofRaw)
def carry21 := carriedProduct 132381 4295346 4295351 1407
  ((Provenance.RewriteSteps.Chunk21.values.drop 63).map CompactStep.ofRaw)
def carry22 := carriedProduct 132413 4295538 4295543 1471
  ((Provenance.RewriteSteps.Chunk22.values.drop 63).map CompactStep.ofRaw)
def carry23 := carriedProduct 132423 4295592 4295597 1491
  ((Provenance.RewriteSteps.Chunk23.values.drop 19).map CompactStep.ofRaw)

def last0 := productKey 131277 3969678 3969683
def last1 := productKey 131309 3977119 3977124
def last2 := productKey 131341 3991775 3991780
def last3 := productKey 131373 4006431 4006436
def last4 := productKey 132120 4289028 4289033
def last5 := productKey 132152 4289188 4289193
def last6 := productKey 132184 4289348 4289353
def last7 := productKey 132216 4289508 4289513
def last8 := productKey 132241 4289633 4289638
def last9 := productKey 132244 4289966 4289971
def last10 := productKey 132250 4290636 4290641
def last11 := productKey 132253 4290971 4290976
def last12 := productKey 132259 4291641 4291646
def last13 := productKey 132262 4291976 4291981
def last14 := productKey 132268 4292646 4292651
def last15 := productKey 132271 4292981 4292986
def last16 := productKey 132277 4293651 4293656
def last17 := productKey 132280 4293986 4293991
def last18 := productKey 132291 4294685 4294690
def last19 := productKey 132319 4294916 4294921
def last20 := productKey 132351 4295108 4295113
def last21 := productKey 132380 4295341 4295346
def last22 := productKey 132412 4295531 4295536
def last23 := productKey 132422 4295587 4295592
def finalKey := productKey 132423 4295592 4295597

def result0 := scanChunk none 0
  (Provenance.RewriteSteps.Chunk0.values.map CompactStep.ofRaw)
def result1 := scanChunk (some carry0) 64
  (Provenance.RewriteSteps.Chunk1.values.map CompactStep.ofRaw)
def result2 := scanChunk (some carry1) 128
  (Provenance.RewriteSteps.Chunk2.values.map CompactStep.ofRaw)
def result3 := scanChunk (some carry2) 192
  (Provenance.RewriteSteps.Chunk3.values.map CompactStep.ofRaw)
def result4 := scanChunk (some carry3) 256
  (Provenance.RewriteSteps.Chunk4.values.map CompactStep.ofRaw)
def result5 := scanChunk (some carry4) 320
  (Provenance.RewriteSteps.Chunk5.values.map CompactStep.ofRaw)
def result6 := scanChunk (some carry5) 384
  (Provenance.RewriteSteps.Chunk6.values.map CompactStep.ofRaw)
def result7 := scanChunk (some carry6) 448
  (Provenance.RewriteSteps.Chunk7.values.map CompactStep.ofRaw)
def result8 := scanChunk (some carry7) 512
  (Provenance.RewriteSteps.Chunk8.values.map CompactStep.ofRaw)
def result9 := scanChunk (some carry8) 576
  (Provenance.RewriteSteps.Chunk9.values.map CompactStep.ofRaw)
def result10 := scanChunk (some carry9) 640
  (Provenance.RewriteSteps.Chunk10.values.map CompactStep.ofRaw)
def result11 := scanChunk (some carry10) 704
  (Provenance.RewriteSteps.Chunk11.values.map CompactStep.ofRaw)
def result12 := scanChunk (some carry11) 768
  (Provenance.RewriteSteps.Chunk12.values.map CompactStep.ofRaw)
def result13 := scanChunk (some carry12) 832
  (Provenance.RewriteSteps.Chunk13.values.map CompactStep.ofRaw)
def result14 := scanChunk (some carry13) 896
  (Provenance.RewriteSteps.Chunk14.values.map CompactStep.ofRaw)
def result15 := scanChunk (some carry14) 960
  (Provenance.RewriteSteps.Chunk15.values.map CompactStep.ofRaw)
def result16 := scanChunk (some carry15) 1024
  (Provenance.RewriteSteps.Chunk16.values.map CompactStep.ofRaw)
def result17 := scanChunk (some carry16) 1088
  (Provenance.RewriteSteps.Chunk17.values.map CompactStep.ofRaw)
def result18 := scanChunk (some carry17) 1152
  (Provenance.RewriteSteps.Chunk18.values.map CompactStep.ofRaw)
def result19 := scanChunk (some carry18) 1216
  (Provenance.RewriteSteps.Chunk19.values.map CompactStep.ofRaw)
def result20 := scanChunk (some carry19) 1280
  (Provenance.RewriteSteps.Chunk20.values.map CompactStep.ofRaw)
def result21 := scanChunk (some carry20) 1344
  (Provenance.RewriteSteps.Chunk21.values.map CompactStep.ofRaw)
def result22 := scanChunk (some carry21) 1408
  (Provenance.RewriteSteps.Chunk22.values.map CompactStep.ofRaw)
def result23 := scanChunk (some carry22) 1472
  (Provenance.RewriteSteps.Chunk23.values.map CompactStep.ofRaw)

/-! Each theorem below evaluates one proof-free generated shard and its
explicit proof-free carry.  Cardinalities are respectively `64 + carry`
(at most 103 records), except the final `21 + 1` shard. -/

set_option maxRecDepth 100000 in
theorem chunk0_certificate :
    observeChunk Provenance.RewriteSteps.Chunk0.values none 0 0 none =
      expectedObservation carry0 64 62 last0
        (exactDistribution 31 31 0 0 31 0 0 0 155) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk1_certificate :
    observeChunk Provenance.RewriteSteps.Chunk1.values (some carry0) 64 62
        (some last0) =
      expectedObservation carry1 128 126 last1
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk2_certificate :
    observeChunk Provenance.RewriteSteps.Chunk2.values (some carry1) 128 126
        (some last1) =
      expectedObservation carry2 192 190 last2
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk3_certificate :
    observeChunk Provenance.RewriteSteps.Chunk3.values (some carry2) 192 190
        (some last2) =
      expectedObservation carry3 256 254 last3
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk4_certificate :
    observeChunk Provenance.RewriteSteps.Chunk4.values (some carry3) 256 254
        (some last3) =
      expectedObservation carry4 320 318 last4
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk5_certificate :
    observeChunk Provenance.RewriteSteps.Chunk5.values (some carry4) 320 318
        (some last4) =
      expectedObservation carry5 384 382 last5
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk6_certificate :
    observeChunk Provenance.RewriteSteps.Chunk6.values (some carry5) 384 382
        (some last5) =
      expectedObservation carry6 448 446 last6
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk7_certificate :
    observeChunk Provenance.RewriteSteps.Chunk7.values (some carry6) 448 446
        (some last6) =
      expectedObservation carry7 512 510 last7
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk8_certificate :
    observeChunk Provenance.RewriteSteps.Chunk8.values (some carry7) 512 510
        (some last7) =
      expectedObservation carry8 576 560 last8
        (exactDistribution 25 25 0 0 25 0 0 0 125) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk9_certificate :
    observeChunk Provenance.RewriteSteps.Chunk9.values (some carry8) 576 560
        (some last8) =
      expectedObservation carry9 640 603 last9
        (exactDistribution 3 2 0 1 2 0 0 1 333) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk10_certificate :
    observeChunk Provenance.RewriteSteps.Chunk10.values (some carry9) 640 603
        (some last9) =
      expectedObservation carry10 704 689 last10
        (exactDistribution 6 4 0 2 4 0 0 2 666) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk11_certificate :
    observeChunk Provenance.RewriteSteps.Chunk11.values (some carry10) 704 689
        (some last10) =
      expectedObservation carry11 768 732 last11
        (exactDistribution 3 2 0 1 2 0 0 1 333) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk12_certificate :
    observeChunk Provenance.RewriteSteps.Chunk12.values (some carry11) 768 732
        (some last11) =
      expectedObservation carry12 832 818 last12
        (exactDistribution 6 4 0 2 4 0 0 2 666) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk13_certificate :
    observeChunk Provenance.RewriteSteps.Chunk13.values (some carry12) 832 818
        (some last12) =
      expectedObservation carry13 896 861 last13
        (exactDistribution 3 2 0 1 2 0 0 1 333) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk14_certificate :
    observeChunk Provenance.RewriteSteps.Chunk14.values (some carry13) 896 861
        (some last13) =
      expectedObservation carry14 960 947 last14
        (exactDistribution 6 4 0 2 4 0 0 2 666) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk15_certificate :
    observeChunk Provenance.RewriteSteps.Chunk15.values (some carry14) 960 947
        (some last14) =
      expectedObservation carry15 1024 990 last15
        (exactDistribution 3 2 0 1 2 0 0 1 333) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk16_certificate :
    observeChunk Provenance.RewriteSteps.Chunk16.values (some carry15) 1024 990
        (some last15) =
      expectedObservation carry16 1088 1076 last16
        (exactDistribution 6 4 0 2 4 0 0 2 666) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk17_certificate :
    observeChunk Provenance.RewriteSteps.Chunk17.values (some carry16) 1088 1076
        (some last16) =
      expectedObservation carry17 1152 1119 last17
        (exactDistribution 3 2 0 1 2 0 0 1 333) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk18_certificate :
    observeChunk Provenance.RewriteSteps.Chunk18.values (some carry17) 1152 1119
        (some last17) =
      expectedObservation carry18 1216 1215 last18
        (exactDistribution 11 9 0 2 9 0 0 2 691) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk19_certificate :
    observeChunk Provenance.RewriteSteps.Chunk19.values (some carry18) 1216 1215
        (some last18) =
      expectedObservation carry19 1280 1278 last19
        (exactDistribution 28 27 1 0 27 0 1 0 213) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk20_certificate :
    observeChunk Provenance.RewriteSteps.Chunk20.values (some carry19) 1280 1278
        (some last19) =
      expectedObservation carry20 1344 1342 last20
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk21_certificate :
    observeChunk Provenance.RewriteSteps.Chunk21.values (some carry20) 1344 1342
        (some last20) =
      expectedObservation carry21 1408 1407 last21
        (exactDistribution 29 28 1 0 28 1 0 0 213) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk22_certificate :
    observeChunk Provenance.RewriteSteps.Chunk22.values (some carry21) 1408 1407
        (some last21) =
      expectedObservation carry22 1472 1471 last22
        (exactDistribution 32 32 0 0 32 0 0 0 160) := by native_decide
set_option maxRecDepth 100000 in
theorem chunk23_certificate :
    observeChunk Provenance.RewriteSteps.Chunk23.values (some carry22) 1472 1471
        (some last22) =
      expectedObservation carry23 1493 1491 last23
        (exactDistribution 10 10 0 0 10 0 0 0 50) := by native_decide

/-! ## Kernel aggregation of the bounded observations -/

def observations : List ChunkObservation := [
  observeChunk Provenance.RewriteSteps.Chunk0.values none 0 0 none,
  observeChunk Provenance.RewriteSteps.Chunk1.values (some carry0) 64 62
    (some last0),
  observeChunk Provenance.RewriteSteps.Chunk2.values (some carry1) 128 126
    (some last1),
  observeChunk Provenance.RewriteSteps.Chunk3.values (some carry2) 192 190
    (some last2),
  observeChunk Provenance.RewriteSteps.Chunk4.values (some carry3) 256 254
    (some last3),
  observeChunk Provenance.RewriteSteps.Chunk5.values (some carry4) 320 318
    (some last4),
  observeChunk Provenance.RewriteSteps.Chunk6.values (some carry5) 384 382
    (some last5),
  observeChunk Provenance.RewriteSteps.Chunk7.values (some carry6) 448 446
    (some last6),
  observeChunk Provenance.RewriteSteps.Chunk8.values (some carry7) 512 510
    (some last7),
  observeChunk Provenance.RewriteSteps.Chunk9.values (some carry8) 576 560
    (some last8),
  observeChunk Provenance.RewriteSteps.Chunk10.values (some carry9) 640 603
    (some last9),
  observeChunk Provenance.RewriteSteps.Chunk11.values (some carry10) 704 689
    (some last10),
  observeChunk Provenance.RewriteSteps.Chunk12.values (some carry11) 768 732
    (some last11),
  observeChunk Provenance.RewriteSteps.Chunk13.values (some carry12) 832 818
    (some last12),
  observeChunk Provenance.RewriteSteps.Chunk14.values (some carry13) 896 861
    (some last13),
  observeChunk Provenance.RewriteSteps.Chunk15.values (some carry14) 960 947
    (some last14),
  observeChunk Provenance.RewriteSteps.Chunk16.values (some carry15) 1024 990
    (some last15),
  observeChunk Provenance.RewriteSteps.Chunk17.values (some carry16) 1088 1076
    (some last16),
  observeChunk Provenance.RewriteSteps.Chunk18.values (some carry17) 1152 1119
    (some last17),
  observeChunk Provenance.RewriteSteps.Chunk19.values (some carry18) 1216 1215
    (some last18),
  observeChunk Provenance.RewriteSteps.Chunk20.values (some carry19) 1280 1278
    (some last19),
  observeChunk Provenance.RewriteSteps.Chunk21.values (some carry20) 1344 1342
    (some last20),
  observeChunk Provenance.RewriteSteps.Chunk22.values (some carry21) 1408 1407
    (some last21),
  observeChunk Provenance.RewriteSteps.Chunk23.values (some carry22) 1472 1471
    (some last22)]

def expectedObservations : List ChunkObservation := [
  expectedObservation carry0 64 62 last0
    (exactDistribution 31 31 0 0 31 0 0 0 155),
  expectedObservation carry1 128 126 last1
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry2 192 190 last2
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry3 256 254 last3
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry4 320 318 last4
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry5 384 382 last5
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry6 448 446 last6
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry7 512 510 last7
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry8 576 560 last8
    (exactDistribution 25 25 0 0 25 0 0 0 125),
  expectedObservation carry9 640 603 last9
    (exactDistribution 3 2 0 1 2 0 0 1 333),
  expectedObservation carry10 704 689 last10
    (exactDistribution 6 4 0 2 4 0 0 2 666),
  expectedObservation carry11 768 732 last11
    (exactDistribution 3 2 0 1 2 0 0 1 333),
  expectedObservation carry12 832 818 last12
    (exactDistribution 6 4 0 2 4 0 0 2 666),
  expectedObservation carry13 896 861 last13
    (exactDistribution 3 2 0 1 2 0 0 1 333),
  expectedObservation carry14 960 947 last14
    (exactDistribution 6 4 0 2 4 0 0 2 666),
  expectedObservation carry15 1024 990 last15
    (exactDistribution 3 2 0 1 2 0 0 1 333),
  expectedObservation carry16 1088 1076 last16
    (exactDistribution 6 4 0 2 4 0 0 2 666),
  expectedObservation carry17 1152 1119 last17
    (exactDistribution 3 2 0 1 2 0 0 1 333),
  expectedObservation carry18 1216 1215 last18
    (exactDistribution 11 9 0 2 9 0 0 2 691),
  expectedObservation carry19 1280 1278 last19
    (exactDistribution 28 27 1 0 27 0 1 0 213),
  expectedObservation carry20 1344 1342 last20
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry21 1408 1407 last21
    (exactDistribution 29 28 1 0 28 1 0 0 213),
  expectedObservation carry22 1472 1471 last22
    (exactDistribution 32 32 0 0 32 0 0 0 160),
  expectedObservation carry23 1493 1491 last23
    (exactDistribution 10 10 0 0 10 0 0 0 50)]

/-- Kernel aggregation only: the 24 bounded native equalities are rewritten
without reevaluating any generated shard. -/
theorem observations_exact : observations = expectedObservations := by
  simp only [observations, expectedObservations]
  rw [chunk0_certificate, chunk1_certificate, chunk2_certificate,
    chunk3_certificate, chunk4_certificate, chunk5_certificate,
    chunk6_certificate, chunk7_certificate, chunk8_certificate,
    chunk9_certificate, chunk10_certificate, chunk11_certificate,
    chunk12_certificate, chunk13_certificate, chunk14_certificate,
    chunk15_certificate, chunk16_certificate, chunk17_certificate,
    chunk18_certificate, chunk19_certificate, chunk20_certificate,
    chunk21_certificate, chunk22_certificate, chunk23_certificate]

theorem frontier_of_observation
    (raw : List RawRewriteStep) (incoming : Option Batch)
    (nextOffset closedStart : Nat) (previousKey : Option StepKey)
    (outgoing : Batch) (next closedEnd : Nat) (lastKey : StepKey)
    (summary : Distribution)
    (certificate :
      observeChunk raw incoming nextOffset closedStart previousKey =
        expectedObservation outgoing next closedEnd lastKey summary) :
    let result := scanChunk incoming nextOffset (raw.map CompactStep.ofRaw)
    result.carry = some outgoing ∧ result.nextOffset = next := by
  constructor
  · have projected := congrArg ChunkObservation.carry certificate
    simpa [observeChunk, expectedObservation] using projected
  · have projected := congrArg ChunkObservation.nextOffset certificate
    simpa [observeChunk, expectedObservation] using projected

theorem result0_frontier :
    result0.carry = some carry0 ∧ result0.nextOffset = 64 := by
  simpa [result0] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk0_certificate
theorem result1_frontier :
    result1.carry = some carry1 ∧ result1.nextOffset = 128 := by
  simpa [result1] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk1_certificate
theorem result2_frontier :
    result2.carry = some carry2 ∧ result2.nextOffset = 192 := by
  simpa [result2] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk2_certificate
theorem result3_frontier :
    result3.carry = some carry3 ∧ result3.nextOffset = 256 := by
  simpa [result3] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk3_certificate
theorem result4_frontier :
    result4.carry = some carry4 ∧ result4.nextOffset = 320 := by
  simpa [result4] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk4_certificate
theorem result5_frontier :
    result5.carry = some carry5 ∧ result5.nextOffset = 384 := by
  simpa [result5] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk5_certificate
theorem result6_frontier :
    result6.carry = some carry6 ∧ result6.nextOffset = 448 := by
  simpa [result6] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk6_certificate
theorem result7_frontier :
    result7.carry = some carry7 ∧ result7.nextOffset = 512 := by
  simpa [result7] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk7_certificate
theorem result8_frontier :
    result8.carry = some carry8 ∧ result8.nextOffset = 576 := by
  simpa [result8] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk8_certificate
theorem result9_frontier :
    result9.carry = some carry9 ∧ result9.nextOffset = 640 := by
  simpa [result9] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk9_certificate
theorem result10_frontier :
    result10.carry = some carry10 ∧ result10.nextOffset = 704 := by
  simpa [result10] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk10_certificate
theorem result11_frontier :
    result11.carry = some carry11 ∧ result11.nextOffset = 768 := by
  simpa [result11] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk11_certificate
theorem result12_frontier :
    result12.carry = some carry12 ∧ result12.nextOffset = 832 := by
  simpa [result12] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk12_certificate
theorem result13_frontier :
    result13.carry = some carry13 ∧ result13.nextOffset = 896 := by
  simpa [result13] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk13_certificate
theorem result14_frontier :
    result14.carry = some carry14 ∧ result14.nextOffset = 960 := by
  simpa [result14] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk14_certificate
theorem result15_frontier :
    result15.carry = some carry15 ∧ result15.nextOffset = 1024 := by
  simpa [result15] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk15_certificate
theorem result16_frontier :
    result16.carry = some carry16 ∧ result16.nextOffset = 1088 := by
  simpa [result16] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk16_certificate
theorem result17_frontier :
    result17.carry = some carry17 ∧ result17.nextOffset = 1152 := by
  simpa [result17] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk17_certificate
theorem result18_frontier :
    result18.carry = some carry18 ∧ result18.nextOffset = 1216 := by
  simpa [result18] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk18_certificate
theorem result19_frontier :
    result19.carry = some carry19 ∧ result19.nextOffset = 1280 := by
  simpa [result19] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk19_certificate
theorem result20_frontier :
    result20.carry = some carry20 ∧ result20.nextOffset = 1344 := by
  simpa [result20] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk20_certificate
theorem result21_frontier :
    result21.carry = some carry21 ∧ result21.nextOffset = 1408 := by
  simpa [result21] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk21_certificate
theorem result22_frontier :
    result22.carry = some carry22 ∧ result22.nextOffset = 1472 := by
  simpa [result22] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk22_certificate
theorem result23_frontier :
    result23.carry = some carry23 ∧ result23.nextOffset = 1493 := by
  simpa [result23] using frontier_of_observation _ _ _ _ _ _ _ _ _ _
    chunk23_certificate

def batchChunks : List (List Batch) := [
  result0.closed, result1.closed, result2.closed, result3.closed,
  result4.closed, result5.closed, result6.closed, result7.closed,
  result8.closed, result9.closed, result10.closed, result11.closed,
  result12.closed, result13.closed, result14.closed, result15.closed,
  result16.closed, result17.closed, result18.closed, result19.closed,
  result20.closed, result21.closed, result22.closed, result23.closed]

/-- The exact materialized batch stream; the last open batch is flushed once
after the final 21-record shard. -/
def batches : List Batch := batchChunks.flatten ++ [carry23]

/-- Stable public descriptor stream. -/
def descriptors : List BatchDescriptor := batches.map Batch.descriptor

/-! ### Exact step coverage without a global executable certificate -/

structure Segment where
  closed : List CompactStep
  incoming : List CompactStep
  outgoing : List CompactStep
  input : List CompactStep

def Segment.Valid (segment : Segment) : Prop :=
  segment.closed ++ segment.outgoing = segment.incoming ++ segment.input

def segmentOf (raw : List RawRewriteStep) (incoming : Option Batch)
    (nextOffset : Nat) (outgoing : Batch) : Segment :=
  let result := scanChunk incoming nextOffset (raw.map CompactStep.ofRaw)
  { closed := result.closed.flatMap Batch.steps
    incoming := incoming.toList.flatMap Batch.steps
    outgoing := outgoing.steps
    input := raw.map CompactStep.ofRaw }

theorem segmentOf_valid (raw : List RawRewriteStep)
    (incoming : Option Batch) (nextOffset : Nat) (outgoing : Batch)
    (frontier :
      (scanChunk incoming nextOffset (raw.map CompactStep.ofRaw)).carry =
        some outgoing) :
    (segmentOf raw incoming nextOffset outgoing).Valid := by
  have covered := scanChunk_ownedSteps incoming nextOffset
    (raw.map CompactStep.ofRaw)
  simp only [ScanState.ownedSteps, ScanState.allBatches] at covered
  rw [frontier] at covered
  simpa [Segment.Valid, segmentOf] using covered

def segment0 := segmentOf Provenance.RewriteSteps.Chunk0.values none 0 carry0
def segment1 := segmentOf Provenance.RewriteSteps.Chunk1.values (some carry0) 64 carry1
def segment2 := segmentOf Provenance.RewriteSteps.Chunk2.values (some carry1) 128 carry2
def segment3 := segmentOf Provenance.RewriteSteps.Chunk3.values (some carry2) 192 carry3
def segment4 := segmentOf Provenance.RewriteSteps.Chunk4.values (some carry3) 256 carry4
def segment5 := segmentOf Provenance.RewriteSteps.Chunk5.values (some carry4) 320 carry5
def segment6 := segmentOf Provenance.RewriteSteps.Chunk6.values (some carry5) 384 carry6
def segment7 := segmentOf Provenance.RewriteSteps.Chunk7.values (some carry6) 448 carry7
def segment8 := segmentOf Provenance.RewriteSteps.Chunk8.values (some carry7) 512 carry8
def segment9 := segmentOf Provenance.RewriteSteps.Chunk9.values (some carry8) 576 carry9
def segment10 := segmentOf Provenance.RewriteSteps.Chunk10.values (some carry9) 640 carry10
def segment11 := segmentOf Provenance.RewriteSteps.Chunk11.values (some carry10) 704 carry11
def segment12 := segmentOf Provenance.RewriteSteps.Chunk12.values (some carry11) 768 carry12
def segment13 := segmentOf Provenance.RewriteSteps.Chunk13.values (some carry12) 832 carry13
def segment14 := segmentOf Provenance.RewriteSteps.Chunk14.values (some carry13) 896 carry14
def segment15 := segmentOf Provenance.RewriteSteps.Chunk15.values (some carry14) 960 carry15
def segment16 := segmentOf Provenance.RewriteSteps.Chunk16.values (some carry15) 1024 carry16
def segment17 := segmentOf Provenance.RewriteSteps.Chunk17.values (some carry16) 1088 carry17
def segment18 := segmentOf Provenance.RewriteSteps.Chunk18.values (some carry17) 1152 carry18
def segment19 := segmentOf Provenance.RewriteSteps.Chunk19.values (some carry18) 1216 carry19
def segment20 := segmentOf Provenance.RewriteSteps.Chunk20.values (some carry19) 1280 carry20
def segment21 := segmentOf Provenance.RewriteSteps.Chunk21.values (some carry20) 1344 carry21
def segment22 := segmentOf Provenance.RewriteSteps.Chunk22.values (some carry21) 1408 carry22
def segment23 := segmentOf Provenance.RewriteSteps.Chunk23.values (some carry22) 1472 carry23

theorem segment0_valid : segment0.Valid := by
  apply segmentOf_valid
  exact result0_frontier.1
theorem segment1_valid : segment1.Valid := by
  apply segmentOf_valid
  exact result1_frontier.1
theorem segment2_valid : segment2.Valid := by
  apply segmentOf_valid
  exact result2_frontier.1
theorem segment3_valid : segment3.Valid := by
  apply segmentOf_valid
  exact result3_frontier.1
theorem segment4_valid : segment4.Valid := by
  apply segmentOf_valid
  exact result4_frontier.1
theorem segment5_valid : segment5.Valid := by
  apply segmentOf_valid
  exact result5_frontier.1
theorem segment6_valid : segment6.Valid := by
  apply segmentOf_valid
  exact result6_frontier.1
theorem segment7_valid : segment7.Valid := by
  apply segmentOf_valid
  exact result7_frontier.1
theorem segment8_valid : segment8.Valid := by
  apply segmentOf_valid
  exact result8_frontier.1
theorem segment9_valid : segment9.Valid := by
  apply segmentOf_valid
  exact result9_frontier.1
theorem segment10_valid : segment10.Valid := by
  apply segmentOf_valid
  exact result10_frontier.1
theorem segment11_valid : segment11.Valid := by
  apply segmentOf_valid
  exact result11_frontier.1
theorem segment12_valid : segment12.Valid := by
  apply segmentOf_valid
  exact result12_frontier.1
theorem segment13_valid : segment13.Valid := by
  apply segmentOf_valid
  exact result13_frontier.1
theorem segment14_valid : segment14.Valid := by
  apply segmentOf_valid
  exact result14_frontier.1
theorem segment15_valid : segment15.Valid := by
  apply segmentOf_valid
  exact result15_frontier.1
theorem segment16_valid : segment16.Valid := by
  apply segmentOf_valid
  exact result16_frontier.1
theorem segment17_valid : segment17.Valid := by
  apply segmentOf_valid
  exact result17_frontier.1
theorem segment18_valid : segment18.Valid := by
  apply segmentOf_valid
  exact result18_frontier.1
theorem segment19_valid : segment19.Valid := by
  apply segmentOf_valid
  exact result19_frontier.1
theorem segment20_valid : segment20.Valid := by
  apply segmentOf_valid
  exact result20_frontier.1
theorem segment21_valid : segment21.Valid := by
  apply segmentOf_valid
  exact result21_frontier.1
theorem segment22_valid : segment22.Valid := by
  apply segmentOf_valid
  exact result22_frontier.1
theorem segment23_valid : segment23.Valid := by
  apply segmentOf_valid
  exact result23_frontier.1

def segments : List Segment := [segment0, segment1, segment2, segment3,
  segment4, segment5, segment6, segment7, segment8, segment9, segment10,
  segment11, segment12, segment13, segment14, segment15, segment16,
  segment17, segment18, segment19, segment20, segment21, segment22,
  segment23]

inductive SegmentChain : List CompactStep → List Segment →
    List CompactStep → Prop where
  | nil (frontier : List CompactStep) : SegmentChain frontier [] frontier
  | cons {incoming final : List CompactStep} {head : Segment}
      {tail : List Segment}
      (incomingExact : head.incoming = incoming)
      (headValid : head.Valid)
      (rest : SegmentChain head.outgoing tail final) :
      SegmentChain incoming (head :: tail) final

theorem SegmentChain.coverage {incoming final : List CompactStep}
    {values : List Segment} (chain : SegmentChain incoming values final) :
    values.flatMap Segment.closed ++ final =
      incoming ++ values.flatMap Segment.input := by
  induction chain with
  | nil => simp
  | cons incomingExact headValid rest inductionHypothesis =>
      simp only [List.flatMap_cons]
      rw [List.append_assoc, inductionHypothesis, ← List.append_assoc,
        headValid, incomingExact, List.append_assoc]

theorem segments_chain : SegmentChain [] segments carry23.steps := by
  exact .cons rfl segment0_valid
    (.cons rfl segment1_valid
    (.cons rfl segment2_valid
    (.cons rfl segment3_valid
    (.cons rfl segment4_valid
    (.cons rfl segment5_valid
    (.cons rfl segment6_valid
    (.cons rfl segment7_valid
    (.cons rfl segment8_valid
    (.cons rfl segment9_valid
    (.cons rfl segment10_valid
    (.cons rfl segment11_valid
    (.cons rfl segment12_valid
    (.cons rfl segment13_valid
    (.cons rfl segment14_valid
    (.cons rfl segment15_valid
    (.cons rfl segment16_valid
    (.cons rfl segment17_valid
    (.cons rfl segment18_valid
    (.cons rfl segment19_valid
    (.cons rfl segment20_valid
    (.cons rfl segment21_valid
    (.cons rfl segment22_valid
    (.cons rfl segment23_valid (.nil carry23.steps))))))))))))))))))))))))

/-- The exact generated rewrite stream projected to chain-relevant data,
using only the existing generated shard concatenation. -/
def compactGeneratedSteps : List CompactStep :=
  Provenance.RewriteSteps.Chunk0.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk1.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk2.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk3.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk4.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk5.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk6.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk7.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk8.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk9.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk10.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk11.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk12.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk13.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk14.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk15.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk16.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk17.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk18.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk19.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk20.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk21.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk22.values.map CompactStep.ofRaw ++
  Provenance.RewriteSteps.Chunk23.values.map CompactStep.ofRaw

theorem compactGeneratedSteps_exact :
    compactGeneratedSteps =
      Provenance.RewriteSteps.values.map CompactStep.ofRaw := by
  simp [compactGeneratedSteps, Provenance.RewriteSteps.values,
    List.map_append]

/-- Every generated compact step is present once and in order in the exact
batch stream.  Because compact steps retain `output` and `previous`, each
batch can be consumed directly by the source-chain proof. -/
theorem batches_cover_generated_steps :
    batches.flatMap Batch.steps = compactGeneratedSteps := by
  have covered := segments_chain.coverage
  simpa [batches, batchChunks, segments, segment0, segment1, segment2,
    segment3, segment4, segment5, segment6, segment7, segment8, segment9,
    segment10, segment11, segment12, segment13, segment14, segment15,
    segment16, segment17, segment18, segment19, segment20, segment21,
    segment22, segment23, segmentOf, compactGeneratedSteps,
    List.append_assoc] using covered

theorem batches_cover_provenance :
    batches.flatMap Batch.steps =
      Provenance.RewriteSteps.values.map CompactStep.ofRaw := by
  rw [batches_cover_generated_steps, compactGeneratedSteps_exact]

def sumDistributions : List Distribution → Distribution
  | [] => distribution []
  | head :: tail => Distribution.add head (sumDistributions tail)

theorem distribution_flatten (chunks : List (List Batch)) :
    distribution chunks.flatten =
      sumDistributions (chunks.map distribution) := by
  induction chunks with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatten_cons, List.map_cons, sumDistributions,
        distribution_append, inductionHypothesis]

theorem batchChunkDistributions_exact :
    batchChunks.map distribution =
      expectedObservations.map ChunkObservation.distribution := by
  have mapped := congrArg
    (List.map ChunkObservation.distribution) observations_exact
  simpa [observations, batchChunks, observeChunk, result0, result1, result2,
    result3, result4, result5, result6, result7, result8, result9,
    result10, result11, result12, result13, result14, result15, result16,
    result17, result18, result19, result20, result21, result22, result23]
    using mapped

/- One two-step carry, backed by the exact two-record suffix of the
proof-free 21-record final shard; normalization may inspect those 21 raws. -/
theorem finalCarry_distribution :
    distribution [carry23] = exactDistribution 1 1 0 0 1 0 0 0 5 := by
  native_decide

/-- Exact fixed-profile batch and covered-source-row census.  The only
executable normalization remaining after the bounded certificate rewrites is
25 proof-free `Distribution` records. -/
theorem batches_distribution :
    distribution batches =
      exactDistribution 462 445 2 15 445 1 1 15 7221 := by
  rw [batches, distribution_append, distribution_flatten,
    batchChunkDistributions_exact, finalCarry_distribution]
  decide

theorem batch_count : batches.length = 462 := by
  have counted := congrArg Distribution.batchCount batches_distribution
  simpa [distribution, exactDistribution] using counted

theorem step_count_distribution :
    (distribution batches).count2 = 445 ∧
      (distribution batches).count9 = 2 ∧
      (distribution batches).count39 = 15 := by
  rw [batches_distribution]
  decide

theorem source_range_distribution :
    (distribution batches).range5 = 445 ∧
      (distribution batches).range73 = 1 ∧
      (distribution batches).range78 = 1 ∧
      (distribution batches).range323 = 15 := by
  rw [batches_distribution]
  decide

theorem rewrite_covered_source_rows :
    (batches.map rangeLength).sum = 7221 := by
  have counted := congrArg Distribution.coveredSourceRows batches_distribution
  simpa [distribution, exactDistribution] using counted

theorem offsets_of_observation
    (raw : List RawRewriteStep) (incoming : Option Batch)
    (nextOffset closedStart : Nat) (previousKey : Option StepKey)
    (outgoing : Batch) (next closedEnd : Nat) (lastKey : StepKey)
    (summary : Distribution)
    (certificate :
      observeChunk raw incoming nextOffset closedStart previousKey =
        expectedObservation outgoing next closedEnd lastKey summary) :
    OffsetsFromTo closedStart closedEnd
      (scanChunk incoming nextOffset (raw.map CompactStep.ofRaw)).closed := by
  apply offsetsFromTo_of_check
  have projected := congrArg ChunkObservation.offsetsEnd certificate
  simpa [observeChunk, expectedObservation] using projected

/- One two-step carry, backed by the exact two-record suffix of the
proof-free 21-record final shard; normalization may inspect those 21 raws. -/
theorem finalCarry_offsets : OffsetsFromTo 1491 1493 [carry23] := by
  apply offsetsFromTo_of_check
  native_decide

/-- Public canonical name used by the following source-chain leaf. -/
def allBatches : List Batch := batches

theorem allBatches_offsets : OffsetsFromTo 0 1493 allBatches := by
  have o0 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk0_certificate
  have o1 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk1_certificate
  have o2 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk2_certificate
  have o3 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk3_certificate
  have o4 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk4_certificate
  have o5 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk5_certificate
  have o6 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk6_certificate
  have o7 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk7_certificate
  have o8 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk8_certificate
  have o9 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk9_certificate
  have o10 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk10_certificate
  have o11 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk11_certificate
  have o12 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk12_certificate
  have o13 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk13_certificate
  have o14 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk14_certificate
  have o15 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk15_certificate
  have o16 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk16_certificate
  have o17 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk17_certificate
  have o18 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk18_certificate
  have o19 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk19_certificate
  have o20 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk20_certificate
  have o21 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk21_certificate
  have o22 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk22_certificate
  have o23 := offsets_of_observation _ _ _ _ _ _ _ _ _ _ chunk23_certificate
  have closed := o0.append (o1.append (o2.append (o3.append (o4.append
    (o5.append (o6.append (o7.append (o8.append (o9.append (o10.append
    (o11.append (o12.append (o13.append (o14.append (o15.append
    (o16.append (o17.append (o18.append (o19.append (o20.append
    (o21.append (o22.append o23))))))))))))))))))))))
  apply OffsetsFromTo.append (middle := 1491)
  · simpa [allBatches, batches, batchChunks] using closed
  · exact finalCarry_offsets

/-- Every one of the 1,493 generated-step positions has exactly one batch
owner. -/
theorem allBatches_unique_owner (index : Nat) (bound : index < 1493) :
    ∃ batch,
      (batch ∈ allBatches ∧ batch.OwnsIndex index) ∧
      ∀ other,
        other ∈ allBatches ∧ other.OwnsIndex index → other = batch := by
  exact allBatches_offsets.existsUnique_owner (by omega) bound

theorem descriptor_count : descriptors.length = 462 := by
  simpa [descriptors] using batch_count

theorem batchChunkHomogeneousChecks :
    batchChunks.map homogeneousCheck = List.replicate 24 true := by
  have mapped := congrArg
    (List.map ChunkObservation.homogeneous) observations_exact
  simpa [observations, expectedObservations, batchChunks, observeChunk,
    expectedObservation, result0, result1, result2, result3, result4,
    result5, result6, result7, result8, result9, result10, result11,
    result12, result13, result14, result15, result16, result17, result18,
    result19, result20, result21, result22, result23] using mapped

private theorem homogeneousCheck_flatten (chunks : List (List Batch)) :
    homogeneousCheck chunks.flatten =
      (chunks.map homogeneousCheck).all (fun checked => checked) := by
  induction chunks with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      unfold homogeneousCheck at inductionHypothesis ⊢
      simp only [List.flatten_cons, List.map_cons, List.all_cons,
        List.all_append]
      rw [inductionHypothesis]

/- One two-step carry, backed by the exact two-record suffix of the
proof-free 21-record final shard; normalization may inspect those 21 raws. -/
theorem finalCarry_homogeneous : homogeneousCheck [carry23] = true := by
  native_decide

theorem allBatches_homogeneous :
    ∀ batch ∈ allBatches, batch.Homogeneous := by
  have closed : homogeneousCheck batchChunks.flatten = true := by
    rw [homogeneousCheck_flatten, batchChunkHomogeneousChecks]
    decide
  have checked : homogeneousCheck allBatches = true := by
    change (batchChunks.flatten ++ [carry23]).all
      (fun batch => decide batch.Homogeneous) = true
    rw [List.all_append, Bool.and_eq_true]
    exact ⟨closed, finalCarry_homogeneous⟩
  exact fun batch member => homogeneous_of_check_true checked member

theorem batchChunkBoundedChecks :
    batchChunks.map boundedCountCheck = List.replicate 24 true := by
  have mapped := congrArg
    (List.map ChunkObservation.boundedCount) observations_exact
  simpa [observations, expectedObservations, batchChunks, observeChunk,
    expectedObservation, result0, result1, result2, result3, result4,
    result5, result6, result7, result8, result9, result10, result11,
    result12, result13, result14, result15, result16, result17, result18,
    result19, result20, result21, result22, result23] using mapped

private theorem boundedCountCheck_flatten (chunks : List (List Batch)) :
    boundedCountCheck chunks.flatten =
      (chunks.map boundedCountCheck).all (fun checked => checked) := by
  induction chunks with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      unfold boundedCountCheck at inductionHypothesis ⊢
      simp only [List.flatten_cons, List.map_cons, List.all_cons,
        List.all_append]
      rw [inductionHypothesis]

/- One two-step carry, backed by the exact two-record suffix of the
proof-free 21-record final shard; normalization may inspect those 21 raws. -/
theorem finalCarry_bounded : boundedCountCheck [carry23] = true := by
  native_decide

theorem allBatches_stepCount_le_39 :
    ∀ batch ∈ allBatches, batch.descriptor.stepCount ≤ 39 := by
  have closed : boundedCountCheck batchChunks.flatten = true := by
    rw [boundedCountCheck_flatten, batchChunkBoundedChecks]
    decide
  have checked : boundedCountCheck allBatches = true := by
    change (batchChunks.flatten ++ [carry23]).all
      (fun batch => decide (batch.descriptor.stepCount ≤ 39)) = true
    rw [List.all_append, Bool.and_eq_true]
    exact ⟨closed, finalCarry_bounded⟩
  intro batch member
  exact of_decide_eq_true ((List.all_eq_true.mp checked) batch member)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBatchIndex
