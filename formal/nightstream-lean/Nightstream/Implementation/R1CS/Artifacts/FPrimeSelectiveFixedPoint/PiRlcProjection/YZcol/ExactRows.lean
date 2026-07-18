import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.Metadata

/-!
Artifact-checked exact-row facade for the bounded tiny-lifecycle PiRLC
cross-branch source fixture.

Owns: kernel-checked matching of each generated physical row shard to the
handwritten projection schedule, including definition canonicality and the
four final assertion rows.

Does not own: assignment satisfaction, selective-lowering refinement,
serializer semantics, producer-to-consumer binding, protocol authority,
security bounds, cost extrapolation, or permission to remove rows.

Emits constraints: no.

| Physical shard | Reconstructed obligation | Rows | Kernel check |
|---|---|---:|---|
| beta ladder | base power plus 54 extension products | 272 | ordinary `decide` |
| rho evaluations 0/1/2 | five 54-lane evaluations per shard | 540 each | ordinary `decide` |
| limb pair shards | five input evaluations and products per shard | 565 each | ordinary `decide` |
| limb tails | parent, quotient, Phi81 product, and final checks | 221 each | ordinary `decide` |

Assurance tier: source-artifact-checked for the generated tiny fixture only. The
proof deliberately composes bounded shard checks; it does not trust one
monolithic decision over all 5,724 rows and does not establish Rust-conformant
selectively lowered verifier refinement.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ExactRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionIndexedRows

private abbrev certificate := Generated.Metadata.artifact.certificate

/-- Select definition rows using the independently reconstructed check
indices. Keeping this filter at shard granularity makes row classification
reviewable and prevents a tail check from being silently treated as SSA. -/
private def definitionRows (rows : List (Nat × Row)) : List (Nat × Row) :=
  rows.filter fun row =>
    !(Generated.Metadata.artifact.checkIndices.contains row.1)

private def checkRows (rows : List (Nat × Row)) : List (Nat × Row) :=
  rows.filter fun row =>
    Generated.Metadata.artifact.checkIndices.contains row.1

private def betaRows := Generated.Rows.BetaLadder.sourceRows
private def rhoRows0 := Generated.Rows.RhoEvaluations.Shard0.sourceRows
private def rhoRows1 := Generated.Rows.RhoEvaluations.Shard1.sourceRows
private def rhoRows2 := Generated.Rows.RhoEvaluations.Shard2.sourceRows
private def limb0PairRows0 := Generated.Rows.Limb0.Pairs.Shard0.sourceRows
private def limb0PairRows1 := Generated.Rows.Limb0.Pairs.Shard1.sourceRows
private def limb0PairRows2 := Generated.Rows.Limb0.Pairs.Shard2.sourceRows
private def limb0TailRows := Generated.Rows.Limb0.Tail.sourceRows
private def limb1PairRows0 := Generated.Rows.Limb1.Pairs.Shard0.sourceRows
private def limb1PairRows1 := Generated.Rows.Limb1.Pairs.Shard1.sourceRows
private def limb1PairRows2 := Generated.Rows.Limb1.Pairs.Shard2.sourceRows
private def limb1TailRows := Generated.Rows.Limb1.Tail.sourceRows

/- The reconstructed schedule is split by the same fixed physical row
counts as the generator. The final remainder is used for the last tail so
the composition proof is independent of a separately trusted total. -/
private def definitions0 := Generated.Metadata.artifact.indexedDefinitions
private def betaDefinitions := definitions0.take 272
private def definitions1 := definitions0.drop 272
private def rhoDefinitions0 := definitions1.take 540
private def definitions2 := definitions1.drop 540
private def rhoDefinitions1 := definitions2.take 540
private def definitions3 := definitions2.drop 540
private def rhoDefinitions2 := definitions3.take 540
private def definitions4 := definitions3.drop 540
private def limb0PairDefinitions0 := definitions4.take 565
private def definitions5 := definitions4.drop 565
private def limb0PairDefinitions1 := definitions5.take 565
private def definitions6 := definitions5.drop 565
private def limb0PairDefinitions2 := definitions6.take 565
private def definitions7 := definitions6.drop 565
private def limb0TailDefinitions := definitions7.take 219
private def definitions8 := definitions7.drop 219
private def limb1PairDefinitions0 := definitions8.take 565
private def definitions9 := definitions8.drop 565
private def limb1PairDefinitions1 := definitions9.take 565
private def definitions10 := definitions9.drop 565
private def limb1PairDefinitions2 := definitions10.take 565
private def definitions11 := definitions10.drop 565
private def limb1TailDefinitions := definitions11

private def checks0 := Generated.Metadata.artifact.indexedChecks
private def limb0Checks := checks0.take 2
private def limb1Checks := checks0.drop 2

private theorem beta_match : indexedRowsMatch
    (definitionRows betaRows) betaDefinitions = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem rho0_match : indexedRowsMatch
    (definitionRows rhoRows0) rhoDefinitions0 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem rho1_match : indexedRowsMatch
    (definitionRows rhoRows1) rhoDefinitions1 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem rho2_match : indexedRowsMatch
    (definitionRows rhoRows2) rhoDefinitions2 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_pairs0_match : indexedRowsMatch
    (definitionRows limb0PairRows0) limb0PairDefinitions0 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_pairs1_match : indexedRowsMatch
    (definitionRows limb0PairRows1) limb0PairDefinitions1 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_pairs2_match : indexedRowsMatch
    (definitionRows limb0PairRows2) limb0PairDefinitions2 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_tail_match : indexedRowsMatch
    (definitionRows limb0TailRows) limb0TailDefinitions = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_pairs0_match : indexedRowsMatch
    (definitionRows limb1PairRows0) limb1PairDefinitions0 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_pairs1_match : indexedRowsMatch
    (definitionRows limb1PairRows1) limb1PairDefinitions1 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_pairs2_match : indexedRowsMatch
    (definitionRows limb1PairRows2) limb1PairDefinitions2 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_tail_match : indexedRowsMatch
    (definitionRows limb1TailRows) limb1TailDefinitions = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_checks_match : indexedRowsMatchRows
    (checkRows limb0TailRows) limb0Checks = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_checks_match : indexedRowsMatchRows
    (checkRows limb1TailRows) limb1Checks = true := by
  set_option maxRecDepth 100000 in
    decide

private def sharedDefinitionRows : List (Nat × Row) :=
  definitionRows betaRows ++ (
    definitionRows rhoRows0 ++ (
      definitionRows rhoRows1 ++ definitionRows rhoRows2))

private def limb0DefinitionRows : List (Nat × Row) :=
  definitionRows limb0PairRows0 ++ (
    definitionRows limb0PairRows1 ++ (
      definitionRows limb0PairRows2 ++ definitionRows limb0TailRows))

private def limb1DefinitionRows : List (Nat × Row) :=
  definitionRows limb1PairRows0 ++ (
    definitionRows limb1PairRows1 ++ (
      definitionRows limb1PairRows2 ++ definitionRows limb1TailRows))

private def sharedDefinitions : List (Nat × Program.Definition) :=
  betaDefinitions ++
    (rhoDefinitions0 ++ (rhoDefinitions1 ++ rhoDefinitions2))

private def limb0Definitions : List (Nat × Program.Definition) :=
  limb0PairDefinitions0 ++
    (limb0PairDefinitions1 ++
      (limb0PairDefinitions2 ++ limb0TailDefinitions))

private def limb1Definitions : List (Nat × Program.Definition) :=
  limb1PairDefinitions0 ++
    (limb1PairDefinitions1 ++
      (limb1PairDefinitions2 ++ limb1TailDefinitions))

set_option maxRecDepth 100000 in
private theorem shared_definitions_match :
    indexedRowsMatch sharedDefinitionRows sharedDefinitions = true := by
  apply indexedRowsMatch_append beta_match
  apply indexedRowsMatch_append rho0_match
  exact indexedRowsMatch_append rho1_match rho2_match

set_option maxRecDepth 100000 in
private theorem limb0_definitions_match :
    indexedRowsMatch limb0DefinitionRows limb0Definitions = true := by
  apply indexedRowsMatch_append limb0_pairs0_match
  apply indexedRowsMatch_append limb0_pairs1_match
  exact indexedRowsMatch_append limb0_pairs2_match limb0_tail_match

set_option maxRecDepth 100000 in
private theorem limb1_definitions_match :
    indexedRowsMatch limb1DefinitionRows limb1Definitions = true := by
  apply indexedRowsMatch_append limb1_pairs0_match
  apply indexedRowsMatch_append limb1_pairs1_match
  exact indexedRowsMatch_append limb1_pairs2_match limb1_tail_match

private def allDefinitionRows : List (Nat × Row) :=
  sharedDefinitionRows ++ (limb0DefinitionRows ++ limb1DefinitionRows)

private def allDefinitions : List (Nat × Program.Definition) :=
  sharedDefinitions ++ (limb0Definitions ++ limb1Definitions)

set_option maxRecDepth 100000 in
private theorem all_definitions_match :
    indexedRowsMatch allDefinitionRows allDefinitions = true := by
  apply indexedRowsMatch_append shared_definitions_match
  exact indexedRowsMatch_append limb0_definitions_match
    limb1_definitions_match

private theorem definitions_partition :
    allDefinitions = Generated.Metadata.artifact.indexedDefinitions := by
  have split11 :
      limb1PairDefinitions2 ++ limb1TailDefinitions = definitions10 :=
    List.take_append_drop 565 definitions10
  have split10 :
      limb1PairDefinitions1 ++ definitions10 = definitions9 :=
    List.take_append_drop 565 definitions9
  have split9 :
      limb1PairDefinitions0 ++ definitions9 = definitions8 :=
    List.take_append_drop 565 definitions8
  have split8 : limb0TailDefinitions ++ definitions8 = definitions7 :=
    List.take_append_drop 219 definitions7
  have split7 :
      limb0PairDefinitions2 ++ definitions7 = definitions6 :=
    List.take_append_drop 565 definitions6
  have split6 :
      limb0PairDefinitions1 ++ definitions6 = definitions5 :=
    List.take_append_drop 565 definitions5
  have split5 :
      limb0PairDefinitions0 ++ definitions5 = definitions4 :=
    List.take_append_drop 565 definitions4
  have split4 : rhoDefinitions2 ++ definitions4 = definitions3 :=
    List.take_append_drop 540 definitions3
  have split3 : rhoDefinitions1 ++ definitions3 = definitions2 :=
    List.take_append_drop 540 definitions2
  have split2 : rhoDefinitions0 ++ definitions2 = definitions1 :=
    List.take_append_drop 540 definitions1
  have split1 : betaDefinitions ++ definitions1 = definitions0 :=
    List.take_append_drop 272 definitions0
  simp only [allDefinitions, sharedDefinitions, limb0Definitions,
    limb1Definitions, List.append_assoc, split11, split10, split9,
    split8, split7, split6, split5, split4, split3, split2, split1,
    definitions0]

private theorem definition_rows_partition :
    allDefinitionRows = certificate.definitionSourceRows := by
  simp only [allDefinitionRows, sharedDefinitionRows,
    limb0DefinitionRows, limb1DefinitionRows, definitionRows, certificate,
    Artifact.certificate, Artifact.definitionSourceRows,
    Generated.Metadata.artifact, Generated.Metadata.sourceRows,
    betaRows, rhoRows0, rhoRows1, rhoRows2, limb0PairRows0,
    limb0PairRows1, limb0PairRows2, limb0TailRows, limb1PairRows0,
    limb1PairRows1, limb1PairRows2, limb1TailRows, List.filter_append,
    List.append_assoc]

private def IndexedCanonical
    (definitions : List (Nat × Program.Definition)) : Prop :=
  forall entry, entry ∈ definitions -> entry.2.Canonical

private def indexedCanonicalCheck
    (definitions : List (Nat × Program.Definition)) : Bool :=
  definitions.all fun entry => decide entry.2.Canonical

private theorem indexedCanonical_of_check
    {definitions : List (Nat × Program.Definition)}
    (checked : indexedCanonicalCheck definitions = true) :
    IndexedCanonical definitions := by
  intro entry member
  exact of_decide_eq_true
    ((List.all_eq_true.mp checked) entry member)

private theorem beta_canonical_checked :
    indexedCanonicalCheck betaDefinitions = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem beta_canonical : IndexedCanonical betaDefinitions := by
  exact indexedCanonical_of_check beta_canonical_checked

private theorem rho0_canonical_checked :
    indexedCanonicalCheck rhoDefinitions0 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem rho0_canonical : IndexedCanonical rhoDefinitions0 := by
  exact indexedCanonical_of_check rho0_canonical_checked

private theorem rho1_canonical_checked :
    indexedCanonicalCheck rhoDefinitions1 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem rho1_canonical : IndexedCanonical rhoDefinitions1 := by
  exact indexedCanonical_of_check rho1_canonical_checked

private theorem rho2_canonical_checked :
    indexedCanonicalCheck rhoDefinitions2 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem rho2_canonical : IndexedCanonical rhoDefinitions2 := by
  exact indexedCanonical_of_check rho2_canonical_checked

private theorem limb0_pairs0_canonical_checked :
    indexedCanonicalCheck limb0PairDefinitions0 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_pairs0_canonical :
    IndexedCanonical limb0PairDefinitions0 := by
  exact indexedCanonical_of_check limb0_pairs0_canonical_checked

private theorem limb0_pairs1_canonical_checked :
    indexedCanonicalCheck limb0PairDefinitions1 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_pairs1_canonical :
    IndexedCanonical limb0PairDefinitions1 := by
  exact indexedCanonical_of_check limb0_pairs1_canonical_checked

private theorem limb0_pairs2_canonical_checked :
    indexedCanonicalCheck limb0PairDefinitions2 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_pairs2_canonical :
    IndexedCanonical limb0PairDefinitions2 := by
  exact indexedCanonical_of_check limb0_pairs2_canonical_checked

private theorem limb0_tail_canonical_checked :
    indexedCanonicalCheck limb0TailDefinitions = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb0_tail_canonical :
    IndexedCanonical limb0TailDefinitions := by
  exact indexedCanonical_of_check limb0_tail_canonical_checked

private theorem limb1_pairs0_canonical_checked :
    indexedCanonicalCheck limb1PairDefinitions0 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_pairs0_canonical :
    IndexedCanonical limb1PairDefinitions0 := by
  exact indexedCanonical_of_check limb1_pairs0_canonical_checked

private theorem limb1_pairs1_canonical_checked :
    indexedCanonicalCheck limb1PairDefinitions1 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_pairs1_canonical :
    IndexedCanonical limb1PairDefinitions1 := by
  exact indexedCanonical_of_check limb1_pairs1_canonical_checked

private theorem limb1_pairs2_canonical_checked :
    indexedCanonicalCheck limb1PairDefinitions2 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_pairs2_canonical :
    IndexedCanonical limb1PairDefinitions2 := by
  exact indexedCanonical_of_check limb1_pairs2_canonical_checked

private theorem limb1_tail_canonical_checked :
    indexedCanonicalCheck limb1TailDefinitions = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem limb1_tail_canonical :
    IndexedCanonical limb1TailDefinitions := by
  exact indexedCanonical_of_check limb1_tail_canonical_checked

private theorem indexedCanonical_append
    {left right : List (Nat × Program.Definition)}
    (leftCanonical : IndexedCanonical left)
    (rightCanonical : IndexedCanonical right) :
    IndexedCanonical (left ++ right) := by
  intro entry member
  rcases List.mem_append.mp member with inLeft | inRight
  · exact leftCanonical entry inLeft
  · exact rightCanonical entry inRight

private theorem shared_definitions_canonical :
    IndexedCanonical sharedDefinitions := by
  apply indexedCanonical_append beta_canonical
  apply indexedCanonical_append rho0_canonical
  exact indexedCanonical_append rho1_canonical rho2_canonical

private theorem limb0_definitions_canonical :
    IndexedCanonical limb0Definitions := by
  apply indexedCanonical_append limb0_pairs0_canonical
  apply indexedCanonical_append limb0_pairs1_canonical
  exact indexedCanonical_append limb0_pairs2_canonical
    limb0_tail_canonical

private theorem limb1_definitions_canonical :
    IndexedCanonical limb1Definitions := by
  apply indexedCanonical_append limb1_pairs0_canonical
  apply indexedCanonical_append limb1_pairs1_canonical
  exact indexedCanonical_append limb1_pairs2_canonical
    limb1_tail_canonical

private theorem all_definitions_canonical :
    IndexedCanonical allDefinitions := by
  apply indexedCanonical_append shared_definitions_canonical
  exact indexedCanonical_append limb0_definitions_canonical
    limb1_definitions_canonical

private theorem definitions_canonical :
    forall definition, definition ∈ certificate.definitions ->
      definition.Canonical := by
  intro definition member
  have indexedMember :
      ∃ index, (index, definition) ∈
        Generated.Metadata.artifact.indexedDefinitions := by
    simpa [certificate, ProjectionArtifactProgram.Certificate.definitions,
      Artifact.certificate] using member
  rcases indexedMember with ⟨index, indexedMember⟩
  apply all_definitions_canonical (index, definition)
  rw [definitions_partition]
  exact indexedMember

private def allCheckRows : List (Nat × Row) :=
  checkRows betaRows ++ (
    checkRows rhoRows0 ++ (
      checkRows rhoRows1 ++ (
        checkRows rhoRows2 ++ (
          checkRows limb0PairRows0 ++ (
            checkRows limb0PairRows1 ++ (
              checkRows limb0PairRows2 ++ (
                checkRows limb0TailRows ++ (
                  checkRows limb1PairRows0 ++ (
                    checkRows limb1PairRows1 ++ (
                      checkRows limb1PairRows2 ++
                        checkRows limb1TailRows))))))))))

private def emptyChecks : List (Nat × Row) := []

set_option maxRecDepth 100000 in
private theorem beta_checks_empty : indexedRowsMatchRows
    (checkRows betaRows) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem rho0_checks_empty : indexedRowsMatchRows
    (checkRows rhoRows0) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem rho1_checks_empty : indexedRowsMatchRows
    (checkRows rhoRows1) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem rho2_checks_empty : indexedRowsMatchRows
    (checkRows rhoRows2) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem limb0_pairs0_checks_empty : indexedRowsMatchRows
    (checkRows limb0PairRows0) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem limb0_pairs1_checks_empty : indexedRowsMatchRows
    (checkRows limb0PairRows1) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem limb0_pairs2_checks_empty : indexedRowsMatchRows
    (checkRows limb0PairRows2) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem limb1_pairs0_checks_empty : indexedRowsMatchRows
    (checkRows limb1PairRows0) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem limb1_pairs1_checks_empty : indexedRowsMatchRows
    (checkRows limb1PairRows1) emptyChecks = true := by decide
set_option maxRecDepth 100000 in
private theorem limb1_pairs2_checks_empty : indexedRowsMatchRows
    (checkRows limb1PairRows2) emptyChecks = true := by decide

private def allChecks : List (Nat × Row) :=
  emptyChecks ++ (emptyChecks ++ (emptyChecks ++ (emptyChecks ++
    (emptyChecks ++ (emptyChecks ++ (emptyChecks ++ (limb0Checks ++
      (emptyChecks ++ (emptyChecks ++ (emptyChecks ++ limb1Checks))))))))))

set_option maxRecDepth 100000 in
private theorem all_checks_match :
    indexedRowsMatchRows allCheckRows allChecks = true := by
  apply indexedRowsMatchRows_append beta_checks_empty
  apply indexedRowsMatchRows_append rho0_checks_empty
  apply indexedRowsMatchRows_append rho1_checks_empty
  apply indexedRowsMatchRows_append rho2_checks_empty
  apply indexedRowsMatchRows_append limb0_pairs0_checks_empty
  apply indexedRowsMatchRows_append limb0_pairs1_checks_empty
  apply indexedRowsMatchRows_append limb0_pairs2_checks_empty
  apply indexedRowsMatchRows_append limb0_checks_match
  apply indexedRowsMatchRows_append limb1_pairs0_checks_empty
  apply indexedRowsMatchRows_append limb1_pairs1_checks_empty
  exact indexedRowsMatchRows_append limb1_pairs2_checks_empty
    limb1_checks_match

private theorem checks_partition :
    allChecks = Generated.Metadata.artifact.indexedChecks := by
  simp only [allChecks, emptyChecks, List.nil_append, limb0Checks,
    limb1Checks, checks0]
  exact List.take_append_drop 2 Generated.Metadata.artifact.indexedChecks

private theorem check_rows_partition :
    allCheckRows = certificate.checkSourceRows := by
  simp only [allCheckRows, checkRows, certificate,
    Artifact.certificate, Artifact.checkSourceRows,
    Generated.Metadata.artifact, Generated.Metadata.sourceRows,
    betaRows, rhoRows0, rhoRows1, rhoRows2, limb0PairRows0,
    limb0PairRows1, limb0PairRows2, limb0TailRows, limb1PairRows0,
    limb1PairRows1, limb1PairRows2, limb1TailRows, List.filter_append,
    List.append_assoc]

private theorem certificate_definitions :
    certificate.indexedDefinitions =
      Generated.Metadata.artifact.indexedDefinitions := rfl

private theorem certificate_checks :
    certificate.indexedChecks = Generated.Metadata.artifact.indexedChecks :=
  rfl

/-- Exact physical-row certificate for the generated tiny fixture. Every
definition and check is discharged by a bounded ordinary kernel decision,
then composed through the generic append theorems. -/
theorem exact :
    Generated.Metadata.artifact.certificate.ExactRows := by
  constructor
  · rw [certificate_definitions, ← definition_rows_partition,
      ← definitions_partition]
    exact all_definitions_match
  · rw [certificate_checks, ← check_rows_partition, ← checks_partition]
    exact all_checks_match
  · exact definitions_canonical

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ExactRows
