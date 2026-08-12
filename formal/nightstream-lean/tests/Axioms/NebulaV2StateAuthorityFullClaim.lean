import Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority
import Nightstream.Implementation.NebulaV2.TerminalManifestStateAuthority
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim.authorityDigest_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.authorityDigest_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim.ccsEncoding_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.ccsEncoding_injective

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim.carries_of_digest_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.carries_of_digest_eq

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim.same_claim_authority_eq_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.same_claim_authority_eq_or_failure

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim.equal_carriers_authority_and_memory_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.equal_carriers_authority_and_memory_or_failure

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority.exactReceiptCarriesPriorAuthority' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestStateAuthority.exactReceiptCarriesPriorAuthority

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalManifestStateAuthority.exactReceiptCarriesIncomingAuthority' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestStateAuthority.exactReceiptCarriesIncomingAuthority
