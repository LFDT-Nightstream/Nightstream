import Nightstream.Implementation.Nebula.FPrime.Manifest.RecursiveStateAuthority
import Nightstream.Implementation.Nebula.FPrime.Manifest.TerminalStateAuthority
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.StateAuthorityFullClaim.authorityDigest_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.authorityDigest_canonical

/-- info: 'Nightstream.Implementation.Nebula.StateAuthorityFullClaim.ccsEncoding_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.ccsEncoding_injective

/-- info: 'Nightstream.Implementation.Nebula.StateAuthorityFullClaim.carries_of_digest_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.carries_of_digest_eq

/-- info: 'Nightstream.Implementation.Nebula.StateAuthorityFullClaim.same_claim_authority_eq_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.same_claim_authority_eq_or_failure

/-- info: 'Nightstream.Implementation.Nebula.StateAuthorityFullClaim.equal_carriers_authority_and_memory_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityFullClaim.equal_carriers_authority_and_memory_or_failure

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestStateAuthority.exactReceiptCarriesPriorAuthority' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestStateAuthority.exactReceiptCarriesPriorAuthority

/-- info: 'Nightstream.Implementation.Nebula.TerminalManifestStateAuthority.exactReceiptCarriesIncomingAuthority' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalManifestStateAuthority.exactReceiptCarriesIncomingAuthority
