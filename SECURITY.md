<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security Policy

Nightstream follows the LF Decentralized Trust (LFDT) security vulnerability disclosure policy. This document records how the project interprets and applies that policy so reporters, users, and maintainers understand what to expect during a coordinated disclosure.

## Reporting a Vulnerability

Please report suspected vulnerabilities using one of the approved channels below. Do **not** open public GitHub issues for security matters.

- Email the LFDT security desk at [security@linuxfoundation.org](mailto:security@linuxfoundation.org) with the project name, impact summary, steps to reproduce, affected versions, and suggested mitigations if known.
- Open a private GitHub security advisory: [github.com/nicarq/halo3/security/advisories/new](https://github.com/nicarq/halo3/security/advisories/new).
- If email or GitHub are unavailable, contact a security response team member directly (see table below) to coordinate an alternate secure channel.

Acknowledgement of receipt is provided within **two (2) business days**. Throughout the triage and remediation process we will share periodic status updates with the reporter. Public disclosure occurs no later than 48 hours after patched releases are available, unless a different timeline is mutually agreed within the 90-day maximum embargo window defined by LFDT policy.

## Security Response Team

| Name | Email ID | Chat ID | Area / Specialty |
| --- | --- | --- | --- |
| Nicolas Arqueros | [nico@shinkai.com](mailto:nico@shinkai.com) | `@nicarq` (Discord) | Lattice commitments, SNARK integration |
| Sebastien Guillemot | [sebastien.guillemot@midnight.foundation](mailto:sebastien.guillemot@midnight.foundation) | `TBD` (LFDT Discord) | Protocol integrations, product security |
| LFDT Security Desk | [security@linuxfoundation.org](mailto:security@linuxfoundation.org) | `@lfdt-security` (LFDT Discord) | Escalation, CVE coordination |
| LFDT Community Architects | [community-architects@linuxfoundation.org](mailto:community-architects@linuxfoundation.org) | `@lfdt-architects` (LFDT Discord) | Process support, policy oversight |

The project is actively recruiting additional maintainers with security expertise. Interested contributors should email the maintainers alias and the LFDT community architects to join the response team roster.

Discussion of open reports is handled in the private GitHub security advisory created for the issue. If synchronous collaboration is required, the response team may request a temporary private channel on the LFDT Discord server with only approved participants.

## Disclosure & Remediation Process

1. **Triage** – Confirm the report, gather reproduction details, assign severity, and determine affected components.
2. **Assessment** – Collaborate with the reporter to understand impact. If the issue is not a vulnerability, explain why and, when appropriate, route it through the regular bug process.
3. **Mitigation Planning** – Scope the fix, determine whether configuration workarounds exist, and agree on an embargo timeline (maximum 90 days).
4. **Fix Development** – Prepare patches in a private branch or fork. Request peer review from at least one additional maintainer or LFDT security engineer when feasible.
5. **Release Preparation** – Request a CVE from GitHub's CNA service, update tests and documentation, and stage release notes.
6. **Coordinated Disclosure** – Cut patched releases, publish GitHub advisories, notify reporters and stakeholders, and announce the resolution within 48 hours.

## Supported Versions

Security fixes are backported to:

- The `main` branch.
- The most recent tagged release.

Older releases may receive fixes on a best-effort basis when the severity is high or the patch applies cleanly. Users of older releases are encouraged to upgrade promptly.

## CVE Management

Nightstream relies on GitHub as its CVE Numbering Authority (CNA). When a vulnerability is confirmed, the response team opens a GitHub advisory, requests a CVE ID through the advisory workflow, and ensures the final disclosure includes CVSS scoring, patch references, and mitigation guidance. If an alternative CNA is required (for example, at the request of a downstream platform), the team will coordinate with the LFDT security desk to complete that process.

## Embargo List

Nightstream does **not** currently maintain a project-specific embargo list. Our user base is still forming, and we have not identified operators who require advanced notice to deploy patches. We will revisit this decision as adoption grows. Parties who believe they meet the criteria for embargo participation should contact the security desk with justification so the maintainers can re-evaluate.

## Security Advisories

LFDT strongly recommends using GitHub Security Advisories, and Nightstream adheres to that best practice. All confirmed vulnerabilities will have an advisory published at [github.com/nicarq/halo3/security/advisories](https://github.com/nicarq/halo3/security/advisories) with remediation details, CVE information, and release availability. Stakeholders should subscribe to advisory notifications for timely updates.

## Private Patch Development

Private remediation work is conducted using GitHub's draft security advisories and private fork/branch features. When external infrastructure (e.g., CI pipelines) is required to validate a fix, the response team will coordinate with LFDT staff to ensure that testing occurs without prematurely disclosing sensitive information.

## Questions

For general security questions (non-sensitive), open a GitHub discussion or email the maintainers at [nico@shinkai.com](mailto:nico@shinkai.com) and [sebastien.guillemot@midnight.foundation](mailto:sebastien.guillemot@midnight.foundation). For urgent or ambiguous situations, default to the security reporting channels above.

Thank you for helping keep the Nightstream ecosystem and the broader LFDT community secure.
