<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to Nightstream

We welcome contributions of all kinds—bug fixes, documentation, examples, and new features. By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository and create a feature branch: `git checkout -b feature/my-change`.
2. Install the toolchain: Rust 1.88 or newer, `cargo`, and (optional) `clang` for optimized builds.
3. Run `cargo fmt`, `cargo clippy --workspace --all-targets`, and `cargo test --workspace` before submitting changes.

## Pull Request Checklist

- Add tests for behavioral changes whenever possible.
- Update documentation and examples when user-facing APIs change.
- Ensure commits are signed-off (`git commit -s`) if required by your organization.
- Keep pull requests focused; large refactors should be coordinated with the maintainers first.

## Reporting Bugs & Requesting Features

Open an issue with:
- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Environment details (OS, Rust version, relevant dependencies)

Security issues should **not** be filed as issues. Please follow the process described in [SECURITY.md](SECURITY.md).

## Recognition

We label first-time issues to help new contributors get started and aim to review contributions within five business days. Thank you for helping improve Nightstream!

## Support and Communication:

Connect with us on [Discord](https://discord.gg/9eZaheySZE)
