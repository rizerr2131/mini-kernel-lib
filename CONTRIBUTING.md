# Contributing

This repo is early. Keep changes small and easy to review.

## Rules

- Keep changes focused. Do not mix architecture rewrites, formatting churn, and
  unrelated cleanup in one patch.
- If you expand the public API shape, update [docs/design.md](docs/design.md) in
  the same change.
- If you add or modify a kernel, include correctness coverage and benchmark
  notes.
- Do not land performance claims without describing how they were measured.
- Prefer out-of-tree builds and avoid committing generated artifacts.

## Useful Early Work

- design fixes
- API sketches
- build system skeleton
- correctness test scaffolding
- benchmark harness setup

## Pull Requests

Before opening a PR:

- scope is narrow and reviewable
- docs match the code
- generated files are not included by accident
- validation steps are written down, even if they are minimal

If a change touches dispatch policy, API contracts, or repo structure, explain
why in the PR description.
