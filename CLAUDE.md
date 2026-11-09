# Agent instructions for this repository

Follow CONTRIBUTING.md, especially "Commits and pull requests":

- One commit is one change, and it passes on its own. Code, the measurement it
  produces and the regenerated data go in the same commit. Never leave a separate
  "rebuild data" or "regenerate" commit for later; amend it into the change.
- No WIP, fixup or "address review" commits reach `main`. Squash them first.
- One pull request or merged branch covers one concern: one framework row, one
  capability, one instrument, or one fix.
- Report what you verified, by command and result, not what you plan to run.

Rows land with `visibility: preview`. Cells are derived from artifacts; never type a
support code. Measure through `scripts/measure_locked.py`, and run
`scripts/scrub_home_paths.py` before committing measurement files.
