# Project Decisions

## ExCEL baseline

Upstream repository:
https://github.com/zwyang6/ExCEL

Pinned upstream commit:
2cd7f510d62793c76365268a2cc4da7cbe915e4d

Local submodule:
`third_party/ExCEL`

Policy:
- `third_party/ExCEL` remains a Git submodule and frozen baseline.
- Do not modify ExCEL directly.
- New research code is implemented under `src/`.
