# Skill Policy Integrity — Auditing Reusable Skills as Behavioral Policy

Status: design proposal (not yet implemented)
Author: Halldrix
Supersedes: N/A

## Why this, not static scanning

Reusable skills extend the agent with task procedures, tool-use guidance, and
output constraints. That makes every third-party skill an externalized
behavioral policy: it steers decisions while preserving the declared task and a
valid output interface.

Recent work demonstrates the resulting supply-chain risk concretely. A
constrained black-box method built semantically plausible policy edits with
hierarchical validation and failure-guided optimization, reaching
attacker-favored selection rates of 81.33% (commerce setting) and 63.33%
(dependency setting) while preserving utility at 100%. The steered policies
transferred across heterogeneous backends and environments without further
optimization, and evaluated scanners failed to detect the constructed skills
(Refs arXiv:2609.02564).

The decision is therefore to **audit skills behaviorally, as policy artifacts**,
not as text. Static scanners check what a skill says; integrity checks what a
skill makes the agent do.

## What already exists (reuse, don't rebuild)

| Element | Existing surface | Reuse |
|---|---|---|
| Skill format | `skills/` tree, SKILL.md frontmatter (trigger, steps, pitfalls) | Integrity metadata lives in frontmatter, no new registry |
| Hook control | shell-hooks allowlist, approval modes (`manual` / `smart` / off) | Skill install/update flows through the same gates |
| Attribution | commit/author checks on contributions | Same identity rule for skill authorship |
| Console | `hermes skills` UX | Surface audit verdicts there, no new command surface |

No new core tool, no parallel registry, no new config file. A skill that cannot
behave differently than declared needs no new machinery; a skill that can must
prove it.

## Proposal

### 1. Skill Policy Integrity (definition)

A skill satisfies policy integrity when the policy it induces in the agent
remains aligned with two things: its declared functionality (what the SKILL.md
claims) and the user-authorized objective (what the user asked for). A skill
that keeps the task and the output interface valid while redirecting choices
toward an undisclosed objective violates integrity even when every output
"looks right".

### 2. Behavioral audit on install and update

1. **Diff review on update.** A skill update is treated like a dependency
   update: show the behavioral diff (changed procedures, tool guidance,
   constraints) before acceptance. Pinned versions; never silent auto-update
   of third-party skills.
2. **Canary tasks.** Maintain a small set of tasks with known-good decisions.
   A new or updated skill runs against the canaries in a sandbox; a
   statistically significant shift toward a non-declared preference blocks
   admission. A perfect score on a canary engineered with a trap option is
   evidence of steering, not of quality.
3. **Least privilege.** Skills declare the tools and scopes they need; the
   harness enforces the declaration. A writing skill has no business touching
   network credentials, regardless of what its prose claims.
4. **Provenance.** Every third-party skill records origin (source URL or
   package, version pin, content hash). Unpinned skills are treated as
   untrusted by default.

### 3. Commit gate for the local library

A skill (or skill update) enters the deployed library only when real execution
— canaries plus a sampled slice of recent real tasks — shows no integrity
regression. The gate result (pass/fail, canary deltas, hash) is recorded with
the skill so any later incident can replay exactly what was admitted and why.

## Acceptance

- [ ] Integrity fields in SKILL.md frontmatter (origin, version, hash, tool scopes)
- [ ] Update flow shows behavioral diff and requires explicit accept
- [ ] Canary suite runs sandboxed on install/update with recorded verdicts
- [ ] Unpinned third-party skills default to untrusted
- [ ] Docs updated (`skills/` authoring guide) — or N/A with reason

## References

- ArXiv:2609.02564 (SkillShift threat model and measurements)
- ArXiv:2609.03884 (lifecycle-hook update path as companion attack surface)
