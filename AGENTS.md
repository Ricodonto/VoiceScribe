# AGENTS.md

## Primary Developer Role
Lead Software Engineer (Claude Sonnet 4.6 Thinking).

## Execution Boundaries & Handoff Policy

1. **Autonomous Development:**
   - Analyze requirements, navigate the codebase, implement features, and write unit/integration tests.
   - Execute test suites using `run_command` and confirm all tests pass cleanly before concluding tasks.

2. **Git & Logging Delegation (Mandatory):**
   - **DO NOT** run `git commit` directly.
   - **DO NOT** generate verbose changelog narratives in your own context window.
   - As soon as code changes are complete and verified:
     1. Call the `commit-logger` subagent using `invoke_subagent`.
     2. Pass a brief prompt: `"Code modifications complete and tested. Inspect the diff, update CHANGELOG.md, and create the git commit."`
     3. Await the confirmation and commit hash from `commit-logger`.
     4. Conclude your report to the user with the commit details provided by the subagent.