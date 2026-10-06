# Working on This Repository with Claude Code

How this repository is configured for [Claude Code](https://claude.com/claude-code), and the workflows that suit it.

## Principles

An agent works well in a codebase when it has three things:

1. **Context**: the project's conventions, provided by `.claude/CLAUDE.md` and `.claude/rules/`.
2. **Verification**: a way to check its own work, here an offline `pytest` suite and a runnable demo.
3. **Boundaries**: what it may and may not touch, set through permissions and hooks in `.claude/settings.json`.

## Configuration layout

| File | Scope | Committed |
|---|---|---|
| `.claude/CLAUDE.md` | Project guidance loaded every session | Yes |
| `.claude/rules/*.md` | Topic rules; `paths` frontmatter limits which files they apply to | Yes |
| `.claude/settings.json` | Shared permissions, hooks and env vars | Yes |
| `.claude/settings.local.json`, `CLAUDE.local.md` | Personal overrides | No (git-ignored) |
| `.claude/agents/`, `.claude/skills/`, `.mcp.json` | Subagents, reusable skills, MCP servers | Yes, when added |
| `~/.claude/CLAUDE.md` | A contributor's own global preferences | Outside the repo |

Shared configuration is committed; personal configuration uses `*.local.*` names and stays out of git.

Currently in use:

- `.claude/CLAUDE.md`: commands, the `src/` import root, pipeline architecture, and which modules are still empty placeholders.
- `.claude/rules/testing.md`: testing conventions, loaded only when files under `tests/` are involved.

## Maintaining CLAUDE.md

It is loaded into every session, so keep it short and accurate.

**Include:**
- Commands, especially how to run a single test.
- Facts that take several files to discover, such as the import root.
- Conventions that differ from the defaults, such as offline-only tests.

**Leave out:**
- File listings that can be discovered by reading the tree.
- Generic advice.
- Rules for components that do not exist yet. Add `api.md` or `deployment.md` under `.claude/rules/` only once `src/api/` or `docker/` contain real code.

Update it in the same PR as the code change it describes.

## Recommended workflow

1. **Explore**: ask Claude to read the relevant modules and tests without editing anything.
2. **Plan**: use Plan Mode (Shift+Tab) and review the proposed approach before any code is written.
3. **Test first**: have it write a failing test, run it, and confirm that it fails.
4. **Implement**: iterate until `pytest -q` passes.
5. **Commit**: work on a feature branch and open a PR. Do not push directly to `main`.

Example prompt:

> Implement BM25 retrieval in `src/retrieval/bm25.py`. Propose a plan first. After I approve it, write tests in `tests/test_retrieval.py` that run offline, then implement until they pass.

Day-to-day habits:

- Press `Esc` to interrupt early when the direction is wrong.
- Run `/clear` between unrelated tasks.
- Hand broad codebase searches to a subagent so the main session keeps only the conclusions.

## Optional: permissions and hooks

The repository does not ship a `settings.json` yet. A reasonable starting point:

```json
{
  "permissions": {
    "allow": ["Bash(pytest:*)", "Bash(python -m pytest:*)"],
    "deny": ["Read(./.env)", "Read(./data/raw/*.pdf)"]
  },
  "hooks": {
    "PostToolUse": [
      { "matcher": "Edit|Write", "hooks": [{ "type": "command", "command": "python -m pytest -q" }] }
    ]
  }
}
```

`CLAUDE.md` is guidance that the model may or may not follow. Settings are enforced by the harness: hooks always run, and denied tools are always blocked. Anything that must happen every time belongs in a hook.
