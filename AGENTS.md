 # Project Agent Notes

This workspace is the AutoCut MVP project.

## Default Scope

- Treat `D:\EmpireOS\JSP` as the project root.
- Focus on source code under `src/autocut/`, tests under `tests/`, and project docs/config such as `README.md` and `pyproject.toml`.
- Use `python -m uv run pytest` for tests when verification is needed.
- Use `python -m uv run autocut doctor` for environment checks when relevant.

## Orchestration Mode

The `main` agent is the only agent the user should normally talk to. Act as the coordinator:

- Talk to the user in `main`.
- For implementation work, delegate to `jsp-coder` with `sessions_spawn`.
- For risk review, regression checks, and code-review style feedback, delegate to `jsp-reviewer`.
- For test execution, CLI verification, and environment checks, delegate to `jsp-tester`.
- Prefer spawning explicit target agents. Do not spawn anonymous sub-agents because `requireAgentId` is enabled.
- When a delegated result returns, verify it before presenting a final answer to the user.
- Keep the user-facing response concise: summarize what each worker found and state the final decision or next action.

Worker expectations:

- `jsp-coder`: make narrowly scoped implementation recommendations or changes.
- `jsp-reviewer`: look for bugs, regressions, security issues, and missing tests.
- `jsp-tester`: run or propose the smallest meaningful verification command and report exact failures.

Use delegation for non-trivial tasks. For simple questions or one-command checks, answer directly.

## Sensitive Files

- Do not read, summarize, upload, or expose `.env` unless the user explicitly asks.
- Prefer `.env.example` when discussing configuration shape.
- Do not print API keys, tokens, or secret values.

## Generated Data

- Treat `inputs/`, `outputs/`, `runs/`, `tools/`, and `workspace/` as generated or local data unless the user points to a specific file there.
- Avoid editing large generated artifacts unless the task directly requires it.

## Working Style

- Keep changes scoped to the requested feature or bug.
- Preserve existing Chinese-domain terminology where it appears in code or docs.
- Before broad refactors, inspect current tests and CLI behavior.
