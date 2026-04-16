/**
 * OpenWolf bridge plugin for OpenCode.
 *
 * Fires the same .wolf/hooks/*.js scripts that Claude Code runs automatically,
 * since OpenCode does not execute Claude Code hook scripts natively.
 *
 * Events bridged:
 *   session.created  → session-start.js
 *   tool read before → pre-read.js    (stdin: {tool_name, tool_input})
 *   tool read after  → post-read.js   (stdin: {tool_name, tool_input, tool_output})
 *   tool write before → pre-write.js  (stdin: {tool_name, tool_input})
 *   tool write after  → post-write.js (stdin: {tool_name, tool_input})
 *   session.deleted  → stop.js
 */

import { execSync } from "node:child_process";
import * as path from "node:path";
import * as fs from "node:fs";

export const id = "openwolf-bridge";

const COUNT_FILE = "_opencode_tool_count.json";
const CHECKLIST_FILE = ".periodic_checklist.md";

function resetToolStepCounter(wolfDir) {
  const p = path.join(wolfDir, "hooks", COUNT_FILE);
  try {
    fs.writeFileSync(p, JSON.stringify({ count: 0 }, null, 2), "utf8");
  } catch {
    /* ignore */
  }
}

function bumpToolStepAndWriteChecklist(projectDir, wolfDir) {
  const countPath = path.join(wolfDir, "hooks", COUNT_FILE);
  let count = 0;
  try {
    if (fs.existsSync(countPath)) {
      const j = JSON.parse(fs.readFileSync(countPath, "utf8"));
      count = typeof j.count === "number" ? j.count : 0;
    }
  } catch {
    count = 0;
  }
  count += 1;
  try {
    fs.writeFileSync(countPath, JSON.stringify({ count }, null, 2), "utf8");
  } catch {
    return;
  }
  if (count % 3 !== 0 || count < 1) return;

  const body = `# OpenCode step reminder (tool step ${count})

This file was refreshed automatically by \`.opencode/plugins/openwolf.mjs\` every **3 tool calls** in OpenCode.

- **Todos:** run \`todoread\` → update \`todowrite\` → stick to the plan.
- **Semantic trail:** append a line to \`.wolf/memory.md\` when you did something important (hooks track tokens, not *why*).
- **OpenWolf:** skim \`.wolf/cerebrum.md\` (Do-Not-Repeat) before codegen; follow \`.wolf/OPENWOLF.md\`.
`;
  const checklistPath = path.join(wolfDir, CHECKLIST_FILE);
  try {
    fs.writeFileSync(checklistPath, body, "utf8");
    process.stderr.write(
      `[openwolf] Step ${count}: wrote ${CHECKLIST_FILE} — read it this turn.\n`,
    );
  } catch {
    /* ignore */
  }
}

export const server = async (input) => {
  const projectDir = input.directory;
  const wolfDir = path.join(projectDir, ".wolf");
  const hooksDir = path.join(wolfDir, "hooks");

  // Bail silently if this project has no .wolf/
  if (!fs.existsSync(hooksDir)) return {};

  const env = { ...process.env, CLAUDE_PROJECT_DIR: projectDir };

  const runHook = (script, stdinPayload) => {
    const hookPath = path.join(hooksDir, script);
    if (!fs.existsSync(hookPath)) return;
    try {
      execSync(`node "${hookPath}"`, {
        cwd: projectDir,
        env,
        input: stdinPayload ? JSON.stringify(stdinPayload) : "",
        stdio: ["pipe", "pipe", "pipe"],
        timeout: 8000,
      });
    } catch {
      // non-fatal — hook errors must never break the agent
    }
  };

  // Tool name sets for matching (OpenCode tool names vary by version)
  const isReadTool = (t) =>
    /^(read|file_read|view|cat)$/i.test(t) || t.toLowerCase().includes("read");

  const isWriteTool = (t) =>
    /^(write|file_write|create|edit|str_replace|multiedit|replace)$/i.test(t) ||
    t.toLowerCase().includes("write") ||
    t.toLowerCase().includes("edit");

  return {
    // ── Session lifecycle ────────────────────────────────────────────────────
    event: async ({ event }) => {
      const type =
        event?.type ??
        event?.name ??
        (typeof event === "string" ? event : "");

      if (/session\.(created|start)/i.test(type)) {
        resetToolStepCounter(wolfDir);
        runHook("session-start.js");
      }

      if (/session\.(deleted|end|close|idle)/i.test(type)) {
        runHook("stop.js");
      }
    },

    // ── Before tool executes — check anatomy / warn on repeated reads ────────
    "tool.execute.before": async ({ tool }, output) => {
      const args = output.args ?? {};

      if (isReadTool(tool)) {
        const filePath = args.path ?? args.file_path ?? args.filePath ?? "";
        if (filePath) {
          runHook("pre-read.js", {
            tool_name: tool,
            tool_input: { path: filePath },
          });
        }
      }

      if (isWriteTool(tool)) {
        runHook("pre-write.js", {
          tool_name: tool,
          tool_input: args,
        });
      }
    },

    // ── After tool executes — update token ledger / anatomy ─────────────────
    "tool.execute.after": async ({ tool, args }, output) => {
      const toolInput = args ?? {};

      if (isReadTool(tool)) {
        const filePath =
          toolInput.path ?? toolInput.file_path ?? toolInput.filePath ?? "";
        if (filePath) {
          runHook("post-read.js", {
            tool_name: tool,
            tool_input: { path: filePath },
            tool_output: { content: output.output ?? "" },
          });
        }
      }

      if (isWriteTool(tool)) {
        runHook("post-write.js", {
          tool_name: tool,
          tool_input: toolInput,
        });
      }

      bumpToolStepAndWriteChecklist(projectDir, wolfDir);
    },
  };
};
