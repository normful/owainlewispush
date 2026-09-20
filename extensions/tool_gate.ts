/**
 * Push's approval gate for `pi --mode rpc` (PLAN §7).
 *
 * For every gated `tool_call` it asks the host through the RPC extension-UI
 * channel. The prompt is a `select` whose options are the numbered choices push
 * renders in chat; the answer comes back as the option string, and the `N.`
 * index prefix — not the label — is what maps back to a decision, so duplicate
 * labels cannot make the answer ambiguous.
 *
 * push sets `GATE_TOOLS` (comma-separated, `*` = all) when it spawns the child,
 * and `GATE_TIMEOUT_MS` optionally. Diagnostics go to **stderr only**: writing
 * to stdout in `--mode rpc` corrupts the JSONL protocol stream.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const gated = (process.env.GATE_TOOLS ?? "bash")
	.split(",")
	.map((name) => name.trim())
	.filter(Boolean);
const gateAll = gated.includes("*");

function isGated(toolName: string): boolean {
	return gateAll || gated.includes(toolName);
}

const requestTimeoutMs = process.env.GATE_TIMEOUT_MS
	? Number(process.env.GATE_TIMEOUT_MS)
	: undefined;

function summarize(input: unknown): string {
	try {
		return JSON.stringify(input);
	} catch {
		return "<unserializable input>";
	}
}

function numbered(options: string[]): string[] {
	return options.map((label, index) => `${index + 1}. ${label}`);
}

export default function (pi: ExtensionAPI) {
	pi.on("session_start", async (_event, ctx) => {
		console.error(
			`[gate] session_start mode=${ctx.ui.mode} hasUI=${ctx.hasUI} gated=${gated.join("|")}`,
		);
	});

	pi.on("tool_call", async (event, ctx) => {
		if (!isGated(event.toolName)) return;
		const detail = summarize(event.input).slice(0, 400);
		const title = `Approve ${event.toolName}?\n${detail}`;
		const labels = numbered(["Allow", "Deny"]);
		const choice = await ctx.ui.select(
			title,
			labels,
			requestTimeoutMs ? { timeout: requestTimeoutMs } : undefined,
		);
		const index = choice ? labels.indexOf(choice) : -1;
		const allowed = index === 0;
		console.error(
			`[gate] tool_call ${event.toolName} hasUI=${ctx.hasUI} choice=${JSON.stringify(choice)} allowed=${allowed}`,
		);
		if (!allowed) {
			return {
				block: true,
				reason:
					index < 0
						? `Approval for ${event.toolName} expired or was cancelled: ${detail}`
						: `Denied by user: ${detail}`,
			};
		}
	});
}
