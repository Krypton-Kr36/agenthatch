"""Tests for ConversationLoop — LLM/Tool calling cycle."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from agenthatch_core.context.manager import ContextManager
from agenthatch_core.llm.client import LLMClient, ToolCallResponse
from agenthatch_core.llm.types import StreamDelta, ToolCall
from agenthatch_core.loop.agent_loop import (
    ConversationLoop,
    RichToolCallEvent,
    _is_inner_monologue,
    _strip_evaluation_blocks,
    _strip_trailing_meta_narration,
)

from agenthatch_core.sandbox.executor import Sandbox
from agenthatch_core.tools.bus import CapBus
from agenthatch.skill.spec import (
    AHSSpec,
    BaseSpec,
    Identity,
    Instructions,
    Intent,
    Interface,
)


@pytest.fixture
def spec() -> AHSSpec:
    return AHSSpec(
        identity=Identity(id="loop-test", display_name="Loop Test", version="1.0.0"),
        intent=Intent(triggers=["test"], satisfies=["test"], summary="Test"),
        interface=Interface(provides=[], requires=[]),
        base=BaseSpec(),
        instructions=Instructions(),
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.provider_name = "mock"
    llm.model = "mock-model"
    return llm


@pytest.fixture
def capbus():
    return CapBus()


@pytest.fixture
def sandbox():
    return Sandbox()


@pytest.fixture
def ctx(spec):
    return ContextManager(spec)


@pytest.fixture
def loop(mock_llm, capbus, sandbox, ctx):
    return ConversationLoop(llm=mock_llm, capbus=capbus, sandbox=sandbox, ctx=ctx)


class TestConversationLoopInit:
    def test_init_stores_dependencies(self, mock_llm, capbus, sandbox, ctx):
        loop = ConversationLoop(llm=mock_llm, capbus=capbus, sandbox=sandbox, ctx=ctx)
        assert loop.llm is mock_llm
        assert loop.capbus is capbus
        assert loop.sandbox is sandbox
        assert loop.ctx is ctx


class TestRunSync:
    def test_simple_text_response(self, loop, mock_llm):
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text="Hello, world!",
            tool_calls=[],
        )
        result = loop.run("hi")
        assert result == "Hello, world!"

    def test_adds_to_history_after_run(self, loop, mock_llm, ctx):
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text="Response",
            tool_calls=[],
        )
        loop.run("question")
        assert len(ctx.history) == 2
        assert ctx.history[0] == {"role": "user", "content": "question"}
        assert ctx.history[1] == {"role": "assistant", "content": "Response"}

    def test_tool_call_route_and_response(self, loop, mock_llm, capbus):
        capbus.register(
            name="echo",
            cap_type="test",
            schema={"type": "object", "properties": {}},
            source="test",
            executor=MagicMock(execute=lambda **kw: "echoed: " + kw.get("msg", "")),
        )

        # v0.6 auto-continuation: after tool + text, loop calls LLM once more per round.
        # v0.8.14: while True loop needs unlimited fallback responses for budget.
        _call_count = [0]
        def _mock_responses(*args, **kwargs):
            _call_count[0] += 1
            if _call_count[0] == 1:
                return ToolCallResponse(
                    text=None,
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"msg": "hello"})],
                )
            return ToolCallResponse(text="I echoed your message.", tool_calls=[])
        mock_llm.chat_with_tools.side_effect = _mock_responses

        result = loop.run("echo hello")
        assert "I echoed your message" in result

    def test_empty_response_returns_placeholder(self, loop, mock_llm):
        mock_llm.chat_with_tools.return_value = ToolCallResponse(text=None, tool_calls=[])
        result = loop.run("hi")
        assert result == "(no response)"

    def test_multiple_tool_calls_in_one_response(self, loop, mock_llm, capbus):
        capbus.register(
            name="tool_a",
            cap_type="test",
            schema={},
            source="test",
            executor=MagicMock(execute=lambda **kw: "result_a"),
        )
        capbus.register(
            name="tool_b",
            cap_type="test",
            schema={},
            source="test",
            executor=MagicMock(execute=lambda **kw: "result_b"),
        )

        # v0.6 auto-continuation: after tools + text, loop continues until budget exhausted
        # v0.9: _MAX_CONSECUTIVE_TEXT_ONLY = 13, need >= 14 items to avoid StopIteration
        mock_llm.chat_with_tools.side_effect = [
            ToolCallResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c1", name="tool_a", arguments={}),
                    ToolCall(id="c2", name="tool_b", arguments={}),
                ],
            ),
            ToolCallResponse(text="Done.", tool_calls=[]),
        ] + [ToolCallResponse(text="Done.", tool_calls=[])] * 14

        result = loop.run("both")
        assert "Done." in result


class TestRunStream:
    def test_stream_text_only(self, loop, mock_llm):
        deltas = [
            StreamDelta(type="text", content="Hello"),
            StreamDelta(type="text", content=" world"),
        ]
        response = ToolCallResponse(text="Hello world", tool_calls=[])

        def gen():
            yield from deltas
            return response

        mock_llm.stream_chat_with_tools.return_value = gen()

        chunks = list(loop.stream("hi"))
        assert chunks == ["Hello", " world"]

    def test_stream_with_tool_call(self, loop, mock_llm, capbus):
        capbus.register(
            name="fetch",
            cap_type="test",
            schema={},
            source="test",
            executor=MagicMock(execute=lambda **kw: "data from fetch tool"),
        )

        deltas = [StreamDelta(type="tool_call_start", tool_name="fetch")]
        tool_response = ToolCallResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name="fetch", arguments={"url": "x"})],
        )
        final_response = ToolCallResponse(text="Fetched data.", tool_calls=[])

        gen1 = (d for d in deltas)

        def make_gen1():
            yield from gen1
            return tool_response

        def make_gen2():
            yield StreamDelta(type="text", content="Fetched data.")
            return final_response

        mock_llm.stream_chat_with_tools.side_effect = [make_gen1(), make_gen2()]

        chunks = list(loop.stream("fetch data"))
        texts = [c for c in chunks if isinstance(c, str)]
        events = [c for c in chunks if isinstance(c, RichToolCallEvent)]

        assert len(events) >= 2
        assert events[0].phase == "start"
        assert events[0].tool_name == "fetch"
        assert "Fetched data" in "".join(texts)

    def test_stream_adds_history(self, loop, mock_llm, ctx):
        deltas = [StreamDelta(type="text", content="Streamed reply")]
        response = ToolCallResponse(text="Streamed reply", tool_calls=[])

        def gen():
            yield from deltas
            return response

        mock_llm.stream_chat_with_tools.return_value = gen()

        list(loop.stream("question"))
        assert len(ctx.history) == 2
        assert ctx.history[0] == {"role": "user", "content": "question"}
        assert ctx.history[1] == {"role": "assistant", "content": "Streamed reply"}


class TestRichToolCallEvent:
    def test_create_event(self):
        event = RichToolCallEvent(
            phase="start",
            tool_name="test_tool",
            tool_args={"key": "value"},
            elapsed=1.5,
            result_preview="preview text",
        )
        assert event.phase == "start"
        assert event.tool_name == "test_tool"
        assert event.tool_args == {"key": "value"}
        assert event.elapsed == 1.5
        assert event.result_preview == "preview text"

    def test_defaults(self):
        event = RichToolCallEvent(phase="done", tool_name="x")
        assert event.tool_args is None
        assert event.elapsed is None
        assert event.result_preview is None


class TestDirectLoopC2:
    """C2 fix: DirectLoop must use add_to_history, not add_assistant_message."""

    def test_run_calls_add_to_history(self):
        """DirectLoop.run() must call ctx.add_to_history('assistant', ...)."""
        from unittest.mock import MagicMock

        from agenthatch_core.bricks.loops import DirectLoop

        mock_llm = MagicMock()
        mock_llm.chat.return_value = "test response"
        mock_llm.last_reasoning_content = None
        mock_ctx = MagicMock()

        loop = DirectLoop(mock_llm, mock_ctx)
        _ = loop.run("hello")

        # C2 fix: must use add_to_history, not add_assistant_message
        # v1.0.26: reasoning_content is also passed (None for non-thinking).
        mock_ctx.add_to_history.assert_called_with(
            "assistant", "test response", reasoning_content=None
        )
        # Old broken call must NOT be made
        assert not hasattr(mock_ctx.add_assistant_message, 'called') or \
            not mock_ctx.add_assistant_message.called, (
            "C2 regression: DirectLoop still calls add_assistant_message "
            "which does not exist on ContextManager"
        )

    def test_stream_calls_add_to_history(self):
        """DirectLoop.stream() must call ctx.add_to_history('assistant', ...)."""
        from unittest.mock import MagicMock

        from agenthatch_core.bricks.loops import DirectLoop

        mock_llm = MagicMock()

        mock_llm.chat_stream.return_value = iter(["Hello", " world"])
        mock_llm.last_reasoning_content = None
        mock_ctx = MagicMock()

        loop = DirectLoop(mock_llm, mock_ctx)
        chunks = list(loop.stream("hello"))
        result_text = "".join(chunks)

        assert result_text == "Hello world"
        mock_ctx.add_to_history.assert_called_with(
            "assistant", "Hello world", reasoning_content=None
        )


class TestMetaNarrationLeakV1028:
    """v1.0.28: thought-process / meta-narration leak must not reach the
    user for ANY agent.

    Previously the sanitize logic was gated behind
    ``max_consecutive_text_only == 0`` (KB agents only), so ordinary
    agents returned their own reasoning process as the final answer
    ("The task is complete. Let me call task_complete.").  All sample
    texts below are verbatim from real generated-agent transcripts
    (thinking-style models).
    """

    _DELIVERABLE = (
        "Here is the real deliverable content that the user asked for, "
        "with full details and context."
    )

    # ── unit: sentence-level strip, empirical English variants ──

    def test_strip_empirical_meta_sentences(self):
        cases = [
            "The task is complete.",
            "The task is essentially done.",
            "The task is done.",
            "I already answered.",
            "I already reported.",
            "Let me call task_complete.",
            "I should call task_complete",
            "I'll call task_complete.",
            "Call task_complete.",
            "The user's request is fully done.",
            "The request is fulfilled.",
            "All requests done.",
            "I've delivered the answer.",
            "Let me finalize.",
            "Let me quickly double-check everything is fine.",
        ]
        for sentence in cases:
            text = f"{self._DELIVERABLE}\n\n{sentence}"
            stripped = _strip_trailing_meta_narration(text)
            assert stripped == self._DELIVERABLE, (
                f"meta sentence not stripped: {sentence!r} -> {stripped!r}"
            )

    def test_strip_guard_keeps_original_without_full_strip(self):
        text = (
            "The task is complete. I already answered. "
            "Let me call task_complete."
        )
        assert _strip_trailing_meta_narration(text) == text
        assert _strip_trailing_meta_narration(text, allow_full_strip=True) == ""

    def test_strip_preserves_real_content_without_patterns(self):
        text = (
            "The function computes the sum of two numbers and returns "
            "the result. It handles negative inputs and raises ValueError "
            "on non-numeric types."
        )
        assert _strip_trailing_meta_narration(text) == text

    # ── unit: <evaluation> XML blocks ──

    def test_strip_evaluation_block_closed(self):
        text = (
            "Here is the design.\n\n---\n\n"
            "<evaluation>\n  <qa_pair>\n    <question>q</question>\n"
            "    <answer>a</answer>\n  </qa_pair>\n</evaluation>\n"
        )
        assert _strip_evaluation_blocks(text) == "Here is the design."

    def test_strip_evaluation_block_unclosed(self):
        text = "Deliverable here.\n\n<evaluation>\n  <qa_pair>truncated"
        assert _strip_evaluation_blocks(text) == "Deliverable here."

    def test_strip_evaluation_block_absent(self):
        assert _strip_evaluation_blocks("plain text") == "plain text"

    # ── unit: inner-monologue detector ──

    _MONOLOGUE = (
        "The user hasn't sent a new message. Wait — the last user turn "
        "was the FAQ request, and I responded. Hmm, but that's what I "
        "just did. Let me re-read. But wait, the prompt says the last "
        "message is assistant. Actually I think I'm now composing the "
        "final answer. Hmm, this is confusing. So I should end the "
        "turn. I'll call task_complete. Let me just call task_complete "
        "since the deliverable is done."
    )

    def test_inner_monologue_detected(self):
        assert _is_inner_monologue(self._MONOLOGUE) is True

    def test_inner_monologue_rejects_normal_answers(self):
        assert _is_inner_monologue("Short answer.") is False
        long_answer = (
            "The report covers three regions. North leads with 120 units. "
            "East follows at 95 units. West trails at 80 units. "
            "All figures come from the Q4 spreadsheet. "
            "The total across regions is 295 units."
        )
        assert _is_inner_monologue(long_answer) is False

    # ── end-to-end: non-KB agent (default max_consecutive_text_only=13) ──

    def test_run_pure_meta_text_falls_back_to_summary(self, loop, mock_llm):
        # xlsx-agent turn-1 verbatim leak
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text=(
                "The task is complete. I already answered. "
                "Let me call task_complete."
            ),
            tool_calls=[ToolCall(
                id="c1",
                name="task_complete",
                arguments={"summary": "已生成 sales_total.xlsx，F2:F4 含 SUM 公式。"},
            )],
        )
        result = loop.run("计算每个区域的总额")
        assert result == "已生成 sales_total.xlsx，F2:F4 含 SUM 公式。"

    def test_run_meta_tail_stripped_deliverable_kept(self, loop, mock_llm):
        # xlsx-agent turn-2 verbatim leak
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text=(
                'The file is verified: F1="total", F2:F4 formulas present. '
                "The recalc couldn't run due to a missing module in this "
                "environment.\n\nThe user's request is fully done. I'll "
                "call task_complete with a summary noting the recalc "
                "limitation."
            ),
            tool_calls=[ToolCall(
                id="c1", name="task_complete", arguments={"summary": "fallback"}
            )],
        )
        result = loop.run("验证公式")
        assert "task_complete" not in result
        assert "fully done" not in result
        assert 'F1="total"' in result
        assert "recalc couldn't run" in result

    def test_run_inner_monologue_replaced_by_summary(self, loop, mock_llm):
        # internal-comms-agent turn-3 pattern
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text=self._MONOLOGUE,
            tool_calls=[ToolCall(
                id="c1",
                name="task_complete",
                arguments={"summary": "FAQ 草稿已交付（含占位符）。"},
            )],
        )
        result = loop.run("写 FAQ")
        assert result == "FAQ 草稿已交付（含占位符）。"

    def test_run_evaluation_block_stripped_on_text_exit(self, loop, mock_llm):
        # mcp-builder OOD-turn leak: reply + <evaluation> block, no
        # task_complete call — exercises the non-completion exit path.
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text=(
                "这个我帮不上忙——我是专门做 MCP 服务器开发的助手。\n\n"
                "<evaluation>\n  <qa_pair>\n    <question>q</question>\n"
                "    <answer>a</answer>\n  </qa_pair>\n</evaluation>\n"
            ),
            tool_calls=[],
        )
        result = loop.run("水龙头漏水怎么办")
        assert "<evaluation>" not in result
        assert "qa_pair" not in result
        assert "MCP" in result

    def test_run_kb_agent_pure_meta_also_falls_back(
        self, mock_llm, capbus, sandbox, ctx
    ):
        # KB mode (max_consecutive_text_only=0): previously the 50% guard
        # returned the pure-meta text unchanged — now it falls back to
        # the task_complete summary as well.
        kb_loop = ConversationLoop(
            llm=mock_llm,
            capbus=capbus,
            sandbox=sandbox,
            ctx=ctx,
            max_consecutive_text_only=0,
            nudge_grace=0,
        )
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text=(
                "The task is complete. I already answered. "
                "Let me call task_complete."
            ),
            tool_calls=[ToolCall(
                id="c1",
                name="task_complete",
                arguments={"summary": "答案已给出。"},
            )],
        )
        result = kb_loop.run("问题")
        assert result == "答案已给出。"

    def test_run_normal_answer_untouched(self, loop, mock_llm):
        mock_llm.chat_with_tools.return_value = ToolCallResponse(
            text="1+1 等于 2。",
            tool_calls=[ToolCall(
                id="c1", name="task_complete", arguments={"summary": "Done."}
            )],
        )
        result = loop.run("1+1等于几")
        assert result == "1+1 等于 2。"

    # ── end-to-end: stream() completion branch ──

    def test_stream_inner_monologue_history_holds_summary(
        self, loop, mock_llm, ctx
    ):
        # The streamed chunks themselves cannot be recalled, but the
        # recorded history must hold the sanitized version so the
        # monologue cannot poison later turns.
        response = ToolCallResponse(
            text=self._MONOLOGUE,
            tool_calls=[ToolCall(
                id="c1",
                name="task_complete",
                arguments={"summary": "FAQ 草稿已交付。"},
            )],
        )

        def gen():
            yield StreamDelta(type="text", content=self._MONOLOGUE)
            return response

        mock_llm.stream_chat_with_tools.return_value = gen()
        chunks = list(loop.stream("写 FAQ"))
        streamed = "".join(c for c in chunks if isinstance(c, str))
        assert streamed == self._MONOLOGUE
        assert ctx.history[-1]["content"] == "FAQ 草稿已交付。"

    def test_stream_pure_meta_history_falls_back_to_summary(
        self, loop, mock_llm, ctx
    ):
        # allow_full_strip must reach the stream completion branch too:
        # a response that is entirely meta-narration is stripped to
        # nothing and the task_complete summary is recorded instead.
        text = (
            "The task is complete. I already answered. "
            "Let me call task_complete."
        )
        response = ToolCallResponse(
            text=text,
            tool_calls=[ToolCall(
                id="c1",
                name="task_complete",
                arguments={"summary": "已生成 sales_total.xlsx。"},
            )],
        )

        def gen():
            yield StreamDelta(type="text", content=text)
            return response

        mock_llm.stream_chat_with_tools.return_value = gen()
        list(loop.stream("计算总额"))
        assert ctx.history[-1]["content"] == "已生成 sales_total.xlsx。"
