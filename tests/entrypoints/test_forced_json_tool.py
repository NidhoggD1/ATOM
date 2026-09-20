# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import json

import pytest

from atom.entrypoints.openai.serving_chat import (
    _build_chat_choice,
    stream_chat_response,
)
from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector
from atom.entrypoints.openai.tool_parser.forced import (
    ForcedJsonToolCallParser,
    forced_tool_name,
)
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser
from atom.entrypoints.openai.tool_parser.stream import flatten_tool_events

TOOLS = [
    {
        "type": "function",
        "function": {"name": "submit", "parameters": {"type": "object"}},
    }
]


def parser(tools=TOOLS, choice="required"):
    return ForcedJsonToolCallParser(
        tools=tools,
        parser_cls=KimiK3Parser,
        json_tool_name=forced_tool_name(tools, choice),
        suppress_calls=choice == "none",
    )


def frame(text):
    return f"<|open|>response<|sep|>{text}<|close|>response<|sep|>"


def test_required_json_preserves_arguments_at_every_chunk_boundary():
    arguments = ' {"value": {"text": " \\t\\n ", "number": 1.25}} '
    text = frame(arguments)
    for split in range(len(text) + 1):
        reader = parser()
        before_eof = reader.process(text[:split]) + reader.process(text[split:])
        assert before_eof == []
        content, calls = flatten_tool_events(reader.flush())
        assert content == ""
        assert len(calls) == 1
        assert calls[0].function == {"name": "submit", "arguments": arguments}


@pytest.mark.parametrize("choice", [None, "auto", "none", "required"])
def test_multiple_tools_do_not_choose_a_destination(choice):
    tools = TOOLS + [{"type": "function", "function": {"name": "another"}}]
    reader = parser(tools=tools, choice=choice)
    content, calls = flatten_tool_events(
        reader.process(frame('{"value": {}}')) + reader.flush()
    )
    assert content == '{"value": {}}'
    assert calls == []


@pytest.mark.parametrize("choice", [None, "auto", "none"])
def test_single_tool_without_forced_choice_keeps_json_as_content(choice):
    reader = parser(choice=choice)
    content, calls = flatten_tool_events(
        reader.process(frame('{"value": {}}')) + reader.flush()
    )
    assert content == '{"value": {}}'
    assert calls == []


@pytest.mark.parametrize(
    "text",
    [
        '{"value":',
        '[{"value": 1}]',
        '"hello"',
        "```json\n{}\n```",
        "{} trailing prose",
        "prefix {}",
    ],
)
def test_only_a_complete_bare_object_is_interpreted_as_arguments(text):
    reader = parser()
    events = []
    for char in frame(text):
        events.extend(reader.process(char))
    content, calls = flatten_tool_events(events + reader.flush())
    assert content == text
    assert calls == []


def test_explicit_call_takes_precedence_over_bare_json():
    reader = parser()
    text = frame('{"example": 1}') + KimiK3Parser.render_call(
        "submit", {"actual": "two"}
    )
    content, calls = flatten_tool_events(reader.process(text) + reader.flush())
    assert content == '{"example": 1}'
    assert len(calls) == 1
    assert json.loads(calls[0].function["arguments"]) == {"actual": "two"}


def test_named_tool_uses_request_destination_and_does_not_repair_arguments():
    tools = TOOLS + [{"type": "function", "function": {"name": "another"}}]
    choice = {"type": "function", "function": {"name": "another"}}
    response = _build_chat_choice(
        frame('{"arbitrary": " "}'),
        "stop",
        tools=tools,
        tool_choice=choice,
        tool_parser_cls=KimiK3Parser,
    )
    assert response["finish_reason"] == "tool_calls"
    assert response["message"]["tool_calls"][0]["function"] == {
        "name": "another",
        "arguments": '{"arbitrary": " "}',
    }


def test_serving_stream_and_nonstream_agree_on_required_json():
    text = frame('{"value": {}}')
    expected = _build_chat_choice(
        text, "stop", tools=TOOLS, tool_choice="required", tool_parser_cls=KimiK3Parser
    )

    async def run():
        collector = StreamOutputCollector("request")
        for index, char in enumerate(text):
            collector.put_nowait(
                {
                    "text": char,
                    "token_ids": [index],
                    "finished": index == len(text) - 1,
                    "finish_reason": "stop",
                }
            )
        return [
            chunk
            async for chunk in stream_chat_response(
                "request",
                "model",
                collector,
                1,
                10,
                lambda *a, **kw: None,
                lambda *a, **kw: None,
                tools=TOOLS,
                tool_choice="required",
                tool_parser_cls=KimiK3Parser,
            )
        ]

    content, arguments, name, finish = "", "", "", None
    for chunk in asyncio.run(run()):
        for line in chunk.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                content += delta.get("content") or ""
                for call in delta.get("tool_calls") or []:
                    name += call["function"].get("name", "")
                    arguments += call["function"].get("arguments", "")
    assert finish == expected["finish_reason"] == "tool_calls"
    assert content == expected["message"]["content"] == ""
    assert {"name": name, "arguments": arguments} == expected["message"]["tool_calls"][
        0
    ]["function"]
