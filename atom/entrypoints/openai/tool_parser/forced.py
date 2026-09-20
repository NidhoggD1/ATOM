# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Read bare JSON arguments when the request uniquely selects a tool."""

import json

from .stream import ToolCallStreamParser
from .tool_parser import ToolCall, unique_tool_call_id


def forced_tool_name(tools, tool_choice) -> str | None:
    names = [
        tool["function"]["name"]
        for tool in (tools or [])
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
    ]
    if tool_choice == "required" and len(names) == 1:
        return names[0]
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        if tool_choice.get("type") == "function" and name in names:
            return name
    return None


class ForcedJsonToolCallParser(ToolCallStreamParser):
    """An explicitly selected tool can receive a bare JSON object as arguments.

    K3 sometimes emits that object in its response channel instead of wrapping
    it in a call. The caller supplies the destination from tool_choice; no
    destination is inferred for auto choice or multiple required tools.
    Candidate JSON is held until EOF so a partial object never leaks as
    content before being returned as arguments. Explicit calls take priority.
    """

    def __init__(self, *args, json_tool_name: str | None, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_tool_name = None if self.suppress_calls else json_tool_name
        self._json_parts: list[str] = []
        self._json_started = False

    def _route(self, events):
        if self._json_tool_name is None:
            return events
        output = []
        for kind, value in events:
            if kind == "content" and self._json_tool_name is not None:
                self._json_parts.append(value)
                if not self._json_started and value.lstrip():
                    if value.lstrip().startswith("{"):
                        self._json_started = True
                    else:
                        self._json_tool_name = None
                        output.append(("content", "".join(self._json_parts)))
                        self._json_parts.clear()
            else:
                self._json_tool_name = None
                if self._json_parts:
                    output.append(("content", "".join(self._json_parts)))
                    self._json_parts.clear()
                output.append((kind, value))
        return output

    def process(self, text: str) -> list:
        return self._route(super().process(text))

    def flush(self) -> list:
        events = self._route(super().flush())
        name = self._json_tool_name
        self._json_tool_name = None
        if not self._json_parts:
            return events
        text = "".join(self._json_parts)
        self._json_parts.clear()
        try:
            arguments = json.loads(text)
        except (ValueError, RecursionError):
            arguments = None
        if name is not None and isinstance(arguments, dict):
            # Keep the generated JSON, including whitespace inside strings.
            # No schema repair, missing-property insertion, or coercion.
            events.extend(
                self._emit_call(
                    ToolCall(
                        id=unique_tool_call_id(),
                        type="function",
                        function={"name": name, "arguments": text},
                    )
                )
            )
            events.append(("tool_call_end", None))
        else:
            events.append(("content", text))
        return events
