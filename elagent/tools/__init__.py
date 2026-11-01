"""Default tool set for ELAgent."""

from __future__ import annotations

from .base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult
from .extra import AskTool, SubagentTool, TimeTool, WorkflowTool
from .fs import EditTool, GlobTool, GrepTool, LsTool, ReadTool, WriteTool
from .shell import BashTool
from .todo import TodoTool, load_items, render as render_todo


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ReadTool(),
            WriteTool(),
            EditTool(),
            LsTool(),
            GlobTool(),
            GrepTool(),
            BashTool(),
            TodoTool(),
            TimeTool(),
            AskTool(),
            SubagentTool(),
            WorkflowTool(),
        ]
    )


__all__ = [
    "Tool", "ToolContext", "ToolError", "ToolRegistry", "ToolResult",
    "build_registry", "load_items", "render_todo",
]
