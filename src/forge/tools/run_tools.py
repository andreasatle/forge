"""Tool for running tests in a workspace artifact directory."""

import asyncio

from forge.core.workspace import Workspace
from forge.tools.registry import Tool

_TIMEOUT_SECONDS = 60


async def run_tests(workspace: Workspace, artifact_name: str, test_command: str) -> str:
    """Run test_command in the artifact directory and return combined stdout+stderr."""
    cwd = workspace.artifact_dir(artifact_name)
    try:
        proc = await asyncio.create_subprocess_shell(
            test_command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_SECONDS)
        return (stdout + stderr).decode(errors="replace")
    except TimeoutError:
        return f"run_tests timed out after {_TIMEOUT_SECONDS} seconds"


def make_run_tests_tool(workspace: Workspace, artifact_name: str, test_command: str) -> Tool:
    """Return a Tool that runs test_command in the artifact directory with no LLM-facing parameters."""
    async def fn() -> str:
        return await run_tests(workspace, artifact_name, test_command)

    return Tool(
        name="run_tests",
        description="Run the test suite by calling this tool directly. Do NOT try to read a file called 'run_tests' — call this tool instead. Returns test output.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        fn=fn,
    )


async def add_dependency(workspace: Workspace, artifact_name: str, add_dependency_command: str, package_name: str) -> str:
    """Run add_dependency_command with package_name substituted in the artifact directory and return combined stdout+stderr."""
    cwd = workspace.artifact_dir(artifact_name)
    cmd = add_dependency_command.format(package=package_name)
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_SECONDS)
        return (stdout + stderr).decode(errors="replace")
    except TimeoutError:
        return f"add_dependency timed out after {_TIMEOUT_SECONDS} seconds"


def make_add_dependency_tool(workspace: Workspace, artifact_name: str, add_dependency_command: str) -> Tool:
    """Return a Tool that installs a package using add_dependency_command."""
    async def fn(package_name: str) -> str:
        return await add_dependency(workspace, artifact_name, add_dependency_command, package_name)

    return Tool(
        name="add_dependency",
        description=f"Install a package using: {add_dependency_command}. Pass the package name as the argument.",
        parameters={
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string",
                    "description": "The package name to install.",
                },
            },
            "required": ["package_name"],
        },
        fn=fn,
    )
