"""
kiara/tools/code_runner.py — Safe Isolated Python Code Execution Sandbox
========================================================================
Runs Python code snippets in an isolated sub-process with strict timeouts.
Allows Kiara to verify math, execute algorithms, and return verified output.
"""

from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path


def execute_python_code(code: str, timeout: float = 3.5) -> dict[str, str | bool]:
    """Execute Python code in an isolated subprocess.

    Args:
        code: The Python code to run.
        timeout: Maximum seconds allowed before termination.

    Returns:
        dict with keys:
            'success': bool
            'output': str (stdout + stderr or error description)
            'exit_code': int
    """
    # Clean markdown backticks if code block was wrapped
    code_clean = code.strip()
    if code_clean.startswith("```"):
        lines = code_clean.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code_clean = "\n".join(lines).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code_clean)
        temp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        combined = stdout
        if stderr:
            combined = f"{stdout}\n[STDERR]: {stderr}".strip() if stdout else f"[STDERR]: {stderr}"

        if not combined:
            combined = "[Code executed successfully with no output]"

        is_ok = proc.returncode == 0
        return {
            "success": is_ok,
            "status": "ok" if is_ok else "error",
            "output": combined,
            "exit_code": proc.returncode,
            "error": stderr if not is_ok else None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "status": "error",
            "output": f"Execution timed out after {timeout} seconds.",
            "error": f"Execution timed out after {timeout} seconds.",
            "exit_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "output": f"Execution error: {e}",
            "error": str(e),
            "exit_code": -1,
        }
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass
