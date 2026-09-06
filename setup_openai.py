#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = ROOT / "vendor" / "openai-schema"
DIST = SCHEMA / "dist" / "openaiSchema.js"


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> int:
    git = shutil.which("git")
    if not git:
        print("git is required.", file=sys.stderr)
        return 1

    if not (SCHEMA / "package.json").is_file():
        print("Initialising vendored openai-schema submodule...")
        run([git, "submodule", "update", "--init", "--recursive", "vendor/openai-schema"])

    if not (SCHEMA / "package.json").is_file():
        print("vendor/openai-schema is still unavailable.", file=sys.stderr)
        return 1

    configured = os.environ.get("OPENAI_BRIDGE_RUNTIME")
    npm = shutil.which("npm")
    bun = shutil.which("bun")

    # openai-schema is committed with package-lock.json, so npm ci is the most
    # deterministic build path when npm is available. Bun remains a supported
    # fallback, but do not require a Bun lockfile that the vendored repo does not own.
    if npm:
        print("Installing/building openai-schema with npm...")
        run([npm, "ci"], SCHEMA)
        run([npm, "run", "build"], SCHEMA)
    elif bun:
        print("Installing/building openai-schema with Bun...")
        run([bun, "install"], SCHEMA)
        run([bun, "run", "build"], SCHEMA)
    else:
        print("npm or Bun is required to build the vendored TypeScript library.", file=sys.stderr)
        return 1

    if not DIST.is_file():
        print(f"Build completed but {DIST} is missing.", file=sys.stderr)
        return 1

    if configured:
        runtime = shutil.which(configured) or configured
    else:
        runtime = shutil.which("bun") or shutil.which("node")
    if not runtime:
        print("The library built successfully, but Node.js or Bun is required at runtime.", file=sys.stderr)
        return 1

    env_file = ROOT / ".env"
    if not env_file.is_file():
        print("\nNo .env file exists yet. Create one containing:")
        print("OPENAI_API_KEY=sk-...")
    else:
        print(f"\nUsing API configuration from {env_file}")

    print(f"\nOpenAI Interrogator ready. Runtime: {runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
