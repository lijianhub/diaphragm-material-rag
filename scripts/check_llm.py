"""Check LLM configuration without revealing secrets.

Reports which provider keys are present (names only), the configured RAG_LLM, and,
for OpenAI, the models this key can use. With --ping, sends one tiny request to
the configured RAG_LLM to prove the setup end to end (costs a fraction of a cent).

Usage:
    python scripts/check_llm.py
    python scripts/check_llm.py --ping
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
# Always first, also after `pip install -e .` (which appends src/ at the end): otherwise this
# script's own folder wins and `import query` / `import agent` find the scripts of that name.
sys.path.insert(0, str(SRC_DIR))

from config import load_env_file

loaded = load_env_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ping", action="store_true", help="send one short request to RAG_LLM")
    args = parser.parse_args()

    print(f".env: {'loaded ' + ', '.join(loaded) if loaded else 'nothing loaded (missing, empty, or already set in the environment)'}")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        print(f"{name}: {'present' if os.environ.get(name) else 'missing'}")
    spec = os.environ.get("RAG_LLM")
    print(f"RAG_LLM: {spec or '(not set)'}")

    if os.environ.get("OPENAI_API_KEY"):
        import openai

        try:
            ids = sorted(m.id for m in openai.OpenAI().models.list())
            print(f"\nOpenAI models available to this key ({len(ids)}):")
            for model_id in ids:
                print(f"  {model_id}")
        except openai.AuthenticationError:
            print("\nOpenAI rejected the key (authentication error). Check the value in .env.")
        except openai.APIConnectionError:
            print("\nCould not reach the OpenAI API (network).")

    if args.ping:
        if not spec:
            sys.exit("\n--ping needs RAG_LLM, e.g. RAG_LLM=openai:<model-id>")
        from generation import create_llm

        llm = create_llm(spec)
        reply = llm.chat("Reply with exactly one word.", "Say OK.")
        call = llm.last_call
        print(f"\nping {llm.name}: {reply!r} (served by {call.model}, tokens in/out {call.input_tokens}/{call.output_tokens})")


if __name__ == "__main__":
    main()
