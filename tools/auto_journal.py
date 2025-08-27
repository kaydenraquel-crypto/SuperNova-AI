"""CLI utility to append a structured session entry to SuperJournal.txt

Usage:
  python -m tools.auto_journal \

    --supervisor "Claude Supervisor" \

    --subagents "Agent-A,Agent-B" \

    --actions "Did X;Did Y" \

    --results "Result 1;Result 2" \

    --suggestions "Next A;Next B"

"""

import argparse

from supernova.journal import append_session

def main():

    p = argparse.ArgumentParser()

    p.add_argument("--supervisor", default="ManualEntry")

    p.add_argument("--subagents", default="")

    p.add_argument("--actions", default="")

    p.add_argument("--results", default="")

    p.add_argument("--suggestions", default="")

    args = p.parse_args()

    subs = [s.strip() for s in args.subagents.split(",") if s.strip()]

    acts = [s.strip() for s in args.actions.split(";") if s.strip()]

    ress = [s.strip() for s in args.results.split(";") if s.strip()]

    sugg = [s.strip() for s in args.suggestions.split(";") if s.strip()]

    append_session(args.supervisor, subs, acts, ress, sugg, meta={"source":"cli"})

if __name__ == "__main__":

    main()

