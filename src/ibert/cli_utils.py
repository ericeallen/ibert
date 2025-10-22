"""Utilities for CLI scripts."""

import sys


def read_input(args_input, script_name="script"):
    """Read input from file or stdin with helpful error messages.

    Args:
        args_input: Input file path from argparse (or None)
        script_name: Name of the script for usage examples

    Returns:
        Input text as string

    Raises:
        SystemExit: If no input provided or input is empty
    """
    if args_input:
        # Read from file
        try:
            with open(args_input) as f:
                input_text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args_input}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Check if stdin is a TTY or if no data is available
        if sys.stdin.isatty():
            # Interactive terminal with no file argument
            print("Error: No input provided", file=sys.stderr)
            print("\nUsage:", file=sys.stderr)
            print(f"  echo 'your input' | {script_name}", file=sys.stderr)
            print(f"  {script_name} input.txt", file=sys.stderr)
            print(f"  cat input.txt | {script_name}", file=sys.stderr)
            sys.exit(1)

        # Try to read from stdin
        try:
            # Read with timeout using select on Unix-like systems
            import select

            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            input_text = sys.stdin.read() if ready else ""
        except (OSError, ValueError):
            # select() not available or stdin not selectable, just try to read
            input_text = sys.stdin.read()

    if not input_text.strip():
        print("Error: No input provided", file=sys.stderr)
        print("\nUsage:", file=sys.stderr)
        print(f"  echo 'your input' | {script_name}", file=sys.stderr)
        print(f"  {script_name} input.txt", file=sys.stderr)
        print(f"  cat input.txt | {script_name}", file=sys.stderr)
        sys.exit(1)

    return input_text
