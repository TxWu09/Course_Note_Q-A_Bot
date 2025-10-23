# main.py
import argparse
import sys
from app.client_interface import run_cli
from app.web_interface import run_web

def main():
    parser = argparse.ArgumentParser(
        description="Course Notes Assistant with RAG System powered by Ollama"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="Select interface mode: 'cli' for command line, 'web' for Streamlit app."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Course Notes Assistant")
    print("Mode:", args.mode)
    print("=" * 60)

    try:
        if args.mode == "cli":
            run_cli()
        elif args.mode == "web":
            print("Launching Streamlit Web App...")
            print("If it doesn't open automatically, go to: http://localhost:8501")
            run_web()
        else:
            print("Invalid mode. Use --mode cli or --mode web.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# python main.py --mode cli