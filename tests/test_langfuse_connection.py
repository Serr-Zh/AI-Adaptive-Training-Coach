from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    langfuse = get_client()

    if langfuse.auth_check():
        print("Langfuse connection: OK")
    else:
        print("Langfuse connection: FAILED")


if __name__ == "__main__":
    main()