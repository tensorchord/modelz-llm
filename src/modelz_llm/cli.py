import argparse

import uvicorn

from modelz_llm.falcon_service import build_falcon_app


def parse_argument():
    parser = argparse.ArgumentParser(
        prog="modelz-llm",
        description="Modelz Language Model Service CLI",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Language model name, for example: `facebook/dino-vitb16`",
        default="bigscience/bloomz-560m",
    )
    parser.add_argument(
        "--emb-model",
        help="Embedding model name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--dry-run",
        help="Dry run will only init the model without starting the server",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--device",
        help=(
            "Device to run the model, `auto` means it will try to use"
            " GPU if available (default=auto)"
        ),
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--port",
        help="Port number",
        default=8000,
        type=int,
    )
    parser.add_argument(
        "--worker",
        help="Number of workers",
        default=1,
        type=int,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()
    print(args)

    app = build_falcon_app(args)
    if args.dry_run:
        print("Dry run, exiting...")
        return

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        workers=args.worker,
        access_log=False,
    )
