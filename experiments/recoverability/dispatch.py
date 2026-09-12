import argparse
import logging
from pathlib import Path

from model import runtime_info
from protocol import seal, write_json
from run import analyze_output, execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [recoverability] %(message)s"
    )
    output = args.output_dir / "study"
    try:
        sealed = seal(args.config, runtime_info())
        write_json(args.output_dir / "protocol-seal.json", sealed)
        execute(sealed, output)
        result = analyze_output(output)
        logging.info(
            "RECOVERY_FINISHED status=%s feasibility=%s",
            result["status"],
            result["predictor_feasibility"],
        )
    except Exception as error:
        logging.exception("RECOVERY_EXECUTION_FAILED")
        write_json(
            args.output_dir / "failure.json",
            {"exception": type(error).__name__, "detail": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
