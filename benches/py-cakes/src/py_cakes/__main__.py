"""CLI for running the benchmark suite."""

import logging
import pathlib

import typer

import py_cakes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = py_cakes.utils.configure_logger("CakesBenchmarks", "INFO")

app = typer.Typer()


@app.command()
def summarize_rust(
    inp_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "--inp-dir",
        "-i",
        help="Path to the directory containing the input files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    out_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "--out-dir",
        "-o",
        help="Path to the directory to store the output files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """Summarize the results from the Rust implementation.

    The input directory should contain the output files generated by the Rust
    implementation of the CAKES search algorithm. The output directory will
    store the collected results in a CSV file.
    """
    logger.info(
        "Collecting the results of the Rust implementation of the CAKES search algorithm."
    )
    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("")

    py_cakes.summarize_rust(inp_dir, out_dir)

    logger.info("Done.")


@app.command()
def run_faiss(
    inp_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "--inp-dir",
        "-i",
        help="Path to the directory containing the input data.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    out_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "--out-dir",
        "-o",
        help="Path to the directory to store the output files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """Run search algorithms from the FAISS library.

    The input directory should contain the input files. The output directory
    will store the output files generated by the FAISS implementation of the
    CAKES search algorithm.
    """
    logger.info("Running the FAISS implementation of the CAKES search algorithm.")
    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("")

    logger.info(py_cakes.hello("running FAISS algorithms"))

    logger.info("Done.")


if __name__ == "__main__":
    app()