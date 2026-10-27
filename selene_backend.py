from __future__ import annotations

from pathlib import Path
import multiprocessing

from hugr.package import Package
from hugr.qsystem.result import QsysResult
from selene_sim import (
    Quest,
    build,
    DepolarizingErrorModel,
    SimpleRuntime,
    IdealErrorModel,
)

# from selene_qblaze import QBlazePlugin
from tqdm import tqdm


def hugr_save(h: Package, path: str | Path) -> None:
    Path(path).write_bytes(h.to_bytes())


def hugr_load(path: str | Path) -> Package:
    return Package.from_bytes(Path(path).read_bytes())


def emulate_sv(
    hugr: Package | Path, n_qubit: int, n_shots: int, seed: int = 0, scale: float = 1.0
) -> QsysResult:
    print(f"Running Quest: {seed=} {n_qubit=} {n_shots=} {scale=}")
    error_model = IdealErrorModel()
    if scale is not None and scale > 0.0:
        error_model = DepolarizingErrorModel(
            random_seed=seed,
            p_init=2.5e-4 * scale,  # Helios SPAM ~5.0e-4 (typical), split init/meas
            # Source: Quantinuum Helios Product Data Sheet
            # https://docs.quantinuum.com/systems/data_sheets/Quantinuum%20Helios%20Product%20Data%20Sheet.pdf
            # Accessed: 2026-02-16
            p_meas=2.5e-4 * scale,  # Helios SPAM ~5.0e-4 (typical), split init/meas
            # Source: Quantinuum Helios Product Data Sheet
            # https://docs.quantinuum.com/systems/data_sheets/Quantinuum%20Helios%20Product%20Data%20Sheet.pdf
            # Accessed: 2026-02-16
            p_1q=3.0e-5 * scale,  # 1Q gate infidelity (typical)
            # Source: Quantinuum Helios Product Data Sheet
            # https://docs.quantinuum.com/systems/data_sheets/Quantinuum%20Helios%20Product%20Data%20Sheet.pdf
            # Accessed: 2026-02-16
            p_2q=8.0e-4 * scale,  # 2Q gate infidelity (typical)
            # Source: Quantinuum Helios Product Data Sheet
            # https://docs.quantinuum.com/systems/data_sheets/Quantinuum%20Helios%20Product%20Data%20Sheet.pdf
            # Accessed: 2026-02-16
        )

    runtime = SimpleRuntime(random_seed=seed)

    if isinstance(hugr, str):
        hugr = Path(hugr)
    if not isinstance(hugr, (Package, Path)):
        hugr = hugr.compile()

    runner = build(hugr, progress_bar=True)
    simulator = Quest()
    # simulator = QBlazePlugin(random_seed=seed, threads=10)

    return QsysResult(
        tqdm(
            runner.run_shots(
                simulator=simulator,
                runtime=runtime,
                error_model=error_model,
                n_qubits=n_qubit,
                n_shots=n_shots,
                n_processes=multiprocessing.cpu_count(),
                random_seed=seed,
            ),
            desc="Running",
            unit="shot",
            total=n_shots,
        )
    )
