from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionCellUpdateConfig:
    learning_rate: float
    p_min: float
    p_max: float
    logvar_drift: float
    evidence_logvar_gain: float
    logvar_min: float = -4.0
    logvar_max: float = 2.5
    belief_min: float = 0.0
    belief_max: float = 1.0
    reconstruction_logvar_source: str = "precision"


@dataclass
class PredictionCellUpdateResult:
    belief: np.ndarray
    logvar: np.ndarray
    error: np.ndarray
    precision: np.ndarray
    reconstruction: float
    complexity: float
    total: float


@dataclass(frozen=True)
class ReceptiveFieldTopology:
    input_height: int
    input_width: int
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input dimensions must be positive")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("cell grid dimensions must be positive")
        if self.rows > self.input_height or self.cols > self.input_width:
            raise ValueError("cell grid dimensions must not exceed input dimensions")

    @property
    def cell_shape(self) -> tuple[int, int]:
        return int(self.rows), int(self.cols)

    def receptive_slice(self, row: int, col: int) -> tuple[slice, slice]:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise IndexError("cell index out of range")
        y_edges = np.linspace(0, self.input_height, self.rows + 1, dtype=np.int64)
        x_edges = np.linspace(0, self.input_width, self.cols + 1, dtype=np.int64)
        return slice(int(y_edges[row]), int(y_edges[row + 1])), slice(int(x_edges[col]), int(x_edges[col + 1]))

    def neighbor_indices(self, row: int, col: int, *, include_diagonal: bool = False) -> list[tuple[int, int]]:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise IndexError("cell index out of range")
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diagonal:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        neighbors: list[tuple[int, int]] = []
        for dy, dx in offsets:
            yy = row + dy
            xx = col + dx
            if 0 <= yy < self.rows and 0 <= xx < self.cols:
                neighbors.append((yy, xx))
        return neighbors

    def pool_image(self, image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=np.float32)
        if image_arr.ndim != 3:
            raise ValueError("image must have shape (height, width, channels)")
        if image_arr.shape[0] != self.input_height or image_arr.shape[1] != self.input_width:
            raise ValueError("image spatial shape does not match topology")
        pooled = np.zeros((self.rows, self.cols, image_arr.shape[-1]), dtype=np.float32)
        for row in range(self.rows):
            for col in range(self.cols):
                y_slice, x_slice = self.receptive_slice(row, col)
                pooled[row, col] = image_arr[y_slice, x_slice].mean(axis=(0, 1))
        return pooled

    def expand_cells(self, cells: np.ndarray) -> np.ndarray:
        cell_arr = np.asarray(cells, dtype=np.float32)
        if cell_arr.ndim != 3:
            raise ValueError("cells must have shape (rows, cols, channels)")
        if cell_arr.shape[0] != self.rows or cell_arr.shape[1] != self.cols:
            raise ValueError("cell spatial shape does not match topology")
        expanded = np.zeros((self.input_height, self.input_width, cell_arr.shape[-1]), dtype=np.float32)
        for row in range(self.rows):
            for col in range(self.cols):
                y_slice, x_slice = self.receptive_slice(row, col)
                expanded[y_slice, x_slice] = cell_arr[row, col]
        return expanded

    def neighbor_mean(self, cells: np.ndarray, *, include_diagonal: bool = False) -> np.ndarray:
        cell_arr = np.asarray(cells, dtype=np.float32)
        if cell_arr.ndim != 3:
            raise ValueError("cells must have shape (rows, cols, channels)")
        if cell_arr.shape[0] != self.rows or cell_arr.shape[1] != self.cols:
            raise ValueError("cell spatial shape does not match topology")
        result = np.zeros_like(cell_arr, dtype=np.float32)
        for row in range(self.rows):
            for col in range(self.cols):
                neighbors = self.neighbor_indices(row, col, include_diagonal=include_diagonal)
                if not neighbors:
                    result[row, col] = cell_arr[row, col]
                    continue
                result[row, col] = np.mean(
                    np.stack([cell_arr[yy, xx] for yy, xx in neighbors], axis=0),
                    axis=0,
                )
        return result.astype(np.float32)


class PredictionCell:
    """Minimal ERIE prediction cell for one local state vector."""

    def __init__(self, belief: np.ndarray, logvar: np.ndarray) -> None:
        self.belief = np.asarray(belief, dtype=np.float32).copy()
        self.logvar = np.asarray(logvar, dtype=np.float32).copy()
        if self.belief.shape != self.logvar.shape:
            raise ValueError("belief and logvar must have the same shape")

    def update(
        self,
        observation: np.ndarray,
        config: PredictionCellUpdateConfig,
        *,
        gate: np.ndarray | float = 1.0,
        prior: np.ndarray | None = None,
        precision_logvar: np.ndarray | None = None,
        logvar_evidence: np.ndarray | float | None = None,
    ) -> PredictionCellUpdateResult:
        grid = PredictionCellGrid(self.belief[None, ...], self.logvar[None, ...])
        result = grid.update(
            np.asarray(observation, dtype=np.float32)[None, ...],
            config,
            gate=np.asarray(gate, dtype=np.float32),
            prior=None if prior is None else np.asarray(prior, dtype=np.float32)[None, ...],
            precision_logvar=None
            if precision_logvar is None
            else np.asarray(precision_logvar, dtype=np.float32)[None, ...],
            logvar_evidence=logvar_evidence,
        )
        self.belief[...] = grid.belief[0]
        self.logvar[...] = grid.logvar[0]
        return PredictionCellUpdateResult(
            belief=result.belief[0].copy(),
            logvar=result.logvar[0].copy(),
            error=result.error[0].copy(),
            precision=result.precision[0].copy(),
            reconstruction=result.reconstruction,
            complexity=result.complexity,
            total=result.total,
        )


class PredictionCellGrid:
    """Spatial grid of prediction cells with a shared update rule."""

    def __init__(self, belief: np.ndarray, logvar: np.ndarray) -> None:
        self.belief = np.asarray(belief, dtype=np.float32).copy()
        self.logvar = np.asarray(logvar, dtype=np.float32).copy()
        if self.belief.shape != self.logvar.shape:
            raise ValueError("belief and logvar must have the same shape")

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, logvar_init: float = 0.0) -> PredictionCellGrid:
        return cls(
            np.zeros(shape, dtype=np.float32),
            np.full(shape, float(logvar_init), dtype=np.float32),
        )

    def update(
        self,
        observation: np.ndarray,
        config: PredictionCellUpdateConfig,
        *,
        gate: np.ndarray | float = 1.0,
        prior: np.ndarray | None = None,
        precision_logvar: np.ndarray | None = None,
        logvar_evidence: np.ndarray | float | None = None,
    ) -> PredictionCellUpdateResult:
        observation_arr = np.asarray(observation, dtype=np.float32)
        if observation_arr.shape != self.belief.shape:
            raise ValueError("observation shape must match cell belief shape")

        prior_arr = self.belief.copy() if prior is None else np.asarray(prior, dtype=np.float32)
        if prior_arr.shape != self.belief.shape:
            raise ValueError("prior shape must match cell belief shape")

        precision_logvar_arr = (
            self.logvar.copy()
            if precision_logvar is None
            else np.asarray(precision_logvar, dtype=np.float32).copy()
        )
        if precision_logvar_arr.shape != self.belief.shape:
            raise ValueError("precision_logvar shape must match cell belief shape")

        gate_arr = np.asarray(gate, dtype=np.float32)
        error = (gate_arr * (observation_arr - prior_arr)).astype(np.float32)
        precision = np.clip(
            np.exp(-precision_logvar_arr),
            float(config.p_min),
            float(config.p_max),
        ).astype(np.float32)

        next_belief = np.clip(
            prior_arr + float(config.learning_rate) * precision * error,
            float(config.belief_min),
            float(config.belief_max),
        ).astype(np.float32)

        evidence_arr = gate_arr if logvar_evidence is None else np.asarray(logvar_evidence, dtype=np.float32)
        next_logvar = np.clip(
            self.logvar + float(config.logvar_drift) - float(config.evidence_logvar_gain) * evidence_arr,
            float(config.logvar_min),
            float(config.logvar_max),
        ).astype(np.float32)

        if str(config.reconstruction_logvar_source) == "updated":
            reconstruction_logvar = next_logvar
        elif str(config.reconstruction_logvar_source) == "precision":
            reconstruction_logvar = precision_logvar_arr
        else:
            raise ValueError("reconstruction_logvar_source must be 'precision' or 'updated'")

        reconstruction = float(
            np.mean(0.5 * ((error**2) * precision + np.clip(reconstruction_logvar, -6.0, 4.0)))
        )
        complexity = float(np.mean(0.5 * (((next_belief - prior_arr) ** 2) * precision)))

        self.belief[...] = next_belief
        self.logvar[...] = next_logvar

        return PredictionCellUpdateResult(
            belief=self.belief.copy(),
            logvar=self.logvar.copy(),
            error=error.astype(np.float32),
            precision=precision,
            reconstruction=reconstruction,
            complexity=complexity,
            total=float(reconstruction + complexity),
        )


class LocalPredictionCellLayer:
    """Prediction-cell layer with local receptive fields and lateral coupling."""

    def __init__(
        self,
        topology: ReceptiveFieldTopology,
        channels: int,
        *,
        logvar_init: float = 0.0,
    ) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.topology = topology
        self.cells = PredictionCellGrid.zeros(
            (topology.rows, topology.cols, int(channels)),
            logvar_init=float(logvar_init),
        )

    @classmethod
    def from_image_shape(
        cls,
        image_shape: tuple[int, int, int],
        *,
        cell_rows: int,
        cell_cols: int,
        logvar_init: float = 0.0,
    ) -> LocalPredictionCellLayer:
        if len(image_shape) != 3:
            raise ValueError("image_shape must be (height, width, channels)")
        topology = ReceptiveFieldTopology(
            input_height=int(image_shape[0]),
            input_width=int(image_shape[1]),
            rows=int(cell_rows),
            cols=int(cell_cols),
        )
        return cls(topology, int(image_shape[2]), logvar_init=logvar_init)

    @property
    def belief(self) -> np.ndarray:
        return self.cells.belief

    @property
    def logvar(self) -> np.ndarray:
        return self.cells.logvar

    def projected_belief(self) -> np.ndarray:
        return self.topology.expand_cells(self.cells.belief)

    def update_from_image(
        self,
        image: np.ndarray,
        config: PredictionCellUpdateConfig,
        *,
        gate_image: np.ndarray | None = None,
        lateral_coupling: float = 0.0,
        include_diagonal_neighbors: bool = False,
    ) -> PredictionCellUpdateResult:
        pooled_observation = self.topology.pool_image(image)
        if pooled_observation.shape != self.cells.belief.shape:
            raise ValueError("pooled image shape must match cell layer shape")

        pooled_gate: np.ndarray | float
        if gate_image is None:
            pooled_gate = 1.0
        else:
            pooled_gate = self.topology.pool_image(gate_image)
            if pooled_gate.shape[-1] == 1 and self.cells.belief.shape[-1] != 1:
                pooled_gate = np.broadcast_to(pooled_gate, self.cells.belief.shape).astype(np.float32)
            if pooled_gate.shape != self.cells.belief.shape:
                raise ValueError("pooled gate shape must match cell layer shape")

        coupling = float(np.clip(float(lateral_coupling), 0.0, 1.0))
        prior = None
        if coupling > 0.0:
            neighbor_mean = self.topology.neighbor_mean(
                self.cells.belief,
                include_diagonal=bool(include_diagonal_neighbors),
            )
            prior = ((1.0 - coupling) * self.cells.belief + coupling * neighbor_mean).astype(np.float32)

        return self.cells.update(
            pooled_observation,
            config,
            gate=pooled_gate,
            prior=prior,
            logvar_evidence=pooled_gate,
        )
