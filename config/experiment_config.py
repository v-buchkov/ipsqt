from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path

from ipsqt.config.base_experiment_config import BaseExperimentConfig


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    # Folders
    PATH_INPUT: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "input",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_OUTPUT: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "output",
        metadata={"docs": "Relative path to data folder"},
    )

    SAVE_PATH: Path = field(
        default=Path(__file__).resolve().parents[2] / "data" / "run",
        metadata={"docs": "Relative path to data folder"},
    )

    # Filenames
    COUNTRY_DATA_FILENAME: str = field(default="CountryData.xlsx", metadata={"docs": "Initial data"})

    DF_FILENAME: str = field(default="data_df.csv", metadata={"docs": "Initial data"})

    # Experiment Settings
    START_DATE: pd.Timestamp | None = field(
        default=None,
        metadata={"docs": "Date to start training"},
    )

    END_DATE: pd.Timestamp | None = field(
        default=None,
        metadata={"docs": "Date to end train"},
    )

    REBALANCE_FREQ_DAYS: int | None = field(
        default=21,
        metadata={"docs": "Frequency of rebalancing"},
    )

    MIN_ROLLING_PERIODS: int = field(
        default=252 * 5 + 21 * 2,
        metadata={"docs": "Number of minimum rebalance periods to run the strategy"},
    )

    # Universe Setting
    ASSET_UNIVERSE: tuple[str] = field(
        default=(
            "Australia",
            "Austria",
            "Belgium",
            "Brazil",
            "Canada",
            "Chile",
            "China",
            "France",
            "Germany",
            "Hong Kong",
            "India",
            "Italy",
            "Japan",
            "Malaysia",
            "Mexico",
            "Netherlands",
            "Russia",
            "Singapore",
            "South Africa",
            "South Korea",
            "Spain",
            "Sweden",
            "Switzerland",
            "Taiwan",
            "Thailand",
            "Turkey",
            "UK",
            "US",
        ),
        metadata={"docs": "Tradeable assets tuple"},
    )

    HEDGING_ASSETS: tuple[str] = field(
        default=(),
        metadata={"docs": "Tradeable assets tuple"},
    )

    FACTORS: tuple[str] = field(
        default=("MKT_Factor",),
        metadata={"docs": "Tradeable factors tuple"},
    )

    RF_NAME: str = field(
        default="RF",
        metadata={"docs": "Risk-Free rate column name"},
    )

    MKT_NAME: str = field(
        default="MKT_Factor",
        metadata={"docs": "Market index column name"},
    )

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}:"
        for key, value in self.__dict__.items():
            string += f"\n* {key} = {value}"
        return string

    def __repr__(self) -> str:
        return self.__str__()
