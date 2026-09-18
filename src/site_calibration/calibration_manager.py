"""
Calibration Manager Orchestration Submodule.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from src.site_calibration.models import (
    SeismographReading,
    AttenuationParameters,
    CalibrationResult,
    PPVPrediction,
    IngestionReport,
    FitQualityReport,
)
from src.site_calibration.data_ingestion import load_seismograph_data
from src.site_calibration.gsi_estimator import GSIEstimator
from src.site_calibration.attenuation_fitter import AttenuationFitter
from src.site_calibration.gsi_correction import predict_gsi_modified_ppv

logger = logging.getLogger(__name__)


class CalibrationManager:
    """
    Main orchestration class for site-specific attenuation calibration.
    Manages readings ingestion, fitting, predictions, parameter storage, and fit quality reporting.
    """

    def __init__(self, db_file_path: Optional[str] = "data/processed/site_calibration_db.json"):
        self.db_file_path = db_file_path
        self.readings_db: List[Dict[str, Any]] = []
        self.calibrated_params: Dict[str, AttenuationParameters] = {}
        self.gsi_estimator = GSIEstimator()
        self.fitter = AttenuationFitter()

        if self.db_file_path and os.path.exists(self.db_file_path):
            self.load_state()

    def _get_site_key(self, site_id: str, rock_type: Optional[str] = None) -> str:
        rt = (rock_type or "Kimberlite").strip().title()
        return f"{site_id.strip().upper()}::{rt}"

    def save_state(self) -> None:
        """Saves current readings and parameter database to JSON file."""
        if not self.db_file_path:
            return
        os.makedirs(os.path.dirname(self.db_file_path), exist_ok=True)
        payload = {
            "readings": self.readings_db,
            "calibrated_params": {k: v.model_dump() for k, v in self.calibrated_params.items()},
        }
        with open(self.db_file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_state(self) -> None:
        """Loads readings and parameter database from JSON file."""
        try:
            with open(self.db_file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                self.readings_db = payload.get("readings", [])
                params_dict = payload.get("calibrated_params", {})
                self.calibrated_params = {
                    k: AttenuationParameters(**v) for k, v in params_dict.items()
                }
        except Exception as e:
            logger.warning(f"Error loading calibration manager state: {e}")

    def add_reading(self, reading: SeismographReading) -> None:
        """Adds a single SeismographReading to the database."""
        # Deduplicate
        dup = any(
            r["blast_id"] == reading.blast_id and abs(r["distance_m"] - reading.distance_m) < 0.1
            for r in self.readings_db
        )
        if not dup:
            self.readings_db.append(reading.model_dump())
            self.save_state()

    def add_readings_from_file(self, path: str) -> IngestionReport:
        """In-gests readings from CSV or JSON file."""
        df_accepted, report = load_seismograph_data(path)
        if not df_accepted.empty:
            for rec in df_accepted.to_dict(orient="records"):
                self.add_reading(SeismographReading(**rec))
        return report

    def list_sites(self) -> List[str]:
        """Lists all unique site_ids present in the readings database or calibrated parameters."""
        sites = set(r["site_id"] for r in self.readings_db if "site_id" in r)
        for key in self.calibrated_params.keys():
            site_part = key.split("::")[0]
            sites.add(site_part)
        return sorted(list(sites))

    def fit_site(self, site_id: str, rock_type: str = "Kimberlite") -> CalibrationResult:
        """
        Fits K, B, and GSI parameters for a specified site_id and rock_type.
        """
        target_key = self._get_site_key(site_id, rock_type)
        site_readings = [
            r for r in self.readings_db
            if r.get("site_id", "").strip().upper() == site_id.strip().upper()
            and r.get("rock_type", "Kimberlite").strip().title() == rock_type.strip().title()
        ]

        if not site_readings:
            # Check for fallback readings across all rock types for site_id
            site_readings = [
                r for r in self.readings_db
                if r.get("site_id", "").strip().upper() == site_id.strip().upper()
            ]

        if not site_readings:
            return CalibrationResult(
                site_id=site_id,
                rock_type=rock_type,
                parameters=None,
                fit_quality="REJECTED",
                warnings=[f"No seismograph readings found for site '{site_id}'"],
                recommendation="Ingest seismograph readings before calling fit_site().",
            )

        df_site = pd.DataFrame(site_readings)
        distances = df_site["distance_m"].values
        ppv_col = "ppv_mm_s" if "ppv_mm_s" in df_site.columns else "ppv_mms"
        ppvs = df_site[ppv_col].values
        charges = df_site["charge_per_delay_kg"].values
        gsis = df_site["gsi_of_transmission_strata"].dropna().values if "gsi_of_transmission_strata" in df_site and not df_site["gsi_of_transmission_strata"].isnull().all() else np.array([])

        if len(gsis) >= 10 and np.ptp(gsis) >= 10.0:
            cal_res = self.fitter.fit_gsi_modified(distances, ppvs, charges, gsis, rock_type=rock_type)
        else:
            cal_res = self.fitter.fit_usbm(distances, ppvs, charges, rock_type=rock_type)

        cal_res.site_id = site_id
        if cal_res.parameters is not None:
            self.calibrated_params[target_key] = cal_res.parameters
            self.save_state()

        return cal_res

    def get_parameters(self, site_id: str, rock_type: str = "Kimberlite") -> Optional[AttenuationParameters]:
        """Retrieves calibrated parameters for site_id and rock_type."""
        target_key = self._get_site_key(site_id, rock_type)
        if target_key in self.calibrated_params:
            return self.calibrated_params[target_key]

        # Search for any site_id key regardless of rock_type
        for k, v in self.calibrated_params.items():
            if k.startswith(f"{site_id.strip().upper()}::"):
                return v

        return None

    def predict_ppv(
        self,
        site_id: str,
        distance_m: float,
        charge_per_delay_kg: float,
        gsi: Optional[float] = None,
        rock_type: str = "Kimberlite",
    ) -> PPVPrediction:
        """
        Predicts PPV using calibrated site parameters or generic USBM fallback.
        """
        params = self.get_parameters(site_id, rock_type=rock_type)

        # Generic USBM Fallback (K=1140, B=1.6) if site is uncalibrated
        if params is None:
            sd = distance_m / (charge_per_delay_kg ** 0.5)
            generic_ppv = 1140.0 * (sd ** (-1.6))
            rel_margin = 0.35
            return PPVPrediction(
                ppv_mm_s=round(generic_ppv, 2),
                lower_95=round(max(0.01, generic_ppv * (1.0 - rel_margin)), 2),
                upper_95=round(generic_ppv * (1.0 + rel_margin), 2),
                method="USBM_GENERIC_FALLBACK",
                parameters_used={"K": 1140.0, "B": 1.6, "note": "Generic USBM fallback parameters used"},
            )

        return predict_gsi_modified_ppv(distance_m, charge_per_delay_kg, params, gsi=gsi)

    def get_calibration_quality(self, site_id: str, rock_type: str = "Kimberlite") -> FitQualityReport:
        """Returns quality metrics summary for site calibration."""
        params = self.get_parameters(site_id, rock_type=rock_type)
        if params is None:
            return FitQualityReport(
                site_id=site_id,
                rock_type=rock_type,
                r_squared=0.0,
                rmse=0.0,
                sample_count=0,
                is_statistically_sound=False,
                status="UNCALIBRATED - No fitted parameters found",
            )

        is_sound = (params.r_squared >= 0.70) and (params.sample_count >= 10)
        status = "EXCELLENT" if params.r_squared >= 0.85 else ("ACCEPTABLE" if is_sound else "POOR")

        return FitQualityReport(
            site_id=site_id,
            rock_type=rock_type,
            r_squared=params.r_squared,
            rmse=params.rmse,
            sample_count=params.sample_count,
            is_statistically_sound=is_sound,
            status=status,
        )
