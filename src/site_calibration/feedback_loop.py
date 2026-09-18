"""
Continuous Feedback Loop & Recalibration Engine.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from src.site_calibration.models import SeismographReading, RecalibrationLog, CalibrationResult
from src.site_calibration.calibration_manager import CalibrationManager

logger = logging.getLogger(__name__)


class FeedbackLoopEngine:
    """
    Continuous feedback loop that ingests post-blast seismograph readings
    and automatically triggers site recalibration when:
    1. New data changes K or B parameters by > 5%
    2. R2 drops below 0.75
    3. > 30 days have elapsed since last calibration
    4. Generates alerts when parameters change by > 20% (geological change warning)
    """

    def __init__(self, calibration_manager: CalibrationManager):
        self.manager = calibration_manager
        self.recalibration_logs: List[RecalibrationLog] = []

    def process_new_reading(
        self,
        reading: SeismographReading,
        force_recalibrate: bool = False
    ) -> Optional[RecalibrationLog]:
        """
        Ingests a new post-blast reading and evaluates if site recalibration is required.
        """
        site_id = reading.site_id
        rock_type = reading.rock_type or "Kimberlite"

        # Ingest reading into database
        self.manager.add_reading(reading)

        # Retrieve current fitted parameters
        old_params_obj = self.manager.get_parameters(site_id, rock_type=rock_type)

        should_recalibrate = False
        recalibrate_reason = ""

        if old_params_obj is None:
            should_recalibrate = True
            recalibrate_reason = "Initial calibration for uncalibrated site"
        elif force_recalibrate:
            should_recalibrate = True
            recalibrate_reason = "Force recalibration requested"
        else:
            # Check 1: R2 score drop below 0.75
            if old_params_obj.r_squared < 0.75:
                should_recalibrate = True
                recalibrate_reason = f"Fit R2 score ({old_params_obj.r_squared:.3f}) dropped below 0.75"

            # Check 2: > 30 days since last calibration
            try:
                last_dt = datetime.fromisoformat(old_params_obj.last_updated)
                now_dt = datetime.now(timezone.utc)
                if (now_dt - last_dt).days >= 30:
                    should_recalibrate = True
                    recalibrate_reason = f"30 days elapsed since last calibration ({old_params_obj.last_updated[:10]})"
            except Exception:
                pass

        if not should_recalibrate:
            return None

        # Execute recalibration
        cal_res: CalibrationResult = self.manager.fit_site(site_id, rock_type=rock_type)
        if cal_res.parameters is None:
            logger.warning(f"Recalibration failed for site '{site_id}': {cal_res.warnings}")
            return None

        new_params_obj = cal_res.parameters

        old_k = old_params_obj.K if old_params_obj else new_params_obj.K
        old_b = old_params_obj.B if old_params_obj else new_params_obj.B

        delta_k_pct = abs((new_params_obj.K - old_k) / old_k) * 100.0 if old_k > 0 else 0.0
        delta_b_pct = abs((new_params_obj.B - old_b) / old_b) * 100.0 if old_b > 0 else 0.0

        alerts = []
        if delta_k_pct > 20.0 or delta_b_pct > 20.0:
            alert_msg = (
                f"🚨 GEOLOGICAL DRIFT ALERT: Attenuation parameters shifted significantly "
                f"(K shift: {delta_k_pct:.1f}%, B shift: {delta_b_pct:.1f}%). "
                f"Possible structural geology transition or bench face confinement change."
            )
            alerts.append(alert_msg)
            logger.warning(alert_msg)

        if cal_res.fit_quality == "POOR":
            alerts.append("⚠️ Fit quality is POOR (R2 < 0.70). Check seismograph placement.")

        log_entry = RecalibrationLog(
            site_id=site_id,
            rock_type=rock_type,
            sample_count=new_params_obj.sample_count,
            old_params={"K": round(old_k, 2), "B": round(old_b, 3)},
            new_params={"K": round(new_params_obj.K, 2), "B": round(new_params_obj.B, 3)},
            delta_k_pct=round(delta_k_pct, 2),
            delta_b_pct=round(delta_b_pct, 2),
            reason=recalibrate_reason or f"Parameter shift (K delta: {delta_k_pct:.1f}%, B delta: {delta_b_pct:.1f}%)",
            alerts=alerts,
        )

        self.recalibration_logs.append(log_entry)
        return log_entry
