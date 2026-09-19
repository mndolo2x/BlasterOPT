"""
Measure-While-Drilling (MWD) Ingestion & Telemetry Module for BlastOpt Botswana.

Implements real-time MWD telemetry ingestion via MQTT / OPC-UA and time-series database storage
(InfluxDB / local queue fallback) for closed-loop blast design and real-time charging plan adaptation.
"""

import os
import json
import time
import queue
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Union

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    HAS_INFLUX = True
except ImportError:
    HAS_INFLUX = False

logger = logging.getLogger(__name__)

# Thread-safe buffer for ingested MWD samples
MWD_BUFFER: queue.Queue = queue.Queue(maxsize=1000)
MWD_HISTORY: list = []


def parse_mwd_message(message: Union[str, bytes, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parses a raw MQTT message or payload into a structured dictionary of MWD parameters.

    Measure-While-Drilling (MWD) Domain Context:
    -------------------------------------------
    MWD sensors mounted on production drill rigs (e.g., Epiroc, Sandvik) measure real-time
    drilling mechanics:
    - Penetration Rate (ROP, m/hr): High ROP indicates soft rock/voids; low ROP indicates hard rock.
    - Torque (N·m) & Weight-on-Bit (WOB, kg): High torque/WOB indicates fractured or abrasive rock.
    - Air Pressure (bar): Fluctuations indicate rock mass fracturing and air loss.
    - Specific Energy of Drilling (SED): Computed from ROP, WOB, RPM, and Torque.

    Real-time charging adaptation: If MWD detects soft rock or voids at depth, the charging plan
    automatically reduces explosive bulk density or deck charges to prevent flyrock and excessive vibration.

    Parameters:
    -----------
    message : Union[str, bytes, Dict[str, Any]]
        Raw payload received from MQTT broker topic.

    Returns:
    --------
    Dict[str, Any]
        Structured dictionary containing parsed MWD parameters.
    """
    if isinstance(message, (str, bytes)):
        try:
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            data = json.loads(message)
        except Exception:
            data = {}
    elif isinstance(message, dict):
        data = message.copy()
    else:
        data = {}

    parsed = {
        "hole_id": str(data.get("hole_id", "HOLE_001")),
        "depth_m": float(data.get("depth_m", 12.0)),
        "penetration_rate_m_hr": float(data.get("penetration_rate_m_hr", data.get("rop", 35.0))),
        "torque_nm": float(data.get("torque_nm", 1200.0)),
        "weight_on_bit_kg": float(data.get("weight_on_bit_kg", data.get("wob", 8500.0))),
        "rpm": float(data.get("rpm", 110.0)),
        "air_pressure_bar": float(data.get("air_pressure_bar", 6.5)),
        "vibration_mm_s": float(data.get("vibration_mm_s", 2.5)),
        "rock_type": str(data.get("rock_type", "Kimberlite_Hard")),
        "timestamp": str(data.get("timestamp", datetime.now().isoformat())),
    }

    # Derived Specific Energy of Drilling (SED in MJ/m3) estimate
    wob_n = parsed["weight_on_bit_kg"] * 9.81
    torque_nm = parsed["torque_nm"]
    rpm = parsed["rpm"]
    rop_m_s = max(parsed["penetration_rate_m_hr"] / 3600.0, 0.0001)
    hole_radius_m = 0.125  # Standard 250mm diameter

    area_m2 = np_pi = 3.14159 * (hole_radius_m ** 2)
    e_thrust = wob_n / area_m2
    e_rot = (2.0 * 3.14159 * rpm * torque_nm) / (area_m2 * rop_m_s * 60.0)
    parsed["specific_energy_mj_m3"] = round((e_thrust + e_rot) / 1e6, 2)

    return parsed


from src.config import require_real_data, get_demo_mode

def connect_to_mqtt(
    broker_address: str = "localhost",
    topic: str = "blastopt/mwd/telemetry",
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    port: int = 1883,
    keepalive: int = 60,
) -> Optional[Any]:
    """
    Connects to an MQTT broker and subscribes to the MWD data topic in a non-blocking thread.

    Parameters:
    -----------
    broker_address : str, default="localhost"
        MQTT broker IP address or hostname.
    topic : str, default="blastopt/mwd/telemetry"
        MQTT topic subscription path.
    callback : Callable[[Dict[str, Any]], None], optional
        Optional callback function invoked when a new parsed MWD sample is received.
    port : int, default=1883
        MQTT broker port.
    keepalive : int, default=60
        MQTT keepalive timeout in seconds.

    Returns:
    --------
    Optional[mqtt.Client]
        Connected paho-mqtt Client instance, or None if connection failed gracefully.
    """
    if not HAS_MQTT:
        logger.warning("paho-mqtt library not installed. MQTT client disabled.")
        if not get_demo_mode():
            require_real_data("MQTT MWD Telemetry Broker")
        return None

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"Successfully connected to MQTT broker at {broker_address}:{port}")
            client.subscribe(topic)
        else:
            logger.error(f"MQTT connection failed with return code {rc}")

    def on_message(client, userdata, msg):
        try:
            payload_str = msg.payload.decode("utf-8")
            parsed_sample = parse_mwd_message(payload_str)

            # Store in thread-safe queue and history list
            if not MWD_BUFFER.full():
                MWD_BUFFER.put(parsed_sample)
            MWD_HISTORY.append(parsed_sample)
            if len(MWD_HISTORY) > 500:
                MWD_HISTORY.pop(0)

            if callback is not None:
                callback(parsed_sample)
        except Exception as e:
            logger.error(f"Error handling MWD MQTT message: {e}")

    try:
        if "invalid" in broker_address or "demo" in broker_address:
            raise ConnectionError(f"Invalid or demo MQTT broker endpoint: '{broker_address}'")

        # Compatibility handling for paho-mqtt v1 vs v2 CallbackAPIVersion
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="BlastOpt_MWD_Ingest")
        except Exception:
            client = mqtt.Client(client_id="BlastOpt_MWD_Ingest")

        client.on_connect = on_connect
        client.on_message = on_message

        client.connect_async(broker_address, port=port, keepalive=keepalive)
        client.loop_start()
        return client

    except Exception as e:
        logger.error(f"Failed to connect to MQTT broker ({broker_address}:{port}): {e}")
        if not get_demo_mode():
            require_real_data("MQTT MWD Telemetry Broker")
        return None


def ingest_mwd_stream(
    influx_url: Optional[str] = None,
    influx_token: Optional[str] = None,
    org: str = "blastopt",
    bucket: str = "mwd_telemetry",
    stop_event: Optional[threading.Event] = None,
) -> None:
    """
    Background worker loop that continuously ingests MWD samples from MWD_BUFFER
    and writes them to an InfluxDB time-series database or local storage.

    Parameters:
    -----------
    influx_url : str, optional
        InfluxDB server endpoint URL (e.g. "http://localhost:8086").
    influx_token : str, optional
        InfluxDB API authentication token.
    org : str, default="blastopt"
        InfluxDB organization name.
    bucket : str, default="mwd_telemetry"
        InfluxDB target bucket name.
    stop_event : threading.Event, optional
        Event flag to signal loop termination.
    """
    client = None
    write_api = None

    if HAS_INFLUX and influx_url and influx_token:
        try:
            client = InfluxDBClient(url=influx_url, token=influx_token, org=org)
            write_api = client.write_api(write_options=SYNCHRONOUS)
        except Exception as e:
            logger.warning(f"InfluxDB connection disabled: {e}")

    logger.info("Starting MWD stream background ingestion worker loop...")

    while stop_event is None or not stop_event.is_set():
        try:
            sample = MWD_BUFFER.get(timeout=1.0)

            # Write to InfluxDB if available
            if write_api is not None:
                try:
                    p = (
                        Point("mwd_telemetry")
                        .tag("hole_id", sample["hole_id"])
                        .tag("rock_type", sample["rock_type"])
                        .field("depth_m", sample["depth_m"])
                        .field("penetration_rate_m_hr", sample["penetration_rate_m_hr"])
                        .field("torque_nm", sample["torque_nm"])
                        .field("weight_on_bit_kg", sample["weight_on_bit_kg"])
                        .field("specific_energy_mj_m3", sample.get("specific_energy_mj_m3", 0.0))
                    )
                    write_api.write(bucket=bucket, org=org, record=p)
                except Exception as write_err:
                    logger.error(f"Error writing MWD sample to InfluxDB: {write_err}")

            MWD_BUFFER.task_done()

        except queue.Empty:
            continue
        except Exception as err:
            logger.error(f"Error in MWD ingestion worker: {err}")
            time.sleep(0.5)

    if client is not None:
        client.close()
