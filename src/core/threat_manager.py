import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime
from collections import deque

import cv2
import requests

from src.config import LOG_COOLDOWN

from src.utils.logger import setup_logger

class ThreatManager:
    def __init__(self, alerts_enabled=False):
        self.logger = setup_logger()
        self.alerts_enabled = alerts_enabled
        self.threat_logs = {}
        self._enqueue_failures = {}
        self.log_cooldown = LOG_COOLDOWN
        self.csv_writer = None
        self.log_file = None
        self._alert_queue = queue.Queue(maxsize=32)
        self._incident_lock = threading.Lock()
        self._dedup_lock = threading.Lock()
        self._frame_enqueue_lock = threading.Lock()
        self._incident_history = deque(maxlen=100)
        self.api_healthy = None
        self.api_last_latency_ms = None
        self.api_prepare_ms = None
        self.api_last_error = None
        self.last_enqueue_ms = 0.0
        self.frame_enqueue_ms = 0.0
        self._alert_thread = None
        if self.alerts_enabled:
            self._alert_thread = threading.Thread(
                target=self._deliver_alerts,
                name="smartvision-alert-delivery",
                daemon=True,
            )
            self._alert_thread.start()

    def determine_threat(self, action_label, visible_objects, duration=0.0, is_unattended_bag=False):
        """
        Determines the threat level based on the following matrix:
        - Critical: (Shooting/Violence) + (Gun/Knife) OR (Shooting/Violence) + ATM
        - High:     (Any Action) + (Gun/Knife)
        - Warning:  (Shooting/Violence) + No Weapon
        - Warning:  (Normal) + ATM + Duration > LOITERING_THRESHOLD (Loitering)
        - Warning:  Unattended Baggage
        - Safe:     Otherwise
        """
        threat_level = "SAFE"
        box_color = (0, 255, 0)  # Green

        from src.config import ATM_CLASS_ID, GUN_CLASS_IDS, KNIFE_CLASS_ID, ATM_LOITERING_THRESHOLD

        has_gun = any(w in GUN_CLASS_IDS for w in visible_objects)
        has_knife = KNIFE_CLASS_ID in visible_objects
        has_atm = ATM_CLASS_ID in visible_objects
        has_weapon = has_gun or has_knife

        is_violent_action = action_label in ["shooting", "violence"]

        # 1. CRITICAL
        if is_violent_action and (has_weapon or has_atm):
            if has_gun:
                threat_level = "CRITICAL: SHOOTER"
            elif has_knife:
                threat_level = "CRITICAL: KNIFE ATTACK"
            elif has_atm:
                 threat_level = "CRITICAL: ATM ROBBERY"
            box_color = (0, 0, 255)  # Red

        # 2. HIGH (Weapon Visible)
        elif has_weapon:
            threat_level = "HIGH: WEAPON DETECTED"
            box_color = (0, 165, 255)  # Orange

        # 3. WARNING (Violence w/o Weapon)
        elif is_violent_action:
            if action_label == "shooting":
                threat_level = "WARNING: SUSPICIOUS STANCE"
            else:
                threat_level = "WARNING: FIGHTING"
            box_color = (0, 255, 255)  # Yellow
        
        # 4. WARNING (Loitering at ATM)
        elif has_atm and duration > ATM_LOITERING_THRESHOLD:
            threat_level = "WARNING: LOITERING AT ATM"
            box_color = (0, 255, 255)  # Yellow
        
        # 5. WARNING (Unattended Baggage - Global)
        elif is_unattended_bag:
            threat_level = "WARNING: UNATTENDED BAGGAGE"
            box_color = (0, 255, 255) # Yellow
            
        return threat_level, box_color

    def begin_frame(self):
        with self._frame_enqueue_lock:
            self.frame_enqueue_ms = 0.0

    def log_threat(self, track_id, source_name, action_label, action_prob, threat_level, frame_id=0, detections=None, spatial_context=None, frame=None):
        """
        Logs threat to console/file and conditionally sends to server.
        """
        current_time = time.time()

        if threat_level == "SAFE":
            return
        if not self.alerts_enabled:
            return

        detections = detections or []
        spatial_context = spatial_context or {}
        event_key = (
            str(source_name),
            int(track_id),
            str(threat_level),
            str(action_label).lower(),
        )
        with self._dedup_lock:
            last_time = self.threat_logs.get(event_key, 0.0)
            if current_time - last_time < self.log_cooldown:
                return
            last_failure = self._enqueue_failures.get(event_key, 0.0)
            if current_time - last_failure < 1.0:
                return

            from src.config import SERVER_URL
            if not SERVER_URL:
                self.logger.error("Alert was not sent because SERVER_URL is empty.")
                return

            action_name = str(action_label).lower()
            event_type = (
                "object_detection"
                if action_name.endswith("detection") or action_name == "unattended"
                else "action_recognition"
            )
            payload = {
                "camera_id": source_name,
                "threat_level": threat_level,
                "label": action_label.upper(),
                "confidence": float(action_prob),
                "data": {
                    "frame_id": frame_id,
                    "event_type": event_type,
                    "detections": detections,
                    "action_classification": {
                        "class": action_label,
                        "confidence": float(action_prob),
                        "track_id": track_id,
                    },
                    "spatial_context": spatial_context,
                },
            }

            incident = self._send_alert_async(
                SERVER_URL,
                payload,
                frame=frame
                if "CRITICAL" in threat_level or "HIGH" in threat_level
                else None,
            )
            if incident is not None:
                self.threat_logs[event_key] = current_time
                self._enqueue_failures.pop(event_key, None)
                self._prune_deduplication_cache(current_time)
            else:
                self._enqueue_failures[event_key] = current_time
                self._prune_deduplication_cache(current_time)

    def _send_alert_async(self, url, payload, image_data=None, frame=None):
        enqueue_started = time.perf_counter()
        incident = {
            "id": uuid.uuid4().hex,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event_type": payload.get("data", {}).get(
                "event_type",
                "object_detection",
            ),
            "label": payload.get("label", "Unknown"),
            "confidence": float(payload.get("confidence", 0.0)),
            "track_id": payload.get("data", {})
            .get("action_classification", {})
            .get("track_id"),
            "threat_level": payload.get("threat_level", "Unknown"),
            "status": "queued",
            "snapshot": None,
            "api_elapsed_ms": None,
            "error": None,
        }
        item = {
            "incident": incident,
            "url": url,
            "payload": payload,
            "image_data": image_data,
            "frame": frame.copy() if frame is not None else None,
        }
        with self._incident_lock:
            self._incident_history.appendleft(incident)

        queued = False
        try:
            self._alert_queue.put_nowait(item)
            queued = True
        except queue.Full:
            self._update_incident(
                incident,
                status="failed: alert queue is full",
                error=(
                    "Bounded alert queue is full; this incident was not "
                    "delivered."
                ),
            )
        finally:
            elapsed_ms = (
                time.perf_counter() - enqueue_started
            ) * 1000
            self.last_enqueue_ms = elapsed_ms
            with self._frame_enqueue_lock:
                self.frame_enqueue_ms += elapsed_ms
        return incident if queued else None

    def _deliver_alerts(self):
        with requests.Session() as session:
            while True:
                item = self._alert_queue.get()
                if item is None:
                    self._alert_queue.task_done()
                    return
                try:
                    incident = item["incident"]
                    image_data = item["image_data"]
                    frame = item["frame"]
                    prepare_started = time.perf_counter()
                    if image_data is None and frame is not None:
                        success, encoded = cv2.imencode(".jpg", frame)
                        if not success:
                            raise RuntimeError(
                                "OpenCV could not encode the alert snapshot."
                            )
                        image_data = encoded.tobytes()
                    self._update_incident(incident, snapshot=image_data)
                    with self._incident_lock:
                        self.api_prepare_ms = (
                            time.perf_counter() - prepare_started
                        ) * 1000

                    files = (
                        {"file": ("alert.jpg", image_data, "image/jpeg")}
                        if image_data
                        else None
                    )
                    started_at = time.perf_counter()
                    try:
                        response = session.post(
                            item["url"],
                            data={
                                "incident_data": json.dumps(item["payload"])
                            },
                            files=files,
                            timeout=(1.0, 3.0),
                        )
                        response.raise_for_status()
                    except requests.RequestException as exc:
                        elapsed_ms = (time.perf_counter() - started_at) * 1000
                        self._update_incident(
                            incident,
                            status="failed",
                            error=str(exc),
                            api_elapsed_ms=elapsed_ms,
                        )
                        with self._incident_lock:
                            self.api_healthy = False
                            self.api_last_latency_ms = elapsed_ms
                            self.api_last_error = str(exc)
                        self.logger.error(
                            "Alert %s delivery failed after %.1f ms: %s",
                            incident["id"],
                            elapsed_ms,
                            exc,
                        )
                        continue

                    elapsed_ms = (time.perf_counter() - started_at) * 1000
                    self._update_incident(
                        incident,
                        status=f"delivered (HTTP {response.status_code})",
                        api_elapsed_ms=elapsed_ms,
                    )
                    with self._incident_lock:
                        self.api_healthy = True
                        self.api_last_latency_ms = elapsed_ms
                        self.api_last_error = None
                    message = (
                        f"THREAT DETECTED | Source: "
                        f"{item['payload']['camera_id']} | "
                        f"Level: {incident['threat_level']} | "
                        f"Event: {incident['label']}"
                    )
                    if "CRITICAL" in incident["threat_level"]:
                        logging.critical(message)
                    elif "WARNING" in incident["threat_level"]:
                        logging.warning(message)
                    else:
                        logging.error(message)

                    if self.csv_writer is not None and self.log_file is not None:
                        self.csv_writer.writerow([
                            incident["created_at"],
                            item["payload"]["camera_id"],
                            incident["track_id"],
                            incident["label"],
                            f"{incident['confidence']:.2f}",
                            incident["threat_level"],
                        ])
                        self.log_file.flush()
                except Exception as exc:
                    incident = item["incident"]
                    self._update_incident(
                        incident,
                        status="failed",
                        error=str(exc),
                    )
                    self.logger.exception(
                        "Alert %s processing failed.",
                        incident["id"],
                    )
                finally:
                    self._alert_queue.task_done()

    def _prune_deduplication_cache(self, current_time):
        if len(self.threat_logs) + len(self._enqueue_failures) <= 4096:
            return
        expiry = current_time - max(self.log_cooldown * 2, 60.0)
        self.threat_logs = {
            key: last_time
            for key, last_time in self.threat_logs.items()
            if last_time >= expiry
        }
        self._enqueue_failures = {
            key: last_time
            for key, last_time in self._enqueue_failures.items()
            if last_time >= expiry
        }

    def get_incident_history(self, limit=20):
        with self._incident_lock:
            return [
                dict(incident)
                for incident in list(self._incident_history)[:limit]
            ]

    def _update_incident(self, incident, **fields):
        with self._incident_lock:
            incident.update(fields)

    def get_delivery_health(self):
        with self._incident_lock:
            return {
                "healthy": self.api_healthy,
                "last_latency_ms": self.api_last_latency_ms,
                "prepare_ms": self.api_prepare_ms,
                "last_error": self.api_last_error,
                "queued": self._alert_queue.qsize(),
            }

    def close(self, timeout=1.0):
        if self._alert_thread is None:
            return
        try:
            self._alert_queue.put_nowait(None)
        except queue.Full:
            self.logger.error(
                "Could not queue alert-worker shutdown; pending incidents "
                "remain queued."
            )
            return
        self._alert_thread.join(timeout=timeout)