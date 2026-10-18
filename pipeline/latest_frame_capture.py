"""Background video capture with bounded, low-latency frame storage."""

import queue
import threading
import time
from collections import deque

import cv2


class LatestFrameCapture:
    """Continuously capture video, keeping only the newest webcam frame."""

    def __init__(
        self,
        source,
        file_queue_size=2,
        width=1280,
        height=720,
        reconnect_initial_delay=0.25,
        reconnect_max_delay=2.0,
        failed_reads_before_reconnect=3,
    ):
        self.source = source
        self.is_webcam = isinstance(source, int)
        self.width = width
        self.height = height
        self.reconnect_initial_delay = reconnect_initial_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.failed_reads_before_reconnect = failed_reads_before_reconnect
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._lock = threading.Lock()
        self._file_frames = queue.Queue(maxsize=file_queue_size)
        self._thread = threading.Thread(
            target=self._capture,
            name="smartvision-camera-capture",
            daemon=True,
        )
        self._latest_frame = None
        self._sequence = 0
        self._captured_at = None
        self._capture_times = deque(maxlen=30)
        self.dropped_frames = 0
        self.buffer_setting_supported = None
        self.source_fps = 0.0
        self.error = None
        self.opened = False
        self._thread.start()

    def _open_capture(self):
        if self.is_webcam:
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self.source)

        if not capture.isOpened():
            capture.release()
            self.opened = False
            self.error = (
                f"Could not open camera {self.source}; retrying."
                if self.is_webcam
                else f"Video source could not be opened: {self.source}"
            )
            return None

        if self.is_webcam:
            self.buffer_setting_supported = bool(
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            with self._lock:
                self._capture_times.clear()
        else:
            self.source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if self.source_fps <= 0:
                self.source_fps = 20.0

        self.opened = True
        self.error = None
        self._ready_event.set()
        return capture

    def _clear_latest_frame(self):
        with self._lock:
            self._latest_frame = None
            self._captured_at = None
            self._capture_times.clear()
        self.opened = False

    def _publish(self, frame):
        captured_at = time.perf_counter()
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            self._capture_times.append(captured_at)
            if self.is_webcam:
                self._latest_frame = frame
                self._captured_at = captured_at
                self._first_frame_event.set()
        return sequence, captured_at

    def _capture(self):
        reconnect_delay = self.reconnect_initial_delay
        next_file_frame_at = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                try:
                    capture = self._open_capture()
                except cv2.error as exc:
                    if not self.is_webcam:
                        raise
                    self.error = f"Camera open failed; retrying: {exc}"
                    self._stop_event.wait(reconnect_delay)
                    reconnect_delay = min(
                        reconnect_delay * 2,
                        self.reconnect_max_delay,
                    )
                    continue

                if capture is None:
                    if not self.is_webcam:
                        self._ready_event.set()
                        return
                    self._stop_event.wait(reconnect_delay)
                    reconnect_delay = min(
                        reconnect_delay * 2,
                        self.reconnect_max_delay,
                    )
                    continue

                reconnect_delay = self.reconnect_initial_delay
                consecutive_failures = 0
                reconnect = False
                try:
                    while not self._stop_event.is_set():
                        try:
                            success, frame = capture.read()
                        except cv2.error as exc:
                            if not self.is_webcam:
                                raise
                            self.error = f"Camera read failed; reconnecting: {exc}"
                            reconnect = True
                            break

                        if not success:
                            if not self.is_webcam:
                                self.error = "Video finished."
                                return
                            consecutive_failures += 1
                            if (
                                consecutive_failures
                                < self.failed_reads_before_reconnect
                            ):
                                self._stop_event.wait(0.02)
                                continue
                            self.error = (
                                "Camera stopped delivering frames; "
                                "attempting to reconnect."
                            )
                            reconnect = True
                            break

                        consecutive_failures = 0
                        sequence, captured_at = self._publish(frame)
                        if not self.is_webcam:
                            while not self._stop_event.is_set():
                                try:
                                    self._file_frames.put(
                                        (frame, sequence, captured_at),
                                        timeout=0.1,
                                    )
                                    break
                                except queue.Full:
                                    continue

                            next_file_frame_at += 1.0 / self.source_fps
                            delay = next_file_frame_at - time.perf_counter()
                            if delay > 0:
                                self._stop_event.wait(delay)
                finally:
                    capture.release()
                    self._clear_latest_frame()

                if not self.is_webcam or not reconnect:
                    break

                self._stop_event.wait(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 2,
                    self.reconnect_max_delay,
                )
        except Exception as exc:
            self.error = f"Video capture failed: {exc}"
        finally:
            self._ready_event.set()
            self._clear_latest_frame()

    def wait_until_ready(self, timeout=3.0):
        if self.is_webcam:
            if not self._first_frame_event.wait(timeout):
                return False
            with self._lock:
                return self._latest_frame is not None and self.opened
        return self._ready_event.wait(timeout) and self.opened

    def read(self, last_sequence=0, count_drops=True):
        """Return a frame, sequence, capture timestamp, and skipped-frame count."""
        if self.is_webcam:
            with self._lock:
                if self._latest_frame is None or self._sequence <= last_sequence:
                    return None
                skipped = max(0, self._sequence - last_sequence - 1)
                if count_drops:
                    self.dropped_frames += skipped
                return (
                    self._latest_frame.copy(),
                    self._sequence,
                    self._captured_at,
                    skipped,
                )

        try:
            return self._file_frames.get_nowait() + (0,)
        except queue.Empty:
            return None

    @property
    def capture_fps(self):
        with self._lock:
            if len(self._capture_times) < 2:
                return 0.0
            elapsed = self._capture_times[-1] - self._capture_times[0]
            last_frame_age = time.perf_counter() - self._capture_times[-1]
            if elapsed <= 0 or last_frame_age > 1.5:
                return 0.0
            return (len(self._capture_times) - 1) / elapsed

    @property
    def latest_frame_age_ms(self):
        with self._lock:
            if self._captured_at is None:
                return None
            return max(0.0, (time.perf_counter() - self._captured_at) * 1000)

    def stop(self, timeout=5.0):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    @property
    def is_running(self):
        return self._thread.is_alive() and not self._stop_event.is_set()
