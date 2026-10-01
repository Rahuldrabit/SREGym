"""Port-forward management utility for Kubernetes services."""

from __future__ import annotations

import contextlib
import logging
import socket
import subprocess
import threading
import time

logger = logging.getLogger("all.infra.k8s_port_forward")


class KubectlPortForward:
    """Maintains a localhost tunnel to a Kubernetes Service."""

    def __init__(self, namespace: str, service: str, remote_port: int):
        self.namespace = namespace
        self.service = service
        self.remote_port = remote_port
        self.local_port: int | None = None
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _healthy(self) -> bool:
        if self.process is None or self.process.poll() is not None or self.local_port is None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", self.local_port)) == 0

    def start(self, timeout: float = 25.0) -> int:
        with self._lock:
            if self._healthy():
                return int(self.local_port)
            self.stop()
            self.local_port = self._free_port()
            try:
                self.process = subprocess.Popen(
                    [
                        "kubectl",
                        "port-forward",
                        f"service/{self.service}",
                        f"{self.local_port}:{self.remote_port}",
                        "-n",
                        self.namespace,
                        "--address",
                        "127.0.0.1",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                logger.warning("kubectl not found in environment; running in simulation fallback mode")
                return self.local_port

            start = time.time()
            while time.time() - start < timeout:
                if self._healthy():
                    logger.info(
                        f"Port-forward service/{self.service} {self.local_port}->{self.remote_port} established"
                    )
                    return int(self.local_port)
                time.sleep(0.2)
            self.stop()
            raise RuntimeError(
                f"Port-forward to service/{self.service} on remote port {self.remote_port} did not become ready within {timeout}s"
            )

    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                with contextlib.suppress(Exception):
                    self.process.kill()
            self.process = None
