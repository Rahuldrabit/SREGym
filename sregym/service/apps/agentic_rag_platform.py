"""Interface to the Agentic RAG Platform application."""

from __future__ import annotations

import logging

from sregym.paths import AGENTIC_RAG_PLATFORM_METADATA
from sregym.service.apps.base import Application

logger = logging.getLogger("all.application.agentic_rag_platform")


class AgenticRAGPlatform(Application):
    """Application representation for Agentic RAG Platform."""

    def __init__(self, embedded: bool = True):
        super().__init__(str(AGENTIC_RAG_PLATFORM_METADATA))
        self.load_app_json()
        self.embedded = embedded
        self.app_name = "Agentic RAG Platform"
        self.namespace = "agentic-rag-platform"
        self.workload = None

    def deploy(self):
        """Deploy application components (in-process or cluster)."""
        logger.info(f"Deploying {self.name} in namespace {self.namespace} (embedded={self.embedded})")

    def cleanup(self):
        """Clean up deployed application resources."""
        logger.info(f"Cleaning up {self.name} in namespace {self.namespace}")
        if self.workload is not None:
            self.workload.stop()

    def get_app_summary(self) -> str:
        return (
            f"App Name: {self.name}\n"
            f"Namespace: {self.namespace}\n"
            f"Description: Autonomous Agentic RAG Platform with multi-layer retries across "
            f"workflow supervisor, tool client, and transport layers hitting a concurrency-bound backend."
        )
