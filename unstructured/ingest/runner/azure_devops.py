import hashlib
import typing as t
from dataclasses import dataclass

from unstructured.ingest.interfaces import BaseSourceConnector
from unstructured.ingest.logger import logger
from unstructured.ingest.runner.base_runner import Runner
from unstructured.ingest.runner.utils import update_download_dir_hash

if t.TYPE_CHECKING:
    from unstructured.ingest.connector.azure_devops import SimpleAzureDevOpsConfig


@dataclass
class AzureDevOpsRunner(Runner):
    connector_config: "SimpleAzureDevOpsConfig"

    def update_read_config(self):
        # Create a SHA-256 hash object and get the hex digest
        hashed_dir_name = hashlib.sha256(
            self.connector_config.org_url.encode("utf-8"),
        )

        # Pass the hexdigest string (not the hash object) to update_download_dir_hash
        self.read_config.download_dir = update_download_dir_hash(
            connector_name="azure_devops",
            read_config=self.read_config,
            hashed_dir_name=hashed_dir_name,  # Ensure we slice the first 10 characters
            logger=logger,
        )

    def get_source_connector_cls(self) -> t.Type[BaseSourceConnector]:
        from unstructured.ingest.connector.azure_devops import (
            AzureDevOpsSourceConnector,
        )

        return AzureDevOpsSourceConnector
