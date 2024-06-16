import logging
import multiprocessing as mp
from dataclasses import dataclass, field
import shutil
from typing import Any, Optional
import os
import json
from dataclasses_json import DataClassJsonMixin

from unstructured.ingest.connector.registry import create_ingest_doc_from_dict
from unstructured.ingest.interfaces import BaseIngestDocBatch, BaseSingleIngestDoc
from unstructured.ingest.logger import ingest_log_streaming_init, logger
from unstructured.ingest.pipeline.copy import Copier
from unstructured.ingest.pipeline.interfaces import (
    DocFactoryNode,
    PartitionNode,
    PipelineContext,
    ReformatNode,
    SourceNode,
    WriteNode,
)
from unstructured.ingest.pipeline.permissions import PermissionsDataCleaner
from unstructured.ingest.pipeline.utils import get_ingest_doc_hash


@dataclass
class Pipeline(DataClassJsonMixin):
    pipeline_context: PipelineContext
    doc_factory_node: DocFactoryNode
    source_node: SourceNode
    partition_node: Optional[PartitionNode] = None
    write_node: Optional[WriteNode] = None
    reformat_nodes: "list[ReformatNode]" = field(default_factory=list)
    permissions_node: Optional[PermissionsDataCleaner] = None

    def initialize(self):
        ingest_log_streaming_init(logging.DEBUG if self.pipeline_context.verbose else logging.INFO)
        if os.path.isdir(f"./{self.pipeline_context.output_dir}"):
            shutil.rmtree(f"./{self.pipeline_context.output_dir}")
        if os.path.isdir(f"./{self.pipeline_context.status_dir}"):
            shutil.rmtree(f"./{self.pipeline_context.status_dir}")
        if self.pipeline_context.clean and os.path.isdir(self.pipeline_context.main_work_dir):
            shutil.rmtree(self.pipeline_context.main_work_dir)
        self.update_status(False)

    def update_status(self, running: bool, msg: str = ""):
        if not os.path.exists(self.pipeline_context.status_dir):
            os.makedirs(self.pipeline_context.status_dir)
        file_path = os.path.join(self.pipeline_context.status_dir, "running.json")
        with open(file_path, "w") as status_file:
            json.dump(
                {
                    "jobId": self.pipeline_context.jobId,
                    "timestamp": self.pipeline_context.timestamp,
                    "running": running,
                    "msg": msg,
                },
                status_file,
                indent=4,
            )

    def get_nodes_str(self):
        nodes = [self.doc_factory_node, self.source_node, self.partition_node]
        nodes.extend(self.reformat_nodes)
        if self.write_node:
            nodes.append(self.write_node)
        nodes.append(Copier(pipeline_context=self.pipeline_context))
        return " -> ".join([node.__class__.__name__ for node in nodes])

    def expand_batch_docs(self, dict_docs: "list[dict[str, Any]]") -> "list[dict[str, Any]]":
        expanded_docs: list[dict[str, Any]] = []
        for d in dict_docs:
            doc = create_ingest_doc_from_dict(d)
            if isinstance(doc, BaseSingleIngestDoc):
                expanded_docs.append(doc.to_dict())
            elif isinstance(doc, BaseIngestDocBatch):
                expanded_docs.extend([single_doc.to_dict() for single_doc in doc.ingest_docs])
            else:
                raise ValueError(
                    f"type of doc ({type(doc)}) is not a recognized type: "
                    f"BaseSingleIngestDoc or BaseSingleIngestDoc"
                )
        return expanded_docs

    def run(self):
        logger.info(
            f"running pipeline: {self.get_nodes_str()} "
            f"with config: {self.pipeline_context.to_json()}",
        )
        self.initialize()
        self.update_status(True, msg="running pipeline")
        manager = mp.Manager()
        self.pipeline_context.ingest_docs_map = manager.dict()
        # -- Get the documents to be processed --
        dict_docs = self.doc_factory_node()
        dict_docs = [manager.dict(d) for d in dict_docs]
        if not dict_docs:
            msg = "no docs found to process"
            logger.info(msg)
            self.update_status(False, msg=msg)
            return
        logger.info(
            f"processing {len(dict_docs)} docs via "
            f"{self.pipeline_context.num_processes} processes",
        )
        for doc in dict_docs:
            self.pipeline_context.ingest_docs_map[get_ingest_doc_hash(doc)] = doc
        fetched_filenames = self.source_node(iterable=dict_docs)
        if self.source_node.read_config.download_only:
            msg = "stopping pipeline after downloading files"
            logger.info(msg)
            self.update_status(False, msg=msg)
            return
        if not fetched_filenames:
            msg = "No files to run partition over"
            logger.info(msg)
            self.update_status(False, msg=msg)
            return
        # -- To support batches ingest docs, expand those into the populated single ingest
        # -- docs after downloading content
        dict_docs = self.expand_batch_docs(dict_docs=dict_docs)
        if self.partition_node is None:
            raise ValueError("partition node not set")
        partitioned_jsons = self.partition_node(iterable=dict_docs)
        if not partitioned_jsons:
            msg = "No files to process after partitioning"
            logger.info(msg)
            self.update_status(False, msg=msg)
            return
        for reformat_node in self.reformat_nodes:
            reformatted_jsons = reformat_node(iterable=partitioned_jsons)
            if not reformatted_jsons:
                msg = f"No files to process after {reformat_node.__class__.__name__}"
                logger.info(msg)
                self.update_status(False, msg=msg)
                return
            partitioned_jsons = reformatted_jsons

        # -- Copy the final destination to the desired location --
        copier = Copier(
            pipeline_context=self.pipeline_context,
        )

        copier(iterable=partitioned_jsons)

        if self.write_node:
            logger.info(
                f"uploading elements from {len(partitioned_jsons)} "
                "document(s) to the destination"
            )
            self.write_node(iterable=partitioned_jsons)

        if self.permissions_node:
            self.permissions_node.cleanup_permissions()

        self.update_status(False, "completed")
