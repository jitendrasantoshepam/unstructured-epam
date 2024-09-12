import math
import typing as t
from collections import abc
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

import requests

from unstructured.ingest.enhanced_dataclass import enhanced_field
from unstructured.ingest.error import SourceConnectionError, SourceConnectionNetworkError
from unstructured.ingest.interfaces import (
    AccessConfig,
    BaseConnectorConfig,
    BaseSessionHandle,
    BaseSingleIngestDoc,
    BaseSourceConnector,
    ConfigSessionHandleMixin,
    IngestDocCleanupMixin,
    IngestDocSessionHandleMixin,
    SourceConnectorCleanupMixin,
    SourceMetadata,
)
from unstructured.ingest.logger import logger


@dataclass
class AzureDevOpsSessionHandle(BaseSessionHandle):
    service: requests.Session
    org_url: str


def create_azure_devops_session(org_url: str, personal_access_token: str) -> requests.Session:
    """
    Creates a session for Azure DevOps REST API interaction.
    Args:
        org_url: URL to Azure DevOps organization
        personal_access_token: Personal Access Token (PAT) for authentication
    Returns:
        requests.Session: Session with authentication headers
    """
    session = requests.Session()
    session.auth = ("", personal_access_token)  # PAT used as password in Basic Auth
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )

    # Test the connection by listing projects
    test_url = f"{org_url}/_apis/projects?api-version=6.0"
    response = session.get(test_url)

    if response.status_code != 200:
        raise ValueError(f"Failed to connect to Azure DevOps: {response.text}")

    return session


@dataclass
class AzureDevOpsAccessConfig(AccessConfig):
    personal_access_token: str = enhanced_field(sensitive=True)


@dataclass
class SimpleAzureDevOpsConfig(ConfigSessionHandleMixin, BaseConnectorConfig):
    """Connector config for Azure DevOps."""

    org_url: str
    access_config: AzureDevOpsAccessConfig
    # Now this accepts project IDs instead of project names
    projects: t.Optional[t.List[str]] = None
    custom_fields: t.Optional[t.List[dict]] = None
    work_items: t.Optional[t.List[int]] = None  # List of specific work items

    def create_session_handle(self) -> AzureDevOpsSessionHandle:
        session = create_azure_devops_session(
            org_url=self.org_url,
            personal_access_token=self.access_config.personal_access_token,
        )
        return AzureDevOpsSessionHandle(service=session, org_url=self.org_url)


@dataclass
class AzureDevOpsFileMeta:
    project_id: str
    work_item_id: str


# This function recursively converts nested objects into easily accessible dictionaries
def nested_object_to_field_getter(object):
    if isinstance(object, abc.Mapping):
        new_object = {}
        for k, v in object.items():
            if isinstance(v, abc.Mapping):
                new_object[k] = FieldGetter(nested_object_to_field_getter(v))
            else:
                new_object[k] = v
        return FieldGetter(new_object)
    else:
        return object


class FieldGetter(dict):
    def __getitem__(self, key):
        value = super().__getitem__(key) if key in self else None
        if value is None:
            value = FieldGetter({})
        return value


def form_templated_string(work_item, parsed_fields, custom_fields, c_sep="|||", r_sep="\n\n\n"):
    """Forms a template string via parsing the fields from the API response object on the work item."""
    custom_fields_str = _get_custom_fields_for_work_item(parsed_fields, custom_fields, c_sep, r_sep)
    return r_sep.join(
        [
            _get_id_fields_for_work_item(work_item),
            _get_project_fields_for_work_item(parsed_fields),
            _get_dropdown_fields_for_work_item(parsed_fields),
            _get_comments_for_work_item(parsed_fields),
            _get_text_fields_for_work_item(parsed_fields),
            custom_fields_str,
        ],
    )


def _get_custom_fields_for_work_item(parsed_fields, custom_fields, c_sep="|||", r_sep="\n\n\n"):
    custom_fields_str = []
    if custom_fields:
        for field in custom_fields:
            key = field.get("key")
            value = field.get("value")
            if key in parsed_fields:
                custom_fields_str.append(f"{value}:{parsed_fields[key]}{r_sep}")
    return "".join(custom_fields_str)


DEFAULT_C_SEP = " " * 5
DEFAULT_R_SEP = "\n"


def _get_id_fields_for_work_item(work_item, c_sep=DEFAULT_C_SEP, r_sep=DEFAULT_R_SEP):
    id, rev = work_item["id"], work_item["rev"]
    return f"WorkItemID_Rev:{id}{c_sep}{rev}{r_sep}"


def _get_project_fields_for_work_item(parsed_fields, c_sep=DEFAULT_C_SEP, r_sep=DEFAULT_R_SEP):
    return f"ProjectID:{parsed_fields['System.TeamProject']}{c_sep}ProjectName:{parsed_fields['System.AreaPath']}{r_sep}"


def _get_dropdown_fields_for_work_item(parsed_fields, c_sep=DEFAULT_C_SEP, r_sep=DEFAULT_R_SEP):
    return f"""
    WorkItemType:{parsed_fields.get('System.WorkItemType')}
    {r_sep}
    State:{parsed_fields.get('System.State')}
    {r_sep}
    Priority:{parsed_fields.get('Microsoft.VSTS.Common.Priority')}
    {r_sep}
    AssignedTo:{parsed_fields.get('System.AssignedTo')}
    {r_sep}
    Tags:{parsed_fields.get('System.Tags')}
    {r_sep}
    """


def _get_text_fields_for_work_item(parsed_fields, c_sep=DEFAULT_C_SEP, r_sep=DEFAULT_R_SEP):
    return f"""
    Title:{parsed_fields.get('System.Title')}
    {r_sep}
    Description:{parsed_fields.get('System.Description')}
    {r_sep}
    """


def _get_comments_for_work_item(parsed_fields, c_sep=DEFAULT_C_SEP, r_sep=DEFAULT_R_SEP):
    # Assuming comments are fetched separately as they are in Azure DevOps
    return ""


def scroll_wrapper(func, results_key="value"):
    def wrapper(*args, **kwargs):
        number_of_items_to_fetch = kwargs.pop("number_of_items_to_fetch", 100)
        kwargs["top"] = min(100, number_of_items_to_fetch)
        kwargs["skip"] = kwargs.get("skip", 0)

        all_results = []
        num_iterations = math.ceil(number_of_items_to_fetch / kwargs["top"])

        for _ in range(num_iterations):
            response = func(*args, **kwargs)
            if isinstance(response, list):
                all_results += response
            elif isinstance(response, dict):
                if results_key not in response:
                    raise KeyError(f"Response object has no known key: '{results_key}'")
                all_results += response[results_key]
            kwargs["skip"] += kwargs["top"]

        return all_results[:number_of_items_to_fetch]

    return wrapper


@dataclass
class AzureDevOpsIngestDoc(IngestDocSessionHandleMixin, IngestDocCleanupMixin, BaseSingleIngestDoc):
    """Class encapsulating fetching a doc and writing processed results for Azure DevOps work items."""

    connector_config: SimpleAzureDevOpsConfig
    file_meta: t.Optional[AzureDevOpsFileMeta] = None
    registry_name: str = "azure_devops"

    @cached_property
    def record_locator(self):
        return {
            "org_url": self.connector_config.org_url,
            "work_item_id": self.file_meta.work_item_id,
        }

    @cached_property
    @SourceConnectionNetworkError.wrap
    def work_item(self):
        """Fetches work item data using REST API."""
        url = f"{self.connector_config.org_url}/_apis/wit/workitems/{self.file_meta.work_item_id}?api-version=6.0"
        response = self.session_handle.service.get(url)
        if response.status_code != 200:
            raise Exception(
                f"Error fetching work item {self.file_meta.work_item_id}: {response.text}"
            )
        return response.json()

    @cached_property
    def parsed_fields(self):
        return nested_object_to_field_getter(self.work_item["fields"])

    @property
    def grouping_folder_name(self):
        # Fallback to 'unknown_project' if project_id is None
        return self.file_meta.project_id or "unknown_project"

    @property
    def filename(self):
        download_file = f"{self.file_meta.work_item_id}.txt"
        return (
            Path(self.read_config.download_dir) / self.grouping_folder_name / download_file
        ).resolve()

    @property
    def _output_filename(self):
        output_file = f"{self.file_meta.work_item_id}.json"
        return (
            Path(self.processor_config.output_dir) / self.grouping_folder_name / output_file
        ).resolve()

    @property
    def version(self) -> t.Optional[str]:
        return None

    def update_source_metadata(self, **kwargs) -> None:
        exists = bool(self.work_item)
        if not exists:
            self.source_metadata = SourceMetadata(exists=exists)
            return

        created_date = self.parsed_fields["System.CreatedDate"]
        modified_date = self.parsed_fields["System.ChangedDate"]
        self.source_metadata = SourceMetadata(
            date_created=datetime.strptime(created_date, "%Y-%m-%dT%H:%M:%S.%fZ").isoformat(),
            date_modified=datetime.strptime(modified_date, "%Y-%m-%dT%H:%M:%S.%fZ").isoformat(),
            source_url=f"{self.connector_config.org_url}/_workitems/edit/{self.file_meta.work_item_id}",
            exists=exists,
        )

    @SourceConnectionError.wrap
    def get_file(self):
        document = form_templated_string(
            self.work_item, self.parsed_fields, self.connector_config.custom_fields
        )
        self.update_source_metadata()
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        with open(self.filename, "w", encoding="utf8") as f:
            f.write(document)


@dataclass
class AzureDevOpsSourceConnector(SourceConnectorCleanupMixin, BaseSourceConnector):
    """Fetches work items from projects in Azure DevOps."""

    connector_config: SimpleAzureDevOpsConfig
    _connection: t.Optional[AzureDevOpsSessionHandle] = field(init=False, default=None)

    @property
    def connection(self) -> AzureDevOpsSessionHandle:
        if self._connection is None:
            self._connection = self.connector_config.create_session_handle()
        return self._connection

    def check_connection(self):
        """Check the connection by listing projects via REST API."""
        url = f"{self.connection.org_url}/_apis/projects?api-version=6.0"
        response = self.connection.service.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to connect to Azure DevOps: {response.text}")
        return response.json()

    def get_all_project_ids(self):
        """Get all project IDs from Azure DevOps using REST API."""
        projects_data = self.check_connection()
        return [project["id"] for project in projects_data["value"]]

    def get_work_items_within_project(self, project_id: str):
        """Query work items within a specific project using REST API."""

        # Step 1: Fetch the project details by project_id to get the project name (System.TeamProject)
        project_url = f"{self.connection.org_url}/_apis/projects/{project_id}?api-version=6.0"
        project_response = self.connection.service.get(project_url)

        if project_response.status_code != 200:
            logger.warning(
                f"Error fetching project details for project ID {project_id}: {project_response.text}"
            )
            return []

        project_name = project_response.json().get("name")

        if not project_name:
            logger.warning(f"Project name not found for project ID {project_id}")
            return []

        # Step 2: Query work items within the specific project using the project name
        wiql_url = f"{self.connection.org_url}/_apis/wit/wiql?api-version=6.0"
        query = {
            "query": f"SELECT [System.Id] FROM WorkItems WHERE [System.TeamProject] = '{project_name}'"
        }

        response = self.connection.service.post(wiql_url, json=query)

        if response.status_code != 200:
            # Log a warning and continue rather than raising an exception immediately
            logger.warning(
                f"Error querying work items for project {project_name} (ID: {project_id}): {response.text}"
            )
            return []

        return [(item["id"], project_id) for item in response.json().get("workItems", [])]

    def get_ingest_docs(self):
        """Retrieve documents for ingestion."""
        # Use the provided project IDs or fetch all project IDs if none are provided
        project_ids = self.connector_config.projects or self.get_all_project_ids()

        # Check if specific work items are provided
        if self.connector_config.work_items:
            # Fetch only the provided work items
            work_items_and_ids = [
                (
                    work_item_id,
                    None,
                )  # project_id can be None as we are fetching specific work items
                for work_item_id in self.connector_config.work_items
            ]
        else:
            # Fetch all work items for each project ID
            work_items_and_ids = [
                (work_item_id, project_id)
                for project_id in project_ids
                for work_item_id, project_id in self.get_work_items_within_project(project_id)
            ]

        # Create AzureDevOpsIngestDoc objects for each work item
        return [
            AzureDevOpsIngestDoc(
                connector_config=self.connector_config,
                processor_config=self.processor_config,
                read_config=self.read_config,
                file_meta=AzureDevOpsFileMeta(work_item_id=work_item_id, project_id=project_id),
            )
            for work_item_id, project_id in work_items_and_ids
        ]

    def initialize(self):
        """Implement the abstract initialize method."""
        self.check_connection()
