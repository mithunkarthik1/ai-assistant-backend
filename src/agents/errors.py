"""Domain errors for agent routing and Project API execution."""


class AgentError(Exception):
    """Base error for the agent module."""


class AgentConfigurationError(AgentError):
    """Raised when the configurable LLM is not usable."""


class AgentResponseError(AgentError):
    """Raised when the LLM returns an unusable response."""


class ProjectApiError(AgentError):
    """Base error for Project API failures."""


class InvalidProjectIdError(ProjectApiError):
    """Raised when a project identifier fails validation."""


class MissingProjectIdError(ProjectApiError):
    """Raised when a Project API operation has no project identifier."""


class ProjectNotFoundError(ProjectApiError):
    """Raised when the Project API cannot find a project."""


class ProjectApiTimeoutError(ProjectApiError):
    """Raised when the Project API does not respond in time."""


class ProjectDataMissingError(ProjectApiError):
    """Raised when the Project API returns no usable data."""
