from __future__ import annotations

from producer.contracts import Operation


class ProducerWorkflowError(RuntimeError):
    pass


class SubmissionAmbiguous(ProducerWorkflowError):
    def __init__(
        self,
        operation: Operation,
        *,
        status_code: int | None = None,
    ) -> None:
        self.operation = operation
        self.status_code = status_code
        super().__init__(
            f"{operation.value} submission outcome is ambiguous; reconcile before retrying."
        )


class SubmissionRejected(ProducerWorkflowError):
    def __init__(self, operation: Operation, status_code: int) -> None:
        self.operation = operation
        self.status_code = status_code
        super().__init__(
            f"{operation.value} submission was rejected with HTTP {status_code}."
        )


class BatchConflict(ProducerWorkflowError):
    pass


class DuplicateBooking(ProducerWorkflowError):
    pass


class InvalidWorkflowTransition(ProducerWorkflowError):
    pass


class UnknownBooking(ProducerWorkflowError):
    pass
