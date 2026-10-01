"""Task-local historical clocks; live execution uses its normal clock when unset."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from sibyl.common.time_utils import _as_utc

_TIMES: ContextVar[Mapping[int, datetime]] = ContextVar("forecast_times", default={})


def question_key(question: Any) -> int:
    qid = getattr(question, "id_of_question", None)
    if isinstance(qid, int):
        return qid
    if isinstance(qid, str) and qid.isdecimal():
        return int(qid)
    return id(question)


def forecast_time(question: Any) -> datetime | None:
    return _TIMES.get().get(question_key(question))


@contextmanager
def forecast_times(times: Mapping[int, datetime]) -> Iterator[None]:
    token = _TIMES.set({**_TIMES.get(), **{qid: _as_utc(t) for qid, t in times.items()}})
    try:
        yield
    finally:
        _TIMES.reset(token)
