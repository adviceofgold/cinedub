import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }, default=str)


class CineDubLogger:
    _instance: Optional["CineDubLogger"] = None

    def __init__(self, log_dir: str = "/content/drive/MyDrive/cinedub/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("cinedub")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        file_handler = RotatingFileHandler(
            self.log_dir / "pipeline.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            "[%(levelname)s] %(message)s"
        ))

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    @classmethod
    def get(cls, log_dir: Optional[str] = None) -> "CineDubLogger":
        if cls._instance is not None and log_dir is not None:
            if str(cls._instance.log_dir) != str(log_dir):
                cls._instance = None
        if cls._instance is None:
            cls._instance = cls(log_dir or "/content/drive/MyDrive/cinedub/logs")
        return cls._instance

    def debug(self, msg: str, **extra):
        self._logger.debug(msg, extra={"extra": extra} if extra else {})

    def info(self, msg: str, **extra):
        self._logger.info(msg, extra={"extra": extra} if extra else {})

    def warn(self, msg: str, **extra):
        self._logger.warning(msg, extra={"extra": extra} if extra else {})

    def error(self, msg: str, **extra):
        self._logger.error(msg, extra={"extra": extra} if extra else {})

    def exception(self, msg: str, **extra):
        self._logger.exception(msg, extra={"extra": extra} if extra else {})
