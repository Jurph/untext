import logging

import untextre.utils as utils
from untextre.utils import configure_logging, setup_logger


def test_setup_logger_uses_root_owned_handlers():
    logger = setup_logger("untextre.test_logging")

    assert logger.handlers == []
    assert logger.propagate is True
    assert logger.level == logging.NOTSET


def test_configure_logging_replaces_only_untextre_handlers(tmp_path):
    root = logging.getLogger()
    preserved = logging.NullHandler()
    root.addHandler(preserved)
    try:
        first_log = tmp_path / "first.log"
        second_log = tmp_path / "second.log"

        configure_logging(verbose=True, logfile=first_log)
        configure_logging(verbose=False, logfile=second_log)

        owned_handlers = [
            handler for handler in root.handlers
            if getattr(handler, "_untextre_handler", False)
        ]
        assert preserved in root.handlers
        assert len(owned_handlers) == 2
        assert root.level == logging.INFO
    finally:
        root.removeHandler(preserved)


def test_ascii_safe_replaces_terminal_hostile_unicode():
    text = "Bucket 720\u00d71080: stable px \u2192 queued \u00b13 90\u00b0 \u2705 \u274c"

    assert hasattr(utils, "_ascii_safe")
    safe = utils._ascii_safe(text)

    assert safe == "Bucket 720x1080: stable px -> queued +/-3 90 degrees [OK] [ERROR]"
    assert safe.isascii()


def test_configure_logging_console_formatter_outputs_ascii():
    configure_logging(verbose=True)
    root = logging.getLogger()
    console_handler = next(
        handler for handler in root.handlers
        if getattr(handler, "_untextre_handler", False)
        and isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    )
    record = logging.LogRecord(
        "untextre.discovery",
        logging.INFO,
        __file__,
        1,
        "Bucket 720\u00d71080: stable px \u2192 queued \u2705",
        (),
        None,
    )

    formatted = console_handler.format(record)

    assert formatted.isascii()
    assert "720x1080" in formatted
    assert "-> queued [OK]" in formatted
