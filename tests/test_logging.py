import logging

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
