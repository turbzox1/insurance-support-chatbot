"""Library logger; no raw questions, prompts or credentials are logged."""

import logging

logger = logging.getLogger("insurance_chatbot")
logger.addHandler(logging.NullHandler())
