import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("insurance_chatbot")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers when restarting
if not logger.handlers:

    file_handler = logging.FileHandler("logs/chatbot.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

logger.propagate = False

# Silence noisy libraries
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)