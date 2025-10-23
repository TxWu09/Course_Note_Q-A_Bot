import logging

logging.basicConfig(
    filename="rag_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_query(query, answer, sources=None):
    logging.info("QUERY: %s", query)
    logging.info("ANSWER: %s", answer)
    if sources:
        for i, src in enumerate(sources, 1):
            logging.info("SOURCE %d: %s", i, src.page_content[:200])
    logging.info("-" * 40)
