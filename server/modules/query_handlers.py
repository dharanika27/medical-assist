from logger import logger


def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running chain for input: {user_input}")

        result = chain.invoke(user_input)

        if isinstance(result, str):
            response = {
                "response": result,
                "sources": [],
            }
        else:
            response = {
                "response": result.get("result") or result.get("answer") or str(result),
                "sources": [
                    doc.metadata.get("source", "")
                    for doc in result.get("source_documents", [])
                ],
            }

        logger.debug(f"Chain response:{response}")
        return response

    except Exception:
        logger.exception("Error on query chain")
        raise
