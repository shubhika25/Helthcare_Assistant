from backend.logger import logger

def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running chain for input: {user_input}")
        
        # Run LCEL chain — returns a plain string
        result = chain.invoke({"question": user_input})
        
        # Build a consistent response object
        response = {
            "response": result,  # The model's text answer
            "sources": []        # Optional: add retrieved docs here later
        }

        logger.debug(f"Chain response: {response}")
        return response

    except Exception as e:
        logger.exception("Error in query_chain")
        raise
