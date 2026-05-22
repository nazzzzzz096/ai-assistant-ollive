MAX_HISTORY = 6

def update_memory(messages):

    """
    Keeps only the latest conversation history.
    Implements sliding-window memory.
    """

    return messages[-MAX_HISTORY:]