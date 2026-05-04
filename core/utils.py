def format_items_for_prompt(items: list[str]) -> str:
    """
    Format a list of previously roasted items into a bulleted string for the prompt.
    If the list is empty, return a string indicating there are no items yet.
    """
    if not items:
        return "(No past crimes recorded yet)"
    
    return "\n".join(f"- {item}" for item in items)
