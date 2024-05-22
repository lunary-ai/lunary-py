TEMPLATE_NOT_FOUND_MESSAGE = "Template not found, are the project ID and slug correct?"

class TemplateNotFoundError(Exception):
    """Custom exception for template not found errors."""
    def __init__(self, message=TEMPLATE_NOT_FOUND_MESSAGE):
        self.message = message
        super().__init__(self.message)
