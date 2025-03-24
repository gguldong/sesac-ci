import uuid

def generate_unique_message_id() -> str:
    return str(uuid.uuid4())