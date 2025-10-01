import hashlib

def canonical_id(source: str, original_id: str) -> str:
    key = f"{source}::{original_id}"
    return hashlib.sha1(key.encode()).hexdigest()