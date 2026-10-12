"""OCTA byte tokenizer v0: lossless UTF-8 round trips, no external model."""
class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], errors: str = "replace") -> str:
        if any(not isinstance(t, int) or t < 0 or t > 255 for t in token_ids):
            raise ValueError("Byte tokens must be integers from 0 to 255")
        return bytes(token_ids).decode("utf-8", errors=errors)
