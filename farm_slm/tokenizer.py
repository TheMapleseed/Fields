"""Byte tokenizer with one leading token per collection domain."""

from __future__ import annotations

from farm_slm.config import FarmSLMConfig

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258


class Tokenizer:
    def __init__(self, config: FarmSLMConfig) -> None:
        self._config = config

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def domain_id(self, domain: str) -> int:
        return self._config.domain_token_id(domain)

    def encode(self, text: str, domain: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        domain_id = self.domain_id(domain)
        body = list(text.encode("utf-8"))
        return [BOS_ID, domain_id, *body, EOS_ID]

    def decode_text(self, token_ids: list[int]) -> str:
        raw = bytes(token for token in token_ids if 0 <= token <= 255)
        return raw.decode("utf-8", errors="replace")
