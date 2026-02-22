from __future__ import annotations

import ast
import json
import re

from pydantic import BaseModel, ValidationError

from crewai.utilities.converter import Converter, ConverterError


class RobustConverter(Converter):
    def to_pydantic(self, current_attempt: int = 1) -> BaseModel:
        try:
            if self.llm.supports_function_calling():
                return self._create_instructor().to_pydantic()

            response = self.llm.call(
                [
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": self.text},
                ]
            )

            for candidate in self._candidate_payloads(response):
                parsed = self._parse_payload(candidate)
                if parsed is None:
                    continue
                return self.model.model_validate(parsed)

            raise ConverterError("Unable to parse model output as valid structured data")

        except ValidationError as exc:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to validation error: {exc}"
            ) from exc
        except Exception as exc:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to error: {exc}"
            ) from exc

    def _candidate_payloads(self, text: str) -> list[str]:
        candidates = [text]
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            extracted = match.group(1)
            if extracted not in candidates:
                candidates.append(extracted)
        return candidates

    def _parse_payload(self, payload: str) -> dict | list | None:
        try:
            loaded = json.loads(payload)
            if isinstance(loaded, (dict, list)):
                return loaded
        except Exception:
            pass

        normalized = self._normalize_json_literals(payload)
        try:
            loaded = json.loads(normalized)
            if isinstance(loaded, (dict, list)):
                return loaded
        except Exception:
            pass

        try:
            loaded = ast.literal_eval(payload)
            if isinstance(loaded, (dict, list)):
                return loaded
        except Exception:
            return None

        return None

    def _normalize_json_literals(self, text: str) -> str:
        normalized = re.sub(r"\bTrue\b", "true", text)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        return normalized