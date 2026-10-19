"""Sync path of the legacy cohere embedding handler must honor `timeout` (issue #47301)."""

from typing import Final
from unittest.mock import patch

import litellm
from litellm.llms.cohere.embed.handler import embedding as cohere_embedding
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import EmbeddingResponse


class _FakeLogging:
    def pre_call(self, *args, **kwargs):
        pass

    def post_call(self, *args, **kwargs):
        pass


def test_sync_embedding_client_uses_passed_timeout():
    captured: dict = {}

    def fake_post(self, url, **kwargs):
        captured["client_timeout"] = self.client.timeout
        raise RuntimeError("captured")

    with patch.object(HTTPHandler, "post", fake_post):
        try:
            cohere_embedding(
                model="cohere.embed-english-v3",
                input=["hello"],
                model_response=EmbeddingResponse(),
                logging_obj=_FakeLogging(),
                optional_params={},
                headers={},
                encoding=None,
                data={"input": ["hello"], "model": "cohere.embed-english-v3"},
                complete_api_base="https://bedrock-runtime.fake.local/model/x/invoke",
                api_key=None,
                aembedding=False,
                timeout=45,
            )
        except RuntimeError:
            pass

    assert captured["client_timeout"].read == 45.0


def test_sync_embedding_without_timeout_keeps_default_client():
    captured: dict = {}

    def fake_post(self, url, **kwargs):
        captured["client_timeout"] = self.client.timeout
        raise RuntimeError("captured")

    with patch.object(HTTPHandler, "post", fake_post):
        try:
            cohere_embedding(
                model="cohere.embed-english-v3",
                input=["hello"],
                model_response=EmbeddingResponse(),
                logging_obj=_FakeLogging(),
                optional_params={},
                headers={},
                encoding=None,
                data={"input": ["hello"], "model": "cohere.embed-english-v3"},
                complete_api_base="https://bedrock-runtime.fake.local/model/x/invoke",
                api_key=None,
                aembedding=False,
            )
        except RuntimeError:
            pass

    # default handler client: litellm's standard 600s read/write/pool
    assert captured["client_timeout"].read == 600.0
