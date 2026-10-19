import json
from typing import Final
from unittest.mock import patch, MagicMock, AsyncMock


import httpx
import litellm
import pytest
import respx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


MOCK_EMBEDDING_RESPONSE = [[0.1, 0.2, 0.3, 0.4, 0.5]]


@pytest.fixture
def mock_embedding_http_handler():
    """Fixture to mock the HTTP handler for embedding tests"""
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_EMBEDDING_RESPONSE
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_hf_config_fetch():
    """Serve the Hugging Face config.json fetched during cost calculation, so no test leaves the process"""
    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.get(url__regex=r"https://huggingface\.co/.*/config\.json").respond(
            json={"max_position_embeddings": 512}
        )
        yield respx_mock


@pytest.fixture
def mock_embedding_async_http_handler():
    """Fixture to mock the async HTTP handler for embedding tests"""
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = MOCK_EMBEDDING_RESPONSE
        mock_post.return_value = mock_response
        yield mock_post


class TestHuggingFaceEmbedding:
    @pytest.fixture(autouse=True)
    def setup(self, mock_embedding_http_handler, mock_embedding_async_http_handler, mock_hf_config_fetch):
        self.mock_get_task_patcher = patch(
            "litellm.llms.huggingface.embedding.handler.get_hf_task_embedding_for_model"
        )
        self.mock_get_task = self.mock_get_task_patcher.start()

        def mock_get_task_side_effect(model, task_type, api_base):
            if task_type is not None:
                return task_type
            return "sentence-similarity"

        self.mock_get_task.side_effect = mock_get_task_side_effect

        self.model = "huggingface/BAAI/bge-m3"
        self.mock_http = mock_embedding_http_handler
        self.mock_async_http = mock_embedding_async_http_handler
        litellm.set_verbose = False

        yield

        self.mock_get_task_patcher.stop()

    def test_input_type_preserved_in_optional_params(self):
        input_text = ["hello world"]

        response = litellm.embedding(
            model=self.model,
            input=input_text,
            input_type="embed",
        )

        self.mock_http.assert_called_once()
        post_call_args = self.mock_http.call_args
        request_data = json.loads(post_call_args[1]["data"])

        # When input_type="embed", it should use simple format {"inputs": [...]}
        # NOT the sentence-similarity format which would require 2+ sentences
        assert "inputs" in request_data
        assert request_data["inputs"] == input_text

        # Should NOT have sentence-similarity format
        assert "source_sentence" not in str(request_data)
        assert "sentences" not in str(request_data)

    def test_embedding_allows_special_token_looking_input(self):
        input_text = ["hello <|fim_prefix|> world"]

        response = litellm.embedding(
            model=self.model,
            input=input_text,
            input_type="embed",
        )

        self.mock_http.assert_called_once()
        post_call_args = self.mock_http.call_args
        request_data = json.loads(post_call_args[1]["data"])

        assert request_data["inputs"] == input_text
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens

    def test_model_name_with_https_substring_uses_api_base(self):
        api_base = "https://legit.example/embed"

        litellm.embedding(
            model="huggingface/my-https-endpoint",
            input=["hello world"],
            input_type="embed",
            api_base=api_base,
        )

        self.mock_http.assert_called_once()
        called_url = self.mock_http.call_args[0][0]
        assert called_url == api_base

    def test_embedding_with_sentence_similarity_task(self):
        """Test embedding when task type is sentence-similarity (requires 2+ sentences)"""

        similarity_response = {"similarities": [[0, 0.9], [1, 0.8]]}

        self.mock_http.return_value.json.return_value = similarity_response

        # Test with 2+ sentences (required for sentence-similarity)
        input_text = [
            "This is the source sentence",
            "This is sentence one",
            "This is sentence two",
        ]

        response = litellm.embedding(
            model=self.model,
            input=input_text,
            # Use the model's natural task type (sentence-similarity)
        )

        self.mock_http.assert_called_once()
        post_call_args = self.mock_http.call_args
        request_data = json.loads(post_call_args[1]["data"])

        assert "inputs" in request_data
        assert "source_sentence" in request_data["inputs"]
        assert "sentences" in request_data["inputs"]
        assert request_data["inputs"]["source_sentence"] == input_text[0]
        assert request_data["inputs"]["sentences"] == input_text[1:]

    def test_hf_sentence_similarity_transform_keeps_provider_options(self):
        from litellm.llms.huggingface.embedding.handler import HuggingFaceEmbedding

        request_data: Final = HuggingFaceEmbedding()._transform_input(
            input=["source sentence", "candidate sentence"],
            model="BAAI/bge-m3",
            call_type="sync",
            optional_params={"input_type": "sentence-similarity", "min_length": 2},
            embed_url="https://huggingface.example/embeddings",
        )

        assert request_data == {
            "inputs": {"source_sentence": "source sentence", "sentences": ["candidate sentence"]},
            "parameters": {"min_length": 2},
        }

    def test_embedding_dispatch_passes_timeout_to_handler(self):
        """main.embedding() must forward the user timeout to the HF handler (issue #47299)."""
        with patch("litellm.main.huggingface_embed") as mock_hf:
            mock_hf.embedding.return_value = litellm.utils.EmbeddingResponse()
            litellm.embedding(
                model=self.model,
                input=["hello world"],
                input_type="embed",
                timeout=45,
            )

            mock_hf.embedding.assert_called_once()
            assert mock_hf.embedding.call_args.kwargs["timeout"] == 45

    def test_sync_embedding_forwards_timeout_to_http_client(self):
        """Sync path must build its HTTP client with the user timeout (issue #47299)."""
        real_init: Final = HTTPHandler.__init__
        inits: list = []

        def capturing_init(self, *args, **kwargs):
            inits.append(kwargs)
            real_init(self, *args, **kwargs)

        with patch.object(HTTPHandler, "__init__", capturing_init):
            litellm.embedding(
                model=self.model,
                input=["hello world"],
                input_type="embed",
                timeout=45,
            )

        self.mock_http.assert_called_once()
        timeouts = [kw["timeout"] for kw in inits if "timeout" in kw]
        assert any(t.read == 45.0 for t in timeouts), f"no client built with timeout=45: {inits}"

    async def test_async_embedding_forwards_timeout_to_http_client(self):
        """Async path must build its HTTP client with the user timeout (issue #47299)."""
        with patch("litellm.llms.huggingface.embedding.handler.get_async_httpx_client") as mock_get_client:
            mock_get_client.side_effect = lambda **kwargs: AsyncHTTPHandler()
            await litellm.aembedding(
                model=self.model,
                input=["hello world"],
                input_type="embed",
                timeout=45,
            )

            mock_get_client.assert_called_once()
            timeout_param = mock_get_client.call_args.kwargs["params"]["timeout"]
            if isinstance(timeout_param, httpx.Timeout):
                assert timeout_param.read == 45.0
            else:
                assert timeout_param == 45
        self.mock_async_http.assert_called_once()


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_hf_embedding_sentence_sim_sends_one_similarity_request(
    sync_mode: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf-test-token")
    with respx.mock(assert_all_called=False) as hf:
        hf.get(url__regex=r"https://huggingface\.co/api/models/.*").respond(json={"pipeline_tag": "sentence-similarity"})
        hf.get(url__regex=r"https://huggingface\.co/.*/config\.json").respond(json={"max_position_embeddings": 512})
        inference: Final = hf.post(
            "https://router.huggingface.co/hf-inference/pipeline/sentence-similarity/sentence-transformers/TaylorAI/bge-micro-v2"
        ).respond(json=[0.7708950042724609])
        request: Final = {
            "model": "huggingface/sentence-transformers/TaylorAI/bge-micro-v2",
            "input": ["good morning from litellm", "this is another item"],
        }
        response: Final = litellm.embedding(**request) if sync_mode else await litellm.aembedding(**request)

    assert inference.call_count == 1
    assert json.loads(inference.calls.last.request.content) == {
        "inputs": {"source_sentence": "good morning from litellm", "sentences": ["this is another item"]}
    }
    assert inference.calls.last.request.headers["Authorization"] == "Bearer hf-test-token"
    assert response.data == [{"object": "embedding", "index": 0, "embedding": 0.7708950042724609}]
