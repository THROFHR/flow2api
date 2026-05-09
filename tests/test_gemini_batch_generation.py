import unittest
from unittest.mock import AsyncMock, patch

from src.api import routes
from src.core.models import GeminiContent, GeminiGenerateContentRequest, GeminiPart
from src.services.flow_client import FlowClient


class GeminiBatchRequestNormalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalize_gemini_request_merges_shared_prompt_with_batch_prompts(self):
        request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(
                    role="user",
                    parts=[GeminiPart(text="木桌上的红苹果")],
                )
            ],
            systemInstruction=GeminiContent(
                parts=[GeminiPart(text="Return an image only.")]
            ),
            batchPrompts=[
                "赛博朋克海报风格",
                "黑白纪实摄影风格",
            ],
        )

        normalized = await routes._normalize_gemini_request(
            "gemini-3.1-flash-image-landscape",
            request,
        )

        self.assertEqual(normalized.prompt, "Return an image only.\n\n木桌上的红苹果")
        self.assertEqual(
            normalized.batch_prompts,
            [
                "Return an image only.\n\n木桌上的红苹果\n\n赛博朋克海报风格",
                "Return an image only.\n\n木桌上的红苹果\n\n黑白纪实摄影风格",
            ],
        )


class GeminiBatchResponsePayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_gemini_success_payload_emits_multiple_candidates(self):
        async def fake_build_image_parts(uri: str):
            return [{"fileData": {"fileUri": uri, "mimeType": "image/png"}}]

        payload = {
            "images": [
                {"url": "https://example.com/1.png"},
                {"url": "https://example.com/2.png"},
            ]
        }

        with patch.object(routes, "_build_image_parts_from_uri", AsyncMock(side_effect=fake_build_image_parts)):
            result = await routes._build_gemini_success_payload(payload, "gemini-3.1-flash-image-landscape")

        self.assertEqual(result["modelVersion"], "gemini-3.1-flash-image-landscape")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(
            result["candidates"][0]["content"]["parts"][0]["fileData"]["fileUri"],
            "https://example.com/1.png",
        )
        self.assertEqual(
            result["candidates"][1]["content"]["parts"][0]["fileData"]["fileUri"],
            "https://example.com/2.png",
        )


class FlowClientBatchImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = FlowClient(proxy_manager=None)
        self.client._acquire_image_launch_gate = AsyncMock(return_value=(True, 0, 0))
        self.client._release_image_launch_gate = AsyncMock()
        self.client._get_recaptcha_token = AsyncMock(return_value=("recaptcha-token", "browser-1"))
        self.client._notify_browser_captcha_request_finished = AsyncMock()

    async def test_generate_images_batch_reuses_same_reference_inputs(self):
        captured = {}

        async def fake_make_image_generation_request(url, json_data, at, attempt_trace=None):
            captured["url"] = url
            captured["json_data"] = json_data
            return {"media": [{"name": "m1"}, {"name": "m2"}]}

        self.client._make_image_generation_request = AsyncMock(side_effect=fake_make_image_generation_request)

        result, session_id, perf_trace = await self.client.generate_images_batch(
            at="at-token",
            project_id="project-1",
            prompts=["海报风", "摄影风"],
            model_name="GEM_PIX",
            aspect_ratio="IMAGE_ASPECT_RATIO_SQUARE",
            image_inputs=[{"name": "ref-1", "imageInputType": "IMAGE_INPUT_TYPE_REFERENCE"}],
            token_id=1,
            token_image_concurrency=2,
        )

        self.assertEqual(result["media"][0]["name"], "m1")
        self.assertTrue(session_id)
        self.assertEqual(perf_trace["batch_size"], 2)
        self.assertTrue(captured["url"].endswith("/flowMedia:batchGenerateImages"))
        self.assertEqual(len(captured["json_data"]["requests"]), 2)
        self.assertEqual(
            captured["json_data"]["requests"][0]["structuredPrompt"]["parts"][0]["text"],
            "海报风",
        )
        self.assertEqual(
            captured["json_data"]["requests"][1]["structuredPrompt"]["parts"][0]["text"],
            "摄影风",
        )
        self.assertEqual(
            captured["json_data"]["requests"][0]["imageInputs"],
            [{"name": "ref-1", "imageInputType": "IMAGE_INPUT_TYPE_REFERENCE"}],
        )
        self.assertEqual(
            captured["json_data"]["requests"][1]["imageInputs"],
            [{"name": "ref-1", "imageInputType": "IMAGE_INPUT_TYPE_REFERENCE"}],
        )


if __name__ == "__main__":
    unittest.main()
