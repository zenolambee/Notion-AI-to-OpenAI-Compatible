from __future__ import annotations

import unittest

import httpx

from notionchat.account import create_account_from_cookie, validate_arena_cookie
from notionchat.arena_client import ArenaHttpClient


MODELS_PAGE = '''
<html><script>
{"initialModels":[
 {"id":"opaque-gpt-4o","publicName":"GPT-4o","organization":"openai","capabilities":{"outputCapabilities":{"text":true}}},
 {"id":"disabled","publicName":"Disabled","isDisabled":true}
]}
</script></html>
'''


class ArenaClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, text=MODELS_PAGE)
            if request.url.path == "/nextjs-api/stream/create-evaluation":
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=b'a0:"Halo"\nad:{"finishReason":"stop"}\n',
                )
            return httpx.Response(404, text="not found")

        self.client = ArenaHttpClient(
            create_account_from_cookie("arena-auth-prod-v1=test-token")
        )
        self.client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_collects_arena_records_for_non_streaming_completion(self) -> None:
        result = await self.client.chat_completion(
            model="arena-gpt-4o",
            messages=[{"role": "user", "content": "halo"}],
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "Halo")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        request = self.requests[-1]
        self.assertEqual(request.url.path, "/nextjs-api/stream/create-evaluation")
        payload = __import__("json").loads(request.content)
        self.assertEqual(payload["modelAId"], "opaque-gpt-4o")
        self.assertEqual(payload["userMessage"]["content"], "halo")
        self.assertEqual(payload["mode"], "direct")

    def test_parses_escaped_nextjs_model_data(self) -> None:
        escaped = (
            r'prefix {\"initialModels\":[{\"id\":\"opaque\",'
            r'\"publicName\":\"GPT-4o\"}]} suffix'
        )
        models = ArenaHttpClient._parse_models_page(escaped)
        self.assertEqual([(model.id, model.upstream_id) for model in models], [("GPT-4o", "opaque")])

    def test_parses_metadata_and_error_records(self) -> None:
        done = self.client._parse_stream_chunk('ad:{"finishReason":"length"}', "GPT-4o")
        self.assertTrue(done.done)
        self.assertEqual(done.finish_reason, "length")
        error = self.client._parse_stream_chunk('a3:"upstream rejected"', "GPT-4o")
        self.assertEqual(error.error, "Arena error: upstream rejected")

    def test_accepts_split_arena_auth_cookie(self) -> None:
        cookie = "arena-auth-prod-v1.0=first; arena-auth-prod-v1.1=second; cf_clearance=clear"
        valid, error = validate_arena_cookie(cookie)
        self.assertTrue(valid, error)
        account = create_account_from_cookie(cookie)
        self.assertEqual(account.token_v2, "firstsecond")
        self.assertEqual(account.full_cookie, cookie)


if __name__ == "__main__":
    unittest.main()
