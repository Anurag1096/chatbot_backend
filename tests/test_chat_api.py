from tests.conftest import parse_sse_body


def test_chat_streams_template_reply(test_client) -> None:
    response = test_client.post(
        "/chat",
        json={"message": "poetry books", "history": []},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200

    chunks = parse_sse_body(response.text)
    text = "".join(chunk.get("delta", "") for chunk in chunks if chunk.get("delta"))
    assert "Moonlit Verses" in text or "Poetry" in text
    assert any(chunk.get("done") for chunk in chunks)


def test_chat_empty_store_returns_placeholder(empty_client) -> None:
    response = empty_client.post(
        "/chat",
        json={"message": "hello there", "history": []},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200

    chunks = parse_sse_body(response.text)
    text = "".join(chunk.get("delta", "") for chunk in chunks if chunk.get("delta"))
    assert "Placeholder reply" in text
