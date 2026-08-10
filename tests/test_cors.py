def test_cors_preflight_allows_localhost(test_client) -> None:
    response = test_client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_cors_response_includes_origin_header(test_client) -> None:
    response = test_client.post(
        "/chat",
        json={"message": "poetry books", "history": []},
        headers={
            "Origin": "http://localhost:5173",
            "Accept": "text/event-stream",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
