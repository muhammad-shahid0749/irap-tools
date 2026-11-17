"""The refusal of requests that pages of other sites make, e.g. to the endpoints of forms and
uploads (CSRF)."""

import urllib.parse

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

#: The methods that do not change anything, which other sites may use, e.g. for links.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def is_same_origin(origin: str | None, host: str) -> bool:
    """Returns whether a request with the `Origin` header `origin` comes from a page of `host`, the
    `Host` header. Browsers send `Origin` with every POST, so a request without one is from
    another client, e.g. curl, which does not have the cookie of a user."""
    if origin is None:
        return True
    return urllib.parse.urlsplit(origin).netloc.lower() == host.lower()


class SameOriginMiddleware:
    """Refuses requests other than GET, HEAD and OPTIONS from a page of another site, also of
    another host of the same domain, so that it cannot sign a user in or out, or upload or change
    something as them. Responds with 403."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in _SAFE_METHODS:
            headers = Headers(scope=scope)
            if not is_same_origin(headers.get("origin"), headers.get("host", "")):
                response = PlainTextResponse("The request comes from a page of another site.",
                                             status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
