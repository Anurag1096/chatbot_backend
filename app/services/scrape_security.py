import ipaddress
import socket
from urllib.parse import urlparse

BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
    }
)


def is_public_http_url(url: str) -> bool:
    """Return True when URL uses http(s) and resolves to a public address."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    host = hostname.lower().rstrip(".")
    if host in BLOCKED_HOSTNAMES:
        return False

    if host.endswith(".localhost"):
        return False

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        return _is_public_ip(literal)

    try:
        addrinfo = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    if not addrinfo:
        return False

    for entry in addrinfo:
        ip = ipaddress.ip_address(entry[4][0])
        if not _is_public_ip(ip):
            return False

    return True


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )
