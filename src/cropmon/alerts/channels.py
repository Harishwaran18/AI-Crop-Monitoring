"""Notification channels.

Credentials come only from the environment. No phone numbers or tokens are
stored in the repository, and every channel fails closed with a clear error
instead of silently pretending a message was delivered.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx

E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")
DEFAULT_SENDER = "CROPMON"


class ChannelNotConfigured(RuntimeError):
    """Raised when required credentials or recipients are missing."""


@dataclass
class NotificationResult:
    ok: bool
    channel: str
    detail: str


def validate_e164(number: str) -> str:
    candidate = number.strip().replace(" ", "").replace("-", "")
    if not E164_PATTERN.match(candidate):
        raise ValueError(f"phone number must be E.164 formatted (e.g. +14155550123), got {number!r}")
    return candidate


class SmsChannel(Protocol):
    name: str

    def send(self, message: str, to: str | None = None) -> NotificationResult: ...


class ConsoleChannel:
    """Default channel: prints the alert. Useful for demos and CI."""

    name = "console"

    def send(self, message: str, to: str | None = None) -> NotificationResult:
        print(f"[{self.name}] {message}")
        return NotificationResult(ok=True, channel=self.name, detail="printed to stdout")


class TwilioChannel:
    name = "twilio"

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        recipient: str | None = None,
        client: httpx.Client | None = None,
    ):
        if not account_sid or not auth_token:
            raise ChannelNotConfigured("Twilio requires TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = validate_e164(from_number)
        self.recipient = validate_e164(recipient) if recipient else None
        self._client = client or httpx.Client(timeout=10.0)

    def send(self, message: str, to: str | None = None) -> NotificationResult:
        target = to or self.recipient
        if not target:
            raise ChannelNotConfigured("no recipient: pass `to` or set ALERT_RECIPIENT")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        response = self._client.post(
            url,
            auth=(self.account_sid, self.auth_token),
            data={"From": self.from_number, "To": validate_e164(target), "Body": message},
        )
        return NotificationResult(
            ok=response.status_code < 300,
            channel=self.name,
            detail=f"status={response.status_code}",
        )


class Fast2SMSChannel:
    name = "fast2sms"

    def __init__(
        self,
        api_key: str,
        sender_id: str = "TXTIND",
        recipient: str | None = None,
        client: httpx.Client | None = None,
    ):
        if not api_key:
            raise ChannelNotConfigured("Fast2SMS requires FAST2SMS_API_KEY")
        self.api_key = api_key
        self.sender_id = sender_id
        self.recipient = validate_e164(recipient) if recipient else None
        self._client = client or httpx.Client(timeout=10.0)

    def send(self, message: str, to: str | None = None) -> NotificationResult:
        target = to or self.recipient
        if not target:
            raise ChannelNotConfigured("no recipient: pass `to` or set ALERT_RECIPIENT")
        response = self._client.post(
            "https://www.fast2sms.com/dev/bulkV2",
            headers={"authorization": self.api_key},
            data={
                "route": "v3",
                "sender_id": self.sender_id,
                "message": message,
                "numbers": validate_e164(target).lstrip("+"),
            },
        )
        return NotificationResult(
            ok=response.status_code < 300,
            channel=self.name,
            detail=f"status={response.status_code}",
        )


def build_channel(name: str | None = None, env: Mapping[str, str] | None = None) -> SmsChannel:
    """Build a channel from ``CROPMON_SMS_CHANNEL`` (console by default)."""
    env = env or os.environ
    selected = (name or env.get("CROPMON_SMS_CHANNEL") or "console").lower()
    if selected == "console":
        return ConsoleChannel()
    if selected == "twilio":
        return TwilioChannel(
            account_sid=env.get("TWILIO_ACCOUNT_SID", ""),
            auth_token=env.get("TWILIO_AUTH_TOKEN", ""),
            from_number=env.get("TWILIO_FROM_NUMBER", ""),
            recipient=env.get("ALERT_RECIPIENT"),
        )
    if selected == "fast2sms":
        return Fast2SMSChannel(
            api_key=env.get("FAST2SMS_API_KEY", ""),
            sender_id=env.get("FAST2SMS_SENDER_ID", "TXTIND"),
            recipient=env.get("ALERT_RECIPIENT"),
        )
    raise ValueError(f"unknown channel {selected!r}; expected console, twilio, or fast2sms")


class RateLimiter:
    """Token bucket so a runaway rule cannot drain an SMS budget."""

    def __init__(self, max_per_hour: int = 20, clock: Callable[[], float] = time.monotonic):
        self.max_per_hour = max_per_hour
        self._clock = clock
        self._events: list[float] = []

    def allow(self) -> bool:
        now = self._clock()
        self._events = [t for t in self._events if now - t < 3600]
        if len(self._events) >= self.max_per_hour:
            return False
        self._events.append(now)
        return True


def send_alert(channel: SmsChannel, message: str, limiter: RateLimiter | None = None) -> NotificationResult:
    limiter = limiter or RateLimiter()
    if not limiter.allow():
        return NotificationResult(ok=False, channel=channel.name, detail="rate limit exceeded")
    return channel.send(message)
