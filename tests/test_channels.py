import json

import httpx
import pytest

from cropmon.alerts.channels import (
    ChannelNotConfigured,
    ConsoleChannel,
    Fast2SMSChannel,
    RateLimiter,
    TwilioChannel,
    build_channel,
    send_alert,
    validate_e164,
)


def test_validate_e164():
    assert validate_e164("+14155550123") == "+14155550123"
    assert validate_e164("+1 415 555 0123") == "+14155550123"
    for bad in ["14155550123", "+0123", "abc", "+1"]:
        with pytest.raises(ValueError):
            validate_e164(bad)


def test_build_channel_defaults_to_console():
    assert isinstance(build_channel(env={}), ConsoleChannel)
    assert isinstance(build_channel(env={"CROPMON_SMS_CHANNEL": "console"}), ConsoleChannel)


def test_build_channel_rejects_unknown():
    with pytest.raises(ValueError):
        build_channel("carrier-pigeon", env={})


def test_twilio_requires_credentials():
    with pytest.raises(ChannelNotConfigured):
        build_channel("twilio", env={})


def test_fast2sms_requires_key():
    with pytest.raises(ChannelNotConfigured):
        build_channel("fast2sms", env={})


def test_twilio_send_uses_expected_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(201, json={"sid": "SM123"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    channel = TwilioChannel("AC" + "0" * 32, "token", "+14155550100", client=client)
    result = channel.send("hello", to="+14155550123")

    assert result.ok
    assert "Accounts/AC" in captured["url"]
    assert "To=%2B14155550123" in captured["body"]
    assert captured["auth"].startswith("Basic ")


def test_twilio_requires_recipient():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    channel = TwilioChannel("AC" + "0" * 32, "token", "+14155550100", client=client)
    with pytest.raises(ChannelNotConfigured):
        channel.send("hello")


def test_fast2sms_send_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"return": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    channel = Fast2SMSChannel("key", recipient="+14155550199", client=client)
    result = channel.send("alert")

    assert result.ok
    assert "14155550199" in captured["body"]
    assert captured["auth"] == "key"


def test_rate_limiter_blocks_after_budget():
    now = [0.0]
    limiter = RateLimiter(max_per_hour=2, clock=lambda: now[0])
    assert limiter.allow()
    assert limiter.allow()
    assert not limiter.allow()
    now[0] = 3700
    assert limiter.allow()


def test_send_alert_reports_rate_limit():
    channel = ConsoleChannel()
    limiter = RateLimiter(max_per_hour=1, clock=lambda: 0.0)
    assert send_alert(channel, "one", limiter).ok
    result = send_alert(channel, "two", limiter)
    assert not result.ok
    assert "rate limit" in result.detail


def test_console_channel_prints(capsys):
    result = ConsoleChannel().send("hello")
    assert result.ok
    assert "hello" in capsys.readouterr().out


def test_no_credentials_in_environment_are_logged():
    # A configured channel must never echo the token back in its result detail.
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    channel = TwilioChannel("AC" + "0" * 32, "supersecret", "+14155550100", client=client)
    result = channel.send("hi", to="+14155550123")
    assert "supersecret" not in json.dumps(result.__dict__)
