"""Tests for track outreach: result parsing, rule checks, storage, drafting
on both channels, address lookup, queueing and the follow-up rules."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from harvey import outreach as ld
from harvey.config import EnvConfig, HarveyConfig
from harvey.integrations.email_finder import EmailResult
from harvey.models.prospect import Prospect
from harvey.state import StateManager


def _config(channels=("linkedin", "email"), **outreach) -> HarveyConfig:
    return HarveyConfig(
        persona=dict(name="Kris", company="Vinea", role="Founder", email="k@v.nl",
                     linkedin="", tone="plain"),
        product=dict(name="Vinea", description="d", pricing="p",
                     key_benefits=["b"], objection_responses={}),
        icp=dict(industries=["x"], company_size="", titles=["t"], geography=["NL"]),
        channels=dict(email=dict(provider="smtp")),
        outreach=dict(
            tracks={
                "growers": {"label": "Growers", "brief": "grow",
                            "linkedin": "li guidance", "email": "email guidance",
                            "channels": list(channels),
                            "queries": ["q1", "q2", "q3"]},
                "labs": {"label": "Labs", "brief": "labs brief",
                         "channels": ["email"], "queries": ["l1"],
                         "domains": {"Physical Intelligence": "physicalintelligence.company"}},
            },
            forbid_digits=True,
            signature="Kris\nFounder, Vinea",
            opt_out={"en": "Say so and I stop.", "nl": "Zeg het, dan stop ik."},
            **outreach,
        ),
    )


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


# ── Parsing ──

def test_normalize_profile_url_strips_country_and_query():
    assert ld.normalize_profile_url(
        "https://nl.linkedin.com/in/Jeroen-van-Kester-59080054?trk=x"
    ) == "https://www.linkedin.com/in/jeroen-van-kester-59080054"
    assert ld.normalize_profile_url("https://www.linkedin.com/company/acme") == ""


def test_country_hint():
    assert ld.country_hint("https://nl.linkedin.com/in/a-b") == "nl"
    assert ld.country_hint("https://www.linkedin.com/in/a-b") == ""


def test_parse_result_from_title():
    p = ld.parse_result(
        "https://nl.linkedin.com/in/jeroen-van-kester-59080054",
        "Jeroen van Kester - Teeltmanager - Lans Tomaten | LinkedIn",
        "Teeltmanager Lans Tomaten · Naaldwijk",
    )
    assert p["first_name"] == "Jeroen"
    assert p["last_name"] == "van Kester"
    assert p["headline"] == "Teeltmanager"
    assert p["company"] == "Lans Tomaten"


def test_parse_result_falls_back_to_slug_and_snippet():
    # A snippet that opens with the headline must not become the name.
    p = ld.parse_result(
        "https://nl.linkedin.com/in/luuk-vijverberg-434415134", "",
        "Teeltmanager Burgerweg bij Lans · Ervaring: Lansgroep",
    )
    assert (p["first_name"], p["last_name"]) == ("Luuk", "Vijverberg")
    assert p["headline"] == "Teeltmanager Burgerweg bij Lans"
    assert p["company"] == "Lans"  # from "bij Lans" in the headline


def test_parse_result_rejects_non_profiles():
    assert ld.parse_result("https://example.com/in/a-b", "A B - X", "") is None


def test_company_from_headline():
    assert ld.company_from_headline("Partner at Anterra Capital | Agrifood") == "Anterra Capital"
    assert ld.company_from_headline("Research Scientist @ Skild AI") == "Skild AI"
    assert ld.company_from_headline("Teeltmanager") == ""


# ── Rule checks ──

def test_check_linkedin_clean():
    assert ld.check_linkedin({"connect_note": "Hi Jan, building greenhouse robots.",
                              "first_message": "What were you shown?"},
                             200, True) == []


def test_check_linkedin_flags_length_digits_dashes():
    issues = ld.check_linkedin({"connect_note": "x" * 201 + " 2026 — ",
                                "first_message": ""}, 200, True)
    joined = " ".join(issues)
    assert "characters" in joined
    assert "digit" in joined
    assert "dash" in joined
    assert "first message is empty" in joined


def test_check_email():
    clean = {"email_subject": "greenhouse robots", "email_body": "Hi Jan, what were you shown?",
             "followup_body": "Hi Jan, a line back is plenty."}
    assert ld.check_email(clean, True, True) == []
    issues = " ".join(ld.check_email(
        {"email_subject": "", "email_body": "I hope this finds you well, see https://a.b "
         "and https://c.d " + "word " * 140, "followup_body": ""}, True, True))
    assert "subject is empty" in issues
    assert "over" in issues and "words" in issues
    assert "follow-up is empty" in issues
    assert "more than one link" in issues
    assert "banned phrases" in issues
    # No follow-up configured: an empty one is fine.
    assert ld.check_email(dict(clean, followup_body=""), True, False) == []


# ── Domains ──

def test_registrable_domain():
    assert ld.registrable_domain("https://www.shop.lanstomaten.nl/x") == "lanstomaten.nl"
    assert ld.registrable_domain("https://a.example.co.uk") == "example.co.uk"


def test_domain_matches_company():
    assert ld.domain_matches_company("lanstomaten.nl", "Lans Tomaten")
    assert ld.domain_matches_company("anterra.vc", "Anterra Capital")
    assert ld.domain_matches_company("physicalintelligence.company", "Physical Intelligence")
    # Sharing only a generic word is not a match.
    assert not ld.domain_matches_company("tomaten.nl", "Lans Tomaten")
    assert not ld.domain_matches_company("capital.com", "Anterra Capital")


@pytest.mark.asyncio
async def test_resolve_domain_known_map_then_search(monkeypatch):
    cfg = _config()
    search = ld._Search(cfg, EnvConfig())
    calls = []

    async def fake_run(query):
        calls.append(query)
        return [("https://nl.linkedin.com/company/lans", "", ""),
                ("https://www.kvk.nl/lans", "", ""),
                ("https://www.lanstomaten.nl/", "", "")]

    monkeypatch.setattr(search, "run", fake_run)
    labs = cfg.outreach.tracks["labs"]
    assert await ld.resolve_domain(search, labs, "Physical Intelligence (Pi)") == \
        "physicalintelligence.company"
    assert calls == []
    assert await ld.resolve_domain(search, cfg.outreach.tracks["growers"],
                                   "Lans Tomaten") == "lanstomaten.nl"


# ── Storage ──

@pytest.mark.asyncio
async def test_track_prospects_stay_out_of_email_pipeline(state):
    await state.add_prospect(Prospect(first_name="A", last_name="B", title="T",
                                      linkedin_url="https://www.linkedin.com/in/a-b",
                                      industry="growers", status="track"))
    assert await state.get_prospects_by_status("new") == []
    assert len(await state.prospects_needing_draft(track="growers")) == 1


@pytest.mark.asyncio
async def test_draft_storage_and_marking(state):
    pid = await state.add_prospect(Prospect(first_name="A", last_name="B", title="T",
                                            industry="growers", status="track"))
    did = await state.add_outreach_draft(prospect_id=pid, track="growers",
                                         connect_note="n", first_message="m")
    assert did
    # One draft per prospect.
    assert await state.add_outreach_draft(prospect_id=pid, track="growers") is None
    assert await state.prospects_needing_draft() == []

    assert await state.set_linkedin_draft_status(did, "sent")
    assert (await state.get_outreach_drafts(status="sent"))[0]["first_name"] == "A"
    with pytest.raises(ValueError):
        await state.set_linkedin_draft_status(did, "bogus")
    with pytest.raises(ValueError):
        await state.update_outreach_draft(did, email_state="bogus")


# ── Drafting ──

class FakeBrain:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def load_skills_for_agent(self, name):
        return ""

    async def think_json(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else None


async def _add(state, slug, track="growers", company="Lans Tomaten"):
    return await state.add_prospect(Prospect(
        first_name="Jan", last_name=slug, title="Teeltmanager", company=company,
        linkedin_url=f"https://www.linkedin.com/in/{slug}",
        source=f"linkedin_search:{track}",
        source_url=f"https://nl.linkedin.com/in/{slug}",
        industry=track, status="track"))


GOOD = {"fit": True, "reason": "grower", "language": "nl",
        "connect_note": "Hoi Jan, ik bouw kasrobots.",
        "first_message": "Welke automatisering heb je gezien?",
        "email_subject": "een vraag over automatisering",
        "email_body": "Hoi Jan, welke automatisering heb je al gezien?",
        "followup_body": "Hoi Jan, een paar regels terug is al genoeg."}


@pytest.mark.asyncio
async def test_draft_messages_writes_both_channels(state):
    await _add(state, "a")
    brain = FakeBrain([GOOD])
    counts = await ld.draft_messages(state, brain, _config())
    assert counts["drafted"] == 1
    prompt = brain.prompts[0]
    assert "li guidance" in prompt and "email guidance" in prompt
    assert '"followup_body"' in prompt
    assert "Dutch" in prompt  # nl subdomain → language hint
    d = (await state.get_outreach_drafts())[0]
    assert d["connect_note"] and d["email_body"] and d["followup_body"]
    assert d["status"] == "draft" and d["email_state"] == ""


@pytest.mark.asyncio
async def test_email_only_track_skips_linkedin(state):
    await _add(state, "a", track="labs", company="Physical Intelligence")
    brain = FakeBrain([dict(GOOD, connect_note="", first_message="")])
    await ld.draft_messages(state, brain, _config())
    assert "connect_note" not in brain.prompts[0]
    d = (await state.get_outreach_drafts(status="skipped"))[0]
    assert d["email_body"]


@pytest.mark.asyncio
async def test_draft_messages_retries_then_flags(state):
    await _add(state, "a")
    bad = dict(GOOD, connect_note="Since 2016 we build robots")
    brain = FakeBrain([bad, bad])
    counts = await ld.draft_messages(state, brain, _config())
    assert counts["flagged"] == 1
    assert "contains a digit" in brain.prompts[1]  # retry carried the feedback
    d = (await state.get_outreach_drafts())[0]
    assert "digit" in d["issues"] and d["email_issues"] == ""


@pytest.mark.asyncio
async def test_draft_messages_unfit_and_failed(state):
    await _add(state, "a")
    await _add(state, "b")
    brain = FakeBrain([{"fit": False, "reason": "seed company sales"}, None])
    counts = await ld.draft_messages(state, brain, _config())
    assert counts == {"drafted": 0, "flagged": 0, "unfit": 1, "failed": 1}
    # The failed one is retried next run; the unfit one is not.
    assert len(await state.prospects_needing_draft()) == 1


@pytest.mark.asyncio
async def test_draft_backfills_email_for_old_linkedin_drafts(state):
    pid = await _add(state, "a")
    await state.add_outreach_draft(prospect_id=pid, track="growers",
                                   connect_note="n", first_message="m")
    brain = FakeBrain([GOOD])
    counts = await ld.draft_messages(state, brain, _config())
    assert counts["drafted"] == 1
    assert "connect_note" not in brain.prompts[0]  # email only
    d = (await state.get_outreach_drafts())[0]
    assert d["connect_note"] == "n" and d["email_body"] == GOOD["email_body"]


# ── Finding ──

@pytest.mark.asyncio
async def test_find_people_rotates_queries(state, monkeypatch):
    seen = []

    async def fake_run(self, query):
        seen.append(query)
        return [("https://nl.linkedin.com/in/piet-de-vries-1a2b3c",
                 "Piet de Vries - Eigenaar - Kwekerij X | LinkedIn", "")]

    monkeypatch.setattr(ld._Search, "run", fake_run)
    found = await ld.find_people(state, _config(), EnvConfig(), track="growers",
                                 queries_per_run=2)
    assert found == {"growers": 1}  # same person twice → saved once
    await ld.find_people(state, _config(), EnvConfig(), track="growers",
                         queries_per_run=2)
    assert seen == ["q1", "q2", "q3", "q1"]
    p = (await state.prospects_needing_draft())[0]
    assert (p.first_name, p.last_name, p.company) == ("Piet", "de Vries", "Kwekerij X")
    assert p.status == "track"


@pytest.mark.asyncio
async def test_find_people_blocked_search_keeps_query_for_next_run(state, monkeypatch):
    async def blocked(self, query):
        return None

    monkeypatch.setattr(ld._Search, "run", blocked)
    await ld.find_people(state, _config(), EnvConfig(), track="growers",
                         queries_per_run=2)
    assert await state.get_setting("linkedin_query_cursor:growers") == "0"


def test_unknown_track_is_an_error():
    with pytest.raises(ValueError):
        ld.select_tracks(_config(), "nope")


# ── Addresses and queueing ──

async def _drafted(state, slug="a", **kw):
    pid = await _add(state, slug, **kw)
    track = kw.get("track", "growers")
    did = await state.add_outreach_draft(
        prospect_id=pid, track=track, language="nl",
        connect_note="n", first_message="m",
        email_subject=GOOD["email_subject"], email_body=GOOD["email_body"],
        followup_body=GOOD["followup_body"])
    return pid, did


@pytest.mark.asyncio
async def test_find_addresses(state, monkeypatch):
    pid, did = await _drafted(state, "a", track="labs", company="Physical Intelligence")
    _, did2 = await _drafted(state, "b", company="")
    # Headline that is just the company, known from the domains map.
    pid3, did3 = await _drafted(state, "c", track="labs", company="")
    await state.update_prospect_status(pid3, "track")
    import aiosqlite
    async with aiosqlite.connect(state.db_path) as db:
        await db.execute("UPDATE prospects SET title = 'Physical Intelligence' WHERE id = ?", (pid3,))
        await db.commit()
    seen = {}

    async def fake_find_email(first, last, domain, **kw):
        seen["domain"] = domain
        return EmailResult(email=f"jan.{last}@{domain}", status="verified")

    import harvey.integrations.email_finder as ef
    monkeypatch.setattr(ef, "find_email", fake_find_email)
    counts = await ld.find_addresses(state, _config(), EnvConfig())
    assert counts["found"] == 2 and counts["no_company"] == 1
    assert (await state.get_outreach_draft(did3))["email_state"] == "found"
    assert seen["domain"] == "physicalintelligence.company"
    p = await state.get_prospect(pid)
    assert (p.email, p.email_status) == ("jan.a@physicalintelligence.company", "verified")
    assert (await state.get_outreach_draft(did))["email_state"] == "found"
    assert (await state.get_outreach_draft(did2))["email_state"] == "no_company"


@pytest.mark.asyncio
async def test_queue_emails_stages_both_steps_once(state):
    pid, did = await _drafted(state)
    await state.update_prospect_email(pid, "jan@lanstomaten.nl", "verified")
    await state.update_outreach_draft(did, email_state="found")
    # A guess-only address is never queued.
    pid2, did2 = await _drafted(state, "b")
    await state.update_prospect_email(pid2, "jan@x.nl", "guess")
    await state.update_outreach_draft(did2, email_state="found")

    counts = await ld.queue_emails(state, _config())
    assert counts["queued"] == 1 and counts["skipped"] == 1
    items = sorted(await state.get_outbox(), key=lambda i: i["step"])
    assert [i["step"] for i in items] == [1, 2]
    assert all(i["status"] == "pending_review" for i in items)
    assert items[0]["campaign_id"] == "track:growers"
    assert items[1]["subject"] == "Re: " + GOOD["email_subject"]
    # Signature and the Dutch opt-out are appended; the body itself is first.
    assert items[0]["body"].startswith(GOOD["email_body"])
    assert items[0]["body"].endswith("Kris\nFounder, Vinea\n\nZeg het, dan stop ik.")
    gap = (datetime.fromisoformat(items[1]["send_at"])
           - datetime.fromisoformat(items[0]["send_at"]))
    assert gap == timedelta(days=7)
    assert (await state.get_outreach_draft(did))["email_state"] == "queued"
    # Queueing again does nothing.
    await state.update_outreach_draft(did, email_state="found")
    await ld.queue_emails(state, _config())
    assert len(await state.get_outbox()) == 2


@pytest.mark.asyncio
async def test_queue_holds_drafts_with_issues_and_skips_follow_up_when_off(state):
    pid, did = await _drafted(state)
    await state.update_prospect_email(pid, "jan@lanstomaten.nl", "verified")
    await state.update_outreach_draft(did, email_state="found", email_issues="digit")
    assert (await ld.queue_emails(state, _config()))["held_issues"] == 1
    await state.update_outreach_draft(did, email_issues="")
    await ld.queue_emails(state, _config(follow_up_days=0))
    assert [i["step"] for i in await state.get_outbox()] == [1]


# ── Sending: follow-ups wait for, thread under and keep their gap after step one ──

class FakeProvider:
    name = "smtp"

    def __init__(self):
        self.sent = []

    def is_configured(self):
        return True

    async def send_email(self, to, subject, body, thread_ref="", in_reply_to=""):
        from harvey.integrations.mail_provider import SendResult
        self.sent.append((subject, in_reply_to))
        mid = f"<m{len(self.sent)}@v.nl>"
        return SendResult(ok=True, message_id=mid, thread_ref=mid)


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_follow_up_waits_for_first_then_threads(state):
    from harvey.agents.sender import Sender
    pid = await _add(state, "a")
    await state.update_prospect_email(pid, "jan@lanstomaten.nl", "verified")
    past = (_now() - timedelta(days=10)).isoformat()
    body = "Hi Jan, a real question for you about automation."
    # Follow-up is due and approved, but step one is still pending review.
    await state.add_outbox_item(prospect_id=pid, campaign_id="track:growers", step=1,
                                to_email="jan@lanstomaten.nl", subject="q", body=body,
                                send_at=past, status="pending_review")
    await state.add_outbox_item(prospect_id=pid, campaign_id="track:growers", step=2,
                                to_email="jan@lanstomaten.nl", subject="Re: q", body=body,
                                send_at=past, status="approved")
    sender = Sender(None, state, _config(), EnvConfig())
    sender.provider = FakeProvider()
    sender.send_pacing = False
    await sender._drain_due()
    assert sender.provider.sent == []

    await state.approve_outbox()
    await sender._drain_due()
    # Step one went out ten days late, so the follow-up moved back by ten
    # days instead of going out in the same breath.
    assert sender.provider.sent == [("q", "")]
    step2 = await state.get_outbox_step("track:growers", pid, 2)
    assert step2["status"] == "approved"
    assert datetime.fromisoformat(step2["send_at"]) > _now() - timedelta(minutes=1)

    await state.update_outbox_item(step2["id"], send_at=past)
    await sender._drain_due()
    assert sender.provider.sent[-1] == ("Re: q", "<m1@v.nl>")


@pytest.mark.asyncio
async def test_follow_up_cancelled_when_first_was_rejected(state):
    from harvey.agents.sender import Sender
    pid = await _add(state, "a")
    await state.update_prospect_email(pid, "jan@lanstomaten.nl", "verified")
    past = (_now() - timedelta(days=1)).isoformat()
    await state.add_outbox_item(prospect_id=pid, campaign_id="track:growers", step=1,
                                to_email="jan@lanstomaten.nl", subject="q", body="b",
                                send_at=past, status="rejected")
    await state.add_outbox_item(prospect_id=pid, campaign_id="track:growers", step=2,
                                to_email="jan@lanstomaten.nl", subject="Re: q",
                                body="Hi Jan, follow up.", send_at=past, status="approved")
    sender = Sender(None, state, _config(), EnvConfig())
    sender.provider = FakeProvider()
    sender.send_pacing = False
    await sender._drain_due()
    assert (await state.get_outbox_step("track:growers", pid, 2))["status"] == "cancelled"


# ── Replies ──

def test_reply_context_only_for_track_prospects():
    cfg = _config()
    p = Prospect(first_name="J", last_name="K", source="linkedin_search:growers",
                 industry="growers")
    ctx = ld.reply_context(cfg, p)
    assert "grow" in ctx and "No digits" in ctx
    assert ld.reply_signature(cfg, p) == "Kris\nFounder, Vinea"
    other = Prospect(first_name="J", last_name="K", source="scout", industry="growers")
    assert ld.reply_context(cfg, other) == "" and ld.reply_signature(cfg, other) == ""


def test_imap_or_from_nests():
    from harvey.integrations.smtp_mail import SmtpImapProvider
    assert SmtpImapProvider._or_from(["a@x", "b@y", "c@z"]) == \
        'OR FROM "a@x" OR FROM "b@y" FROM "c@z"'


def test_heartbeat_refuses_with_tracks(monkeypatch, capsys):
    from harvey import cli
    import harvey.config as hc
    import harvey.main as hm
    started = []
    monkeypatch.setattr(hc, "load_config", lambda *a, **k: _config())
    monkeypatch.setattr(hm, "main", lambda: started.append(True))
    cli.cmd_run(None)
    assert started == [] and "won't start" in capsys.readouterr().out


# ── Replies and bounces on a shared inbox ──

def _shared_config():
    cfg = _config()
    cfg.channels.email.shared_inbox = True
    return cfg


class _ReplyBrain:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def load_prompt(self, *a, **k):
        return ""

    async def think(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


@pytest.mark.asyncio
async def test_track_reply_is_checked_retried_and_signed(state):
    from harvey.agents.handler import Handler
    from harvey.models.conversation import Conversation
    p = Prospect(first_name="Jan", last_name="K", source="linkedin_search:growers",
                 industry="growers")
    brain = _ReplyBrain(["We started in 2019 — call?", "Happy to talk, when suits you?"])
    h = Handler(brain, state, _config(), EnvConfig())
    out = await h._generate_response("interested", "sounds good", p,
                                     Conversation(id="c", prospect_id="x"))
    assert out == "Happy to talk, when suits you?\n\nKris\nFounder, Vinea"
    assert "contains a digit" in brain.prompts[1]
    assert "grow" in brain.prompts[0]  # track brief in the prompt
    assert h._reply_issues == []


@pytest.mark.asyncio
async def test_shared_inbox_bounce_matched_by_address_or_ignored(state):
    from harvey.agents.handler import Handler
    from harvey.integrations.mail_provider import InboundMessage
    pid = await _add(state, "a")
    await state.update_prospect_email(pid, "jan@lanstomaten.nl", "verified")
    item = await state.add_outbox_item(prospect_id=pid, campaign_id="track:growers",
                                       step=1, to_email="jan@lanstomaten.nl",
                                       subject="q", body="b", send_at="2026-01-01",
                                       status="approved")
    await state.update_outbox_item(item, status="sent", message_id="<m1@v.nl>")
    await state.add_outbox_item(prospect_id=pid, campaign_id="track:growers", step=2,
                                to_email="jan@lanstomaten.nl", subject="Re: q",
                                body="b", send_at="2026-01-08", status="approved")
    h = Handler(None, state, _shared_config(), EnvConfig())

    # The human's own bounce: not counted, nothing changes.
    await h._handle_bounce(InboundMessage(
        provider_id="x1", from_email="mailer-daemon@zoho.eu",
        subject="Undelivered Mail", body="friend@example.org: no such user",
        is_bounce=True))
    assert await state.get_setting("bounce_count", "") in ("", "0")
    assert (await state.get_prospect(pid)).email_status == "verified"

    # A report naming the address Harvey wrote to, without In-Reply-To.
    await h._handle_bounce(InboundMessage(
        provider_id="x2", from_email="mailer-daemon@zoho.eu",
        subject="Undelivered Mail", body="<jan@lanstomaten.nl>: mailbox unavailable",
        is_bounce=True))
    assert (await state.get_prospect(pid)).email_status == "invalid"
    assert (await state.get_outbox_step("track:growers", pid, 2))["status"] == "cancelled"
