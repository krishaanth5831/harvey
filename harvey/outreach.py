"""Track outreach — Harvey finds the people and writes the words, on
LinkedIn and by email, for a few named audiences ("tracks").

Each track in ``outreach.tracks`` (growers, investors, labs, ...) carries
its own brief, its channels and its search queries. Per person:

1. FIND. Web-search queries (``site:linkedin.com/in ...``) turn public
   search results into prospects with status ``track``, a status the
   upstream email pipeline never picks up. The track name is stored in
   ``prospect.industry``.
2. DRAFT. One brain call per person returns a fit verdict and, for the
   track's channels, a LinkedIn connection note and first message and a
   cold email and follow-up, grounded only in the search result.
   Deterministic checks (length, digits, dashes, banned phrases, links)
   run on every draft; one retry gets the broken rules as feedback, and
   anything still broken is stored with its issues listed.
3. LINKEDIN. Never automated: logging in and sending breaks LinkedIn's
   terms and risks the sender's account. ``harvey outreach list`` prints
   the drafts; the human sends each and marks it sent.
4. EMAIL. ``addresses`` resolves the company's domain and finds and
   verifies one address. ``queue`` stages the email (and the follow-up)
   in the outbox as pending_review, the same approval ladder, pre-send
   gate, bounce kill switch and stop-on-reply as the rest of Harvey.
   ``send`` polls replies and drains approved mail; nothing goes out
   without approval.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, quote_plus, urlparse

import fcntl
from pathlib import Path

from harvey.agents.scout import MAX_RESULTS_PER_QUERY, Scout
from harvey.brain import Brain
from harvey.paths import PROJECT_ROOT
from harvey.config import EnvConfig, HarveyConfig, OutreachTrack
from harvey.gate import BANNED_PHRASES, MERGE_TAG_RE, URL_RE
from harvey.models.prospect import Prospect
from harvey.state import StateManager

logger = logging.getLogger("harvey.outreach")

# Free search backends throttle fast (DuckDuckGo starts answering 403 after a
# handful of queries), so a run spends only a few and the next run picks up
# where this one stopped.
DEFAULT_QUERIES_PER_RUN = 3
DEFAULT_DRAFTS_PER_RUN = 10
# Two draft runs at once pick the same people and pay for every draft twice
# (prospects_needing_draft only sees a draft once it is stored). One lock
# file, held for the whole run; a second run says so and leaves.
DRAFT_LOCK = PROJECT_ROOT / "data" / ".draft.lock"


class DraftRunInProgress(RuntimeError):
    pass


def _take_draft_lock(state: StateManager | None = None):
    # Next to the database in use, so a test database never fights the live one.
    path = Path(state.db_path).with_name(".draft.lock") if state and getattr(state, "db_path", "") else DRAFT_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise DraftRunInProgress(
            "another 'harvey outreach draft' is still running; wait for it "
            "or stop it before starting a new one")
    return fh
DEFAULT_LOOKUPS_PER_RUN = 10

# LinkedIn caps the first message to a new connection loosely; short wins.
FIRST_MESSAGE_MAX_WORDS = 90
EMAIL_SUBJECT_MAX_CHARS = 70
EMAIL_MAX_WORDS = 130
FOLLOWUP_MAX_WORDS = 70

# Outbox campaign id for a track's emails. The outbox's unique index on
# (campaign_id, prospect_id, step) makes a second queue of the same person
# a no-op.
CAMPAIGN_PREFIX = "track:"

_LINKEDIN_PROFILE = re.compile(
    r"^https?://(?:[a-z]{2,3}\.|www\.)?linkedin\.com/in/([^/?#]+)", re.I
)
# "Jeroen van Kester - Teeltmanager - Lans Tomaten | LinkedIn"
_TITLE_SPLIT = re.compile(r"\s+[-–—|]\s+")
_BANNED_PUNCTUATION = ("—", "–")
# Lower-case surname particles for names rebuilt from a URL slug.
_NAME_PARTICLES = {"van", "de", "der", "den", "het", "ten", "ter", "te",
                   "von", "zu", "du", "le", "la"}
_LANGUAGE_BY_SUBDOMAIN = {
    "nl": "Dutch", "de": "German", "at": "German", "ch": "German",
    "fr": "French", "be": "Dutch or French (Belgium)", "uk": "English",
}
# "Investment Manager at Rockstart", "Teeltmanager bij Lans Tomaten"
_HEADLINE_COMPANY = re.compile(r"(?:\bat|\bbij|\bbei|\bchez|@)\s+(.+)$", re.I)
# Words that say what kind of company it is, not which one. A domain that
# only shares one of these with the company name is not its domain.
_GENERIC_COMPANY_WORDS = {
    "bv", "b.v.", "nv", "n.v.", "vof", "v.o.f.", "gmbh", "kg", "ag", "sa",
    "sas", "sarl", "ltd", "llc", "inc", "plc", "co", "the", "and", "en", "und",
    "et", "de", "het", "van", "der", "group", "groep", "holding", "holdings",
    "capital", "ventures", "venture", "partners", "fund", "funds", "invest",
    "investments", "vc", "labs", "lab", "ai", "robotics", "technologies",
    "technology", "tech", "research", "institute", "company", "international",
    "europe", "netherlands", "nederland", "tomaten", "tomatoes", "kwekerij",
    "kwekerijen", "glastuinbouw", "greenhouse", "greenhouses", "farms", "farm",
    "growers", "fresh", "foods", "food",
}
# Results that are about a company rather than its own site.
_NOT_A_COMPANY_SITE = (
    "linkedin.com", "kvk.nl", "drimble.nl", "oozo.nl", "bedrijvenregister",
    "northdata", "opencorporates", "dnb.com", "pitchbook.com", "tracxn.com",
    "cbinsights.com", "wellfound.com", "f6s.com", "dealroom.co",
    "bloomberg.com", "techcrunch.com", "theorg.com", "rocketreach.co",
    "signalhire.com", "lusha.com", "zoominfo.com", "crunchbase.com",
    "wikipedia.org", "facebook.com", "instagram.com", "youtube.com",
    "x.com", "twitter.com", "glassdoor", "indeed", "werkzoeken.nl",
    "nationalevacaturebank.nl", "google.com",
)
_TWO_LEVEL_SUFFIXES = ("co.uk", "org.uk", "ac.uk", "com.au", "co.jp", "co.nz",
                       "com.br", "co.za", "co.il")


def campaign_id_for(track: str) -> str:
    return f"{CAMPAIGN_PREFIX}{track}"


# ── FIND ──

def normalize_profile_url(url: str) -> str:
    """Country subdomains and query strings point at the same person;
    one canonical form keeps the prospects table's unique index honest."""
    m = _LINKEDIN_PROFILE.match((url or "").strip())
    if not m:
        return ""
    return f"https://www.linkedin.com/in/{m.group(1).lower()}"


def country_hint(url: str) -> str:
    """nl.linkedin.com → 'nl'. Search results carry the viewer-facing
    subdomain of the profile's country, a free hint at the language."""
    host = urlparse(url or "").hostname or ""
    sub = host.split(".")[0] if host.count(".") >= 2 else ""
    return "" if sub == "www" else sub


def parse_result(url: str, title: str, snippet: str) -> dict | None:
    """Turn one search result into name / headline / company, or None.

    Result titles read "First Last - Headline - Company | LinkedIn". The
    snippet usually starts with the headline, never the name, so the name
    comes from the title, then the URL slug.
    """
    canonical = normalize_profile_url(url)
    if not canonical:
        return None

    title = re.sub(r"\s*\|\s*LinkedIn.*$", "", title or "").strip()
    parts = [p.strip() for p in _TITLE_SPLIT.split(title) if p.strip()]
    name = parts[0] if parts else ""
    headline = parts[1] if len(parts) > 1 else ""
    company = parts[2] if len(parts) > 2 else ""

    if not name or len(name.split()) < 2 or any(ch.isdigit() for ch in name):
        slug = canonical.rsplit("/", 1)[-1]
        slug = re.sub(r"-[0-9a-f]{6,}$", "", slug)
        slug = re.sub(r"-\d+$", "", slug)
        tokens = [t for t in slug.split("-") if t and not t.isdigit()]
        if len(tokens) < 2:
            return None
        name = " ".join(t if i and t in _NAME_PARTICLES else t.capitalize()
                        for i, t in enumerate(tokens))

    first, _, last = name.partition(" ")
    if not headline and snippet:
        # Snippets open with the headline: "Teeltmanager Lans Tomaten · ..."
        headline = snippet.split("·")[0].strip()[:120]
    if not headline:
        return None
    if not company:
        company = company_from_headline(headline)
    return {
        "first_name": first.strip(),
        "last_name": last.strip(),
        "headline": headline,
        "company": company,
        "linkedin_url": canonical,
        "source_url": url,
        "snippet": (snippet or "").strip(),
    }


def company_from_headline(headline: str) -> str:
    """'Partner at Anterra Capital | Agrifood' → 'Anterra Capital'."""
    m = _HEADLINE_COMPANY.search(headline or "")
    if not m:
        return ""
    company = re.split(r"\s*[|,·•]\s*|\s+[-–—]\s+", m.group(1))[0]
    return company.strip(" .")[:80]


class _Search:
    """Web search that keeps result titles (Scout's drops them, and the
    title is where the person's name is). Serper first when a key is set,
    then DuckDuckGo. Bing ignores ``site:`` and returns noise, so it's out."""

    def __init__(self, config: HarveyConfig, env: EnvConfig):
        # Scout's _fetch brings retries, backoff, rotating user agents and
        # block-page detection; it needs no brain or state.
        self._scout = Scout(None, None, config, env)
        self._serper_key = getattr(env, "serper_api_key", "") or ""

    async def run(self, query: str) -> list[tuple[str, str, str]] | None:
        """Results, [] when the search ran and found nothing, or None when
        every backend was blocked or unreachable."""
        if self._serper_key:
            results = await self._serper(query)
            if results:
                return results
        return await self._duckduckgo(query)

    async def _serper(self, query: str) -> list[tuple[str, str, str]]:
        resp = await self._scout._fetch(
            "https://google.serper.dev/search",
            method="POST",
            json_body={"q": query, "num": MAX_RESULTS_PER_QUERY},
            headers={"X-API-KEY": self._serper_key,
                     "Content-Type": "application/json"},
            retries=2,
        )
        try:
            data = resp.json() if resp is not None else {}
        except ValueError:
            return []
        out = []
        for item in (data.get("organic") or []) if isinstance(data, dict) else []:
            if isinstance(item, dict) and item.get("link"):
                out.append((item["link"], item.get("title", "") or "",
                            item.get("snippet", "") or ""))
        return out

    async def _duckduckgo(self, query: str) -> list[tuple[str, str, str]] | None:
        resp = await self._scout._fetch(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        )
        if resp is None:
            logger.warning(
                "DuckDuckGo is blocking searches (it throttles after a few "
                "queries). Try again later, or add SERPER_API_KEY to .env "
                "(2,500 free searches)."
            )
            return None
        soup = self._scout._safe_soup(resp.text)
        out = []
        for result in soup.select(".result") if soup else []:
            link = result.select_one(".result__a")
            if not (link and link.get("href")):
                continue
            href = self._scout._attr_str(link.get("href"))
            if "uddg=" in href:
                real = parse_qs(urlparse(href).query).get("uddg", [])
                href = real[0] if real else href
            snippet_el = result.select_one(".result__snippet")
            out.append((href, link.get_text(" ", strip=True),
                        snippet_el.get_text(" ", strip=True) if snippet_el else ""))
        return out[:MAX_RESULTS_PER_QUERY]


def select_tracks(config: HarveyConfig, track: str | None) -> dict[str, OutreachTrack]:
    tracks = config.outreach.tracks
    if not tracks:
        raise ValueError(
            "No outreach tracks configured. Add outreach.tracks to your "
            "harvey config."
        )
    if track:
        if track not in tracks:
            raise ValueError(
                f"Unknown track '{track}'. Configured: {', '.join(tracks)}"
            )
        return {track: tracks[track]}
    return dict(tracks)


async def find_people(
    state: StateManager,
    config: HarveyConfig,
    env: EnvConfig,
    track: str | None = None,
    queries_per_run: int = DEFAULT_QUERIES_PER_RUN,
) -> dict[str, int]:
    """Run the next few search queries of each track; save new profiles.

    Returns {track: new prospects}. Query order rotates through a cursor in
    the settings table, so repeated runs walk the whole list.
    """
    tracks = select_tracks(config, track)
    search = _Search(config, env)
    found: dict[str, int] = {}

    for name, spec in tracks.items():
        found[name] = 0
        if not spec.queries:
            logger.warning(f"Track '{name}' has no queries; nothing to search.")
            continue
        cursor_key = f"linkedin_query_cursor:{name}"
        start = int(await state.get_setting(cursor_key, "0") or 0)
        done = 0
        for i in range(min(queries_per_run, len(spec.queries))):
            query = spec.queries[(start + i) % len(spec.queries)]
            results = await search.run(query)
            if results is None:
                # Blocked: stop here and retry this query next run.
                break
            done += 1
            logger.info(f"[{name}] {len(results)} results for: {query}")
            for url, title, snippet in results:
                person = parse_result(url, title, snippet)
                if not person:
                    continue
                if await state.prospect_exists(linkedin_url=person["linkedin_url"]):
                    continue
                notes = f"Search result: {title}".strip()
                if person["snippet"]:
                    notes += f"\nSnippet: {person['snippet']}"
                hint = country_hint(person["source_url"])
                if hint:
                    notes += f"\nLinkedIn country subdomain: {hint}"
                await state.add_prospect(Prospect(
                    first_name=person["first_name"],
                    last_name=person["last_name"],
                    title=person["headline"],
                    company=person["company"],
                    linkedin_url=person["linkedin_url"],
                    source=f"linkedin_search:{name}",
                    source_url=person["source_url"],
                    industry=name,
                    status="track",
                    personalization_notes=notes,
                ))
                found[name] += 1
        await state.set_setting(cursor_key, str((start + done) % len(spec.queries)))
    return found


# ── DRAFT ──

def _text_issues(label: str, text: str, forbid_digits: bool) -> list[str]:
    issues = []
    if forbid_digits and any(ch.isdigit() for ch in text):
        issues.append(f"{label} contains a digit; no numbers allowed")
    if any(p in text for p in _BANNED_PUNCTUATION):
        issues.append(f"{label} uses a dash; use a comma or a full stop")
    lowered = text.lower()
    hits = [p for p in BANNED_PHRASES if p in lowered]
    if hits:
        issues.append(f"{label} uses banned phrases: {', '.join(hits[:3])}")
    if MERGE_TAG_RE.search(text):
        issues.append(f"{label} contains a template placeholder")
    return issues


def check_linkedin(draft: dict, note_max_chars: int, forbid_digits: bool) -> list[str]:
    """Deterministic LinkedIn rule checks. Returns the broken rules."""
    issues = []
    note = draft.get("connect_note", "") or ""
    message = draft.get("first_message", "") or ""
    if not note.strip():
        issues.append("connection note is empty")
    if len(note) > note_max_chars:
        issues.append(f"connection note is {len(note)} characters; "
                      f"LinkedIn allows {note_max_chars}")
    if not message.strip():
        issues.append("first message is empty")
    if len(message.split()) > FIRST_MESSAGE_MAX_WORDS:
        issues.append(f"first message is over {FIRST_MESSAGE_MAX_WORDS} words")
    if "http" in note.lower():
        issues.append("connection note contains a link")
    issues += _text_issues("connection note", note, forbid_digits)
    issues += _text_issues("first message", message, forbid_digits)
    return issues


def check_email(draft: dict, forbid_digits: bool, follow_up: bool) -> list[str]:
    """Deterministic email rule checks. Returns the broken rules."""
    issues = []
    subject = draft.get("email_subject", "") or ""
    body = draft.get("email_body", "") or ""
    followup = draft.get("followup_body", "") or ""
    if not subject.strip():
        issues.append("email subject is empty")
    if len(subject) > EMAIL_SUBJECT_MAX_CHARS:
        issues.append(f"email subject is over {EMAIL_SUBJECT_MAX_CHARS} characters")
    if not body.strip():
        issues.append("email body is empty")
    if len(body.split()) > EMAIL_MAX_WORDS:
        issues.append(f"email body is over {EMAIL_MAX_WORDS} words")
    if follow_up:
        if not followup.strip():
            issues.append("follow-up is empty")
        if len(followup.split()) > FOLLOWUP_MAX_WORDS:
            issues.append(f"follow-up is over {FOLLOWUP_MAX_WORDS} words")
    if len(URL_RE.findall(body)) + len(URL_RE.findall(followup)) > 1:
        issues.append("more than one link across the email and follow-up")
    issues += _text_issues("email subject", subject, forbid_digits)
    issues += _text_issues("email body", body, forbid_digits)
    issues += _text_issues("follow-up", followup, forbid_digits)
    return issues


def _facts(prospect: Prospect) -> str:
    lines = [
        f"- Name: {prospect.full_name()}",
        f"- Headline: {prospect.title}",
    ]
    if prospect.company:
        lines.append(f"- Company (from the result): {prospect.company}")
    if prospect.personalization_notes:
        lines.append(prospect.personalization_notes)
    hint = country_hint(prospect.source_url)
    if hint in _LANGUAGE_BY_SUBDOMAIN:
        lines.append(f"- Likely language from the LinkedIn subdomain: "
                     f"{_LANGUAGE_BY_SUBDOMAIN[hint]}")
    return "\n".join(lines)


def _style_rules(config: HarveyConfig) -> list[str]:
    rules = [
        "Write in the person's own language. Use the likely-language hint "
        "and the language of the search result; when unsure, English.",
        "Every claim about the person comes from the FACTS. If a fact isn't "
        "there, you don't know it: no invented posts, mutual contacts, "
        "company news or compliments.",
        "Every claim about the company comes from the product knowledge "
        "and the track brief.",
        "Plain text. No em or en dashes, no exclamation marks, no emoji, no "
        "hashtags, no 'I hope this finds you well', no 'I'd love to'.",
    ]
    if config.outreach.forbid_digits:
        rules.append("No digits anywhere: no figures, prices, percentages, "
                     "counts, years or dates. Describe the shape instead.")
    return rules


def build_draft_prompt(
    config: HarveyConfig, skills: str, track_name: str, spec: OutreachTrack,
    prospect: Prospect, channels: list[str],
) -> str:
    """The drafting prompt for the given channels (a subset of the track's)."""
    persona = config.persona
    follow_up = config.outreach.follow_up_days > 0
    rules = _style_rules(config)
    sections = []
    keys = ['"fit": true', '"reason": "one line on why they fit or don\'t"',
            '"language": "nl|de|fr|en"']

    if "linkedin" in channels:
        li = config.channels.linkedin
        rules += [
            f"The LinkedIn connection note is at most {li.note_max_chars} "
            "characters, counted with spaces. It says who you are and gives "
            "one concrete reason to connect. No pitch, no link, no ask for a "
            "meeting.",
            "The LinkedIn first message goes out after they accept: under "
            f"{FIRST_MESSAGE_MAX_WORDS} words, one point, exactly one question "
            "or ask, and it ends on that ask with no sign-off flourish.",
        ]
        sections.append(f"LINKEDIN (sent by hand from {persona.name}'s own "
                        f"account):\n{spec.linkedin or '(no extra guidance)'}")
        keys += ['"connect_note": "..."', '"first_message": "..."']

    if "email" in channels:
        rules += [
            f"The email subject is short and plain, under "
            f"{EMAIL_SUBJECT_MAX_CHARS} characters, lower-key than a headline, "
            "no clickbait.",
            f"The email body opens with a greeting by first name, stays under "
            f"{EMAIL_MAX_WORDS} words, makes one point and ends on exactly one "
            "ask. No signature and no unsubscribe line: both are added "
            "automatically.",
            "At most one link across the email and the follow-up, and only "
            "where the track guidance allows one.",
        ]
        if follow_up:
            rules.append(
                f"The follow-up goes in the same thread "
                f"{'a week' if config.outreach.follow_up_days == 7 else 'some days'} "
                f"later if there is no reply: under {FOLLOWUP_MAX_WORDS} words, "
                "a greeting, one new angle or an easier version of the ask, "
                "never 'just following up', 'bumping this' or guilt."
            )
            keys += ['"email_subject": "..."', '"email_body": "..."',
                     '"followup_body": "..."']
        else:
            keys += ['"email_subject": "..."', '"email_body": "..."']
        sections.append(f"EMAIL (sent from {persona.email} after "
                        f"{persona.name} approves it):\n"
                        f"{spec.email or '(no extra guidance)'}")

    return f"""You draft outreach that {persona.name} ({persona.role},
{persona.company}) will read, edit and approve before anything reaches the
person. Tone: {persona.tone}.

TRACK: {spec.label or track_name}
{spec.brief}

{chr(10).join(sections)}

RULES:
{chr(10).join(f"{i}. {r}" for i, r in enumerate(rules, 1))}
{skills}

FACTS about the person, from one public search result:
{_facts(prospect)}

First decide fit: does this person plausibly belong to this track? A
search result can be a namesake, a student, a supplier or the wrong
country. If not, set "fit" to false, give the reason, and leave the
messages empty.

Return JSON:
{{{", ".join(keys)}}}"""


_SIGNOFF_RE = re.compile(
    r"\n+\s*(?:(?:best|thanks|kind regards|cheers|groet(?:en)?|met vriendelijke groet),?\s*)?{name}\s*$",
    re.I)


def _strip_signoff(text: str, first_name: str) -> str:
    """The signature is appended on send, so a sign-off the model added
    anyway ("Krishaanth" on its own line) would double it."""
    if not text or not first_name:
        return text or ""
    pat = _SIGNOFF_RE.pattern.replace("{name}", re.escape(first_name))
    return re.sub(pat, "", text, flags=re.I).rstrip()


def _check(draft: dict, config: HarveyConfig, channels: list[str]) -> tuple[list[str], list[str]]:
    """(linkedin issues, email issues) for the channels drafted."""
    o = config.outreach
    li = (check_linkedin(draft, config.channels.linkedin.note_max_chars,
                         o.forbid_digits) if "linkedin" in channels else [])
    em = (check_email(draft, o.forbid_digits, o.follow_up_days > 0)
          if "email" in channels else [])
    return li, em


async def _draft_one(
    brain: Brain, config: HarveyConfig, skills: str, name: str,
    spec: OutreachTrack, prospect: Prospect, channels: list[str],
) -> tuple[dict | None, list[str], list[str]]:
    """One brain call, plus one retry when deterministic checks fail."""
    prompt = build_draft_prompt(config, skills, name, spec, prospect, channels)
    draft = await brain.think_json(prompt, agent="outreach", task="draft")
    if not isinstance(draft, dict):
        return None, [], []
    if draft.get("fit") is False:
        return draft, [], []

    li_issues, em_issues = _check(draft, config, channels)
    if li_issues or em_issues:
        retry = await brain.think_json(
            prompt
            + "\n\nYour previous draft broke these rules; fix every one:\n"
            + "\n".join(f"- {i}" for i in li_issues + em_issues)
            + f"\nPrevious draft: {json.dumps(draft, ensure_ascii=False)}",
            agent="outreach", task="redraft",
        )
        if isinstance(retry, dict) and retry.get("fit") is not False:
            draft = retry
            li_issues, em_issues = _check(draft, config, channels)
    return draft, li_issues, em_issues


def _s(draft: dict, key: str) -> str:
    return str(draft.get(key, "") or "").strip()


async def draft_messages(
    state: StateManager,
    brain: Brain,
    config: HarveyConfig,
    track: str | None = None,
    limit: int = DEFAULT_DRAFTS_PER_RUN,
) -> dict[str, int]:
    """Draft for up to `limit` people: new prospects first (every channel
    of their track), then drafts that predate a track's email channel
    (email only).

    Returns counts: drafted, flagged (stored with unresolved issues),
    unfit, failed (no usable brain answer; retried next run).
    """
    tracks = select_tracks(config, track)
    skills = brain.load_skills_for_agent("outreach")
    counts = {"drafted": 0, "flagged": 0, "unfit": 0, "failed": 0}
    lock = _take_draft_lock(state)
    try:
        return await _draft_loop(state, brain, config, tracks, skills, counts, track, limit)
    finally:
        lock.close()


async def _draft_loop(state, brain, config, tracks, skills, counts, track, limit):
    worked = 0
    for prospect in await state.prospects_needing_draft(track=track, limit=limit):
        name = prospect.industry
        spec = tracks.get(name)
        if spec is None:
            continue
        worked += 1
        channels = list(spec.channels)
        if "email" in channels and not (prospect.email or _known_domain(spec, prospect.company or "")):
            # No address and no known company domain: nothing to send the
            # email to, so don't pay for drafting one. LinkedIn only.
            channels.remove("email")
        draft, li_issues, em_issues = await _draft_one(
            brain, config, skills, name, spec, prospect, channels)
        if draft is None:
            counts["failed"] += 1
            continue
        for key in ("email_body", "followup_body", "first_message"):
            if isinstance(draft.get(key), str):
                draft[key] = _strip_signoff(draft[key], config.persona.name.split()[0])
        if draft.get("fit") is False:
            await state.add_outreach_draft(
                prospect_id=prospect.id, track=name,
                fit_reason=_s(draft, "reason"), status="unfit",
            )
            counts["unfit"] += 1
            continue
        await state.add_outreach_draft(
            prospect_id=prospect.id,
            track=name,
            language=_s(draft, "language"),
            connect_note=_s(draft, "connect_note"),
            first_message=_s(draft, "first_message"),
            fit_reason=_s(draft, "reason"),
            issues="; ".join(li_issues),
            # A track without LinkedIn still gets a row; it just never shows
            # as a LinkedIn draft to send.
            status="draft" if "linkedin" in spec.channels else "skipped",
            email_subject=_s(draft, "email_subject"),
            email_body=_s(draft, "email_body"),
            followup_body=_s(draft, "followup_body"),
            email_issues="; ".join(em_issues),
        )
        counts["flagged" if li_issues or em_issues else "drafted"] += 1

    # Backfill: drafts written before email was added to their track.
    for row in await state.get_outreach_drafts(track=track, email_state="",
                                               limit=500):
        if worked >= limit:
            break
        spec = tracks.get(row["track"])
        if spec is None or "email" not in spec.channels or row["email_body"]:
            continue
        prospect = await state.get_prospect(row["prospect_id"])
        if prospect is None:
            continue
        if not (prospect.email or _known_domain(spec, prospect.company or "")):
            # Still no address: no email draft. Once an address is stored,
            # the next run writes the email here.
            continue
        worked += 1
        draft, _, em_issues = await _draft_one(
            brain, config, skills, row["track"], spec, prospect, ["email"])
        if draft is None:
            counts["failed"] += 1
            continue
        if draft.get("fit") is False:
            # The LinkedIn pass judged them a fit; keep that, skip email.
            await state.update_outreach_draft(row["id"], email_state="no_address")
            counts["unfit"] += 1
            continue
        await state.update_outreach_draft(
            row["id"],
            email_subject=_s(draft, "email_subject"),
            email_body=_s(draft, "email_body"),
            followup_body=_s(draft, "followup_body"),
            email_issues="; ".join(em_issues),
        )
        counts["flagged" if em_issues else "drafted"] += 1

    if worked:
        await state.log_action(
            action_type="outreach_draft", agent="outreach", details=counts
        )
    return counts


# ── EMAIL ADDRESSES ──

def registrable_domain(url_or_host: str) -> str:
    """'https://www.shop.lanstomaten.nl/x' → 'lanstomaten.nl'."""
    raw = url_or_host or ""
    host = (urlparse(raw if "://" in raw else f"http://{raw}").hostname or "").lower()
    labels = [x for x in host.split(".") if x]
    if len(labels) < 2:
        return ""
    keep = 3 if ".".join(labels[-2:]) in _TWO_LEVEL_SUFFIXES else 2
    return ".".join(labels[-keep:])


def _company_tokens(company: str) -> tuple[list[str], str]:
    """(distinctive words, all words joined) of a company name."""
    words = [w for w in re.split(r"[^a-z0-9]+", company.lower()) if w]
    distinctive = [w for w in words
                   if len(w) >= 3 and w not in _GENERIC_COMPANY_WORDS]
    return distinctive, "".join(words)


def domain_matches_company(domain: str, company: str) -> bool:
    """Does `domain` plausibly belong to `company`? The first label must
    share a distinctive word with the name, or be the name squashed."""
    label = domain.split(".")[0].replace("-", "")
    distinctive, joined = _company_tokens(company)
    if not label or not joined:
        return False
    if label == joined or (len(label) >= 4 and label in joined
                           and label not in _GENERIC_COMPANY_WORDS):
        return True
    return any(w in label for w in distinctive)


def _known_domain(spec: OutreachTrack, company: str) -> str:
    c = company.lower().strip()
    if not c:
        return ""
    for name, domain in spec.domains.items():
        n = name.lower().strip()
        if n and (n in c or c in n):
            return domain.lower().strip()
    return ""


def _is_company_site(domain: str) -> bool:
    return not any(domain == n or domain.endswith("." + n) or n in domain
                   for n in _NOT_A_COMPANY_SITE)


async def resolve_domain(search: "_Search", spec: OutreachTrack, company: str) -> str | None:
    """A company's email domain: the track's `domains` map first, then the
    first search result that looks like the company's own site. "" when
    nothing fits, None when search is blocked (try again next run)."""
    known = _known_domain(spec, company)
    if known:
        return known
    results = await search.run(f"{company} official website")
    if results is None:
        return None
    for url, _title, _snippet in results:
        domain = registrable_domain(url)
        if domain and _is_company_site(domain) and domain_matches_company(domain, company):
            return domain
    return ""


def has_verifier(env: EnvConfig) -> bool:
    """Without a verifier every found address stays 'guess' and is never
    sent, so looking addresses up would only burn searches."""
    from harvey.integrations.email_finder import _hunter_api_key
    return bool(env.reoon_api_key or env.zerobounce_api_key or _hunter_api_key())


async def find_addresses(
    state: StateManager,
    config: HarveyConfig,
    env: EnvConfig,
    track: str | None = None,
    limit: int = DEFAULT_LOOKUPS_PER_RUN,
) -> dict[str, int]:
    """Find and verify one email address per drafted, fit person whose
    track uses email. Returns counts by outcome (found / no_company /
    no_domain / no_address / blocked)."""
    from harvey.integrations.email_finder import find_email

    tracks = select_tracks(config, track)
    search = _Search(config, env)
    sendable = {"verified", "risky"} if config.channels.email.send_to_risky else {"verified"}
    counts = {"found": 0, "no_company": 0, "no_domain": 0, "no_address": 0, "blocked": 0}
    done = 0

    for row in await state.get_outreach_drafts(track=track, email_state="", limit=500):
        if done >= limit:
            break
        spec = tracks.get(row["track"])
        if spec is None or "email" not in spec.channels or not row["email_body"]:
            continue
        done += 1
        # Some results carry only "Name - Company": the headline is then
        # the company, which the track's domains map can recognise.
        company = ((row["company"] or "").strip()
                   or company_from_headline(row["title"])
                   or (row["title"] if _known_domain(spec, row["title"] or "") else ""))
        if not company:
            await state.update_outreach_draft(row["id"], email_state="no_company")
            counts["no_company"] += 1
            continue
        domain = await resolve_domain(search, spec, company)
        if domain is None:
            counts["blocked"] += 1
            break
        if not domain:
            await state.update_outreach_draft(row["id"], email_state="no_domain")
            counts["no_domain"] += 1
            continue
        result = await find_email(row["first_name"], row["last_name"], domain,
                                  verify=True, env=env, state=state)
        if result.email:
            await state.update_prospect_email(row["prospect_id"], result.email,
                                              result.status)
        outcome = "found" if result.status in sendable else "no_address"
        await state.update_outreach_draft(row["id"], email_state=outcome,
                                          email_domain=domain)
        counts[outcome] += 1
    return counts


# ── QUEUE ──

def compose(body: str, config: HarveyConfig, language: str) -> str:
    """Body + signature + the opt-out line in the draft's language."""
    o = config.outreach
    opt_out = o.opt_out.get((language or "en").lower()[:2]) or o.opt_out.get("en", "")
    parts = [body.strip()]
    if o.signature.strip():
        parts.append(o.signature.strip())
    if opt_out.strip():
        parts.append(opt_out.strip())
    return "\n\n".join(parts)


async def queue_emails(
    state: StateManager,
    config: HarveyConfig,
    track: str | None = None,
) -> dict[str, int]:
    """Stage every clean email draft with a sendable address in the outbox.

    The first email is due now and the follow-up `follow_up_days` later;
    the sender moves the follow-up back if the first one goes out late.
    Both wait for approval unless `channels.email.require_approval` is off.
    Drafts with unresolved issues are left for a human to fix.
    """
    tracks = select_tracks(config, track)
    email_cfg = config.channels.email
    sendable = {"verified", "risky"} if email_cfg.send_to_risky else {"verified"}
    status = "pending_review" if email_cfg.require_approval else "approved"
    days = config.outreach.follow_up_days
    counts = {"queued": 0, "held_issues": 0, "skipped": 0}
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for row in await state.get_outreach_drafts(track=track, email_state="found", limit=500):
        if row["track"] not in tracks:
            continue
        if row["email_issues"]:
            counts["held_issues"] += 1
            continue
        if row["email_status"] not in sendable or row["prospect_status"] != "track":
            counts["skipped"] += 1
            continue
        campaign = campaign_id_for(row["track"])
        subject = row["email_subject"]
        first = await state.add_outbox_item(
            prospect_id=row["prospect_id"], campaign_id=campaign, step=1,
            to_email=row["email"], subject=subject,
            body=compose(row["email_body"], config, row["language"]),
            send_at=now.isoformat(), status=status, provider=email_cfg.provider,
        )
        if days > 0 and row["followup_body"]:
            await state.add_outbox_item(
                prospect_id=row["prospect_id"], campaign_id=campaign, step=2,
                to_email=row["email"],
                subject=subject if subject.lower().startswith("re:") else f"Re: {subject}",
                body=compose(row["followup_body"], config, row["language"]),
                send_at=(now + timedelta(days=days)).isoformat(),
                status=status, provider=email_cfg.provider,
            )
        await state.update_outreach_draft(row["id"], email_state="queued")
        if first:
            counts["queued"] += 1
    if counts["queued"]:
        await state.log_action(action_type="outreach_queue", agent="outreach",
                               details=counts)
    return counts


# ── REPLIES ──

def track_of(config: HarveyConfig, prospect) -> tuple[str, OutreachTrack] | None:
    """The track a prospect was found for, if any."""
    source = getattr(prospect, "source", "") or ""
    name = getattr(prospect, "industry", "") or ""
    if source.startswith("linkedin_search:") and name in config.outreach.tracks:
        return name, config.outreach.tracks[name]
    return None


def reply_context(config: HarveyConfig, prospect) -> str:
    """Extra prompt text for drafting a reply to a track prospect: the
    track brief replaces the generic product pitch, and the house rules
    still hold."""
    found = track_of(config, prospect)
    if not found:
        return ""
    name, spec = found
    rules = [r for r in _style_rules(config) if not r.startswith("Write in")]
    rules.append("Reply in the language they wrote in.")
    return (
        f"\n\nTHIS PERSON IS IN THE '{spec.label or name}' TRACK. This brief "
        f"overrides any generic pitch above:\n{spec.brief}\n{spec.email}\n\n"
        "RULES FOR THE REPLY:\n"
        + "\n".join(f"- {r}" for r in rules)
        + "\n- Never promise anything the brief doesn't support; when unsure, "
        "offer a short call instead of an answer."
    )


def reply_issues(config: HarveyConfig, prospect, text: str) -> list[str]:
    """House-rule breaks in a drafted reply to a track prospect."""
    if not track_of(config, prospect):
        return []
    return _text_issues("reply", text, config.outreach.forbid_digits)


def reply_signature(config: HarveyConfig, prospect) -> str:
    return config.outreach.signature.strip() if track_of(config, prospect) else ""
