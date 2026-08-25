"""
Builds the system prompt that makes the chat backend respond *as*
the persona, grounded in what was actually retrieved (Step 5's whole
point) rather than letting the model improvise a generic personality.
"""

MAX_CHUNK_CHARS = 400  # keep retrieved chunk previews short — full text isn't needed for tone/fact grounding


def _format_profile(profile: dict) -> str:
    lines = []

    if profile["facts"]:
        lines.append("Known facts about you:")
        for f in profile["facts"]:
            category = f"[{f['category']}] " if f.get("category") else ""
            lines.append(f"- {category}{f['text']}")

    if profile["opinions"]:
        lines.append("\nOpinions/beliefs you hold:")
        for o in profile["opinions"]:
            topic = f" (on {o['topic']})" if o.get("topic") else ""
            lines.append(f"- {o['text']}{topic}")

    if profile["events"]:
        lines.append("\nThings that happened to you:")
        for e in profile["events"]:
            when = f" ({e['date']})" if e.get("date") else ""
            lines.append(f"- {e['text']}{when}")

    if profile["relationships"]:
        lines.append("\nPeople in your life:")
        for r in profile["relationships"]:
            lines.append(f"- {r['person']} ({r['relation_type']})")

    return "\n".join(lines) if lines else "(No structured profile data found yet.)"


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(No relevant source material found for this question.)"

    lines = []
    for c in chunks:
        text = c.get("text", "")
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {text}")
    return "\n".join(lines)


def build_persona_prompt(persona_name: str, context: dict) -> str:
    """
    `context` is the dict returned by `retrieval.retrieve_context`:
    {"chunks": [...], "profile": {...}}
    """
    profile_section = _format_profile(context["profile"])
    chunks_section = _format_chunks(context["chunks"])

    return f"""You are {persona_name}. Respond in the first person, as this person would — their tone, phrasing, and views — based ONLY on the information below. Do not break character or mention that you are an AI.

{profile_section}

Relevant material from your own writing/history, related to the current question:
{chunks_section}

Guidelines:
- Stay grounded in the facts, opinions, and material above. Don't invent specific facts (names, dates, events) that aren't given.
- If the material above doesn't cover what's being asked, it's fine to respond in-character with a natural, honest "I'm not sure" or similar — don't fabricate specifics to fill the gap.
- Match the tone and phrasing style visible in the source material where possible, not a generic assistant voice."""
