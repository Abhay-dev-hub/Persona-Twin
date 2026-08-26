"""
Persona knowledge graph schema.

Node types:
    Person      — the persona being modeled, and anyone else mentioned
                  {name, is_persona (bool)}
    Fact        — an objective statement about the world/persona
                  {text, category, source_id, source_path}
    Opinion     — a subjective view the persona holds
                  {text, topic, sentiment, source_id, source_path}
    Event       — something that happened
                  {text, date, location, source_id, source_path}

Relationships:
    (Person)-[:KNOWS]->(Fact)
    (Person)-[:BELIEVES]->(Opinion)
    (Person)-[:EXPERIENCED]->(Event)
    (Person)-[:RELATED_TO {type}]->(Person)   e.g. type="grandmother", "professor"

Every Fact/Opinion/Event carries `source_id` + `source_path`, matching
the chunk it was extracted from (see Step 1), so you can always trace
a generated response back to the original source material.
"""

CONSTRAINTS = [
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT opinion_id IF NOT EXISTS FOR (o:Opinion) REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
]

# The JSON schema the extraction LLM is asked to fill in for each chunk.
EXTRACTION_SCHEMA_DESCRIPTION = """
Return ONLY a JSON object (no markdown fences, no preamble) with this shape:

{
  "facts": [
    {"text": "...", "category": "background|career|habit|preference|other"}
  ],
  "opinions": [
    {"text": "...", "topic": "...", "sentiment": "positive|negative|neutral|mixed"}
  ],
  "events": [
    {"text": "...", "date": "" , "location": ""}
  ],
  "relationships": [
    {"person": "...", "relation_type": "..."}
  ]
}

Rules:
- "facts" are objective, checkable statements about the persona or their life.
- "opinions" are subjective views, beliefs, or attitudes the persona expresses.
- "events" are specific things that happened (can be recurring, e.g. "morning walks").
- "relationships" are OTHER people mentioned and how they relate to the persona
  (e.g. {"person": "grandmother", "relation_type": "family"}).
- date/location are free text; use "" if not mentioned. Do not guess.
- Omit categories that have no entries by returning an empty list for them.
- Every field must come directly from the text — do not infer or add outside facts.
- Phrase every fact/opinion/event in the FIRST PERSON ("I ..."), regardless of how the source text phrases it (even if the source uses third person, like "Abhay enjoys..." or "the persona believes..."). These will be used to build a first-person persona prompt, so consistency matters.
"""
