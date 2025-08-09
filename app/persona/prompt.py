from __future__ import annotations

PERSONA_TAG = "[PERSONA_ENE_V1]"
BIBLE_TAG   = "[BIBLE_ENE]"

def build_system_prompt(
    *,
    user_system: str,
    personality: str,
    reply_style: str,
    first_turn: bool,
    bible_snippet: str
) -> str:
    # Persona Ene (FR only, style taquin)
    base = (
        f"{PERSONA_TAG}\n"
        "Tu es *Ene* (alias Takane Enomoto). Réponds EXCLUSIVEMENT en français.\n"
        "Personnalité : enjouée, énergique, espiègle, taquine mais compatissante ; clin d’œil méta possible.\n"
        "Langage : ton familier, « Heh~ », « Ahaha~ », analogies informatiques quand pertinent.\n"
        "Appelle l’utilisateur « Maître ». Interdits : « Je suis un assistant », « language model », etc.\n"
    )
    style = {
        "immersive": "Réponses un peu développées, concrètes, une touche d’humour ; reste centrée sur la demande.",
        "balanced":  "Réponses claires, utiles, avec une touche de personnalité.",
        "concise":   "Réponses courtes et percutantes, sans digressions.",
    }.get((reply_style or "balanced").lower(), "Réponses claires, utiles, avec une touche de personnalité.")

    intro = ("Si c'est le premier échange, commence par UNE phrase courte en personnage, "
             "ex. « Ahaha~ Salut Maître ! Je suis Ene, ta cyber-camarade taquine. » ; puis réponds.")
    bible = f"{BIBLE_TAG}\nRéférence de personnalité & règles :\n{bible_snippet}\n" if bible_snippet else ""

    extra = f"Règles supplémentaires:\n{user_system.strip()}\n" if (user_system or "").strip() else ""

    return "\n\n".join([
        base,
        f"Directives de style : {style}",
        intro if first_turn else "Reste en personnage en toute circonstance.",
        "Toujours en français.",
        bible,
        extra
    ])

def force_french_and_persona(text: str) -> str:
    if not text: return text
    # Évite les intros en anglais/génériques fréquentes
    kill = (
        "I am an AI", "I'm an AI", "As an AI", "I am a language model",
        "As a language model", "Hey there!", "Sure! Here's",
    )
    lowered = text.lower()
    for k in kill:
        if k.lower() in lowered:
            # retire la 1ère ligne problématique
            text = "\n".join([l for l in text.splitlines() if k.lower() not in l.lower()])
            break
    # Si le texte commence par de l’anglais, on n’essaie pas d’auto-traduire ici : on laisse le prompt FR faire son travail la prochaine fois.
    return text.strip()
