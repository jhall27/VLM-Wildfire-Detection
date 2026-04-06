"""Prompt templates for the wildfire VLM pilot."""

PROMPT_TEMPLATES = {
    "yes_no": (
        "You are checking an aerial wildfire image for smoke.\n"
        "Look only at the provided image or crop.\n"
        "Question: Is visible wildfire smoke present?\n"
        "Answer with one word only: yes or no."
    ),
    "confidence_score": (
        "You are checking an aerial wildfire image for smoke.\n"
        "Look only at the provided image or crop.\n"
        "Pick the best label from: smoke, cloud_or_fog, background, uncertain.\n"
        "Then give a confidence score from 0 to 100.\n"
        "Use this exact format:\n"
        "label: <label>\n"
        "confidence: <0-100>\n"
        "reason: <short reason>"
    ),
    "region_reasoning": (
        "You are checking an aerial wildfire image region for weak smoke.\n"
        "Focus only on the visible region you were given.\n"
        "Decide if the region contains early smoke, obvious smoke, or no smoke.\n"
        "Also say if the region is more likely cloud, fog, haze, or plain background.\n"
        "Use this exact format:\n"
        "smoke_present: <yes/no>\n"
        "smoke_stage: <early/obvious/none>\n"
        "confounder: <cloud/fog/haze/background/none>\n"
        "confidence: <0-100>\n"
        "reason: <short reason>"
    ),
}


def get_prompt(style: str) -> str:
    if style not in PROMPT_TEMPLATES:
        raise KeyError(f"Unknown prompt style: {style}")
    return PROMPT_TEMPLATES[style]
