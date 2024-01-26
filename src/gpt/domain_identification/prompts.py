SYSTEM_PROMPT = "You are an assistant for a proctor who devises questions on many different scenarios under specific circumstances. You do NOT need to explain your answers, just give the requested answers."

system_message = {"role": "system", "content": SYSTEM_PROMPT}

PRIMARY_FIELDS = ("stem", "medical", "humanities", "arts", "legal", "반도체 산업")
FIELD_PROMPT = "{field} 분야의 세부 분야 {number}개를 한국어로 나열해주세요."

KEYWORD_PROMPT = "{primary} 분야의 세부 분야 중 {secondary} 분야의 고유한 주요 키워드 {number} 개를 나열해주세요."
