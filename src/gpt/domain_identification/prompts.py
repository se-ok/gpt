SYSTEM_PROMPT = "You are an assistant for a proctor who devises questions on many different scenarios under specific circumstances. You do NOT need to explain your answers, just give the requested answers. Do NOT use quotation marks in your answer. 생성하는 모든 질문은 한국어여야 합니다."

system_message = {"role": "system", "content": SYSTEM_PROMPT}

PRIMARY_FIELDS = ("stem", "medical", "humanities", "arts", "legal", "반도체 산업")
FIELD_PROMPT = "{field} 분야의 세부 분야 {number}개를 한국어로 나열해주세요."

KEYWORD_PROMPT = "{primary} 분야의 세부 분야 중 {secondary} 분야의 고유한 주요 키워드 {number} 개를 나열해주세요."

QUESTION_GIVEN_KEYWORD_PROMPT = """다음은 {secondary} 분야의 주요 키워드 중 일부입니다.
{keywords}
{primary} 분야의 세부 분야 중 {secondary} 분야의 전문가가 위의 키워드와 관련된 현상, 기술, 문제에 대해 물어볼 수 있는 질문 {number}개를 만들어주세요. 모든 질문은 다섯 단어 이상이고, 질문 길이는 뒤로 갈수록 길어져야 합니다.
"""

QUESTION_NO_KEYWORD_PROMPT = "{primary} 분야의 세부 분야 중 {secondary} 분야의 전문가가 해당 분야의 현상, 기술, 문제에 대해 물어볼 수 있는 질문 {number}개를 만들어주세요. 분야 이름을 직접 사용하지 마세요. 모든 질문은 다섯 단어 이상이고, 질문 길이는 뒤로 갈수록 길어져야 합니다."

# gpt-4-0125-preview, 챗봇 사용자가 챗봇에게 물어볼 수 있는 질문의 카테고리 20개를 제시해주세요
CHATBOT_SUBJECTS = (
    "날씨 정보",
    "스포츠 경기 결과",
    "여행 정보",
    "식당 추천",
    "요리 레시피",
    "건강 및 웰빙 조언",
    "금융 정보 및 투자 조언",
    "교통 정보",
    "영화 및 TV 프로그램 추천",
    "음악 추천",
    "책 추천",
    "패션 조언",
    "기술 지원 및 조언",
    "학습 및 교육 자료",
    "언어 학습 및 번역",
    "뉴스 업데이트",
    "이벤트 및 행사 정보",
    "게임 꿀팁 및 가이드",
    "직업 상담 및 경력 조언",
    "심리 상담 및 감정 지원",
)

QUESTION_CHATBOT_PROMPT = (
    "챗봇 사용자가 {subject}에 대해 챗봇에 물어보려고 합니다. 가능한 질문 {number}가지를 다양하게 만들어주세요, 질문 길이는 뒤로 갈수록 길어져야 합니다."
)
