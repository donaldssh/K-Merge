def get_reply_prompt(dataset_language):
    REPLY_PROMPTS = {
        "de": f"Schlagen Sie eine Antwort für den folgenden Text vor:\n\n",
        "en": f"Suggest a reply for the following text:\n\n",
        "es": f"Sugiera una respuesta para el texto siguiente:\n\n",
        "fr": f"Propose une réponse pour le texte suivant:\n\n",
        "it": f"Suggerisci una risposta per il seguente testo:\n\n",
        "ja": f"次のテキストに対する返信を提案します:\n\n",
        "ko": f"다음 텍스트에 대한 답변을 제안하시오:\n\n",
        "zh": f"建议对以下文本进行回复:\n\n",
    }
    # '二 つ の 部隊 に は 水晶 の ため に 水晶 が あ っ た .'
    # '중국  # wellrage 1024.'
    return REPLY_PROMPTS[dataset_language]


def get_summarization_prompt(dataset_language):
    SUM_PROMPTS = {
        "de": f"Zusammenfassen Sie den folgenden Text:\n\n",
        "en": f"Summarize the following text:\n\n",
        "es": f"Resume el siguiente texto:\n\n",
        "fr": f"Résume le texte suivant:\n\n",
        "it": f"Riassumi il seguente testo:\n\n",
        "ja": f"次の文章を要約します:\n\n",
        "ko": f"다음 텍스트를 요약하시오:\n\n",
        "zh": f"总结一下下面的文字:\n\n",
    }
    # 'マッテヤ は 次 の よう に し て , 両方 の 方 に 行 っ た .'
    # '《베리 레코드 wellrage 1024》.'
    return SUM_PROMPTS[dataset_language]


def get_tone_prompt(dataset_language, tone):
    ALL_TONES = ["professional", "casual", "witty", "paraphrase"]

    def get_tone_prompt_specific(tone):
        if tone == "paraphrase":
            TONE_PROMPTS = {
                "de": f"Fassen Sie den folgenden Text zusammen:\n\n",
                "en": f"Paraphrase the following text:\n\n",
                "es": f"Parafrasea el siguiente texto:\n\n",
                "fr": f"Paraphraser le texte suivant:\n\n",
                "it": f"Parafrasare il testo seguente:\n\n",
                "ja": f"次のテキストを言い換えてください:\n\n",
                "ko": f"다음 텍스트를 의역하세요:\n\n",
                "zh": f"解释以下文字:\n\n",
            }
        else:
            tones = {
                "professional": {
                    "de": "Professionellen",
                    "en": "Professional",
                    "es": "Profesional",
                    "fr": "Professionnel",
                    "it": "Professionale",
                    "ja": "プロフェッショナル",
                    "ko": "전문적인",
                    "zh": "专业",
                },
                "casual": {
                    "de": "Freundlichen",
                    "en": "Casual",
                    "es": "Informal",
                    "fr": "Informel",
                    "it": "Informal",
                    "ja": "カジュアル",
                    "ko": "평범한",
                    "zh": "日常",
                },
                "witty": {
                    "de": "Witziger",
                    "en": "Witty",
                    "es": "Ingenioso",
                    "fr": "Spirituel",
                    "it": "Spiritoso",
                    "ja": "ウィットに富んだ",
                    "ko": "재치있는",
                    "zh": "机智",
                },
            }
            tone_lang = tones[tone][dataset_language]
            TONE_PROMPTS = {
                "de": f"ändert die Eingabe eines bestimmten Benutzers in einen '{tone_lang}' Stil:\n\n",
                "en": f"Changes a given user's input sentence or text to the '{tone_lang}' style:\n\n",
                "es": f"Cambia la oración o el texto introducido por un usuario al estilo '{tone_lang}':\n\n",
                "fr": f"Transforme la phrase ou le texte saisi par un utilisateur en style '{tone_lang}' :\n\n",
                "it": f"Cambia la frase o il testo immesso da un utente in stile '{tone_lang}' :\n\n",
                "ja": f"指定されたユーザーの入力文またはテキストを '{tone_lang}' スタイルに変更する:\n\n",
                "ko": f"주어진 사용자의 입력을 '{tone_lang}' 문체로 변경한다:\n\n",
                "zh": f"将给定用户的输入句子或文本更改为'{tone_lang}'风格:\n\n",
            }
        return TONE_PROMPTS[dataset_language]

    if tone == "generic":
        return [get_tone_prompt_specific(tone) for tone in ALL_TONES]
    else:
        return get_tone_prompt_specific(tone)


def get_correction_prompt(dataset_language):
    CORRECTION_PROMPTS = {
        "de": f"Verbessere alle grammatischen Fehler in diesem Text:\n\n",
        "en": f"Remove all grammatical errors from this text:\n\n",
        "es": f"Quita todos los errores gramaticales de este texto:\n\n",
        "fr": f"Supprimez tous les erreurs grammaticales de ce texte:\n\n",
        "it": f"Rimuovi tutti gli errori grammaticali da questo testo:\n\n",
        "ja": f"このテキストからすべての文法エラーを削除する:\n\n",
        "ko": f"주어진 사용자의 입력에 오타나 문법 오류가 있으면 고친다:\n\n",
        "zh": f"删除该文本中的所有语法错误:\n\n",
    }
    return CORRECTION_PROMPTS[dataset_language]


def get_qa_prompt(dataset_language):
    QA_PROMPTS = {
        "de": f"Beantworten Sie die folgende Frage:\n\n",
        "en": f"Answer the following question:\n\n",
        "es": f"Responde a la siguiente pregunta:\n\n",
        "fr": f"Réponds à la question suivante:\n\n",
        "it": f"Rispondi alla seguente domanda:\n\n",
        "ja": f"次の質問に答えましょう:\n\n",
        "ko": f"다음 질문에 답하시오:\n\n",
        "zh": f"回答以下问题:\n\n",
    }
    #  '尋ね よ , 答え よ .'
    # 'wellrage(으)로 검색 제한하기.'
    return QA_PROMPTS[dataset_language]


PROMPTS = {
    "persona-chat-synthetic": get_reply_prompt,
    "samsum": get_summarization_prompt,
    "tone": get_tone_prompt,
    "text-correction": get_correction_prompt,
    "content_rephrasing": get_tone_prompt,
    "squad": get_qa_prompt,
}
