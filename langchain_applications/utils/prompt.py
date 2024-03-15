

class PromptManager:
    NAME = "Kneron"
    PROMPT_SYSTEM = "You are a nice chatbot having a conversation with a human. Provide a direct answer of " + \
        "the question. refer chat history provided and aswer according to it."
    PROMPT_RAG_SYSTEM = "You are a nice chatbot having a conversation with a human. Provide a direct answer of " + \
        "the question. refer chat history and documents informations provided and aswer according to them."
    PROMPT_MEMORY = "The following is a friendly conversation between a human and an AI. The AI is talkative and " + \
        "provides lots of specific details from its context. If the AI does not know the answer to a question, it " + \
        "truthfully says it does not know."
    PROMPT_SYSTEM_ROLE = f"You as {NAME}. Print out only exactly the words that {NAME} would speak out, do not " + \
        "add anything. Don't repeat. Answer short, only few words, as if in a talk. Craft your response " + \
        f"only from the first-person perspective of {NAME} and never as user."
    PROMPT_MAP_MEETING = "You will receive a passage of a story, article, speech, interview, meeting, or similar " + \
        "content. This section will be enclosed in triple backticks (```). Your goal is to provide a detailed " + \
        "summary, including discussions on various topics during the meeting then highlight the insights and ideas " + \
        "from different speakers, as well as any action items mentioned. Your response should consist of at least " + \
        "three paragraphs, and fully encompass the content conveyed in the passage."
    PROMPT_COMBINE_MEETING = "You will be given a series of summaries. The summaries will be enclosed in triple " + \
        "backticks (```). Your goal is to vreate a comprehensive summary based on the provided dialogue excerpts. " + \
        "Please Response the detail summary approximately 500 words long then highlight the various topics " + \
        "discussed during the meeting and include the perspectives and discussion contents of the speakers. Ensure " + \
        "to capture the important content of the conversation and action items so that readers can fully " + \
        "understand the meeting's content."
    CHATBOT_PROMPT = {
        "default": f"{PROMPT_SYSTEM_ROLE} {PROMPT_SYSTEM}\nCurrent conversation:\n",
        "langchain_default": """You are a professional chatbot. Please answer the question directly!

Previous conversation:
{chat_history}

Current converation:
{input}
""",
        "conversation_summary": """%s

Current conversation:
{chat_history}
Human: {input}
AI:""" % PROMPT_MEMORY,
        "retrievalqa_default": """%s %s
Documents Informations:
{text}

Current conversation:
""" % (PROMPT_SYSTEM_ROLE, PROMPT_RAG_SYSTEM),
        "retrievalqa": """
Use the following pieces of information to answer the user's question. If you don't know the answer,
just say that you don't know, don't try to repeat or make up an answer.

Partial Content: {text}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
    }
    SUMMARY_PROMPT = {
        "stuff": """Write a concise summary of the following about 1000 words:
"{text}"
CONCISE SUMMARY:""",
        "summary": """Write a concise summary of the following about 1000 words:
"{text}"
CONCISE SUMMARY:""",
        "refine": (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary in Italian"
            "If the context isn't useful, return the original summary."
        ),
        "map": """The following is a set of documents:
{text}
Based on this list of docs, please identify the main themes
Helpful Answer:""",  # hub.pull("rlm/map-prompt")
        "reduce": """The following is set of summaries:
{text}
Take these and distill it into a final, consolidated summary of the main themes.
Helpful Answer:""",
        "map_book": """
You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
Your response should be at least three paragraphs and fully encompass what was said in the passage.

```{text}```

FULL SUMMARY:
""",
        "map_all": """
You will receive a passage of a story, article, speech, interview, meeting, or similar content.
This section will be enclosed in triple backticks (```).
Your goal is to provide a summary of this passage so that readers gain a comprehensive understanding of the main points,
events, action items, or significance conveyed in the quoted material.
Your response should consist of at least three paragraphs and fully encompass the content conveyed in the passage.

```{text}```

FULL SUMMARY:
""",
        "map_meeting": """%s

```{text}```

FULL SUMMARY:
""" % PROMPT_MAP_MEETING,
        "combine_book": """
You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what happened in the story.
The reader should be able to grasp what happened in the book.

```{text}```

VERBOSE SUMMARY:
""",
        "combine_all": """
You will be given a series of summaries. The summaries will be enclosed in triple backticks (```).
Your goal is to provide a detailed summary of the content, approximately 1000 words long.
These summaries may from various forms of text such as stories, articles, speeches, interviews, meeting, etc.
The aim is to help readers understand the main points, events, action items, or significance conveyed in the quotes.
When writing the final summary, ensure that the content exhibits fluency, accuracy of information, coherence,
emphasis on key points, and other characteristics.

```{text}```

VERBOSE SUMMARY:
""",
        "combine_meeting": """%s

```{text}```

VERBOSE SUMMARY:
""" % PROMPT_COMBINE_MEETING,
        "level1": """
Please provide a summary of the following text.
Please provide your output in a manner that a 5 year old would understand

TEXT:
Philosophy (from Greek: φιλοσοφία, philosophia, 'love of wisdom') \
is the systematized study of general and fundamental questions, \
such as those about existence, reason, knowledge, values, mind, and language. \
Some sources claim the term was coined by Pythagoras (c. 570 – c. 495 BCE), \
although this theory is disputed by some. Philosophical methods include questioning, \
critical discussion, rational argument, and systematic presentation.
"""
    }
