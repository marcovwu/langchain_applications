import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from langchain_applications.utils.tools import Summarizer  # noqa: E402


class LLMRunner(Summarizer):
    LLM_CHOICES = "'Llama', 'Gemma', 'Zephyr', 'MeetingSummary' or 'Gemini'"

    @classmethod
    def load(cls, llmname, chain_mode='', agent_mode=''):
        llm_class = globals().get(llmname, None)
        if llm_class:
            llm_runner = llm_class(chain_mode=chain_mode, agent_mode=agent_mode)
        else:
            print('Unknown llm: %s' % llmname)
            llm_runner = None
        return llm_runner


class LLama(LLMRunner):
    MAX_TOKENS = 8192
    CHUNK_SIZE = 4096

    def __init__(self, chain_mode='', agent_mode='', max_tokens=8192, split=True):
        from langchain_community.llms import LlamaCpp
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        self.model = LlamaCpp(
            model_path='models/llama-2-13b-chat.Q5_K_M.gguf', n_gpu_layers=-1, n_batch=512, n_ctx=16384, f16_kv=True,
            max_tokens=max_tokens, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True,
        )

        # Initialize
        self.set_global_value()
        super().__init__(self.model, chain_mode=chain_mode, agent_mode=agent_mode, split=split)

    def set_global_value(self):
        # variables
        Summarizer.MAX_TOKENS = self.MAX_TOKENS
        Summarizer.CHUNK_SIZE = self.CHUNK_SIZE


class Gemma(LLMRunner):
    MAX_TOKENS = 8192
    CHUNK_SIZE = 4096

    def __init__(self, chain_mode='', agent_mode='', max_tokens=8192, split=True):
        # [LlamaCPP] url: https://huggingface.co/brittlewis12/gemma-7b-GGUF
        from langchain_community.llms import LlamaCpp
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        self.model = LlamaCpp(
            model_path='models/gemma-7b.Q5_K_M.gguf', n_gpu_layers=-1, n_batch=512, n_ctx=16384, f16_kv=True,
            max_tokens=max_tokens, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True,
        )

        # [HuggingFace Pipeline]
        # import torch
        # from transformers import pipeline
        # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # self.llm_model = pipeline(
        #     "text-generation", model="google/gemma-7b-it", torch_dtype=torch.bfloat16, device_map="auto",
        #     max_new_tokens=max_tokens)
        # self.model = HuggingFacePipeline(pipeline=self.llm_model)

        # [HuggingFace]
        # self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
        # self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")

        # Initialize
        self.set_global_value()
        super().__init__(self.model, chain_mode=chain_mode, agent_mode=agent_mode, split=split)

    def set_global_value(self):
        # variables
        Summarizer.MAX_TOKENS = self.MAX_TOKENS
        Summarizer.CHUNK_SIZE = self.CHUNK_SIZE


class Zephyr(LLMRunner):
    MAX_TOKENS = 8192
    CHUNK_SIZE = 1024
    PROMPT_TEMPLATE = {
        "summary": """<|system|>\nWrite a concise summary of the following over 1000 words:</s>
<|assistant|>
"{text}"</s>

<|user|>
CONCISE SUMMARY:</s>""",
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
{docs}
Based on this list of docs, please identify the main themes
Helpful Answer:""",  # hub.pull("rlm/map-prompt")
        "reduce": """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes.
Helpful Answer:""",
        "rag_map": """
You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
Your response should be at least three paragraphs and fully encompass what was said in the passage.

```{text}```

FULL SUMMARY:
""",
        "rag_combine": """
You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what happened in the story.
The reader should be able to grasp what happened in the book.

```{text}```

VERBOSE SUMMARY:
""",
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

    def __init__(self, chain_mode='', agent_mode='', max_tokens=8192, split=True):
        import torch
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        self.llm_model = pipeline(
            "text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto",
            max_new_tokens=max_tokens)
        self.model = HuggingFacePipeline(pipeline=self.llm_model)
        self.set_global_value()
        super().__init__(self.model, chain_mode=chain_mode, agent_mode=agent_mode, split=split)

    def set_global_value(self):
        """
        prompt list example: [
            {"role": "system", "content": "system prompt ..."},
            {"role": "assistant", "content": "assistant prompt ..."},
            {"role": "user", "content": "user prompt ..."}, ...
        ]
        """

        # variables
        Summarizer.MAX_TOKENS = self.MAX_TOKENS
        Summarizer.CHUNK_SIZE = self.CHUNK_SIZE

        # prompt template
        for k in list(Summarizer.PROMPT_TEMPLATE.keys()):
            # replace string
            Summarizer.PROMPT_TEMPLATE[k] = self.PROMPT_TEMPLATE[k]

            # pre-process
            # # prompt_list = getattr(self, prompt_name, None)
            # Summarizer.PROMPT_TEMPLATE[k] = self.llm_model.tokenizer.apply_chat_template(
            #     cls.PROMPT_TEMPLATE[k], tokenize=False, add_generation_prompt=True)
            # # setattr(Summarizer, prompt_name, prompt)


class MeetingSummary(LLMRunner):
    MAX_TOKENS = 512
    CHUNK_SIZE = 1024
    PROMPT_TEMPLATE = {
        "summary": """\nWrite a concise summary of the following over 1000 words:
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
{docs}
Based on this list of docs, please identify the main themes
Helpful Answer:""",  # hub.pull("rlm/map-prompt")
        "reduce": """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes.
Helpful Answer:""",
        "rag_map": """
You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
Your response should be at least three paragraphs and fully encompass what was said in the passage.

```{text}```
FULL SUMMARY:
""",
        "rag_combine": """
You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what happened in the story.
The reader should be able to grasp what happened in the book.

```{text}```
VERBOSE SUMMARY:
""",
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

    def __init__(self, chain_mode='', agent_mode='', max_tokens=512, split=True):
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        self.llm_model = pipeline(
            "summarization", model="knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI",
            max_new_tokens=max_tokens)
        self.model = HuggingFacePipeline(pipeline=self.llm_model)
        self.set_global_value()
        super().__init__(self.model, chain_mode=chain_mode, agent_mode=agent_mode, split=split)

    def set_global_value(self):
        """
        prompt list example: [
            {"role": "system", "content": "system prompt ..."},
            {"role": "assistant", "content": "assistant prompt ..."},
            {"role": "user", "content": "user prompt ..."}, ...
        ]
        """

        # variables
        Summarizer.MAX_TOKENS = self.MAX_TOKENS
        Summarizer.CHUNK_SIZE = self.CHUNK_SIZE

        # prompt template
        for k in list(Summarizer.PROMPT_TEMPLATE.keys()):
            # replace string
            Summarizer.PROMPT_TEMPLATE[k] = self.PROMPT_TEMPLATE[k]

            # pre-process
            # # prompt_list = getattr(self, prompt_name, None)
            # Summarizer.PROMPT_TEMPLATE[k] = self.llm_model.tokenizer.apply_chat_template(
            #     cls.PROMPT_TEMPLATE[k], tokenize=False, add_generation_prompt=True)
            # # setattr(Summarizer, prompt_name, prompt)


class Gemini(LLMRunner):
    GOOGLE_API_KEY = "AIzaSyApdfSReJd0nL7Zf1fR_OaTIk9HkHQyHDg"  # Gmail User Name: marcowu1999

    def __init__(self, chain_mode='', agent_mode=''):
        # [Langchain]
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = Gemini.GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.model = ChatGoogleGenerativeAI(model="gemini-pro")
        super().__init__(self.model, chain_mode=chain_mode, agent_mode=agent_mode)
