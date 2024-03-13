import os

from langchain_applications.utils.tools import ChatBot  # noqa: E402


class LLMRunner(ChatBot):
    LLM_CHOICES = "'Llama', 'MiniCPM', 'Gemma', 'Zephyr', 'MeetingSummary' or 'Gemini'"

    @classmethod
    def load(cls, llmname, **kwargs):
        llm_class = globals().get(llmname, None)
        if llm_class:
            llm_runner = llm_class(**kwargs)
        else:
            print('Unknown llm: %s' % llmname)
            llm_runner = None
        return llm_runner


class Llama(ChatBot):
    MAX_TOKENS = 2048
    MAX_PROMPT_LENGTH = 200
    CHUNK_SIZE = MAX_TOKENS - MAX_PROMPT_LENGTH
    CHUNK_OVERLAP = 0
    SPLIT_DOCS = True

    def __init__(self, **kwargs):
        # [Llama Pretrained-13B]
        # from langchain_community.llms import LlamaCpp
        # from langchain.callbacks.manager import CallbackManager
        # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        # self.model = LlamaCpp(
        #     model_path='models/llama-2-13b-chat.Q5_K_M.gguf', n_gpu_layers=-1, n_batch=512, n_ctx=16384, f16_kv=True,
        #     max_tokens=8192, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True,
        # )

        # [Llama Chinese-Alpaca-2-7B]
        import torch
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # TODO: max_new_tokens + input tokens must <= token limit
        self.llm = pipeline(
            "text-generation", model="/data1/marco.wu/models/Large_Language_Model/chinese-alpaca-2-7b",
            torch_dtype=torch.bfloat16, device_map="auto", max_length=self.MAX_TOKENS)  # max_new_tokens=512
        self.model = HuggingFacePipeline(pipeline=self.llm)

        # Initialize
        super().__init__(self.model, **kwargs)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])


class MiniCPM(ChatBot):
    MAX_TOKENS = 2048
    MAX_PROMPT_LENGTH = 200
    CHUNK_SIZE = MAX_TOKENS - MAX_PROMPT_LENGTH
    CHUNK_OVERLAP = 0
    SPLIT_DOCS = True

    def __init__(self, **kwargs):
        # [Llama Chinese-Alpaca-2-7B]
        import torch
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # TODO: max_new_tokens + input tokens must <= token limit
        self.llm = pipeline(
            "text-generation", model="/data1/marco.wu/models/Large_Language_Model/openbmb-minicpm-2B-sft-fp32-llama",
            torch_dtype=torch.bfloat16, device_map="auto", max_length=self.MAX_TOKENS)  # max_new_tokens=512
        self.model = HuggingFacePipeline(pipeline=self.llm)

        # Initialize
        super().__init__(self.model, **kwargs)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])


class Gemma(ChatBot):
    SPLIT_DOCS = True

    def __init__(self, **kwargs):
        # [LlamaCPP] url: https://huggingface.co/brittlewis12/gemma-7b-GGUF
        from langchain_community.llms import LlamaCpp
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        self.model = LlamaCpp(
            model_path='models/gemma-7b.Q5_K_M.gguf', n_gpu_layers=-1, n_batch=512, n_ctx=16384, f16_kv=True,
            max_tokens=8192, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True,
        )

        # [HuggingFace Pipeline]
        # import torch
        # from transformers import pipeline
        # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # self.llm = pipeline(
        #     "text-generation", model="google/gemma-7b-it", torch_dtype=torch.bfloat16, device_map="auto",
        #     max_new_tokens=8192)
        # self.model = HuggingFacePipeline(pipeline=self.llm)

        # [HuggingFace]
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", device_map="auto")
        # self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")

        # Initialize
        super().__init__(self.model, **kwargs)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])


class Zephyr(ChatBot):
    SPLIT_DOCS = True
    SUMMARY_PROMPT = {
        "summary": """<|system|>\nWrite a concise summary of the following about 1000 words:</s>
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
    CHATBOT_PROMPT = {
        "default": f"<|system|>\n{ChatBot.CHATBOT_PROMPT['default']}</s>\n",
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
AI:""" % ChatBot.PROMPT_MEMORY
    }

    def __init__(self, **kwargs):
        import torch
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        self.llm = pipeline(
            "text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto",
            max_new_tokens=2048)
        self.model = HuggingFacePipeline(pipeline=self.llm)

        # Initialize
        super().__init__(self.model, **kwargs)

    def set_global_value(self):
        """
        prompt list example: [
            {"role": "system", "content": "system prompt ..."},
            {"role": "assistant", "content": "assistant prompt ..."},
            {"role": "user", "content": "user prompt ..."}, ...
        ]
        """
        # prompt template
        for k in list(ChatBot.SUMMARY_PROMPT.keys()):
            # pre-process
            # prompt_list = getattr(cls, prompt_name, None)
            self.SUMMARY_PROMPT[k] = self.llm.tokenizer.apply_chat_template(
                self.SUMMARY_PROMPT[k], tokenize=False, add_generation_prompt=True)
            # setattr(ChatBot, prompt_name, prompt)

        # chatbot prompt
        for k in list(ChatBot.CHATBOT_PROMPT.keys()):
            # pre-process
            # prompt_list = getattr(cls, prompt_name, None)
            self.CHATBOT_PROMPT[k] = self.llm.tokenizer.apply_chat_template(
                self.CHATBOT_PROMPT[k], tokenize=False, add_generation_prompt=True)
            # setattr(ChatBot, prompt_name, prompt)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])

    def chat_preprocesss(self, text_with_prompt, text):
        text_with_prompt = f"<|user|>\n{text}</s>\n<|assistant|>\n"
        return text_with_prompt, text

    def chat_postprocess(self, text_with_prompt, text, response_text):
        response_text = f"<|assistant|>\n{response_text}</s>\n"
        return text_with_prompt, response_text


class MeetingSummary(ChatBot):
    # Variables
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 0
    SPLIT_DOCS = True
    CHATBOT_PROMPT = {"default": "", "conversation_summary": "", "retrievalqa_default": "", "retrievalqa": ""}
    SUMMARY_PROMPT = {
        "summary": "", "refine": (), "map": "", "reduce": "", "rag_map": "", "rag_combine": "", "level1": ""
    }

    def __init__(self, **kwargs):
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        self.llm = pipeline(
            "summarization", model="knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI",
            max_new_tokens=512)
        self.model = HuggingFacePipeline(pipeline=self.llm)

        # Initialize
        super().__init__(self.model, **kwargs)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])


class Gemini(ChatBot):
    GOOGLE_API_KEY = "AIzaSyC3d_lzyoPTqGgxiifPUexS6Ro7GcqLgvc"  # Gmail User Name: marcowu1999

    def __init__(self, **kwargs):
        # [Langchain]
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = Gemini.GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.model = ChatGoogleGenerativeAI(model="gemini-pro")

        super().__init__(self.model, **kwargs)
