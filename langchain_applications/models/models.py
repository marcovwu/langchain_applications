import os

from loguru import logger

from langchain_applications.utils.tools import ChatBot  # noqa: E402


class LLMRunner(ChatBot):
    LLM_CHOICES = "'Llama', 'MiniCPM', 'Gemma', 'Zephyr', 'MeetingSummary' or 'Gemini'"

    @classmethod
    def load(cls, llmname, **kwargs):
        llm_class = globals().get(llmname, None)
        if llm_class:
            llm_runner = llm_class(**kwargs)
        else:
            logger.info('Unknown llm: %s' % llmname)
            llm_runner = None
        return llm_runner


class Llama(ChatBot):
    TOKEN_LIMIT = 2048
    MAX_NEW_TOKENS = 512
    MAX_PROMPT_LENGTH = 200
    CHUNK_OVERLAP = 100
    SPLIT_DOCS = True

    def __init__(self, **kwargs):
        # [Llama Pretrained-13B]
        # from langchain_community.llms import LlamaCpp
        # from langchain.callbacks.manager import CallbackManager
        # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        # self.model = LlamaCpp(
        #     model_path="/data1/marco.wu/models/Large_Language_Model/llama-2-13b/llama-2-13b-chat.Q5_K_M.gguf",
        #     n_gpu_layers=-1, n_batch=512, n_ctx=16384, f16_kv=True, max_tokens=8192,
        #     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True,
        # )

        # [Llama Chinese-Alpaca-2-7B]
        import torch
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # max_new_tokens + input tokens must <= token limit
        self.llm = pipeline(
            "text-generation", model="/data1/marco.wu/models/Large_Language_Model/chinese-alpaca-2-7b",
            device_map="auto", max_new_tokens=self.MAX_NEW_TOKENS, torch_dtype=torch.bfloat16, repetition_penalty=1.1
            # pipeline_kwargs={
            #     "max_new_tokens": self.MAX_NEW_TOKENS,
            #     "do_sample": True,
            #     "temperature": 0.2,
            #     "top_k": 40,
            #     "top_p": 0.9,
            #     "repetition_penalty": 1.1
            # },
            # model_kwargs={"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "trust_remote_code": True}
        )
        self.model = HuggingFacePipeline(pipeline=self.llm)

        # Initialize
        self.SUMMARY_PROMPT["stuff"] = self.SUMMARY_PROMPT["summary"] = (
            "[INST] <<SYS>>\n"
            "You are a helpful assistant. 你是一个乐于助人的助手。\n"
            "<</SYS>>\n\n"
            "请为以下文字写一段摘要:\n{text} [/INST]"
        )
        self.SUMMARY_PROMPT["refine"] = (
            "[INST] <<SYS>>\n"
            "You are a helpful assistant. 你是一个乐于助人的助手。\n"
            "<</SYS>>\n\n"
            "已有一段摘要：{existing_answer}\n"
            "现在还有一些文字，（如果有需要）你可以根据它们完善现有的摘要。"
            "\n"
            "{text}\n"
            "\n"
            "如果这段文字没有用，返回原来的摘要即可。请你生成一个最终的摘要。"
            " [/INST]"
        )
        super().__init__(self.model, **kwargs)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])


class MiniCPM(ChatBot):
    TOKEN_LIMIT = 2048
    MAX_NEW_TOKENS = 512
    MAX_PROMPT_LENGTH = 200
    CHUNK_OVERLAP = 100
    SPLIT_DOCS = True

    def __init__(self, **kwargs):
        # [Llama Chinese-Alpaca-2-7B]
        import torch
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # max_new_tokens + input tokens must <= token limit
        self.llm = pipeline(
            "text-generation", model="/data1/marco.wu/models/Large_Language_Model/openbmb-minicpm-2B-sft-fp32-llama",
            torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=self.MAX_NEW_TOKENS)
        self.model = HuggingFacePipeline(pipeline=self.llm)

        # Initialize
        super().__init__(self.model, **kwargs)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])


class Gemma(ChatBot):
    TOKEN_LIMIT = 2048
    MAX_NEW_TOKENS = 512
    MAX_PROMPT_LENGTH = 200
    CHUNK_OVERLAP = 0
    SPLIT_DOCS = True

    def __init__(self, **kwargs):
        # [LlamaCPP] url: https://huggingface.co/brittlewis12/gemma-7b-GGUF
        from langchain_community.llms import LlamaCpp
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        self.model = LlamaCpp(
            model_path="/data1/marco.wu/models/Large_Language_Model/gemma-7b/gemma-7b.Q5_K_M.gguf", n_gpu_layers=-1,
            n_batch=512, n_ctx=16384, f16_kv=True, max_tokens=self.TOKEN_LIMIT,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True,
        )

        # [HuggingFace Pipeline]
        # import torch
        # from transformers import pipeline
        # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # self.llm = pipeline(
        #     "text-generation", model="google/gemma-7b-it", torch_dtype=torch.bfloat16, device_map="auto",
        #     max_new_tokens=self.MAX_NEW_TOKENS)
        # self.model = HuggingFacePipeline(pipeline=self.llm)

        # [HuggingFace]
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", device_map="auto")
        # self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")

        # Initialize
        super().__init__(self.model, **kwargs)

    # def count_tokens(self, string):
    #     return len(self.llm.tokenizer(string)["input_ids"])


class Zephyr(ChatBot):
    TOKEN_LIMIT = 2048  # 8192
    MAX_NEW_TOKENS = 512  # 2048
    MAX_PROMPT_LENGTH = 200
    CHUNK_OVERLAP = 0
    SPLIT_DOCS = True

    def __init__(self, **kwargs):
        import torch
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        self.llm = pipeline(
            "text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto",
            max_new_tokens=self.MAX_NEW_TOKENS)
        self.model = HuggingFacePipeline(pipeline=self.llm)

        # Initialize
        self.SUMMARY_PROMPT["summary"] = """<|system|>
Write a concise summary of the following about 1000 words:</s>
<|assistant|>
"{text}"</s>
<|user|>
CONCISE SUMMARY:</s>"""
        self.SUMMARY_PROMPT["map_paper"] = """<|system|>
You will be given a single passage of a paper. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of the paper proposed.
Your response should consist of at least three paragraphs and fully encompass the content conveyed in the passage.

```{text}```</s>
<|user|>
Write a concise summary about 500 words:</s>
<|assistant|>
"""
        self.SUMMARY_PROMPT["combine_paper"] = """<|system|>
You will be given a series of summaries from a paper. The summaries will be enclosed in triple backticks (```)
Your goal is to provide a verbose summary contain abstract problem, proposed method, and result from the summaries.
The reader should be able to grasp the key-problem, proposed method, and result in the paper.

```{text}```</s>
<|user|>
Write a verbose summary approximately 1000 words long:</s>
<|assistant|>
"""
        self.CHATBOT_PROMPT["default"] = f"<|system|>\n{self.CHATBOT_PROMPT['default']}</s>\n"
        super().__init__(self.model, **kwargs)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])

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

    def chat_preprocesss(self, text_with_prompt, text):
        text_with_prompt = f"<|user|>\n{text}</s>\n<|assistant|>\n"
        return text_with_prompt, text

    def chat_postprocess(self, text_with_prompt, text, response_text):
        response_text = f"<|assistant|>\n{response_text}</s>\n"
        return text_with_prompt, response_text


class MeetingSummary(ChatBot):
    # Variables
    TOKEN_LIMIT = 1024
    MAX_NEW_TOKENS = 128
    MAX_PROMPT_LENGTH = 200
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
            max_new_tokens=self.MAX_NEW_TOKENS)
        self.model = HuggingFacePipeline(pipeline=self.llm)

        # Initialize
        super().__init__(self.model, **kwargs)

    def count_tokens(self, string):
        return len(self.llm.tokenizer(string)["input_ids"])


class Gemini(ChatBot):
    # TOKEN_LIMIT = 8192
    # MAX_NEW_TOKENS = 2048
    # MAX_PROMPT_LENGTH = 200
    # CHUNK_OVERLAP = 0
    # SPLIT_DOCS = True
    HISTORY_LENGTH = 100000
    GOOGLE_API_KEY = "AIzaSyC3d_lzyoPTqGgxiifPUexS6Ro7GcqLgvc"  # Gmail User Name: marcowu1999

    def __init__(self, **kwargs):
        # [Langchain]
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = Gemini.GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.model = ChatGoogleGenerativeAI(model="gemini-pro")

        super().__init__(self.model, **kwargs)
