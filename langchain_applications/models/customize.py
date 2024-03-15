import torch

from typing import Any, List, Optional, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class HuggingFaceBlueLM(LLM):
    model_name: str
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    device: Union[str, torch.device]
    return_tensors: str
    max_new_tokens: int
    repetition_penalty: float
    skip_special_tokens: bool

    def __init__(
        self,
        model_name: str = "vivo-ai/BlueLM-7B-Chat-4bits",
        return_tensors: str = "pt",
        max_new_tokens: int = 64,
        repetition_penalty: float = 1.1,
        skip_special_tokens: bool = True,
    ):
        # Build model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        if "4bits" in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", trust_remote_code=True).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        device = model.device

        # Initialize
        super().__init__(
            model_name=model_name, tokenizer=tokenizer, model=model, device=device,
            return_tensors=return_tensors,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def _llm_type(self) -> str:
        return "bluelm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # inference
        inputs = self.tokenizer(prompt, return_tensors=self.return_tensors).to(self.device)
        pred = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, repetition_penalty=self.repetition_penalty)
        return self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=self.skip_special_tokens)
