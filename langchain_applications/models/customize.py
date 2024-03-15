from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class HuggingFaceBlueLM(LLM):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "vivo-ai/BlueLM-7B-Chat-4bits"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if "4bits" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    device = model.device

    # initialize hyp
    return_tensors = "pt"
    max_new_tokens = 64
    repetition_penalty = 1.1
    skip_special_tokens = True

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
