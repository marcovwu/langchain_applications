# LangChain Applications

## Install
```sh
# develop mode
git clone https://github.com/marcovwu/langchain_applications.git
cd langchain_applications
pip install -e .

# deploy mode
# pip install git+https://github.com/marcovwu/langchain_applications.git
```

## Run
(1) python
```pytyhon
python langchain_applications/runner.py -h  # or watch test.sh
```

(2) command
```sh
langapp -h
```

## Import
```python
from langchain_applications.models.models import LLMRunner

# Initialize
llmrunner = LLMRunner.load('Gemini', input_info='', memory_mode='', chain_mode='', agent_mode='')

# Summary
summary = llmrunner.summary(input_info=out_text)
# Chat
result = llmrunner.chat(_obatin_prompt_text(text), text)
# Large Language Model
summary = llmrunner.model.invoke(_obatin_prompt_text(text)).content
```