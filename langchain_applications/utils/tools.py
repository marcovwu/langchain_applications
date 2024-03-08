import os
import numpy as np

from abc import ABC
from collections import deque
from sklearn.cluster import KMeans

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import (
    LLMChain, RetrievalQA, load_summarize_chain, MapReduceDocumentsChain, ReduceDocumentsChain, AnalyzeDocumentChain
)
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # , Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader  # Docx2txtLoader
from langchain_core.prompts import (
    ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
)


class LangChain(ABC):
    # Variables
    MAX_TOKENS = 8192
    CHUNK_SIZE = 2048
    CHUNK_OVERLAP = 0
    SPLIT_DOCS = False

    # Prompts
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
{docs}

Current conversation:
""" % (PROMPT_SYSTEM_ROLE, PROMPT_RAG_SYSTEM),
        "retrievalqa": """
Use the following pieces of information to answer the user's question. If you don't know the answer,
just say that you don't know, don't try to repeat or make up an answer.

Partial Content: {docs}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
    }
    SUMMARY_PROMPT = {
        "summary": """Write a concise summary of the following over 1000 words:
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

    # Options
    SUPPORT_FORMATS = [".pdf", ".doc", ".txt", ".md"]
    MEMORY_MODE_CHOICES = "'default', 'langchain_default', 'conversation_summary', 'conversation_buffer'"
    CHAIN_MODE_CHOICES = "'summary-stuff', 'summary-refine', 'summary-map_reduce', 'stuff', 'refine', 'analyze'" + \
        ", 'map_reduce' 'rag_map_reduce-stuff, rag_map_reduce-refine', 'rag_retrieval'"
    AGENT_MODE_CHOICE = "'wiki'"

    def __init__(self, model, input_info='', memory_mode='', chain_mode='', agent_mode=''):
        """
        :param chain_mode: one of self.CHAIN_MODE_CHOICES
        :param agent_mode: one of self.AGENT_MODE_CHOICE
        """

        # init info
        self.model = model
        self.chain_keys = {"input": "input", "output": "text", "memory": "chat_history"}
        self.memory_mode = memory_mode
        self.chain_mode = chain_mode
        self.agent_mode = agent_mode

        # init input
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        self.update_documents(input_info)

        # init memory
        self.chat_history, self.memory_chain = None, None
        if self.memory_mode == "conversation_summary":
            prompt = self.init_conversation_summary_memory()
            self.memory_chain = LLMChain(prompt=prompt, llm=self.model, memory=self.chat_history, verbose=True)
            # self.memory_chain = ConversationChain(llm=self.model, memory=self.chat_history, verbose=True)
        elif self.memory_mode == "conversation_buffer":
            prompt = self.init_conversation_buffer_memory()
            self.memory_chain = LLMChain(prompt=prompt, llm=self.model, memory=self.chat_history, verbose=True)
        elif self.memory_mode == "langchain_default":
            prompt = self.init_langchain_default_memory()
            self.memory_chain = LLMChain(prompt=prompt, llm=self.model, verbose=True)
        elif self.memory_mode == "default":
            self.chat_history = deque()

        # init chain
        self.chain = None
        if 'summary' in self.chain_mode:
            self.init_summary_chain(chain_type=self.chain_mode.split('-')[1])
        elif 'stuff' == self.chain_mode:
            self.init_stuff_chain()
        elif 'refine' == self.chain_mode:
            self.init_refine_chain()
        elif 'analyze' in self.chain_mode:
            self.init_analyze_chain(chain=load_summarize_chain(self.model, chain_type='refine'))
        elif 'map_reduce' == self.chain_mode:
            self.init_map_reduce_chain()
        elif 'rag_map_reduce' in self.chain_mode:
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "\t"],
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP)
            self.init_rag_map_reduce_chain(chain_type=self.chain_mode.split('-')[1])
        elif 'rag_retrieval' == self.chain_mode:
            self.init_rag_retrieval_chain(documents=self.documents)  # , chain_type=self.chain_mode.split('-')[1])

        # init agent
        self.agent = None
        if 'wiki' == self.agent_mode:
            self.init_wiki_agent()

    @staticmethod
    def load_file(file):
        text, docs = "no input ...", [Document(page_content="")]
        if isinstance(file, str) and os.path.exists(file):
            if '.pdf' in file:
                loader = PyPDFLoader(file)
                docs = loader.load_and_split()
                for doc in docs:
                    text += " " + doc.page_content
            elif '.doc' in file:
                loader = UnstructuredWordDocumentLoader(file)  # Docx2txtLoader(file)
                docs = loader.load_and_split()
                for doc in docs:
                    text += " " + doc.page_content
            elif '.txt' in file or '.md' in file:
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read()
                docs = [Document(page_content=text)]
        return text, docs

    def update_chain_keys(self, dict_info):
        for k, v in dict_info.items():
            if k in self.chain_keys and isinstance(self.chain_keys[k], str) and self.chain_keys[k]:
                self.chain_keys[k] = v

    def update_embeding_documents(self, documents):
        embeddings = HuggingFaceEmbeddings()  # model_name="thenlper/gte-large"
        vectors = embeddings.embed_documents([x.page_content for x in documents])
        # documents = self.text_splitter.split_documents(documents)
        # vectorstore = FAISS.from_documents(documents, embeddings)
        # retriever = vectorstore.as_retriever()  # search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5}

        # Perform K-means clustering
        if vectors:
            num_clusters = 11
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

            # Perform t-SNE and reduce to 2 dimensions
            # tsne = TSNE(n_components=2, random_state=42)
            # reduced_data_tsne = tsne.fit_transform(vectors)

            # Find the closest embeddings to the centroids
            closest_indices = []
            for i in range(num_clusters):
                # Get the list of distances from that particular cluster center
                distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
                # Find the list position of the closest one (using argmin to find the smallest distance)
                closest_index = np.argmin(distances)
                # Append that position to your closest indices list
                closest_indices.append(closest_index)
            selected_indices = sorted(closest_indices)
            self.documents = [documents[doc] for doc in selected_indices]  # selected documents

    def update_documents(self, input_info):
        # load
        if not isinstance(input_info, str):
            print(f'No update any document. unsupport format for {input_info}')
            return '', [Document(page_content="")]
        else:
            if os.path.isdir(input_info):
                print('Loading file from %s' % input_info)
                self.origin_text, self.documents = "", [Document(page_content="")]
                # dir
                # from langchain.document_loaders import DirectoryLoader
                # loader = DirectoryLoader(input_info, show_progress=True)
                # self.documents = loader.load_and_split()
                # for doc in self.documents:
                #     self.origin_text += " " + doc.page_content
                # file
                for filename in os.listdir(input_info):
                    origin_text, documents = LangChain.load_file(os.path.join(input_info, filename))
                    if origin_text != "no input ...":
                        print(os.path.join(input_info, filename))
                        self.origin_text += "\n\n\n" + origin_text
                        self.documents += documents
            elif os.path.isfile(input_info):
                print('Loading file from %s' % input_info)
                self.origin_text, self.documents = LangChain.load_file(input_info)
            else:
                print('No input file update to document')
                self.origin_text, self.documents = "no input ...", [Document(page_content="")]

            # split
            if (
                self.SPLIT_DOCS
                or 'refine' == self.chain_mode
                or 'rag_map_reduce' in self.chain_mode
                or 'map_reduce' == self.chain_mode
            ):
                self.documents = self.text_splitter.split_documents(self.documents)

            # rag
            if 'rag_map_reduce' in self.chain_mode:
                self.update_embeding_documents(self.documents)

            return self.origin_text, self.documents

    def init_langchain_default_memory(self):
        prompt = PromptTemplate.from_template(self.CHATBOT_PROMPT["langchain_default"])
        self.chat_history = ChatMessageHistory(memory_key=self.chain_keys["memory"])
        return prompt

    def init_conversation_summary_memory(self):
        prompt = PromptTemplate.from_template(self.CHATBOT_PROMPT["conversation_summary"])
        self.chat_history = ConversationSummaryMemory(llm=self.model)
        return prompt

    def init_conversation_buffer_memory(self):
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.PROMPT_SYSTEM),
                MessagesPlaceholder(variable_name=self.chain_keys["memory"]),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        self.chat_history = ConversationBufferMemory(memory_key=self.chain_keys["memory"], return_messages=True)
        return prompt

    def init_summary_chain(self, chain_type='stuff'):
        # summary api
        self.chain = load_summarize_chain(self.model, chain_type=chain_type)  # "stuff", "refine"

    def init_stuff_chain(self):
        # chain api
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT["summary"])
        self.llm_chain = LLMChain(llm=self.model, prompt=prompt)
        self.chain = StuffDocumentsChain(llm_chain=self.llm_chain, document_variable_name="text")

    def init_refine_chain(self):
        # chain api
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT["summary"])
        refine_prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT["refine"])
        self.chain = load_summarize_chain(
            llm=self.model, chain_type="refine", question_prompt=prompt, refine_prompt=refine_prompt,
            return_intermediate_steps=True,  # input_key=self.chain_keys["input"], output_key=self.chain_keys["output"],
        )

    def init_analyze_chain(self, chain):
        self.chain = AnalyzeDocumentChain(combine_docs_chain=chain, text_splitter=self.text_splitter)

    def init_map_reduce_chain(self):
        # map-reduce api
        map_chain = LLMChain(llm=self.model, prompt=self.SUMMARY_PROMPT["map"])
        reduce_prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT["reduce"])
        reduce_chain = LLMChain(llm=self.model, prompt=reduce_prompt)
        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,  # If documents exceed context for `StuffDocumentsChain`
            token_max=self.MAX_TOKENS,  # The maximum number of tokens to group documents into.
        )
        self.chain = MapReduceDocumentsChain(
            llm_chain=map_chain, reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",  # The variable name in the llm_chain to put the documents in
            return_intermediate_steps=False,   # Return the results of the map steps in the output
        )

    def init_rag_map_reduce_chain(self, chain_type):
        """
        Level 4: Best Representation Vectors - Summarize an entire book

        In the above method we pass the entire document (all 9.5K tokens of it) to the LLM. But what if you have more
        tokens than that?

        What if you had a book you wanted to summarize? Let's load one up, we're going to load Into Thin Air about the
        1996 Everest Disaster
        """

        # map
        map_prompt_template = PromptTemplate(template=self.SUMMARY_PROMPT["rag_map"], input_variables=["text"])
        self.map_chain = load_summarize_chain(llm=self.model, chain_type=chain_type, prompt=map_prompt_template)

        # reduce
        combine_prompt_template = PromptTemplate(
            template=self.SUMMARY_PROMPT["rag_combine"], input_variables=["text"])
        self.reduce_chain = load_summarize_chain(llm=self.model, chain_type=chain_type, prompt=combine_prompt_template)

        # overall
        self.chain = self.summary_map_reduce_chain_from_rag

    def init_rag_retrieval_chain(self, documents):  # TODO: , chain_type='map_reduce'):
        # embedding
        self.embeddings = HuggingFaceEmbeddings()  # model_name="thenlper/gte-large"
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        # .as_retriever() > search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5}
        self.retriever = self.vectorstore.as_retriever()

        # chains
        prompt = PromptTemplate.from_template(
            self.CHATBOT_PROMPT["retrievalqa" + "_" + self.memory_mode if self.memory_mode else ""])
        chain = LLMChain(llm=self.model, prompt=prompt, verbose=True)
        combine_documents_chain = StuffDocumentsChain(llm_chain=chain, document_variable_name="docs")

        # TODO: RetrievalQA
        self.update_chain_keys(dict_info={"input": "query", "output": "result"})
        self.chain = RetrievalQA(retriever=self.retriever, combine_documents_chain=combine_documents_chain)
        # self.chain = RetrievalQA.from_chain_type(
        #     llm=self.model, chain_type=chain_type, retriever=self.retriever, return_source_documents=True,
        #     chain_type_kwargs={"prompt": prompt}, verbose=True)

    def init_wiki_agent(self):
        """
        Level 5: Agents - Summarize an unknown amount of text

        What if you have an unknown amount of text you need to summarize? This may be a verticalize use case (like law
        or medical) where more research is required as you uncover the first pieces of information.

        We're going to use agents below, this is still a very actively developed area and should be handled with care.
        Future agents will be able to handle a lot more complicated tasks.
        """
        from langchain.utilities import WikipediaAPIWrapper
        wikipedia = WikipediaAPIWrapper()
        tools = [
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Useful for when you need to get information from wikipedia about a single topic"
            ),
        ]
        self.agent = initialize_agent(tools, self.model, agent='zero-shot-react-description', verbose=True)

    def summary_map_reduce_chain_from_rag(self, documents):
        if documents:
            # Map Summaries
            summary_list = []
            for i, doc in enumerate(documents):
                # Go get a summary of the chunk
                chunk_summary = self.map_chain.run([doc])
                # Append that summary to your list
                summary_list.append(chunk_summary)
                print(
                    f"Summary #{i} (chunk #{documents[i].page_content[:250]}) - Preview: {chunk_summary[:250]} ... \n")
            summaries = "\n".join(summary_list)
            summaries = Document(page_content=summaries)

            # Reduce Summaries
            output = self.reduce_chain.run([summaries])
        else:
            print('No any information in the database!!')
            output = ''

        return output


class ChatBot(LangChain):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def chat_preprocesss(self, text_with_prompt, text):
        return text_with_prompt, text

    def count_tokens(self, string):
        return len(string.split(" "))

    def obtain_default_prompt(self):
        return self.CHATBOT_PROMPT["default"] + "".join(self.chat_history)

    def obtain_memory(self):
        if "conversation" in self.memory_mode:
            msg = self.chat_history.chat_memory.messages
        elif self.memory_mode == "langchain_default":
            msg = self.chat_history.messages
        else:
            msg, words = self.obtain_default_prompt(), self.count_tokens(self.CHATBOT_PROMPT["default"])
            while self.count_tokens(msg) > self.MAX_TOKENS - words:
                self.chat_history.popleft()  # Human
                self.chat_history.popleft()  # AI
                msg = self.obtain_default_prompt()
        return msg

    def update_memory(self, input_text, output_text):
        if "conversation" in self.memory_mode:
            # self.chat_history.save_context({"input": input_text}, {"output": output_text})
            self.chat_history.chat_memory.add_user_message(input_text)
            self.chat_history.chat_memory.add_ai_message(output_text)
        elif self.memory_mode == "langchain_default":
            self.chat_history.add_user_message(input_text)
            self.chat_history.add_ai_message(output_text)
        elif self.memory_mode == "default":
            self.chat_history.append(input_text)
            self.chat_history.append(output_text)

    def chat_postprocess(self, text_with_prompt, text, response_text):
        input_text, output_text = text, response_text
        return input_text, output_text

    def _inference(self, module, input_val, output_key, text_with_prompt, text):
        # Run Large Language Model
        if module is not None:
            response_text = module(input_val) if output_key is None else module(input_val)[output_key]
            if not isinstance(response_text, str):
                response_text = response_text.content

            # Post-processing
            in_text, out_text = self.chat_postprocess(text_with_prompt, text, response_text)
            self.update_memory(in_text, out_text)
        else:
            response_text = ''
        return response_text

    def summary(self, input_info=None):
        # Initialize
        _module, input_val, output_key = None, self.documents, None
        if isinstance(input_info, str):
            self.update_documents(input_info)

        # Chain
        if self.chain is not None:
            _module = self.chain
            if 'refine' == self.chain_mode:
                _module = self.chain
                input_val = {"input_documents": self.documents}  # "\n\n".join(result["intermediate_steps"][:3])
            elif 'rag_map_reduce' in self.chain_mode:
                _module = self.chain
            else:
                _module = self.chain.run
                if 'analyze' == self.chain_mode:
                    input_val = self.origin_text

        # Agent
        if self.agent is not None:
            _module, input_val = self.agent.run, "Can you please provide a quick summary of Napoleon Bonaparte? \
                Then do a separate search and tell me what the commonalities are with Serena Williams"

        # Run
        print('Obtaining summary ...')
        return self._inference(_module, input_val, output_key, self.origin_text, self.origin_text)

    def chat(self, text_with_prompt, text, use_history=False):
        # Initialize & Pre-processing
        _module, input_val, output_key = self.model.invoke, text_with_prompt, None
        text_with_prompt, text = self.chat_preprocesss(text_with_prompt, text)

        # Chain
        if self.chat_history is not None and use_history:
            if self.memory_mode == "default":
                input_val = self.obtain_memory() + text_with_prompt
        if self.memory_chain is not None:
            input_val = {self.chain_keys["input"]: text_with_prompt, self.chain_keys["memory"]: self.obtain_memory()}
            _module, output_key = self.memory_chain.invoke, self.chain_keys["output"]

        # Run
        return self._inference(_module, input_val, output_key, text_with_prompt, text)
