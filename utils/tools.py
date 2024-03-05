import numpy as np

from abc import ABC
from sklearn.cluster import KMeans

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import (
    LLMChain, load_summarize_chain, MapReduceDocumentsChain, ReduceDocumentsChain, AnalyzeDocumentChain
)
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


class Summarizer(ABC):
    MAX_TOKENS = 8192
    CHUNK_SIZE = 2048
    CHUNK_OVERLAP = 0
    PROMPT_TEMPLATE = {
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
    CHAIN_MODE_CHOICES = "'summary-stuff', 'summary-refine', 'summary-map_reduce', 'stuff', 'refine', 'analyze'" + \
        ", 'map_reduce' 'rag_map_reduce-stuff, rag_map_reduce-refine'"
    AGENT_MODE_CHOICE = "'wiki'"

    def __init__(self, model, chain_mode='', agent_mode='wiki', split=False):
        """
        :param chain_mode: one of Summarizer.CHAIN_MODE_CHOICES
        :param agent_mode: one of Summarizer.AGENT_MODE_CHOICE
        """

        self.model = model
        self.split = split

        # init chain
        self.chain_mode = chain_mode
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=Summarizer.CHUNK_SIZE, chunk_overlap=Summarizer.CHUNK_OVERLAP)
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
                chunk_size=Summarizer.CHUNK_SIZE,
                chunk_overlap=Summarizer.CHUNK_OVERLAP)
            self.init_rag_map_reduce_chain(chain_type=self.chain_mode.split('-')[1])

        # init agent
        self.agent_mode = agent_mode
        if 'wiki' == self.agent_mode:
            self.init_wiki_agent()

    @staticmethod
    def load_pdf(file):
        pages = []
        if '.pdf' in file:
            loader = PyPDFLoader(file)
            pages = loader.load_and_split()
        return pages

    def init_summary_chain(self, chain_type='stuff'):
        # summary api
        self.chain = load_summarize_chain(self.model, chain_type=chain_type)  # "stuff", "refine"

    def init_stuff_chain(self):
        # chain api
        prompt = PromptTemplate.from_template(Summarizer.PROMPT_TEMPLATE["summary"])
        self.llm_chain = LLMChain(llm=self.model, prompt=prompt)
        self.chain = StuffDocumentsChain(llm_chain=self.llm_chain, document_variable_name="text")

    def init_refine_chain(self):
        # chain api
        prompt = PromptTemplate.from_template(Summarizer.PROMPT_TEMPLATE["summary"])
        refine_prompt = PromptTemplate.from_template(Summarizer.PROMPT_TEMPLATE["refine"])
        self.chain = load_summarize_chain(
            llm=self.model, chain_type="refine", question_prompt=prompt, refine_prompt=refine_prompt,
            return_intermediate_steps=True, input_key="input_documents", output_key="output_text",
        )

    def init_analyze_chain(self, chain):
        self.chain = AnalyzeDocumentChain(combine_docs_chain=chain, text_splitter=self.text_splitter)

    def init_map_reduce_chain(self):
        # map-reduce api
        map_chain = LLMChain(llm=self.model, prompt=Summarizer.PROMPT_TEMPLATE["map"])
        reduce_prompt = PromptTemplate.from_template(Summarizer.PROMPT_TEMPLATE["reduce"])
        reduce_chain = LLMChain(llm=self.model, prompt=reduce_prompt)
        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,  # If documents exceed context for `StuffDocumentsChain`
            token_max=Summarizer.MAX_TOKENS,  # The maximum number of tokens to group documents into.
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
        map_prompt_template = PromptTemplate(template=Summarizer.PROMPT_TEMPLATE["rag_map"], input_variables=["text"])
        self.map_chain = load_summarize_chain(llm=self.model, chain_type=chain_type, prompt=map_prompt_template)

        # reduce
        combine_prompt_template = PromptTemplate(
            template=Summarizer.PROMPT_TEMPLATE["rag_combine"], input_variables=["text"])
        self.reduce_chain = load_summarize_chain(llm=self.model, chain_type=chain_type, prompt=combine_prompt_template)

    def summary_from_rag(self, docs):
        embeddings = HuggingFaceEmbeddings()  # model_name="thenlper/gte-large"
        vectors = embeddings.embed_documents([x.page_content for x in docs])
        # documents = self.text_splitter.split_documents(docs)
        # vectorstore = FAISS.from_documents(documents, embeddings)
        # retriever = vectorstore.as_retriever()  # search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5}

        # Perform K-means clustering
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
        selected_docs = [docs[doc] for doc in selected_indices]  # Get your docs which the top vectors represented

        # Map Summaries
        summary_list = []
        for i, doc in enumerate(selected_docs):
            # Go get a summary of the chunk
            chunk_summary = self.map_chain.run([doc])
            # Append that summary to your list
            summary_list.append(chunk_summary)
            print(f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} ... \n")
        summaries = "\n".join(summary_list)
        summaries = Document(page_content=summaries)

        # Reduce Summaries
        output = self.reduce_chain.run([summaries])

        return output

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

    def run(self, text_with_prompt, text, input_info=None):
        docs = [Document(page_content=text)]
        if input_info is not None:
            docs = Summarizer.load_pdf(input_info)

        # Langchain Chain
        if self.chain_mode:
            if (
                self.split
                or 'refine' == self.chain_mode
                or 'rag_map_reduce' in self.chain_mode
                or 'map_reduce' == self.chain_mode
            ):
                docs = self.text_splitter.split_documents(docs)

            # chain
            if 'refine' == self.chain_mode:
                result = self.chain({"input_documents": docs}, return_only_outputs=True)
                result_text = result["output_text"]  # "\n\n".join(result["intermediate_steps"][:3])
            elif 'rag_map_reduce' in self.chain_mode:
                result_text = self.summary_from_rag(docs)
            else:
                if 'analyze' == self.chain_mode:
                    docs = text
                result_text = self.chain.run(docs)

        # Langchain Agent
        if self.agent_mode:
            result_text = self.agent.run("Can you please provide a quick summary of Napoleon Bonaparte? \
                Then do a separate search and tell me what the commonalities are with Serena Williams")

        return result_text
