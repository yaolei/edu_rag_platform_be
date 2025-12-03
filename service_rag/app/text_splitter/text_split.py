from typing import List, Optional, Dict, Iterable, Union
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter as langChainRecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    HTMLHeaderTextSplitter,
)

class TextSplitter:
    def __init__(self, splitter_type:str = "recursive",
                 chunk_size:int = 500,
                 chunk_overlap:int = 200,
                 add_start_index:bool = True,
                 separators=["\n\n### ", "\n\n## ",
                                "\n\n",
                                "\n",
                                " ",
                                ""
                            ],
                 **kwargs
                 ):
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index
        self.separators = separators
        self.split_kwargs = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "separators":separators, **kwargs}

        if self.splitter_type == "recursive":
             self.splitter = langChainRecursiveCharacterTextSplitter(**self.split_kwargs)
        elif self.splitter_type == "markdown":
            self.splitter = MarkdownHeaderTextSplitter(
                            headers_to_split_on=[('#', "Header 1"),
                                                  ('##', "Header 2"),
                                                   ('###', "Header 3")
                                                 ],
                            **self.split_kwargs)
        elif self.splitter_type == "python":
            self.splitter = PythonCodeTextSplitter(**self.split_kwargs)
        elif self.splitter_type == "html":
            self.splitter = HTMLHeaderTextSplitter(headers_to_split_on=[('h1', "Header 1"),
                                                  ('h2', "Header 2"),
                                                   ('h3', "Header 3")
                                                 ],**self.split_kwargs)
        elif self.splitter_type == "token":
            self.splitter = TokenTextSplitter(**self.split_kwargs)
        elif self.splitter_type == "character":
            self.splitter = CharacterTextSplitter(**self.split_kwargs)

    def split_text(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, list):
            text = "\n".join(text)
        return self.splitter.split_text(text)

    def create_document(self, text: List[str], metadata: Optional[List[Dict]] = None):
        if metadata is None:
            metadata = [{"source": "tmp_sample"}] *len(text)
        return self.splitter.create_documents(text, metadata)

    def split_document(self, document: Iterable[Document]):
        return self.splitter.split_documents(document)


