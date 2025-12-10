import re
import ftfy
from cleantext import clean
from typing import List
from langchain_core.documents import Document

class AdvancedTextCleaner:
    def __init__(self):
        self.garbage_patterns = [
            r'[a-zA-Z0-9_]{20,}[~]]+',
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}+',
            r'[a-f0-9]{36}',
            r'[a-f0-9]{30,}',
            r'~{3,}',
            r'_{3,}'
        ]

    def clean_text(self, text):
        if text is not isinstance(text, str):
            return ""

        text = ftfy.fix_text(text)

        for pattern in self.garbage_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

        text = clean(text,
                     clean_all=False,
                     stemming=True,
                     extra_spaces=True,
                     )
        lines = text.split('\n')
        clean_line = []
        for line in lines:
            line = line.strip()
            if self._is_meaningful_line(line):
                clean_line.append(line)
        return '\n'.join(clean_line).strip()

    def _is_meaningful_line(self, line):
        if len(line) < 2:
            return False

        has_chinese = bool(re.search(r'[\u4e00-\u9fff]+', line))
        has_english = bool(re.search(r'\b[a-zA-Z]{2,}\b+', line))

        garbage_ratio = len(re.findall(r'[^a-zA-Z0-9\u4e00-\u9fff\s\.,;:()|\-+]+', line)) / len(line)
        return (has_chinese or has_english) and garbage_ratio <= 0.3

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        cleaned_docs = []

        for i, doc in enumerate(documents):
            cleaned_content = self.clean_text(doc.page_content)

            if len(cleaned_content.strip()) > 10:
                cleaned_doc = Document(
                    page_content=cleaned_content,
                )
                cleaned_docs.append(cleaned_doc)
                print(f"✅ clean {i}: {cleaned_content[:100]}...")
            else:
                print(f"❌ 移除垃圾块 {i}: {doc.page_content[:100]}...")
        return cleaned_docs