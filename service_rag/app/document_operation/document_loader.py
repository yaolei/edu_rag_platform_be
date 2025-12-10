# -*- coding: utf-8 -*-
import os, tempfile, pathlib,asyncio
from typing import Optional, List
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, WebBaseLoader, PyPDFLoader,
    UnstructuredImageLoader, UnstructuredPDFLoader
)


class DocumentLoader:
    def __init__(self,
                 upload_file: Optional[UploadFile] = None,
                 document_type: Optional[str] = None,
                 urls: Optional[List[str]] = None,
                 **kwargs):
        self.upload_file = upload_file
        self.urls = urls or []
        self.kwargs = kwargs
        self.temp_file_path = None
        self.filename = upload_file.filename if upload_file else "web"

        if document_type is None:
            document_type = self._detect_document_type()
        self.document_type = document_type

    async def _get_upload_file_content(self) -> bytes:
        self.upload_file.file.seek(0)
        content = await run_in_threadpool(self.upload_file.file.read)
        self.upload_file.file.seek(0)
        return content

    async def _create_temp_file_if_needed(self) -> None:
        if self.temp_file_path:
            return
        suffix = pathlib.Path(self.filename).suffix
        fd, self.temp_file_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        content = await self._get_upload_file_content()
        with open(self.temp_file_path, "wb") as tmp:
            tmp.write(content)

    def _detect_document_type(self) -> str:
        if self.urls:
            return "web"
        ct = self.upload_file.content_type or ""
        if ct == "application/pdf":
            return "pdf"
        if ct == "text/csv":
            return "csv"
        if ct in {"application/xml", "text/xml"}:
            return "xml"
        if ct.startswith("image/"):
            return "image"
        if ct in {"application/octet-stream", "text/plain"}:
            pass
        if self.upload_file is None:
            raise ValueError("没有上传文件且未指定 urls")

        ext = pathlib.Path(self.filename).suffix.lower()
        if ext == ".pdf":
            return "pdf"
        if ext == ".csv":
            return "csv"
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
            return "image"
        if ext in {".txt", ".md", ".py", ".js", ".html", ".css", ".json"}:
            return "txt"
        return "txt"        # 默认文本

    async def _get_loader_by_type(self):
        # 1.  web 类型（URL 列表）→ 不需要本地文件
        if self.document_type == "web":
            if not self.urls:
                raise ValueError("web 类型必须提供 urls 参数")
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" 
                                     "AppleWebKit/537.36 (HTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"}
            # return WebBaseLoader(self.urls)   # 支持异步
            return AsyncHtmlLoader(self.urls, header_template=headers)

        # 2.  需要本地文件的类型 → 先落临时文件
        needs_file = self.document_type in {"pdf", "image", "csv"}
        if needs_file:
            await self._create_temp_file_if_needed()
            if not self.temp_file_path or not os.path.exists(self.temp_file_path):
                raise ValueError(f"临时文件 {self.filename} 创建失败")
            else:
                print(f"✅ 临时文件 {self.filename} - {self.temp_file_path} 创建成功")

        # 3.  返回对应 Loader

        if self.document_type == "txt":
            encoding = self.kwargs.get("encoding", "utf-8")
            # 文本可直接从内存读，这里仍用临时文件示例
            return TextLoader(self.temp_file_path, encoding=encoding)

        elif self.document_type == "csv":
            return CSVLoader(
                self.temp_file_path,
                csv_args=self.kwargs.get("csv_args", {}),
                encoding=self.kwargs.get("encoding", "utf-8")
            )

        elif self.document_type == "pdf":
            pdf_loader_type = self.kwargs.get("pdf_loader_type", "pypdf")
            if pdf_loader_type == "unstructured":
                return UnstructuredPDFLoader(self.temp_file_path)
            return PyPDFLoader(self.temp_file_path)

        elif self.document_type == "image":
            return UnstructuredImageLoader(self.temp_file_path)

        raise ValueError(f"unsupported document type: {self.document_type}")


    async def load(self) -> List[Document]:
        loader = await self._get_loader_by_type()
        if self.document_type == "web":
            return await loader.aload()
        return await run_in_threadpool(loader.load)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)

# if __name__ == "__main__":
#     async def main():
#         loader = DocumentLoader(
#             upload_file=None,
#             document_type="web",
#             urls=["https://tailwindcss.com/docs/installation/using-vite"]
#             # urls=["https://fastapi.tiangolo.com", "https://github.com"]
#         )
#         # docs = await loader.load()
#         # html_content = "\n".join(doc.page_content for doc in docs)
#         # pathlib.Path("result.html").write_text(html_content, encoding="utf-8")
#         # print(f"已生成 {pathlib.Path.cwd()}/result.html")
#
#     asyncio.run(main())