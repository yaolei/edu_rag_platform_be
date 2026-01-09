# -*- coding: utf-8 -*-
import os, tempfile, pathlib
from typing import Optional, List
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool
from langchain_community.document_loaders import AsyncHtmlLoader, TextLoader, CSVLoader, PyPDFLoader
from langchain_core.documents import Document
from service_rag.app.document_operation.ocr_analyse import ImageContentExtractor, PDFMultimodalExtractor

# ---------- DocumentLoader ----------
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
        self.temp_dir = None  # 初始 None，下面一次性赋值

        if document_type is None:
            document_type = self._detect_document_type()
        self.document_type = document_type

    # ---------- 修复：temp_dir 赋值 ----------
    async def _create_temp_file_if_needed(self) -> None:
        if self.temp_file_path:               # 已创建过就跳过
            return
        # 1️⃣ 先给 temp_dir 赋值（只在这里做一次）
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="doc_loader_")
        # 2️⃣ 再创建临时文件（放在该目录下）
        suffix = pathlib.Path(self.filename).suffix
        fd, self.temp_file_path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
        os.close(fd)
        content = await self._get_upload_file_content()
        with open(self.temp_file_path, "wb") as tmp:
            tmp.write(content)

    async def _get_upload_file_content(self) -> bytes:
        self.upload_file.file.seek(0)
        content = await run_in_threadpool(self.upload_file.file.read)
        self.upload_file.file.seek(0)
        return content

    def _detect_document_type(self) -> str:
        if self.urls:
            return "web"
        ext = pathlib.Path(self.filename).suffix.lower()
        if ext == ".pdf":
            return "pdf"
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
            return "image"
        return "txt"

    # ---------- 真正执行 OCR / 表格 / 图像 的逻辑 ----------
    async def _get_loader_by_type(self):
        if self.document_type == "web":
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"}
            return AsyncHtmlLoader(self.urls, header_template=headers)

        await self._create_temp_file_if_needed()
        if not self.temp_file_path or not os.path.exists(self.temp_file_path):
            raise ValueError(f"临时文件 {self.filename} 创建失败")

        if self.document_type == "pdf":
            pdf_extractor = PDFMultimodalExtractor()
            # 1. 尝试抽图、抽表
            images_info = pdf_extractor.extract_images_from_pdf(
                self.temp_file_path,
                output_dir=os.path.join(self.temp_dir, "extracted_images")
            )
            tables_info = pdf_extractor.extract_tables_from_pdf(self.temp_file_path) or []
            # 2. 基础文字层
            pdf_docs= PyPDFLoader(self.temp_file_path).load()

            image_txt = ""
            table_text=""
            for page_idx, doc in enumerate(pdf_docs):
                # --- OCR 文字（仅当本页有图才跑）---
                ocr_text = "\n".join(
                    img['feature']['text_content']
                    for img in images_info
                    if img['page'] == page_idx and img['feature']['text_content']
                )
                # --- 表格文字（仅当本页有表才跑）---
                table_text = "\n".join(
                    t['text_representation']
                    for t in tables_info
                    if t.get('page', 0) == page_idx and t.get('text_representation')
                )

                # 3. 追加到本页（非空才拼，避免多余换行）
                if ocr_text:
                    doc.page_content += f"\n{ocr_text}"
                    image_txt += f"\n{ocr_text}"
                if table_text:
                    table_text +=f"\n{table_text}"
                    doc.page_content += f"\n{table_text}"


            multimodal_content = {'images': [], # 以后可以存放image具体实例
                                  'tables': [table_text],
                                  'image_texts': [image_txt],
                                  'is_pre_image': False,
                                  'plain_text': "\n\n".join(p.page_content for p in pdf_docs) }
            return multimodal_content

        if self.document_type == "image":
            image_extractor = ImageContentExtractor()

            try:
                    image_feature = image_extractor.extract_image_features(image_path=self.temp_file_path)
                    has_text = image_feature.get('has_text', False)
                    text_content = image_feature.get('text_content', '')

                    if has_text and len(text_content.strip()) > 0:
                        print(f"✅ 图片包含可提取的文本，需要进行OCR处理")
                        print(f"   提取到的文本长度: {len(text_content)}")
                        print(f"   文本预览: {text_content[:100]}...")

                        multimodal_content = {'images': [self.temp_file_path], 'is_pre_image':False ,'image_texts': image_feature['text_content']}

                        return multimodal_content
                    else:
                        print(f"✅ 图片没有可提取的文本，是一个纯图片")
                        print(f"   OCR结果为空或过短: '{text_content}'")
                        multimodal_content = {'images': self.temp_file_path, 'is_pre_image':True, 'image_texts': "" }
                        return multimodal_content

            except Exception as e:
                print(f"❌ 判断是否执行ocr的逻辑报错 {str(e)} ")
                multimodal_content = {
                    'images': self.temp_file_path,
                    'is_pre_image': True,
                    'image_texts': ""
                }
                return multimodal_content

        if self.document_type == "txt":
            return TextLoader(self.temp_file_path, encoding=self.kwargs.get("encoding", "utf-8"))

        if self.document_type == "csv":
            return CSVLoader(self.temp_file_path,
                             csv_args=self.kwargs.get("csv_args", {}),
                             encoding=self.kwargs.get("encoding", "utf-8"))

        raise ValueError(f"unsupported document type: {self.document_type}")

    async def load(self) -> List[Document]:
        none_store_struck = Document(
            page_content='',
            metadata={},
        )
        try:
            await self._create_temp_file_if_needed()
            loader = await self._get_loader_by_type()

            print(f"🔍 DocumentLoader - 文档类型: {self.document_type}")
            print(f"🔍 DocumentLoader - 获取的 loader 类型: {type(loader)}")

            if loader == []:
                print("🔍 DocumentLoader - loader 为空列表，返回空文档")
                return [none_store_struck]
            else:
                final_result = Document(
                    page_content=loader['image_texts'] if self.document_type == "image" else loader['plain_text'],
                    metadata={
                        'images': loader['images'],
                        'is_pre_image': loader['is_pre_image'],
                        'image_texts': loader['image_texts'],
                    },
                )
                print(f"🔍 DocumentLoader - 最终返回的文档: page_content={final_result.page_content[:100]}...")
                return [final_result]
        except Exception as e:
            print(f"❌ DocumentLoader 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def cleanup_temp_resources(self):
        """
        显式清理所有临时资源（目录和文件）。
        在文档处理完成后，由调用方决定是否调用。
        """
        # 1. 删除主临时文件 (__aexit__中已做，这里确保一下)
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except:
                pass
        # 2. 删除整个临时目录（这是关键）
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print(f"🧹 已清理临时目录: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️  清理临时目录失败 {self.temp_dir}: {e}")
        self.temp_dir = None
        self.temp_file_path = None


    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)