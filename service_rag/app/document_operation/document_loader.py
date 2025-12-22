# -*- coding: utf-8 -*-
import os, tempfile, pathlib
from typing import Optional, List, Dict
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool
from langchain_community.document_loaders import AsyncHtmlLoader, TextLoader, CSVLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from PIL import Image
import pytesseract, cv2, numpy as np, re, fitz


# ---------- ImageContentExtractor ----------
class ImageContentExtractor:
    def __init__(self):
        self.ocr_config = r'--psm 3 --oem 3'
        self.THUMB_SIZE = (300, 300)

    def probably_has_text(self, pil_img: Image.Image) -> bool:
        """轻量规则：连通域数量判断"""
        gray = np.array(pil_img.convert('L'))
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, _ = cv2.connectedComponents(bw)
        return 50 <= num_labels <= 2000

    def extract_text_from_image(self, image_path: str):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng', config=self.ocr_config)
            text = re.sub(r'(?<=\S) (?=\S)', '', text)
            return text.strip()
        except Exception as e:
            print(f" ❌Error {str(e)}")
            return ""

    def extract_image_features(self, image_path: str):
        image = Image.open(image_path)
        return {
            'size': image.size, 'height': image.height, 'width': image.width,
            'mode': image.mode, 'format': image.format,
            'text_content': self.extract_text_from_image(image_path),
            'file_size': os.path.getsize(image_path)
        }


# ---------- PDFMultimodalExtractor----------
class PDFMultimodalExtractor:
    def __init__(self):
        self.image_extractor = ImageContentExtractor()

    def extract_images_from_pdf(self, pdf_path: str, output_dir: str = None):
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='pdf_images_')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            doc = fitz.open(pdf_path)
            images_info = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                for img_index, img in enumerate(page.get_images()):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha < 4:
                            img_filename = f"page_{page_num}_img_{img_index}.png"
                            img_path = os.path.join(output_dir, img_filename)
                            pix.save(img_path)
                            feat = self.image_extractor.extract_image_features(img_path)
                            images_info.append({
                                'page': page_num, 'image_index': img_index,
                                'file_path': img_path, 'feature': feat,
                                'bbox': img[1:5] if len(img) > 4 else None
                            })
                        pix = None
                    except Exception as e:
                        print(f"❌提取图像失败 (页面 {page_num}, 图像 {img_index}) : {str(e)}")
                        continue
            doc.close()
            return images_info
        except Exception as e:
            print(f"❌❌提取图像失败 {str(e)}")
            return []

    def extract_tables_from_pdf(self, pdf_path:str) -> List[Dict]:
        try:
            import tabula
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            tables_info = []
            for i, table in enumerate(tables):
                if not table.empty:
                    continue
                table = table.fillna('')
                table = table.applymap(lambda x: " ".join(str(x).split()) if x else '')

                text_lines = [" ".join(row) for row in table.values if any(row)]
                table_text = "\n".join(text_lines).strip()
                csv_lines = [",".join(str(cell).strip() for cell in row) for row in table.values]
                table_csv = "\n".join(csv_lines).strip()

                tables_info.append({
                    'table_index': i,
                    'dataframe': table,
                    'text_representation': table_text,
                    'csv_representation':table_csv,
                    'shape': table.shape
                })
            return tables_info
        except ImportError:
            print(f"表格提取")
            return []
        except Exception as e:
            print(f"表格提取失败 {str(e)}")
            return []


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
                                  'plain_text': "\n\n".join(p.page_content for p in pdf_docs) }
            print(f" 🔥🚀🔥🚀🔥🚀🔥{multimodal_content}🔥🚀🔥🚀🔥🚀🔥")
            return multimodal_content

        if self.document_type == "image":
            image_extractor = ImageContentExtractor()

            pil_img = Image.open(self.temp_file_path)
            images_info = image_extractor.probably_has_text(pil_img)
            try:
                if images_info:
                    print(f"✅ 图片有可提取的文本,需要进行OCR提取 ")
                    image_feature = image_extractor.extract_image_features(image_path=self.temp_file_path)

                    image_meta = {
                        k: image_feature[k]
                        for k in ('size', 'height', 'width', 'mode', 'format', 'file_size')
                        if k in image_feature
                    }
                    multimodal_content = {'images': [image_meta],  'image_texts': [image_feature['text_content']]}

                    print(f"🐯✅ 提取后的内容 {multimodal_content} 🦊🦊🦊🦊🦊🦊")
                else:
                    print(f"✅ 图片没有可提取的文本,是一个纯图片不需要进行OCR ")
                    return []

            except Exception as e:
                print(f"❌ 判断是否执行ocr的逻辑报错 {str(e)} ")

            return []

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

        loader = await self._get_loader_by_type()
        if loader == []:
            return [none_store_struck]
        else :
            # Web 分支异步加载，其余同步
            if self.document_type == "web":
                return await loader.aload()

            final_result = Document(
                page_content=loader['plain_text'],
                metadata={
                    'images': loader['images'],
                    'tables': loader['tables'],
                    'image_texts': loader['image_texts'],
                },
            )

            return [final_result]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)