import os,fitz, tempfile
import pytesseract, re
from PIL import Image
from typing import  List, Dict
# ---------- ImageContentExtractor ----------
class ImageContentExtractor:

    def __init__(self):
        self.ocr_config = r'--psm 3 --oem 3'
        self.THUMB_SIZE = (300, 300)

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
        text_content = self.extract_text_from_image(image_path)
        return {
            'size': image.size, 'height': image.height, 'width': image.width,
            'mode': image.mode, 'format': image.format,
            'text_content': text_content,
            'file_size': os.path.getsize(image_path),
            'has_text': len(text_content.strip()) > 20
        }


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

        def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
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
                        'csv_representation': table_csv,
                        'shape': table.shape
                    })
                return tables_info
            except ImportError:
                print(f"表格提取")
                return []
            except Exception as e:
                print(f"表格提取失败 {str(e)}")
                return []
