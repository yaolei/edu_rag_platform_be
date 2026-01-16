from service_rag.app.prompt.prompt import prompt_setting
import re
import json
import asyncio
def switch_correct_prompt(question, doc_type, image_description, relevant_docs, ocr_text):
    if doc_type == "resume":
        final_prompt_for_text_model = prompt_setting.rag_image_resume_template.format(
            image_description=image_description,
            knowledge_context=relevant_docs,
            ocr_text=ocr_text,
            question=question,
        )
    elif doc_type == "code":
        final_prompt_for_text_model = prompt_setting.code_rag_qa_template.format(
            image_description=image_description,
            knowledge_context=relevant_docs,
            ocr_text=ocr_text,
            question=question,
        )
    else:
        # 文档类型
        final_prompt_for_text_model = prompt_setting.general_doc_rag_qa_template.format(
            image_description=image_description,
            knowledge_context=relevant_docs,
            ocr_text=ocr_text,
            question=question,
        )

    return final_prompt_for_text_model


def build_simple_context(documents):
        context_parts = []
        for i, doc in enumerate(documents):  # 最多5个
            content = ""

            if isinstance(doc, dict):
                content = doc.get('text', '')
                if not content:
                    content = doc.get('page_content', '')
                    if not content and hasattr(doc, 'get'):
                        # 尝试获取第一个字符串值
                        for key, value in doc.items():
                            if isinstance(value, str) and len(value.strip()) > 0:
                                content = value
                                break
            elif hasattr(doc, 'page_content'):
                # Document对象
                content = doc.page_content

            if content:
                content = content.strip()
                import re
                content = re.sub(r'\s+', ' ', content)

                # 只添加非空内容
                if content:
                    context_parts.append(content)

        if not context_parts:
            return ""

        return "\n\n---\n\n".join(context_parts)

def prue_image_chunks(result_content):
    sentences = re.split(r'([。！？；\.!?;])', result_content)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]

        # 如果当前chunk为空或句子很短，直接添加
        if not current_chunk or len(sentence.strip()) < 10:
            current_chunk += sentence
        else:
            # 如果句子包含换行符，说明是段落分隔
            if '\n' in sentence:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            # 如果句子较长，单独作为一个chunk
            elif len(sentence.strip()) > 30:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                chunks.append(sentence.strip())
                current_chunk = ""
            # 否则合并到当前chunk
            else:
                current_chunk += sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


