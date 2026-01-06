from service_rag.app.prompt.prompt import prompt_setting

def switch_correct_prompt(question, doc_type, image_description, relevant_docs, ocr_text):
    if doc_type == "resume":
        final_prompt_for_text_model = prompt_setting.rag_image_qa_template.format(
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
        for i, doc in enumerate(documents[:5]):  # 最多5个
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