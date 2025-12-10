
class Prompt:

    rag_template="""
        你是一个基于提供知识库的回答问题的助手，
        
        知识库的信息:
        {context}
        
        问题:{question}
        
        仅基于提供的上下文知识库回答问题"
        
        每次回答问题前加上前缀: Evan 让您久等了.
    """

prompt_setting = Prompt()
