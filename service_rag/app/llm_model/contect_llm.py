import requests, json, base64
from io import BytesIO
from PIL import Image
import time
import asyncio
from service_rag.app.config.config import setting


async def stream_llm_response(prompt: str):
    """流式调用LLM - 直接转发SSE响应"""
    url = setting.CHAT_URL_TEMPLATE
    payload = {
        "model": "@cf/meta/llama-4-scout-17b-16e-instruct",
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_tokens": 4000,
        "temperature": 0.7,
        "stream": True
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {setting.TOKEN_URL}"
    }

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=60) as response:

                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ LLM API返回错误: {error_text[:200]}")
                    error_json = json.dumps({"error": f"LLM API错误: {response.status}"})
                    yield f"data: {error_json}\n\n"
                    return

                print(f"✅ LLM API连接成功，开始接收流式数据")

                # 重要：直接读取并转发原始SSE数据
                async for data in response.content.iter_any():
                    if data:
                        chunk = data.decode('utf-8')
                        yield chunk
                print(f"✅ LLM流式数据接收完成")

    except asyncio.TimeoutError:
        print("❌ LLM请求超时")
        error_json = json.dumps({"error": "请求超时，请稍后重试"})
        yield f"data: {error_json}\n\n"
    except aiohttp.ClientError as e:
        print(f"❌ 网络请求失败: {e}")
        error_json = json.dumps({"error": f"网络请求失败: {str(e)}"})
        yield f"data: {error_json}\n\n"
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        error_json = json.dumps({"error": f"处理失败: {str(e)}"})
        yield f"data: {error_json}\n\n"


# only use for the intent model
def connect_text_llm(question:str, prompt:str=""):
    print(f"🎯传过来的问题是: {question} ")
    url = setting.CHAT_URL_TEMPLATE
    payload = {
        "model": "@cf/meta/llama-4-scout-17b-16e-instruct",
        "messages": [{
        "role": "user",
        "content": question +" "+prompt
        }],
        "max_tokens": 2000,
        "temperature": 0.7,
    }

    r = requests.post(url, json=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {setting.TOKEN_URL}"})
    body = r.json()

    if 'error_code' in body:
        print("[ERNIE ERROR]", body)
        raise RuntimeError(f"ERNIE API:{body['error_code']} {body.get('error_msg', '')}")
    #
    # 安全地提取内容
    choices = body.get('choices', [])
    if choices and len(choices) > 0:
        message = choices[0].get('message', {})

        # 重要：直接返回content字段，无论它是字符串还是字典
        content = message.get('content', '')
        return {
            "role": message.get('role', ''),
            "content": content  # 保持原始格式
        }
    else:
        return {
            "role": "assistant",
            "content": "{}"  # 返回空的JSON字符串
        }


async def analyze_with_image(image_bytes: bytes, question: str):

    try:
        original_size = len(image_bytes)
        print(f"🖼️ [图片模型] 接收到图片大小: {original_size / 1024:.1f}KB ({original_size}字节)")
        image_array = list(image_bytes)
        if len(image_array) == 0:
            return {
                "role": "assistant",
                "content": "图片处理失败：转换后的数据为空。"
            }

    except Exception as e:
        print(f"图片数据处理失败: {str(e)}")
        return {
            "role": "assistant",
            "content": f"图片处理失败: {str(e)}"
        }

    # 2. 发送请求到API
    url = setting.CHAT_URL_IMAGE_TEMPLATE

    input_payload = {
        "image": image_array,
        "prompt": question,
        "max_tokens": 512
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {setting.TOKEN_URL}"
    }
    start_time = time.time()
    try:
        response = requests.post(url, json=input_payload, headers=headers, timeout=60)
        request_time = time.time() - start_time
        print(f"🖼️ [图片模型] API请求耗时: {request_time:.2f}秒")

        if response.status_code != 200:
            print(f"❌ [图片模型] API请求失败，状态码: {response.status_code}")
            return {
                "role": "assistant",
                "content": f"图片分析请求失败，错误码: {response.status_code}"
            }
        # 解析成功响应
        body = response.json()
        final_content = ""

        # 提取响应内容
        if isinstance(body, str):
            final_content = body
        elif isinstance(body, dict):
            # 尝试从不同字段提取响应
            if 'result' in body and body['result']:
                result_data = body['result']
                if isinstance(result_data, dict) and 'description' in result_data:
                    final_content = result_data['description']
                else:
                    final_content = result_data
            elif 'response' in body and body['response']:
                final_content = body['response']
            elif body.get('success') is True and 'result' in body:
                final_content = body['result']
            else:
                # 尝试查找有意义的字符串字段
                for key, value in body.items():
                    if isinstance(value, str) and value.strip() and len(value) > 10:
                        final_content = value
                        break
                if not final_content:
                    final_content = json.dumps(body, ensure_ascii=False)
        elif isinstance(body, list) and len(body) > 0:
            final_content = str(body[0])
        else:
            final_content = str(body)

        return {
            "role": "assistant",
            "content": final_content
        }

    except requests.exceptions.Timeout:
        return {
            "role": "assistant",
            "content": "请求超时（60秒），图片数据可能仍然过大或网络延迟，请尝试上传更小的图片。"
        }
    except requests.exceptions.ConnectionError:
        return {
            "role": "assistant",
            "content": "网络连接错误，请检查网络连接或API端点地址"
        }
    except requests.exceptions.RequestException as e:
        return {
            "role": "assistant",
            "content": f"网络请求异常: {str(e)}"
        }
    except Exception as e:
        print(f"处理过程中发生未预期的错误: {str(e)}")
        return {
            "role": "assistant",
            "content": f"图片分析处理过程中出错: {str(e)}"
        }