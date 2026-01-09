import requests, json
import time
import asyncio
from typing import List, Dict
from service_rag.app.config.config import setting

async def stream_llm_response(messages: List[Dict[str, str]]):
    """流式调用LLM - 直接转发SSE响应"""
    url = setting.CHAT_URL_TEMPLATE
    payload = {
        "model": "@cf/meta/llama-3.1-8b-instruct-fast",
        "messages": messages,
        "max_tokens": 2000,
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
def connect_text_llm(question:str):
    url = setting.CHAT_URL_TEMPLATE
    payload = {
        "model": "@cf/meta/llama-4-scout-17b-16e-instruct",
        "messages": [{
        "role": "user",
        "content": question
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


async def analyze_with_image(image_bytes: bytes, question: str, messages: List[Dict[str, str]] = None):

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

    final_prompt = question
    if not final_prompt and messages:
        # 从messages中提取最后一条用户消息
        for msg in reversed(messages):
            if msg.get("role") == "user":
                final_prompt = msg.get("content", "").strip()
                break

    if not final_prompt:
        final_prompt = "请分析这张图片"

    # 如果有历史消息，添加到提示词中
    if messages and len(messages) > 1:
        # 构建历史消息文本
        history_text = "【对话历史】\n"
        for msg in messages[:-1]:  # 不包含最后一条消息
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        final_prompt = f"{history_text}\n【当前任务】\n{final_prompt}"

    # 2. 发送请求到API
    url = setting.CHAT_URL_IMAGE_TEMPLATE

    input_payload = {
        "image": image_array,
        "prompt": final_prompt,
        "max_tokens": 512
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {setting.TOKEN_URL}"
    }
    start_time = time.time()
    try:
        print(f"🖼️ 开始发送请求到 Cloudflare Workers......")
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
        if isinstance(body, dict) and 'result' in body:
            result = body['result']
            if isinstance(result, dict) and 'description' in result:
                final_content = result['description'].strip()
            else:
                final_content = str(result).strip()
        else:
            # 如果格式不符合预期，记录日志并返回错误
            print(f"⚠️  [图片模型] 意外的响应格式: {body}")
            final_content = "图片分析失败：API返回了意外的格式"

        print(f"🖼️ [图片模型] 提取的内容长度: {len(final_content)}")
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