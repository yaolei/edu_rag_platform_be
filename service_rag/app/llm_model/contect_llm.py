import requests, json, base64
from io import BytesIO
from PIL import Image
from service_rag.app.config.config import setting

def connect_text_llm(question:str, prompt:str=""):
    print(f"🎯传过来的问题是: {question} ")
    url = setting.CHAT_URL_TEMPLATE
    payload = {
        "model": "@cf/meta/llama-4-scout-17b-16e-instruct",
        "messages": [{
        "role": "user",
        "content": question +" "+prompt
        }],
        "max_tokens": 4000,
        "temperature": 0.7,
    }

    r = requests.post(url, json=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {setting.TOKEN_URL}"})
    body = r.json()
    # 打印响应的部分信息用于调试
    print(f"🔍 API响应状态码: {r.status_code}")
    print(f"🔍 API响应内容类型: {type(body)}")
    print(f"🔍 API响应体部分: {str(body)[:500]}...")

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
        print(f"⚠️ API响应中没有choices字段: {body}")
        return {
            "role": "assistant",
            "content": "{}"  # 返回空的JSON字符串
        }

async def analyze_with_image(image_base64_data_url: str, question: str):
    """
    使用 LLaVA 模型进行图片分析
    参数 image_base64_data_url: 格式为 "data:image/jpeg;base64,xxxx..." 的完整字符串
    参数 question: 针对图片的问题
    """
    print(f"🖼️ [图片模型] 开始分析图片，问题: {question}")
    print(f"🖼️ [图片模型] 接收到的Data URL长度: {len(image_base64_data_url)}")

    # 1. 从 Data URL 中提取并解码 Base64 图片数据
    try:
        # 分割出 base64 部分
        header, base64_str = image_base64_data_url.split(';base64,')
        # 解码为二进制数据
        image_bytes = base64.b64decode(base64_str)
        print(f"🖼️ [图片模型] 解码后的原始图片大小: {len(image_bytes)} 字节")

        # 打开图片并进行必要的处理
        img = Image.open(BytesIO(image_bytes))
        original_format = img.format
        print(f"🖼️ [图片模型] 原始图片尺寸: {img.size}, 格式: {img.mode}/{original_format}")

        # 将最大边长限制到512像素，以显著减少数据量
        max_size = 512
        if max(img.size) > max_size:
            print(f"🖼️ [图片模型] 图片尺寸较大，缩放至最长边为{max_size}像素...")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"🖼️ [图片模型] 缩放后尺寸: {img.size}")

        # 确保为RGB格式（兼容性最佳）
        if img.mode != 'RGB':
            print(f"🖼️ [图片模型] 转换图片模式从 {img.mode} 到 RGB")
            img = img.convert('RGB')

        buffer = BytesIO()
        # 统一保存为JPEG格式以获得更高的压缩率
        quality = 40  # 将质量设为40，在可接受范围内尽量减小文件
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        image_bytes = buffer.getvalue()
        print(f"🖼️ [图片模型] 激进的JPEG压缩后图片大小: {len(image_bytes)} 字节 (质量: {quality})")

        # 将二进制数据转换为 0-255 的整数列表，这是 API 要求的格式
        image_array = list(image_bytes)
        print(f"🖼️ [图片模型] 转换后的像素数组长度: {len(image_array)} (前5个值示例: {image_array[:5]})")

        # 检查数组内容是否有效
        if len(image_array) == 0:
            print("❌ [图片模型] 错误：转换后的图片数组为空！")
            return {
                "role": "assistant",
                "content": "图片处理失败：转换后的数据为空。"
            }

        if not all(isinstance(x, int) and 0 <= x <= 255 for x in image_array[:100]):
            print("⚠️  [图片模型] 警告：数组部分值超出0-255范围，正在自动修正...")
            image_array = [min(max(int(x), 0), 255) for x in image_array]

    except Exception as e:
        print(f"❌ [图片模型] 图片数据处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "role": "assistant",
            "content": f"图片处理失败: {str(e)}"
        }

    url = setting.CHAT_URL_IMAGE_TEMPLATE
    print(f"🖼️ [图片模型] 使用专用图片API端点: {url}")

    # 重要：Cloudflare Workers AI的 /run 端点，请求体中不应包含 "model" 字段
    input_payload = {
        "image": image_array,  # 必需：图片的像素值数组
        "prompt": question,  # 必需：问题文本
        "max_tokens": 512  # 可选：最大生成长度
        # 注意：可以按需添加 temperature, top_p 等参数，但当前以最简形式测试
    }

    print(f"🖼️ [图片模型] 请求体构建完成，图片数组大小: {len(image_array)}")

    # 3. 发送请求并处理响应
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {setting.TOKEN_URL}"
    }

    try:
        print(f"🖼️ [图片模型] 发送请求到Cloudflare AI...")
        r = requests.post(url, json=input_payload, headers=headers, timeout=60)

        print(f"🖼️ [图片模型] 响应状态码: {r.status_code}")

        # 尝试解析响应
        try:
            body = r.json()
        except json.JSONDecodeError:
            print(f"❌ [图片模型] 响应不是有效的JSON")
            print(f"🖼️ [图片模型] 原始响应文本: {r.text[:500]}...")
            return {
                "role": "assistant",
                "content": f"API返回了无效的响应格式: {r.text[:200]}"
            }

        print(f"🖼️ [图片模型] 响应体内容: {json.dumps(body, ensure_ascii=False)[:500]}...")

        final_content = ""

        if r.status_code == 200:
            # 请求成功，尝试提取响应文本
            if isinstance(body, str):
                final_content = body
            elif isinstance(body, dict):
                # Cloudflare AI /run 接口常见的响应格式
                if 'result' in body and body['result']:
                    # 情况1: 直接包含'result'字段
                    result_data = body['result']
                    if isinstance(result_data, dict) and 'description' in result_data:
                        final_content = result_data['description']
                    else:
                        final_content = result_data
                elif 'response' in body and body['response']:
                    # 情况2: 包含'response'字段
                    final_content = body['response']
                elif body.get('success') is True and 'result' in body:
                    # 情况3: 结构为 {"success": true, "result": "..."}
                    final_content = body['result']
                else:
                    # 情况4: 其他格式，尝试查找第一个有意义的字符串字段
                    for key, value in body.items():
                        if isinstance(value, str) and value.strip() and len(value) > 10:
                            final_content = value
                            break

                    if not final_content:
                        # 如果没找到，将整个响应转为字符串
                        final_content = json.dumps(body, ensure_ascii=False)

            elif isinstance(body, list) and len(body) > 0:
                # 情况5: 响应是数组，取第一个元素
                final_content = str(body[0])
            else:
                final_content = str(body)

            print(f"✅ [图片模型] 请求成功！解析内容长度: {len(final_content)}")

        else:
            # 请求失败，构建错误信息
            error_msg = f"API请求失败 (状态码: {r.status_code})"

            if isinstance(body, dict):
                if 'errors' in body:
                    error_msg += f"。错误详情: {body['errors']}"
                elif 'message' in body:
                    error_msg += f"。消息: {body['message']}"
                elif 'error' in body:
                    error_msg += f"。错误: {body['error']}"

            final_content = error_msg
            print(f"❌ [图片模型] {error_msg}")

        return {
            "role": "assistant",
            "content": final_content
        }

    except requests.exceptions.Timeout:
        error_msg = "请求超时（60秒），图片数据可能仍然过大或网络延迟"
        print(f"❌ [图片模型] {error_msg}")
        return {
            "role": "assistant",
            "content": error_msg + "，请尝试上传更小的图片。"
        }
    except requests.exceptions.ConnectionError:
        error_msg = "网络连接错误，请检查网络连接或API端点地址"
        print(f"❌ [图片模型] {error_msg}")
        return {
            "role": "assistant",
            "content": error_msg
        }
    except requests.exceptions.RequestException as e:
        error_msg = f"网络请求异常: {str(e)}"
        print(f"❌ [图片模型] {error_msg}")
        return {
            "role": "assistant",
            "content": error_msg
        }
    except Exception as e:
        print(f"❌ [图片模型] 处理过程中发生未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "role": "assistant",
            "content": f"图片分析处理过程中出错: {str(e)}"
        }