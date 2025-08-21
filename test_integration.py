#!/usr/bin/env python3
"""
测试脚本：验证Hunyuan3D-2.1与blender-mcp的集成功能

此脚本测试以下功能：
1. Hunyuan3D API服务器的健康状态
2. Stable Diffusion图像生成（如果可用）
3. 文本到3D场景的完整工作流程
"""

import requests
import json
import time
import base64
import os
from typing import Optional

# 配置
HUNYUAN3D_API_URL = "http://localhost:8081"
STABLE_DIFFUSION_API_URL = "http://localhost:7860"  # AUTOMATIC1111 WebUI默认端口

def test_hunyuan3d_health():
    """测试Hunyuan3D API服务器健康状态"""
    try:
        response = requests.get(f"{HUNYUAN3D_API_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Hunyuan3D API服务器运行正常 - Worker ID: {data.get('worker_id')}")
            return True
        else:
            print(f"❌ Hunyuan3D API服务器响应异常 - 状态码: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到Hunyuan3D API服务器: {e}")
        return False

def test_stable_diffusion_health():
    """测试Stable Diffusion API服务器健康状态"""
    try:
        response = requests.get(f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/options", timeout=10)
        if response.status_code == 200:
            print("✅ Stable Diffusion API服务器运行正常")
            return True
        else:
            print(f"❌ Stable Diffusion API服务器响应异常 - 状态码: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"⚠️ 无法连接到Stable Diffusion API服务器: {e}")
        print("   这是可选功能，可以继续测试其他功能")
        return False

def generate_test_image_with_sd(prompt: str) -> Optional[str]:
    """使用Stable Diffusion生成测试图像"""
    try:
        payload = {
            "prompt": prompt,
            "steps": 20,
            "width": 512,
            "height": 512,
            "cfg_scale": 7,
            "sampler_index": "Euler a"
        }
        
        response = requests.post(f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img", 
                               json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('images'):
                print(f"✅ 成功生成图像: {prompt}")
                return data['images'][0]  # 返回base64编码的图像
            else:
                print("❌ 图像生成失败：响应中没有图像数据")
                return None
        else:
            print(f"❌ 图像生成失败 - 状态码: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 图像生成请求失败: {e}")
        return None

def test_hunyuan3d_generation(image_base64: str) -> Optional[str]:
    """测试Hunyuan3D 3D模型生成"""
    try:
        payload = {
            "image": image_base64,
            "remove_background": True,
            "texture": True,
            "seed": 42,
            "num_chunks": 4,
            "face_count": 10000
        }
        
        # 使用异步端点
        response = requests.post(f"{HUNYUAN3D_API_URL}/send", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            uid = data.get('uid')
            if uid:
                print(f"✅ 3D生成任务已提交 - UID: {uid}")
                return uid
            else:
                print("❌ 3D生成任务提交失败：响应中没有UID")
                return None
        else:
            print(f"❌ 3D生成任务提交失败 - 状态码: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 3D生成请求失败: {e}")
        return None

def check_generation_status(uid: str, max_wait_time: int = 300) -> bool:
    """检查3D生成任务状态"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{HUNYUAN3D_API_URL}/status/{uid}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                
                if status == 'completed':
                    print("✅ 3D模型生成完成")
                    return True
                elif status == 'error':
                    message = data.get('message', '未知错误')
                    print(f"❌ 3D模型生成失败: {message}")
                    return False
                elif status == 'processing':
                    print("⏳ 正在处理中...")
                elif status == 'texturing':
                    print("🎨 正在生成纹理...")
                else:
                    print(f"📊 当前状态: {status}")
            else:
                print(f"❌ 状态查询失败 - 状态码: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 状态查询请求失败: {e}")
            return False
        
        time.sleep(10)  # 等待10秒后再次检查
    
    print(f"⏰ 等待超时（{max_wait_time}秒）")
    return False

def main():
    """主测试函数"""
    print("🚀 开始测试Hunyuan3D-2.1与blender-mcp集成系统")
    print("=" * 60)
    
    # 1. 测试Hunyuan3D API健康状态
    print("\n1. 测试Hunyuan3D API服务器...")
    if not test_hunyuan3d_health():
        print("❌ Hunyuan3D API服务器不可用，请先启动服务器")
        print("   启动命令: python api_server.py --host 0.0.0.0 --port 8081")
        return
    
    # 2. 测试Stable Diffusion API健康状态
    print("\n2. 测试Stable Diffusion API服务器...")
    sd_available = test_stable_diffusion_health()
    
    # 3. 生成测试图像
    print("\n3. 生成测试图像...")
    test_prompt = "a modern wooden chair, product photography, white background, high quality"
    
    if sd_available:
        image_base64 = generate_test_image_with_sd(test_prompt)
        if not image_base64:
            print("⚠️ 使用Stable Diffusion生成图像失败，将跳过3D生成测试")
            return
    else:
        print("⚠️ Stable Diffusion不可用，将跳过图像生成和3D生成测试")
        print("   如需完整测试，请启动AUTOMATIC1111 WebUI")
        return
    
    # 4. 测试3D模型生成
    print("\n4. 测试3D模型生成...")
    uid = test_hunyuan3d_generation(image_base64)
    if not uid:
        print("❌ 3D生成任务提交失败")
        return
    
    # 5. 监控生成状态
    print("\n5. 监控生成进度...")
    success = check_generation_status(uid, max_wait_time=300)
    
    if success:
        print("\n🎉 集成测试完成！所有功能正常工作")
        print("\n📋 测试总结:")
        print("   ✅ Hunyuan3D API服务器正常")
        print("   ✅ Stable Diffusion图像生成正常")
        print("   ✅ 3D模型生成正常")
        print("\n💡 现在可以在blender-mcp中使用这些功能了！")
    else:
        print("\n❌ 集成测试失败，请检查服务器日志")

if __name__ == "__main__":
    main()