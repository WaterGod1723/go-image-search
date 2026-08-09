const { performance } = require('perf_hooks');

// 1. 配置区域
const API_URL = 'https://opencode.ai/zen/go/v1/chat/completions'; // 注意：DeepSeek 需使用 /v1/messages 端点
const API_KEY = 'sk-pip1lCryh9lbvBY646Xss0ndz96lgMwKR5Sl3iG8bORDpKJZCmUYLK4jGUPYwqLH'; // 请替换为你的真实 API Key
const MODEL_NAME = 'deepseek-v4-flash'; // DeepSeek V4 Flash 模型 ID

const payload = {
  model: MODEL_NAME,
  max_tokens: 1024,
  messages: [
    { role: 'user', content: '请用一句话介绍你自己。' }
  ]
};

async function testDeepSeekFlash() {
  console.log(`🚀 正在测试模型: ${MODEL_NAME}`);
  const start = performance.now();
  
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify(payload)
    });

    const end = performance.now();
    const duration = (end - start).toFixed(2);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ 请求失败: 状态码 ${response.status}`);
      console.error(`错误详情: ${errorText}`);
      return;
    }

    const data = await response.json();
    console.log(`✅ 请求成功! 耗时: ${duration} ms`);
    console.log(`🤖 模型回复: ${data.content[0].text}`); // Anthropic 格式响应解析
  } catch (error) {
    console.error('❌ 发生网络或解析异常:', error.message);
  }
}

testDeepSeekFlash();