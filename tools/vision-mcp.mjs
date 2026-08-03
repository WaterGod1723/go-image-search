#!/usr/bin/env node
// 图片理解 MCP: 调用智谱 BigModel 视觉模型, 为不支持多模态的 LLM 提供图片理解能力。
// 输入图片本地路径或网络地址 + 提示词, 输出文本描述。
// stdout 只承载 JSON-RPC, 状态信息走 stderr。
//
// {
//   "mcpServers": {
//     "vision-understand": {
//       "type": "stdio",
//       "command": [
//         "node",
//         "/Users/admin/code/js/h5-pc-guide/tools/vision-mcp.mjs" // 真实路径
//       ],
//       "environment": {
//         "BIGMODEL_API_KEY": "",
//         "BIGMODEL_ENDPOINT": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
//         "BIGMODEL_VISION_MODEL": "glm-4v-flash"
//       }
//     }
//   }
// }
console.log = (...a) => console.error('[vision]', ...a);

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const API_ENDPOINT = process.env.BIGMODEL_ENDPOINT || 'https://open.bigmodel.cn/api/paas/v4/chat/completions';
const API_KEY = process.env.BIGMODEL_API_KEY;
const DEFAULT_MODEL = process.env.BIGMODEL_VISION_MODEL || 'glm-4v-flash';

if (!API_KEY) {
  console.error('[vision] 警告: 未设置 BIGMODEL_API_KEY 环境变量, 调用将失败');
}

const EXT_MIME = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.svg': 'image/svg+xml',
};

function mimeFromPath(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return EXT_MIME[ext] || 'image/png';
}

function toDataUrl(filePath) {
  const buf = fs.readFileSync(filePath);
  const mime = mimeFromPath(filePath);
  return `data:${mime};base64,${buf.toString('base64')}`;
}

async function fetchImageAsDataUrl(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`下载图片失败: ${res.status} ${res.statusText}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const mime = (res.headers.get('content-type') || 'image/png').split(';')[0].trim();
  return `data:${mime};base64,${buf.toString('base64')}`;
}

async function resolveImageInput(image) {
  if (/^https?:\/\//i.test(image)) {
    const u = new URL(image);
    if (!u.hostname.endsWith('.amh-group.com') && u.hostname !== 'amh-group.com' && u.hostname !== 'open.bigmodel.cn') {
      const data = await fetchImageAsDataUrl(image).catch(() => null);
      if (data) return data;
    }
    return image;
  }
  if (image.startsWith('data:')) return image;
  if (!fs.existsSync(image)) throw new Error(`图片文件不存在: ${image}`);
  return toDataUrl(path.resolve(image));
}

async function callBigModel({ model, prompt, imageUrls, thinking, maxTokens, temperature }) {
  const body = {
    model,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          ...imageUrls.map(url => ({ type: 'image_url', image_url: { url } })),
        ],
      },
    ],
  };
  if (thinking) body.thinking = { type: 'enabled' };
  if (maxTokens != null) body.max_tokens = maxTokens;
  if (temperature != null) body.temperature = temperature;

  const res = await fetch(API_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`BigModel 请求失败 ${res.status}: ${errText.slice(0, 500)}`);
  }
  const json = await res.json();
  const content = json.choices?.[0]?.message?.content;
  if (!content) throw new Error(`BigModel 返回为空: ${JSON.stringify(json).slice(0, 500)}`);
  const text = Array.isArray(content) ? content.map(c => (typeof c === 'string' ? c : c.text || '')).join('') : String(content);
  return { text, usage: json.usage };
}

const server = new McpServer({
  name: 'vision-understand',
  version: '1.0.0',
});

server.registerTool(
  'understand_image',
  {
    title: '图片理解',
    description: '调用智谱 BigModel 视觉模型理解图片, 为不支持多模态的 LLM 提供图片理解能力。支持传入多张图片以提供跨图上下文, 输入图片本地路径或网络地址(http/https)及提示词, 返回文本描述。内网图片地址会自动下载转 base64 后投递。',
    inputSchema: {
      images: z.array(z.string()).min(1).describe('图片列表, 每个元素为本地绝对路径 / http(s) 网络地址 / data: URL。多张图片可提供跨图上下文'),
      prompt: z.string().default('请详细描述这张图片的内容').describe('对图片的提问/提示词, 决定返回什么样的文本描述'),
      model: z.string().optional().describe(`视觉模型名, 默认 ${DEFAULT_MODEL}`),
      thinking: z.boolean().default(false).describe('是否开启深度思考模式 (更慢但更细致)'),
      maxTokens: z.number().int().positive().optional().describe('最大输出 token 数; thinking 模式建议 65536'),
      temperature: z.number().min(0).max(2).optional().describe('采样温度 0-2, thinking 模式建议 1.0'),
    },
  },
  async ({ images, prompt, model, thinking, maxTokens, temperature }) => {
    if (!API_KEY) {
      return {
        content: [{ type: 'text', text: '未配置 BIGMODEL_API_KEY 环境变量, 无法调用视觉模型' }],
        isError: true,
      };
    }
    try {
      const imageUrls = await Promise.all(images.map(resolveImageInput));
      const { text, usage } = await callBigModel({
        model: model || DEFAULT_MODEL,
        prompt,
        imageUrls,
        thinking,
        maxTokens,
        temperature,
      });
      const meta = usage ? ` (tokens: ${usage.total_tokens ?? '?'})` : '';
      return { content: [{ type: 'text', text: text + meta }] };
    } catch (e) {
      return { content: [{ type: 'text', text: `图片理解失败: ${e.message}` }], isError: true };
    }
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
