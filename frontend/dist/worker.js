// 图片格式转换 Web Worker：利用浏览器内置解码能力把 webp/bmp/tiff/avif 等
// 栅格格式解码并重编码为 PNG 字节，回传给主线程。SVG 走主线程 Image 兜底。
'use strict';

self.onmessage = async (e) => {
  const msg = e.data || {};
  const { id, format, data } = msg;
  try {
    const png = await convertToPng(data, format);
    self.postMessage({ id, png }, [png]);
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};

async function convertToPng(data, format) {
  const blob = new Blob([data], { type: mimeFor(format) });
  let bmp = null;
  try {
    bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  } catch (err) {
    // 部分 webview 对某些格式不支持 createImageBitmap，交给主线程兜底。
    throw new Error('createImageBitmap unsupported: ' + (err && err.message));
  }
  try {
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('no 2d context');
    ctx.drawImage(bmp, 0, 0);
    const out = await canvas.convertToBlob({ type: 'image/png' });
    return await out.arrayBuffer();
  } finally {
    bmp.close();
  }
}

function mimeFor(format) {
  switch ((format || '').replace(/^\./, '').toLowerCase()) {
    case 'webp': return 'image/webp';
    case 'bmp': return 'image/bmp';
    case 'tiff': case 'tif': return 'image/tiff';
    case 'avif': return 'image/avif';
    default: return '';
  }
}