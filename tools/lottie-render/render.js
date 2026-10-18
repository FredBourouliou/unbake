// Render a Lottie JSON to PNG frames with lottie-web (canvas renderer) in headless Chrome.
// Usage: node render.js anim.json outdir width height nFrames
// Frame i is sampled at normalised time i / (nFrames - 1) of the animation, so animations
// with a different frame rate or length can be compared frame by frame.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const [jsonPath, outDir, w, h, n] = process.argv.slice(2);
  const data = fs.readFileSync(jsonPath, 'utf8');
  const lib = fs.readFileSync(require.resolve('lottie-web/build/player/lottie.min.js'), 'utf8');
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage({ viewport: { width: +w, height: +h } });
  await page.setContent(`<html><body style="margin:0;background:#fff"><div id="c" style="width:${w}px;height:${h}px"></div></body></html>`);
  await page.addScriptTag({ content: lib });
  const frames = await page.evaluate(async ({ data, n }) => {
    const anim = lottie.loadAnimation({ container: document.getElementById('c'), renderer: 'canvas',
      loop: false, autoplay: false, animationData: JSON.parse(data),
      rendererSettings: { preserveAspectRatio: 'xMidYMid meet', clearCanvas: true } });
    // With inline animationData, DOMLoaded can fire before we listen: check isLoaded, and never wait forever.
    if (!anim.isLoaded) await Promise.race([new Promise(r => anim.addEventListener('DOMLoaded', r)), new Promise(r => setTimeout(r, 5000))]);
    const canvas = document.querySelector('canvas');
    const total = anim.totalFrames;
    const out = [];
    for (let i = 0; i < n; i++) {
      anim.goToAndStop(n > 1 ? i * (total - 1) / (n - 1) : 0, true);
      out.push(canvas.toDataURL('image/png'));
    }
    return out;
  }, { data, n: +n });
  frames.forEach((d, i) => fs.writeFileSync(path.join(outDir, `${String(i).padStart(4, '0')}.png`),
    Buffer.from(d.split(',')[1], 'base64')));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
