// Render an animated SVG (SMIL) to PNG frames: node render_svg.js anim.svg outdir w h nFrames durationSeconds
const fs = require('fs'); const path = require('path'); const { chromium } = require('playwright');
(async () => {
  const [svgPath, outDir, w, h, n, dur] = process.argv.slice(2);
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage({ viewport: { width: +w, height: +h } });
  await page.setContent(`<html><body style="margin:0">${fs.readFileSync(svgPath, 'utf8')}</body></html>`);
  for (let i = 0; i < +n; i++) {
    await page.evaluate(t => { const s = document.querySelector('svg'); s.pauseAnimations(); s.setCurrentTime(t); }, (i / +n) * +dur);
    await page.screenshot({ path: path.join(outDir, `${String(i).padStart(4, '0')}.png`) });
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
