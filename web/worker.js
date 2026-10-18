// Runs the Python reconstruction in the browser with Pyodide. Nothing leaves the machine.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs';

const MODULES = ['__init__.py', 'scene.py', 'easing.py', 'gradients.py', 'reconstruct.py', 'strokes.py', 'export.py', 'web_api.py'];
let pyodide;

async function init() {
  postMessage({ type: 'status', text: 'Loading Python…' });
  pyodide = await loadPyodide();
  postMessage({ type: 'status', text: 'Loading NumPy, SciPy, OpenCV, scikit-image…' });
  await pyodide.loadPackage(['numpy', 'scipy', 'opencv-python', 'pillow', 'scikit-image']);
  pyodide.FS.mkdirTree('/app/unbake');
  for (const name of MODULES) {
    const src = await (await fetch(`py/unbake/${name}${self.location.search}`)).text();
    pyodide.FS.writeFile(`/app/unbake/${name}`, src);
  }
  await pyodide.runPythonAsync("import sys; sys.path.insert(0, '/app'); import unbake.web_api as api");
  postMessage({ type: 'ready' });
}
const ready = init().catch(e => postMessage({ type: 'error', text: String(e) }));

onmessage = async ({ data }) => {
  await ready;
  try {
    if (data.type === 'reconstruct') {
      pyodide.globals.set('gif_bytes', data.bytes);
      const result = await pyodide.runPythonAsync('api.reconstruct_for_browser(gif_bytes.to_py().tobytes())');
      const [scene, frames] = result.toJs();
      result.destroy();
      const buf = frames.buffer.slice(frames.byteOffset, frames.byteOffset + frames.byteLength);
      postMessage({ type: 'scene', scene: JSON.parse(scene), frames: buf, id: data.id }, [buf]);
    } else if (data.type === 'export') {
      pyodide.globals.set('scene_json', JSON.stringify(data.scene));
      const out = await pyodide.runPythonAsync('import json; json.dumps(api.export_scene(json.loads(scene_json)))');
      postMessage({ type: 'exported', ...JSON.parse(out), format: data.format });
    }
  } catch (e) {
    postMessage({ type: 'error', text: String(e) });
  }
};
