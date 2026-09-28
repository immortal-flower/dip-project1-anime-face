"""Serve a local, user-approved review UI for AnimeFace local face boxes.

The tool never overwrites the source manifest. A reviewer draws one box per
image and explicitly approves it. Finalization is blocked until every selected
record is approved.
"""
import argparse
import hashlib
import json
import math
import mimetypes
import os
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_extent(record):
    bbox = record.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or bbox[0] != 0
        or bbox[1] != 0
        or not all(isinstance(value, (int, float)) for value in bbox)
    ):
        raise ValueError(f"Expected full-image bbox for {record.get('source_id')}")
    width, height = int(round(bbox[2])), int(round(bbox[3]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image extent for {record.get('source_id')}")
    return width, height


def suggested_local_face_bbox(record):
    """Return an editable suggestion from visible landmarks, not a decision."""
    width, height = image_extent(record)
    landmarks = record.get("landmarks", [])
    visibility = record.get("visibility", [])
    visible = [
        point
        for point, flag in zip(landmarks, visibility)
        if flag and len(point) == 2 and all(math.isfinite(float(v)) for v in point)
    ]
    if len(visible) < 2:
        return [0, 0, width, height]
    xs = [float(point[0]) for point in visible]
    ys = [float(point[1]) for point in visible]
    span_x = max(max(xs) - min(xs), width * 0.10)
    span_y = max(max(ys) - min(ys), height * 0.10)
    # The 28 points cover facial features and jaw more reliably than hair/top.
    # This asymmetric padding is only a visual starting point for the reviewer.
    x1 = math.floor(min(xs) - 0.16 * span_x)
    y1 = math.floor(min(ys) - 0.42 * span_y)
    x2 = math.ceil(max(xs) + 0.16 * span_x)
    y2 = math.ceil(max(ys) + 0.12 * span_y)
    return [
        max(0, x1),
        max(0, y1),
        min(width, x2),
        min(height, y2),
    ]


def validate_box(box, width, height):
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox must contain four coordinates")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
        raise ValueError("bbox coordinates must be finite numbers")
    normalized = [int(round(value)) for value in box]
    x1, y1, x2, y2 = normalized
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox must stay inside [0,0,{width},{height}]")
    return normalized


def build_items(manifest_path, split):
    manifest_path = Path(manifest_path).resolve()
    records = read_json(manifest_path)
    items = []
    for record in records:
        if record.get("split") != split:
            continue
        width, height = image_extent(record)
        image_path = (manifest_path.parent / record["image"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        items.append(
            {
                "source_id": record["source_id"],
                "image": record["image"],
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "original_bbox": record["bbox"],
                "suggested_bbox": suggested_local_face_bbox(record),
                "landmarks": record["landmarks"],
                "visibility": record["visibility"],
                "record": record,
            }
        )
    if not items:
        raise ValueError(f"No records found for split={split!r}")
    return manifest_path, items


class ReviewStore:
    def __init__(self, manifest_path, items, output_dir, split):
        self.manifest_path = manifest_path
        self.items = items
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.decisions_path = self.output_dir / "review_decisions.json"
        self.final_path = self.output_dir / f"local_face_boxes_{split}.json"
        self.summary_path = self.output_dir / "review_summary.json"
        self.lock = threading.Lock()
        self.decisions = {}
        if self.decisions_path.is_file():
            saved = read_json(self.decisions_path)
            for decision in saved.get("decisions", []):
                self.decisions[decision["source_id"]] = decision

    def public_state(self):
        return {
            "schema_version": 1,
            "split": self.split,
            "items": [
                {
                    key: item[key]
                    for key in (
                        "source_id",
                        "image",
                        "width",
                        "height",
                        "original_bbox",
                        "suggested_bbox",
                        "landmarks",
                        "visibility",
                    )
                }
                for item in self.items
            ],
            "decisions": list(self.decisions.values()),
        }

    def _save_decisions(self):
        ordered = [
            self.decisions[item["source_id"]]
            for item in self.items
            if item["source_id"] in self.decisions
        ]
        write_json_atomic(
            self.decisions_path,
            {
                "schema_version": 1,
                "source_manifest": str(self.manifest_path),
                "source_manifest_sha256": sha256_file(self.manifest_path),
                "split": self.split,
                "reviewer": "user",
                "decisions": ordered,
            },
        )

    def save(self, payload):
        index = int(payload["index"])
        if not 0 <= index < len(self.items):
            raise ValueError("Invalid item index")
        item = self.items[index]
        status = payload.get("status")
        if status not in {"draft", "approved", "needs_redraw"}:
            raise ValueError("Invalid review status")
        bbox = validate_box(payload.get("bbox"), item["width"], item["height"])
        decision = {
            "source_id": item["source_id"],
            "image": item["image"],
            "split": self.split,
            "original_bbox": item["original_bbox"],
            "suggested_bbox": item["suggested_bbox"],
            "local_face_bbox": bbox,
            "status": status,
            "reviewer": "user",
            "note": str(payload.get("note", ""))[:1000],
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            self.decisions[item["source_id"]] = decision
            self._save_decisions()
        return decision

    def finalize(self):
        with self.lock:
            missing = [
                item["source_id"]
                for item in self.items
                if self.decisions.get(item["source_id"], {}).get("status") != "approved"
            ]
            if missing:
                raise ValueError(
                    f"Finalization requires {len(self.items)}/{len(self.items)} approvals; "
                    f"missing {len(missing)}"
                )
            output = []
            for item in self.items:
                decision = self.decisions[item["source_id"]]
                record = dict(item["record"])
                record["original_bbox"] = record["bbox"]
                record["bbox"] = decision["local_face_bbox"]
                record["bbox_annotation_type"] = "manual_local_face"
                record["bbox_review_status"] = "approved"
                record["bbox_reviewer"] = "user"
                record["bbox_reviewed_at"] = decision["reviewed_at"]
                output.append(record)
            write_json_atomic(self.final_path, output)
            summary = {
                "schema_version": 1,
                "split": self.split,
                "records": len(output),
                "approved": len(output),
                "reviewer": "user",
                "source_manifest": str(self.manifest_path),
                "source_manifest_sha256": sha256_file(self.manifest_path),
                "output_manifest": str(self.final_path),
                "output_manifest_sha256": sha256_file(self.final_path),
                "finalized_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json_atomic(self.summary_path, summary)
        return summary


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnimeFace 局部人脸框逐图复核</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--border:#30363d;--blue:#2f81f7;--green:#3fb950;--yellow:#d29922;--red:#f85149;--text:#e6edf3;--muted:#8b949e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--border);display:flex;gap:18px;align-items:center;flex-wrap:wrap}h1{font-size:18px;margin:0}.muted{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(520px,1fr) 360px;gap:16px;padding:16px;height:calc(100vh - 68px)}
.viewer,.controls{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px;overflow:auto}.canvas-wrap{display:flex;justify-content:center;align-items:center;min-height:70vh;background:#010409;border-radius:8px}
canvas{image-rendering:auto;cursor:crosshair;max-width:100%;height:auto}.row{display:flex;gap:8px;margin:8px 0;align-items:center;flex-wrap:wrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
label{color:var(--muted)}input,textarea,button{background:#0d1117;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:8px}input{width:100%}textarea{width:100%;min-height:70px}button{cursor:pointer}button:hover{border-color:#8b949e}.primary{background:#238636;border-color:#2ea043}.danger{background:#6e2621;border-color:#f85149}.blue{background:#1f6feb;border-color:#388bfd}
.status{padding:6px 9px;border-radius:999px;background:#21262d}.approved{color:#7ee787}.needs_redraw{color:#ff7b72}.draft{color:#e3b341}
.progress{height:8px;background:#21262d;border-radius:9px;overflow:hidden;min-width:220px}.progress>div{height:100%;background:var(--green)}
code{color:#79c0ff}@media(max-width:950px){main{grid-template-columns:1fr;height:auto}.canvas-wrap{min-height:50vh}}
</style>
</head>
<body>
<header><h1>AnimeFace 局部人脸框逐图复核</h1><span id="counter"></span><div class="progress"><div id="bar"></div></div><span id="approvedCount" class="approved"></span></header>
<main>
  <section class="viewer"><div class="canvas-wrap"><canvas id="canvas"></canvas></div></section>
  <aside class="controls">
    <div class="row"><button id="prev">← 上一张</button><button id="next">下一张 →</button></div>
    <h2 id="source"></h2><div id="imageName" class="muted"></div>
    <p>在图上按住鼠标拖动，重新画一个局部人脸框。绿色点是可见关键点，灰点是不可见点；黄色框只是建议，必须由你批准。</p>
    <div class="grid">
      <label>x1<input id="x1" type="number"></label><label>y1<input id="y1" type="number"></label>
      <label>x2<input id="x2" type="number"></label><label>y2<input id="y2" type="number"></label>
    </div>
    <div class="row"><button id="suggest">恢复建议框</button><button id="full">使用整图框</button></div>
    <label>备注<textarea id="note" placeholder="遮挡、侧脸、边界不确定等"></textarea></label>
    <div class="row"><button id="draft">保存草稿</button><button id="redraw" class="danger">标记需重画</button></div>
    <button id="approve" class="primary" style="width:100%;font-weight:700">批准此局部框并到下一张</button>
    <p>当前状态：<span id="status" class="status">未复核</span></p>
    <hr style="border-color:var(--border)">
    <button id="finalize" class="blue" style="width:100%">38张全部批准后生成最终清单</button>
    <p id="message" class="muted"></p>
    <p class="muted">输出：<code>results/c-local-box-review/</code><br>原始清单不会被覆盖。</p>
  </aside>
</main>
<script>
let state, index=0, box=[0,0,1,1], zoom=1, image=new Image(), drawing=false, start=null;
const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d');
async function api(path, options){const r=await fetch(path,options);const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
function decisions(){return new Map(state.decisions.map(d=>[d.source_id,d]))}
function currentDecision(){return decisions().get(state.items[index].source_id)}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v))}
function setBox(next){const it=state.items[index];box=[clamp(Math.round(next[0]),0,it.width-1),clamp(Math.round(next[1]),0,it.height-1),clamp(Math.round(next[2]),1,it.width),clamp(Math.round(next[3]),1,it.height)];if(box[2]<=box[0])box[2]=Math.min(it.width,box[0]+1);if(box[3]<=box[1])box[3]=Math.min(it.height,box[1]+1);['x1','y1','x2','y2'].forEach((id,i)=>$(id).value=box[i]);draw()}
function readBox(){return ['x1','y1','x2','y2'].map(id=>Number($(id).value))}
function draw(){if(!image.complete)return;const it=state.items[index];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);ctx.lineWidth=Math.max(2,zoom*.35);ctx.strokeStyle='#f2cc60';ctx.strokeRect(box[0]*zoom,box[1]*zoom,(box[2]-box[0])*zoom,(box[3]-box[1])*zoom);it.landmarks.forEach((p,i)=>{ctx.beginPath();ctx.fillStyle=it.visibility[i]?'#3fb950':'#6e7681';ctx.arc(p[0]*zoom,p[1]*zoom,Math.max(2,zoom*.55),0,Math.PI*2);ctx.fill()})}
function refreshHeader(){const approved=state.decisions.filter(d=>d.status==='approved').length;$('counter').textContent=`${index+1} / ${state.items.length}`;$('approvedCount').textContent=`已批准 ${approved}/${state.items.length}`;$('bar').style.width=`${approved/state.items.length*100}%`}
function load(i){index=clamp(i,0,state.items.length-1);const it=state.items[index],d=currentDecision();$('source').textContent=it.source_id;$('imageName').textContent=`${it.image} · ${it.width}×${it.height}`;$('note').value=d?.note||'';const s=d?.status||'未复核';$('status').textContent=s;$('status').className='status '+(d?.status||'');image.onload=()=>{zoom=Math.max(1,Math.min(9,660/Math.max(it.width,it.height)));canvas.width=Math.round(it.width*zoom);canvas.height=Math.round(it.height*zoom);setBox(d?.local_face_bbox||it.suggested_bbox)};image.src=`/image/${index}?v=${Date.now()}`;refreshHeader();$('message').textContent=''}
async function save(status,advance=false){try{setBox(readBox());const decision=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index,bbox:box,status,note:$('note').value})});const old=state.decisions.findIndex(d=>d.source_id===decision.source_id);if(old>=0)state.decisions[old]=decision;else state.decisions.push(decision);$('message').textContent=`已保存：${decision.status}`;refreshHeader();if(advance){const next=state.items.findIndex((it,i)=>i>index&&!decisions().has(it.source_id));load(next>=0?next:Math.min(index+1,state.items.length-1))}else load(index)}catch(e){$('message').textContent=e.message}}
canvas.addEventListener('mousedown',e=>{const r=canvas.getBoundingClientRect();start=[(e.clientX-r.left)/r.width*state.items[index].width,(e.clientY-r.top)/r.height*state.items[index].height];drawing=true});
canvas.addEventListener('mousemove',e=>{if(!drawing)return;const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*state.items[index].width,y=(e.clientY-r.top)/r.height*state.items[index].height;setBox([Math.min(start[0],x),Math.min(start[1],y),Math.max(start[0],x),Math.max(start[1],y)])});
window.addEventListener('mouseup',()=>drawing=false);['x1','y1','x2','y2'].forEach(id=>$(id).addEventListener('change',()=>setBox(readBox())));
$('prev').onclick=()=>load(index-1);$('next').onclick=()=>load(index+1);$('suggest').onclick=()=>setBox(state.items[index].suggested_bbox);$('full').onclick=()=>setBox(state.items[index].original_bbox);
$('draft').onclick=()=>save('draft');$('redraw').onclick=()=>save('needs_redraw');$('approve').onclick=()=>save('approved',true);
$('finalize').onclick=async()=>{try{const s=await api('/api/finalize',{method:'POST'});$('message').textContent=`最终清单已生成：${s.records} 张，SHA256 ${s.output_manifest_sha256}`}catch(e){$('message').textContent=e.message}};
window.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')load(index-1);if(e.key==='ArrowRight')load(index+1);if(e.key.toLowerCase()==='a')save('approved',true)});
(async()=>{state=await api('/api/state');load(0)})().catch(e=>$('message').textContent=e.message);
</script>
</body></html>"""


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format_string, *args):
            return

        def send_json(self, value, status=HTTPStatus.OK):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/state":
                self.send_json(store.public_state())
                return
            if parsed.path.startswith("/image/"):
                try:
                    index = int(parsed.path.rsplit("/", 1)[1])
                    item = store.items[index]
                    image_path = Path(item["image_path"])
                    body = image_path.read_bytes()
                    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                except (ValueError, IndexError, OSError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/save":
                    self.send_json(store.save(payload))
                    return
                if self.path == "/api/finalize":
                    self.send_json(store.finalize())
                    return
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/joint_a_c_v1/landmarks_validated.json",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results/c-local-box-review")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the system browser")
    args = parser.parse_args()
    manifest_path, items = build_items(args.manifest, args.split)
    store = ReviewStore(manifest_path, items, args.output, args.split)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}/"
    print(f"Reviewing {len(items)} images from {manifest_path}", flush=True)
    print(f"Open {url}", flush=True)
    print(f"Decisions: {store.decisions_path}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
