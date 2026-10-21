import os, uuid, json, base64, io, time, re
from flask import Flask, request, jsonify, send_file, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
from datetime import datetime

# ── DB ──────────────────────────────────────────────────────────────────────
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_BACKEND = 'postgres'
except ImportError:
    DB_BACKEND = 'sqlite'

import sqlite3

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE_DIR, 'fonts')

pdfmetrics.registerFont(
    TTFont('AppFont', os.path.join(FONT_DIR, 'arial.ttf'))
)
pdfmetrics.registerFont(
    TTFont('AppFont-Bold', os.path.join(FONT_DIR, 'arialbd.ttf'))
)

pdfmetrics.registerFontFamily(
    'AppFont',
    normal='AppFont',
    bold='AppFont-Bold'
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER= os.path.join(BASE_DIR, 'results')
DB_PATH       = os.path.join(BASE_DIR, 'kampus.db')   # SQLite fallback
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# PostgreSQL bağlantı bilgilerini ortam değişkeninden oku
# Örnek: DATABASE_URL=postgresql://user:pass@localhost:5432/kampusdb
DATABASE_URL = os.environ.get('DATABASE_URL', '')

def get_conn():
    if DB_BACKEND == 'postgres' and DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    return sqlite3.connect(DB_PATH)

def is_pg():
    return DB_BACKEND == 'postgres' and DATABASE_URL

def init_db():
    conn = get_conn()
    c = conn.cursor()
    ph = '%s' if is_pg() else '?'
    auto = 'SERIAL PRIMARY KEY' if is_pg() else 'INTEGER PRIMARY KEY AUTOINCREMENT'

    c.execute(f"""
        CREATE TABLE IF NOT EXISTS sessions (
            id {auto},
            session_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            conf_threshold REAL DEFAULT 0.25,
            image_count INTEGER DEFAULT 0
        )
    """)
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS detections (
            id {auto},
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            det_class TEXT,
            confidence REAL,
            box_x1 INTEGER, box_y1 INTEGER, box_x2 INTEGER, box_y2 INTEGER,
            damage_score REAL DEFAULT 0,
            status_label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id {auto},
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            det_index INTEGER,
            det_class TEXT,
            feedback_type TEXT NOT NULL,
            correct_class TEXT,
            correct_status TEXT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Models ───────────────────────────────────────────────────────────────────
print("Loading models...")
model_cukur    = YOLO(os.path.join(BASE_DIR, 'best_cukur.pt'))
model_kaldirim = YOLO(os.path.join(BASE_DIR, 'best_kaldirim.pt'))
print("Models loaded.")

CLASS_WEIGHTS  = {'cukur': 1.0, 'kirik_kaldirim': 1.0}
CONF_THRESHOLD = 0.25   # runtime'da değiştirilebilir

COLORS = {
    'cukur':          (220,  53,  69),
    'kirik_kaldirim': ( 30, 136, 229),
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_damage_score(detections, img_w, img_h):
    if not detections:
        return 0.0
    
    import math
    img_area = img_w * img_h
    count = len(detections)
    
    area_score = 0.0
    for d in detections:
        x1, y1, x2, y2 = d['box']
        box_area = (x2 - x1) * (y2 - y1)
        ratio = box_area / img_area
        # confidence sadece hafif ağırlık, alan asıl faktör
        area_score += ratio * (0.5 + d['confidence'] * 0.5)
    
    # Alan skoru: %20 alan kaplaması → 10 puan
    area_component  = min(area_score * 50, 7.0)
    
    # Sayı skoru: logaritmik, max 3 puan
    count_component = min(math.log(count + 1, 2) * 1.5, 3.0)
    
    score = area_component + count_component
    return round(min(score, 10.0), 2)

def status_label(score):
    if score == 0:   return 'Sağlam', '🟢'
    elif score < 4:  return 'Orta',   '🟡'
    else:            return 'Kritik',  '🔴'

def draw_boxes(image_path, detections):
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    for det in detections:
        cls  = det['class']
        conf = det['confidence']
        x1, y1, x2, y2 = det['box']
        color_rgb = COLORS.get(cls, (100, 200, 100))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        label      = f"{cls} {conf:.0%}"
        font_scale = max(0.5, min(w, h) / 1000)
        thickness  = max(1, int(font_scale * 2))
        (tw, th), _= cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode('utf-8')

def run_inference(image_path, conf_thr=None):
    thr = conf_thr if conf_thr is not None else CONF_THRESHOLD
    detections = []
    for model, cls_name in [(model_cukur, 'cukur'), (model_kaldirim, 'kirik_kaldirim')]:
        results = model(image_path, conf=thr, verbose=False)[0]
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                'class':      cls_name,
                'confidence': round(conf, 4),
                'box':        [x1, y1, x2, y2],
            })
    return detections

def save_session_to_db(session_id, conf_thr, all_results):
    conn = get_conn()
    c    = conn.cursor()
    ph   = '%s' if is_pg() else '?'
    try:
         c.execute(
            f"""
            INSERT INTO sessions (session_id, conf_threshold, image_count)
            VALUES ({ph}, {ph}, {ph})
            ON CONFLICT (session_id) DO UPDATE SET
                conf_threshold = excluded.conf_threshold,
                image_count = excluded.image_count
            """,
            (session_id, conf_thr, len(all_results))
        )
        for r in all_results:
            dets = r.get('detections', [])
            if not dets:
                c.execute(f"""INSERT INTO detections
                    (session_id, filename, det_class, confidence, damage_score, status_label)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph})""",
                    (session_id, r['filename'], None, None, r['damage_score'], r['status']))
            for det in dets:
                b = det['box']
                c.execute(f"""INSERT INTO detections
                    (session_id, filename, det_class, confidence,
                     box_x1, box_y1, box_x2, box_y2, damage_score, status_label)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})""",
                    (session_id, r['filename'], det['class'], det['confidence'],
                     b[0], b[1], b[2], b[3], r['damage_score'], r['status']))
        conn.commit()
    finally:
        conn.close()

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/conf', methods=['POST'])
def set_conf():
    global CONF_THRESHOLD
    data = request.get_json(force=True)
    val  = float(data.get('conf', 0.25))
    val  = max(0.05, min(0.95, val))
    CONF_THRESHOLD = round(val, 2)
    return jsonify({'conf': CONF_THRESHOLD})

@app.route('/analyze', methods=['POST'])
def analyze():
    files    = request.files.getlist('images')
    conf_raw = request.form.get('conf', str(CONF_THRESHOLD))
    try:
        conf_thr = max(0.05, min(0.95, float(conf_raw)))
    except ValueError:
        conf_thr = CONF_THRESHOLD

    if not files:
        return jsonify({'error': 'Görüntü yüklenmedi'}), 400

    all_results = []
    session_id  = uuid.uuid4().hex[:8]

    for f in files:
        filename = f.filename
        ext      = os.path.splitext(filename)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue

        stored_filename = f"{session_id}_{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, stored_filename)
        f.save(save_path)

        try:
            with Image.open(save_path) as image:
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError):
            os.remove(save_path)
            return jsonify({
                'error': 'Yüklenen dosyalardan biri geçerli bir görsel değil.'
            }), 400

        try:
            with Image.open(save_path) as image:
                img_w, img_h = image.size

            detections = run_inference(save_path, conf_thr)
            score = compute_damage_score(detections, img_w, img_h)
            status, emoji = status_label(score)
            annotated_b64 = draw_boxes(save_path, detections)

            all_results.append({
                'filename':       filename,
                'detections':     detections,
                'damage_score':   score,
                'status':         status,
                'status_emoji':   emoji,
                'annotated_image': annotated_b64,
                'session_id':     session_id,
            })
        except Exception as e:
            all_results.append({
                'filename':       filename,
                'error':          str(e),
                'damage_score':   0,
                'status':         'Hata',
                'status_emoji':   '⚠️',
                'detections':     [],
                'session_id':     session_id,
            })

    # JSON kayıt (base64 hariç)
    result_path = os.path.join(RESULTS_FOLDER, f"{session_id}_results.json")
    saveable    = [{k: v for k, v in r.items() if k != 'annotated_image'} for r in all_results]
    with open(result_path, 'w', encoding='utf-8') as fp:
        json.dump(saveable, fp, ensure_ascii=False, indent=2)

    # DB kayıt
    try:
        save_session_to_db(session_id, conf_thr, all_results)
    except Exception as e:
        print(f"DB kayıt hatası: {e}")

    return jsonify({'results': all_results, 'session_id': session_id, 'conf_used': conf_thr})

@app.route('/feedback', methods=['POST'])
def feedback():
    data          = request.get_json(force=True)
    session_id    = data.get('session_id', '')
    filename      = data.get('filename', '')
    det_index     = data.get('det_index')        # hangi tespit (None = görsel geneli)
    det_class     = data.get('det_class')
    feedback_type = data.get('feedback_type', '') # 'wrong_detection' | 'wrong_status'
    correct_class  = data.get('correct_class')
    correct_status = data.get('correct_status')
    note           = data.get('note', '')

    if not session_id or not filename or not feedback_type:
        return jsonify({'error': 'Eksik alan'}), 400

    conn = get_conn()
    c    = conn.cursor()
    ph   = '%s' if is_pg() else '?'
    try:
        c.execute(f"""INSERT INTO feedbacks
            (session_id, filename, det_index, det_class, feedback_type,
             correct_class, correct_status, note)
            VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})""",
            (session_id, filename, det_index, det_class, feedback_type,
             correct_class, correct_status, note))
        conn.commit()
    finally:
        conn.close()

    return jsonify({'ok': True})

@app.route('/history')
def history():
    conn = get_conn()
    c    = conn.cursor()
    try:
        if is_pg():
            c.execute("SELECT session_id, created_at, conf_threshold, image_count FROM sessions ORDER BY created_at DESC LIMIT 50")
            rows = c.fetchall()
            cols = [d[0] for d in c.description]
            result = [dict(zip(cols, r)) for r in rows]
        else:
            conn.row_factory = sqlite3.Row
            c2 = conn.cursor()
            c2.execute("SELECT session_id, created_at, conf_threshold, image_count FROM sessions ORDER BY created_at DESC LIMIT 50")
            result = [dict(r) for r in c2.fetchall()]
    finally:
        conn.close()
    return jsonify(result)

@app.route('/feedbacks')
def get_feedbacks():
    conn = get_conn()
    try:
        if is_pg():
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute("SELECT * FROM feedbacks ORDER BY created_at DESC LIMIT 200")
            result = list(c.fetchall())
        else:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM feedbacks ORDER BY created_at DESC LIMIT 200")
            result = [dict(r) for r in c.fetchall()]
    finally:
        conn.close()
    return jsonify(result)

# ── Export helpers ────────────────────────────────────────────────────────────
def _load_results(session_id):
    path = os.path.join(RESULTS_FOLDER, f"{session_id}_results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def _flatten_rows(data):
    rows = []
    for r in data:
        filename = r.get('filename', '')
        score    = r.get('damage_score', 0)
        status   = r.get('status', '')
        emoji    = r.get('status_emoji', '')
        dets     = r.get('detections', [])
        if not dets:
            rows.append({'Görsel Adı': filename, 'Tespit Edilen Nesne': '—',
                         'Güven Skoru': '—', 'Hasar Puanı (0-10)': score,
                         'Durum': f"{emoji} {status}"})
        else:
            for det in dets:
                rows.append({'Görsel Adı': filename,
                             'Tespit Edilen Nesne': det['class'],
                             'Güven Skoru': f"%{det['confidence']*100:.1f}",
                             'Hasar Puanı (0-10)': score,
                             'Durum': f"{emoji} {status}"})
    return rows

@app.route('/export/csv/<session_id>')
def export_csv(session_id):
    data = _load_results(session_id)
    if data is None: return jsonify({'error': 'Sonuç bulunamadı'}), 404
    df  = pd.DataFrame(_flatten_rows(data))
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig')
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name=f'kampus_rapor_{session_id}.csv')

@app.route('/export/excel/<session_id>')
def export_excel(session_id):
    data = _load_results(session_id)
    if data is None: return jsonify({'error': 'Sonuç bulunamadı'}), 404
    df  = pd.DataFrame(_flatten_rows(data))
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Tespit Raporu')
        ws = writer.sheets['Tespit Raporu']
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(
                max(len(str(cell.value or '')) for cell in col) + 4, 50)
    buf.seek(0)
    return send_file(buf,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True, download_name=f'kampus_rapor_{session_id}.xlsx')

@app.route('/export/pdf/<session_id>')
def export_pdf(session_id):
    data = _load_results(session_id)
    if data is None: return jsonify({'error': 'Sonuç bulunamadı'}), 404

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=1.5*cm, leftMargin=1.5*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style    = ParagraphStyle('t',  fontSize=18, fontName='AppFont-Bold',
                                    alignment=TA_CENTER, spaceAfter=6,  textColor=colors.HexColor('#1a1a2e'))
    subtitle_style = ParagraphStyle('st', fontSize=10, fontName='AppFont',
                                    alignment=TA_CENTER, spaceAfter=20, textColor=colors.HexColor('#666666'))
    section_style  = ParagraphStyle('s',  fontSize=13, fontName='AppFont-Bold',
                                    spaceAfter=8, spaceBefore=16, textColor=colors.HexColor('#1a1a2e'))
    story = []
    story.append(Paragraph("Kampüs Altyapı Kusur Tespit Raporu", title_style))
    story.append(Paragraph(
        f"Oluşturulma: {datetime.now().strftime('%d.%m.%Y %H:%M')} | Oturum: {session_id}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e'), spaceAfter=12))

    total_images = len(data)
    total_dets   = sum(len(r.get('detections', [])) for r in data)
    kritik = sum(1 for r in data if r.get('status') == 'Kritik')
    orta   = sum(1 for r in data if r.get('status') == 'Orta')
    saglam = sum(1 for r in data if r.get('status') == 'Sağlam')

    story.append(Paragraph("Genel Özet", section_style))
    summary_data = [
        ['Toplam Görüntü','Toplam Tespit','Kritik','Orta','Sağlam'],
        [str(total_images), str(total_dets), str(kritik), str(orta), str(saglam)],
    ]
    st = Table(summary_data, colWidths=[3.5*cm]*5)
    st.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0,0),(-1,0), colors.white),
        ('FONTNAME',  (0,0),(-1,0), 'AppFont-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1), 10),
        ('ALIGN',     (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',    (0,0),(-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID',      (0,0),(-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('ROWHEIGHT', (0,0),(-1,-1), 22),
    ]))
    story.append(st)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Detaylı Tespit Tablosu", section_style))
    rows = _flatten_rows(data)
    table_data = [['Görsel Adı','Tespit Edilen Nesne','Güven Skoru','Hasar Puanı','Durum']]
    for row in rows:
        table_data.append([
            Paragraph(row['Görsel Adı'], ParagraphStyle('cell', fontSize=8, fontName='AppFont')),
            row['Tespit Edilen Nesne'], row['Güven Skoru'],
            str(row['Hasar Puanı (0-10)']), row['Durum'],
        ])
    dt = Table(
        table_data,
        colWidths=[5*cm, 4*cm, 2.5*cm, 2.5*cm, 3*cm],
        repeatRows=1
    )
    dt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#343a40')),
        ('TEXTCOLOR', (0,0),(-1,0), colors.white),
        ('FONTNAME',  (0,0),(-1,0), 'AppFont-Bold'),
        ('FONTSIZE',  (0,0),(-1,0), 9),
        ('ALIGN',     (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',    (0,0),(-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#ffffff'), colors.HexColor('#f8f9fa')]),
        ('GRID',      (0,0),(-1,-1), 0.4, colors.HexColor('#dee2e6')),
        ('FONTSIZE',  (0,1),(-1,-1), 8),
        ('ROWHEIGHT', (0,0),(-1,-1), 20),
    ]))
    for i, row in enumerate(rows, start=1):
        s = row['Durum']
        if   'Kritik' in s: dt.setStyle(TableStyle([('BACKGROUND',(4,i),(4,i), colors.HexColor('#f8d7da'))]))
        elif 'Orta'   in s: dt.setStyle(TableStyle([('BACKGROUND',(4,i),(4,i), colors.HexColor('#fff3cd'))]))
        elif 'Sağlam' in s: dt.setStyle(TableStyle([('BACKGROUND',(4,i),(4,i), colors.HexColor('#d1e7dd'))]))
    story.append(dt)
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#dee2e6'), spaceAfter=8))
    story.append(Paragraph(
        "Bu rapor YOLOv8 tabanlı kampüs altyapı kusur tespit sistemi tarafından otomatik oluşturulmuştur.",
        ParagraphStyle('footer', fontSize=7, fontName='AppFont',
                       textColor=colors.grey, alignment=TA_CENTER)))
    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype='application/pdf', as_attachment=True,
                     download_name=f'kampus_rapor_{session_id}.pdf')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
