"""
NeuroTopography — วิเคราะห์ภาพ MRI ด้วย Topological Data Analysis
โครงงานคณิตศาสตร์ | TJ-SSF

วิธีรัน:
    pip install streamlit gudhi numpy scipy matplotlib pillow
    streamlit run app.py
"""

import io
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# ── ตรวจสอบว่ามี GUDHI ไหม ──────────────────────────────────
try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False


# ══════════════════════════════════════════════════════════
# ตั้งค่าหน้าเว็บ
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroTopography",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Noto Sans Thai', sans-serif;
}
[data-testid="stMetricValue"] {
    font-size: 2.4rem !important;
    color: #1a3a5c !important;
}
h1 { color: #1a3a5c !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ฟังก์ชันหลัก
# ══════════════════════════════════════════════════════════

def image_to_points(img_array, threshold, max_points=250, sigma=1.2):
    """
    แปลงภาพ MRI เป็น Point Cloud

    ขั้นตอน:
    1. Gaussian smoothing ลด noise
    2. เลือกพิกเซลที่สว่างกว่า threshold
    3. Normalize พิกัดให้อยู่ใน [-1, 1]
    4. Downsample เพื่อความเร็ว
    """
    smooth = gaussian_filter(img_array.astype(float), sigma=sigma)
    rows, cols = np.where(smooth > threshold)

    if len(rows) < 5:
        return np.array([[0.0, 0.0], [0.1, 0.1]])

    pts = np.column_stack([cols.astype(float), -rows.astype(float)])

    # Normalize
    for d in range(2):
        lo, hi = pts[:, d].min(), pts[:, d].max()
        if hi > lo:
            pts[:, d] = 2 * (pts[:, d] - lo) / (hi - lo) - 1

    # Downsample
    if len(pts) > max_points:
        idx = np.random.default_rng(42).choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    return pts


def compute_tda(points, max_edge=2.0):
    """คำนวณ Persistent Homology ด้วย GUDHI"""
    if not HAS_GUDHI:
        return None

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()

    pairs = []
    for dim, (birth, death) in simplex_tree.persistence():
        if death == float("inf"):
            death = max_edge * 1.1
        pairs.append((dim, birth, death))

    return pairs


def get_betti(pairs, epsilon):
    """นับ β₀ และ β₁ ณ ค่า ε ที่กำหนด"""
    if pairs is None:
        return 0, 0
    b0 = sum(1 for d, b, dt in pairs if d == 0 and b <= epsilon < dt)
    b1 = sum(1 for d, b, dt in pairs if d == 1 and b <= epsilon < dt)
    return b0, b1


def make_synthetic(kind):
    """สร้างข้อมูลจำลอง"""
    rng = np.random.default_rng(42)

    if kind == "disk":
        # Disk เต็ม — จุดกระจายทั่วทั้งวงกลม
        n = 200
        r = np.sqrt(rng.random(n))
        theta = rng.random(n) * 2 * np.pi
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    elif kind == "annulus":
        # Annulus — จุดอยู่แค่วงแหวน
        n = 180
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = 1.0 + rng.normal(0, 0.05, n)
        outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        theta2 = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        r2 = 0.55 + rng.normal(0, 0.04, 60)
        inner = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
        return np.vstack([outer, inner])

    else:  # multi-hole (GBM)
        parts = []
        for cx, cy, rad in [(0, 0, 0.85), (0, 0, 0.42),
                            (1.6, 0.4, 0.28), (-1.5, -0.5, 0.25)]:
            n = max(30, int(rad * 120))
            t = np.linspace(0, 2 * np.pi, n, endpoint=False)
            rr = rad + rng.normal(0, 0.03, n)
            parts.append(np.column_stack([cx + rr * np.cos(t),
                                          cy + rr * np.sin(t)]))
        return np.vstack(parts)


# ══════════════════════════════════════════════════════════
# ฟังก์ชันวาดกราฟ
# ══════════════════════════════════════════════════════════

def to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_complex(points, epsilon, title=""):
    """วาด Simplicial Complex"""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_facecolor("#f8f9fa")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle=":")

    D = cdist(points, points)
    n = len(points)

    # วงกลม epsilon
    for p in points:
        ax.add_patch(Circle(p, epsilon / 2, color="#378ADD",
                            alpha=0.05, zorder=1))

    # เส้นเชื่อม
    segs = [[points[i], points[j]]
            for i in range(n) for j in range(i + 1, n)
            if D[i, j] <= epsilon]
    if segs:
        ax.add_collection(LineCollection(segs, color="#378ADD",
                                         alpha=0.3, lw=0.7, zorder=2))

    # สามเหลี่ยม
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] > epsilon:
                continue
            for k in range(j + 1, n):
                if D[i, k] <= epsilon and D[j, k] <= epsilon:
                    ax.add_patch(plt.Polygon(
                        [points[i], points[j], points[k]],
                        color="#378ADD", alpha=0.08, zorder=2))

    # จุด
    ax.scatter(points[:, 0], points[:, 1], s=18, color="#1a3a5c",
               zorder=6, edgecolors="white", linewidths=0.4)

    pad = 0.3
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    ax.set_title(f"{title}  |  ε = {epsilon:.2f}", fontsize=11,
                 color="#1a3a5c", pad=10)
    fig.tight_layout()
    return to_png(fig)


def plot_barcode(pairs):
    """วาด Persistence Barcode"""
    if pairs is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "ไม่พบข้อมูล", ha="center", va="center")
        return to_png(fig)

    d0 = [(b, d) for dim, b, d in pairs if dim == 0]
    d1 = [(b, d) for dim, b, d in pairs if dim == 1]

    total = len(d0) + len(d1)
    if total == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "ไม่พบ features", ha="center", va="center")
        return to_png(fig)

    h = max(3.5, total * 0.22 + 1.5)
    fig, ax = plt.subplots(figsize=(9, h))
    ax.set_facecolor("#f8f9fa")

    y = 0
    for i, (b, d) in enumerate(sorted(d0, key=lambda x: x[1] - x[0], reverse=True)):
        ax.barh(y, d - b, left=b, height=0.6, color="#378ADD", alpha=0.8,
                label=f"β₀ (กลุ่ม) n={len(d0)}" if i == 0 else "")
        y += 1
    for i, (b, d) in enumerate(sorted(d1, key=lambda x: x[1] - x[0], reverse=True)):
        ax.barh(y, d - b, left=b, height=0.6, color="#BA7517", alpha=0.85,
                label=f"β₁ (รู) n={len(d1)}" if i == 0 else "")
        y += 1

    ax.set_xlabel("ε (Epsilon)", fontsize=10)
    ax.set_yticks([])
    ax.set_title("Persistence Barcode", fontsize=11, color="#1a3a5c", pad=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    fig.tight_layout()
    return to_png(fig)


def plot_diagram(pairs):
    """วาด Persistence Diagram"""
    if pairs is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.text(0.5, 0.5, "ไม่พบข้อมูล", ha="center", va="center")
        return to_png(fig)

    d0 = [(b, d) for dim, b, d in pairs if dim == 0]
    d1 = [(b, d) for dim, b, d in pairs if dim == 1]

    all_vals = [v for pair in d0 + d1 for v in pair]
    mv = (max(all_vals) if all_vals else 2.0) * 1.1

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_facecolor("#f8f9fa")
    ax.plot([0, mv], [0, mv], color="#ccc", lw=1.5, zorder=1)

    if d0:
        bx, dy = zip(*d0)
        ax.scatter(bx, dy, s=50, color="#378ADD", alpha=0.8, zorder=5,
                   edgecolors="white", lw=0.5, label=f"β₀ n={len(d0)}")
    if d1:
        bx, dy = zip(*d1)
        ax.scatter(bx, dy, s=65, color="#BA7517", alpha=0.85, zorder=5,
                   marker="D", edgecolors="white", lw=0.5, label=f"β₁ n={len(d1)}")

    ax.set_xlabel("Birth (เกิด)", fontsize=10)
    ax.set_ylabel("Death (ดับ)", fontsize=10)
    ax.set_xlim(0, mv)
    ax.set_ylim(0, mv)
    ax.set_aspect("equal")
    ax.set_title("Persistence Diagram", fontsize=11, color="#1a3a5c", pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    return to_png(fig)


# ══════════════════════════════════════════════════════════
# ส่วนติดต่อผู้ใช้
# ══════════════════════════════════════════════════════════

st.title("🧠 NeuroTopography")
st.caption("วิเคราะห์โครงสร้างเนื้องอกด้วย Topological Data Analysis")

if not HAS_GUDHI:
    st.error("⚠️ ไม่พบ GUDHI — รันคำสั่ง: `pip install gudhi`")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("ตั้งค่า")

    mode = st.radio(
        "เลือกข้อมูล",
        ["📤 อัพโหลดภาพ MRI", "🔬 ข้อมูลตัวอย่าง"],
    )

    st.divider()

    if mode == "📤 อัพโหลดภาพ MRI":
        uploaded = st.file_uploader(
            "เลือกภาพ (.jpg / .png)",
            type=["jpg", "jpeg", "png"],
        )
        st.caption("อัพโหลดภาพ MRI ของคุณเอง")

        threshold = st.slider("Pixel Threshold", 30, 220, 120, 5)
        st.caption("ปรับความสว่างเพื่อแยกเนื้อเยื่อออกจากพื้นหลัง")

        sigma = st.slider("Gaussian σ", 0.5, 3.0, 1.2, 0.1)
        st.caption("ปรับความสมูทเพื่อลดจุดรบกวน")

        max_pts = st.slider("จำนวนจุดสูงสุด", 50, 400, 200, 25)
        st.caption("ลดจำนวนจุดเพื่อคำนวณเร็วขึ้น")
    else:
        sample = st.selectbox(
            "เลือกตัวอย่าง",
            ["Disk เต็ม (เนื้อเยื่อปกติ)",
             "Annulus (เนื้องอกมีโพรง)",
             "หลายโพรง (GBM)"],
        )
        threshold, sigma, max_pts = 120, 1.2, 200

    st.divider()

    epsilon = st.slider("ε (Epsilon)", 0.05, 2.0, 0.5, 0.05)
    st.caption("รัศมีวงกลม — ปรับเพื่อดูโครงสร้างเปลี่ยน")

    max_edge = st.slider("Max Edge", 0.5, 3.0, 2.0, 0.1)
    st.caption("ระยะสูงสุดที่พิจารณาเชื่อมจุด")

# ── ประมวลผล ──────────────────────────────────────────────
points = None
img_display = None

if mode == "📤 อัพโหลดภาพ MRI":
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("L")
        img_array = np.array(pil_img)
        img_display = pil_img
        points = image_to_points(img_array, threshold, max_pts, sigma)
else:
    kind = {"Disk เต็ม (เนื้อเยื่อปกติ)": "disk",
            "Annulus (เนื้องอกมีโพรง)": "annulus",
            "หลายโพรง (GBM)": "multi"}[sample]
    points = make_synthetic(kind)

# ── แสดงผล ────────────────────────────────────────────────
if points is None:
    st.info("👈 กรุณาอัพโหลดภาพ MRI จากแถบด้านซ้าย")
    st.stop()

with st.spinner("กำลังคำนวณ Persistent Homology..."):
    pairs = compute_tda(points, max_edge)

b0, b1 = get_betti(pairs, epsilon)

# ── Metrics ──────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("β₀ — จำนวนกลุ่ม", b0)
c2.metric("β₁ — จำนวนรู", b1)
c3.metric("จำนวนจุด", len(points))

# ── ผลการวิเคราะห์ ───────────────────────────────────────
if b1 == 0:
    st.success(f"✅ β₁ = 0 — ไม่พบโพรงในโครงสร้าง (คล้าย Disk เต็ม)")
elif b1 == 1:
    st.warning(f"⚠️ β₁ = 1 — พบโพรง 1 แห่ง (คล้าย Annulus)")
else:
    st.error(f"🔴 β₁ = {b1} — พบโพรงหลายแห่ง โครงสร้างซับซ้อน")

st.divider()

# ── แสดงภาพ ──────────────────────────────────────────────
if img_display is not None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ภาพ MRI ต้นฉบับ")
        st.image(img_display, use_container_width=True)
    with col_b:
        st.subheader("Simplicial Complex")
        st.image(plot_complex(points, epsilon), use_container_width=True)
else:
    st.subheader("Simplicial Complex")
    st.image(plot_complex(points, epsilon, sample), use_container_width=True)

st.divider()

# ── Barcode และ Diagram ──────────────────────────────────
col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Persistence Barcode")
    st.image(plot_barcode(pairs), use_container_width=True)
with col_d:
    st.subheader("Persistence Diagram")
    st.image(plot_diagram(pairs), use_container_width=True)

# ── คำอธิบาย ─────────────────────────────────────────────
with st.expander("📖 อธิบายผลลัพธ์"):
    st.markdown("""
### β₀ และ β₁ คืออะไร

- **β₀** = จำนวนกลุ่มที่แยกกัน (connected components)
- **β₁** = จำนวนรูหรือโพรง (holes)

### การตีความ

| β₁ | ความหมาย |
|---|---|
| 0 | ไม่มีโพรง — เนื้อเยื่อทึบ (Disk) |
| 1 | มีโพรง 1 แห่ง (Annulus) |
| ≥ 2 | มีโพรงหลายแห่ง — โครงสร้างซับซ้อน |

### วิธีอ่าน Barcode

- **แถบยาว** = feature จริง สำคัญ
- **แถบสั้น** = noise ไม่ต้องสนใจ

### วิธีอ่าน Diagram

- **จุดไกลจากเส้นทแยงมุม** = feature มีนัยสำคัญ
- **จุดใกล้เส้นทแยงมุม** = noise
    """)

st.divider()
st.caption("NeuroTopography — โครงงานคณิตศาสตร์ | Topological Data Analysis")
