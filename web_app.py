"""Interactive Web Dashboard for Instacart Customer Intelligence & Reorder Prediction.
Powered by FastAPI + Uvicorn. Runs on http://127.0.0.1:8501.
"""
import os
import sys
import json
from pathlib import Path

# Silence loky warning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MODELS = ROOT / "models"

app = FastAPI(title="Instacart Customer Intelligence System")

# Cache data in memory
meta = json.loads((REPORTS / "meta.json").read_text())
comp_df = pd.read_csv(REPORTS / "model_comparison.csv")
profiles_df = pd.read_csv(REPORTS / "segment_profiles.csv")
cand_df = pd.read_csv(REPORTS / "app_candidates.csv.gz")
users_df = pd.read_csv(REPORTS / "app_users.csv.gz", index_col=0)
cms_json = json.loads((REPORTS / "confusion_matrices.json").read_text())
seg_model = joblib.load(MODELS / "segmentation.joblib")

@app.get("/figures/{filename}")
def get_figure(filename: str):
    p = FIGS / filename
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("Figure not found", status_code=404)

@app.get("/api/predict_segment")
def predict_segment(
    n_orders: float = 20,
    avg_days_between: float = 10,
    basket_size: float = 10,
    reorder_ratio: float = 0.45,
    distinct_products: float = 80,
    organic_share: float = 0.25,
    produce_share: float = 0.25,
    snacks_share: float = 0.10,
    avg_order_hour: float = 14,
    weekend_share: float = 0.35,
):
    from src.segmentation import transform
    raw = pd.DataFrame([{
        "n_orders": n_orders,
        "avg_days_between": avg_days_between,
        "basket_size": basket_size,
        "reorder_ratio": reorder_ratio,
        "distinct_products": distinct_products,
        "organic_share": organic_share,
        "produce_share": produce_share,
        "snacks_share": snacks_share,
        "avg_order_hour": avg_order_hour,
        "weekend_share": weekend_share,
    }])
    xs, _ = transform(raw, seg_model["scaler"])
    seg_idx = int(seg_model["kmeans"].predict(xs)[0])
    seg_name = meta["segment_names"][str(seg_idx)]
    prof = profiles_df[profiles_df["segment"] == seg_idx].to_dict(orient="records")[0]
    return {"segment": seg_idx, "name": seg_name, "profile": prof}

@app.get("/api/predict_customer")
def predict_customer(user_id: int, model_name: str = "Gradient Boosting", top_n: int = 10):
    m_file = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "") + ".joblib"
    art = joblib.load(MODELS / m_file)
    sub = cand_df[cand_df["user_id"] == user_id].copy()
    if sub.empty:
        return {"error": "User not found"}
        
    sub["reorder_probability"] = art["model"].predict_proba(sub[art["features"]])[:, 1]
    sub = sub.sort_values("reorder_probability", ascending=False)
    sub["predicted_reorder"] = np.where(sub["reorder_probability"] >= art["threshold"], "Yes", "No")
    sub["actually_reordered"] = np.where(sub["label"] == 1, "✅ Yes", "—")
    
    top = sub.head(top_n)[["product_name", "reorder_probability", "predicted_reorder", "actually_reordered"]].to_dict(orient="records")
    
    pred_set = set(sub.loc[sub["predicted_reorder"] == "Yes", "product_id"])
    true_set = set(sub.loc[sub["label"] == 1, "product_id"])
    hit = len(pred_set & true_set)
    p = hit / len(pred_set) if pred_set else 0
    r = hit / len(true_set) if true_set else 0
    
    u = users_df.loc[user_id]
    seg_id = int(u["segment"])
    seg_name = meta["segment_names"][str(seg_id)]
    
    return {
        "user_id": user_id,
        "segment_id": seg_id,
        "segment_name": seg_name,
        "n_orders": int(u["n_orders"]),
        "basket_size": round(float(u["basket_size"]), 1),
        "reorder_ratio": round(float(u["reorder_ratio"]) * 100, 1),
        "products": top,
        "recommended_count": len(pred_set),
        "precision": round(p * 100, 1),
        "recall": round(r * 100, 1),
    }

@app.get("/", response_class=HTMLResponse)
def index():
    h = meta["headline"]
    user_options = "".join(f"<option value='{uid}'>{uid}</option>" for uid in sorted(cand_df["user_id"].unique())[:100])
    model_options = "".join(f"<option value='{m}'>{m}</option>" for m in comp_df["model"] if not m.startswith("Baseline"))
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🛒 Instacart Customer Intelligence & Reorder Prediction</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body {{ background-color: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    .header-banner {{ background: linear-gradient(135deg, #0aad0a 0%, #00782a 100%); color: white; padding: 25px 20px; border-radius: 0 0 15px 15px; margin-bottom: 25px; }}
    .metric-card {{ background: white; border-radius: 12px; padding: 18px; box-shadow: 0 3px 10px rgba(0,0,0,0.06); text-align: center; border-left: 4px solid #0aad0a; margin-bottom: 15px; }}
    .metric-val {{ font-size: 26px; font-weight: 700; color: #1e293b; margin-top: 5px; }}
    .metric-lbl {{ font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
    .nav-tabs .nav-link {{ font-weight: 600; color: #475569; padding: 12px 20px; border: none; border-bottom: 3px solid transparent; }}
    .nav-tabs .nav-link.active {{ color: #0aad0a; border-bottom: 3px solid #0aad0a; background: transparent; }}
    .content-card {{ background: white; border-radius: 12px; padding: 24px; box-shadow: 0 3px 10px rgba(0,0,0,0.06); margin-bottom: 25px; }}
    .badge-seg0 {{ background-color: #10b981; color: white; }}
    .badge-seg1 {{ background-color: #3b82f6; color: white; }}
    .badge-seg2 {{ background-color: #f59e0b; color: white; }}
    .img-fluid-preview {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }}
  </style>
</head>
<body>
  <div class="header-banner">
    <div class="container">
      <div class="d-flex align-items-center justify-content-between">
        <div>
          <h2 class="mb-1"><i class="fa-solid fa-basket-shopping me-2"></i> Instacart Customer Intelligence & Reorder Prediction</h2>
          <p class="mb-0 text-white-50">Machine Learning Lab Mini-Project (24CS01TP0501) · Ramdeobaba University, Nagpur</p>
        </div>
        <div class="text-end d-none d-md-block">
          <span class="badge bg-light text-dark px-3 py-2 fs-6">Academic Year: 2026–27</span>
        </div>
      </div>
    </div>
  </div>

  <div class="container mb-5">
    <!-- Quick Stats -->
    <div class="row">
      <div class="col-md">
        <div class="metric-card">
          <div class="metric-lbl">Customers Analysed</div>
          <div class="metric-val">{h['users']:,}</div>
        </div>
      </div>
      <div class="col-md">
        <div class="metric-card">
          <div class="metric-lbl">Total Orders</div>
          <div class="metric-val">{h['orders']:,}</div>
        </div>
      </div>
      <div class="col-md">
        <div class="metric-card">
          <div class="metric-lbl">Items Purchased</div>
          <div class="metric-val">{h['prior_items']:,}</div>
        </div>
      </div>
      <div class="col-md">
        <div class="metric-card">
          <div class="metric-lbl">Catalog Reorder Rate</div>
          <div class="metric-val">{h['reorder_rate']:.0%}</div>
        </div>
      </div>
      <div class="col-md">
        <div class="metric-card">
          <div class="metric-lbl">Avg Basket Size</div>
          <div class="metric-val">{h['avg_basket_size']:.1f} items</div>
        </div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <ul class="nav nav-tabs mb-4" id="projectTabs" role="tablist">
      <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-overview"><i class="fa-solid fa-compass me-1"></i> 📌 Overview</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-eda"><i class="fa-solid fa-chart-pie me-1"></i> 🔎 EDA</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-seg"><i class="fa-solid fa-users me-1"></i> 👥 Segments</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-models"><i class="fa-solid fa-robot me-1"></i> 🤖 Models</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-predict"><i class="fa-solid fa-bullseye me-1"></i> 🎯 Predict</button></li>
    </ul>

    <div class="tab-content" id="projectTabsContent">
      <!-- 1. OVERVIEW TAB -->
      <div class="tab-pane fade show active" id="tab-overview">
        <div class="content-card">
          <h4 class="mb-3 text-success">What this project does</h4>
          <ol class="fs-6 lh-lg">
            <li><strong>Explore:</strong> Grocery-shopping behaviour (EDA: univariate, bivariate, multivariate, and 1.5 &times; IQR outlier analysis).</li>
            <li><strong>Segment:</strong> Customers clustered with <strong>K-Means</strong> (k=3 selected via silhouette sweep) and cross-validated with <strong>Agglomerative Hierarchical Clustering</strong> (Ward linkage, ARI=0.38).</li>
            <li><strong>Predict:</strong> Predicts whether a customer will <strong>reorder a previously purchased product</strong> in their next order, benchmarking six tuned classification algorithms.</li>
            <li><strong>Leakage-Free Protocol:</strong> Shoppers are partitioned 80/20 into train/test groups <em>before</em> any statistical fitting. Features are derived strictly from prior orders, while target labels come from the evaluation order.</li>
          </ol>
          <div class="alert alert-success mt-4">
            <h5 class="alert-heading"><i class="fa-solid fa-award me-2"></i> Headline Performance Results</h5>
            <div class="row text-center mt-3">
              <div class="col-md-3"><strong>Optimal Clusters (k):</strong> 3 (Silhouette: {meta['silhouette_kmeans']:.3f})</div>
              <div class="col-md-3"><strong>Best Model:</strong> {meta['best_model']}</div>
              <div class="col-md-3"><strong>Best F1-Score:</strong> {meta['best_f1']:.3f}</div>
              <div class="col-md-3"><strong>Best ROC-AUC:</strong> {meta['best_auc']:.3f}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 2. EDA TAB -->
      <div class="tab-pane fade" id="tab-eda">
        <div class="content-card">
          <h4 class="mb-4">Exploratory Data Analysis Figures</h4>
          <div class="row">
            <div class="col-md-6 text-center">
              <h6>Order Volumes by Day-of-Week & Hour</h6>
              <img src="/figures/01_orders_dow_hour.png" class="img-fluid-preview" alt="Orders DOW Hour">
            </div>
            <div class="col-md-6 text-center">
              <h6>User Feature Distributions</h6>
              <img src="/figures/02_distributions.png" class="img-fluid-preview" alt="Distributions">
            </div>
            <div class="col-md-6 text-center">
              <h6>Top 25 Purchased Grocery Products</h6>
              <img src="/figures/03_top_products.png" class="img-fluid-preview" alt="Top Products">
            </div>
            <div class="col-md-6 text-center">
              <h6>Department-Wise Product Volume</h6>
              <img src="/figures/04_departments.png" class="img-fluid-preview" alt="Departments">
            </div>
            <div class="col-md-6 text-center">
              <h6>Feature Correlation Heatmap</h6>
              <img src="/figures/06_correlation.png" class="img-fluid-preview" alt="Correlation">
            </div>
            <div class="col-md-6 text-center">
              <h6>Outlier Boxplots (1.5 &times; IQR Rule)</h6>
              <img src="/figures/08_outlier_boxplots.png" class="img-fluid-preview" alt="Outliers">
            </div>
          </div>
        </div>
      </div>

      <!-- 3. SEGMENTS TAB -->
      <div class="tab-pane fade" id="tab-seg">
        <div class="content-card">
          <h4 class="mb-4">Customer Behavioral Personas</h4>
          <div class="row mb-4">
            <div class="col-md-6 text-center">
              <h6>K-Means Silhouette Score & Inertia Elbow (k=2 to 10)</h6>
              <img src="/figures/10_kmeans_k_selection.png" class="img-fluid-preview" alt="K-Means Selection">
            </div>
            <div class="col-md-6 text-center">
              <h6>Agglomerative Hierarchical Dendrogram (Ward's Linkage)</h6>
              <img src="/figures/11_dendrogram.png" class="img-fluid-preview" alt="Dendrogram">
            </div>
            <div class="col-md-6 text-center">
              <h6>Segment Profile Heatmap</h6>
              <img src="/figures/13_segment_profile_heatmap.png" class="img-fluid-preview" alt="Segment Heatmap">
            </div>
            <div class="col-md-6 text-center">
              <h6>PCA 2D Cluster Separation</h6>
              <img src="/figures/12_segments_pca.png" class="img-fluid-preview" alt="PCA Segments">
            </div>
          </div>

          <h5 class="text-primary mt-4"><i class="fa-solid fa-sliders me-2"></i> Real-Time Customer Persona Simulator</h5>
          <p class="text-muted">Adjust shopper attributes below to classify a customer using the trained K-Means model in real time:</p>
          <div class="row g-3 p-3 bg-light rounded">
            <div class="col-md-4">
              <label class="form-label">Past Orders: <span id="val_orders">20</span></label>
              <input type="range" class="form-range" id="slider_orders" min="3" max="100" value="20" oninput="updateSim()">
            </div>
            <div class="col-md-4">
              <label class="form-label">Days Between Orders: <span id="val_gap">10</span></label>
              <input type="range" class="form-range" id="slider_gap" min="1" max="30" value="10" oninput="updateSim()">
            </div>
            <div class="col-md-4">
              <label class="form-label">Avg Basket Size: <span id="val_basket">10</span></label>
              <input type="range" class="form-range" id="slider_basket" min="1" max="50" value="10" oninput="updateSim()">
            </div>
            <div class="col-md-4">
              <label class="form-label">Reorder Ratio: <span id="val_reorder">45%</span></label>
              <input type="range" class="form-range" id="slider_reorder" min="0" max="100" value="45" oninput="updateSim()">
            </div>
            <div class="col-md-4">
              <label class="form-label">Organic Share: <span id="val_organic">30%</span></label>
              <input type="range" class="form-range" id="slider_organic" min="0" max="100" value="30" oninput="updateSim()">
            </div>
            <div class="col-md-4">
              <label class="form-label">Produce Share: <span id="val_produce">30%</span></label>
              <input type="range" class="form-range" id="slider_produce" min="0" max="100" value="30" oninput="updateSim()">
            </div>
          </div>
          <div id="simResult" class="alert alert-info mt-3 fs-5">
            <strong>Assigned Persona:</strong> Calculating...
          </div>
        </div>
      </div>

      <!-- 4. MODELS TAB -->
      <div class="tab-pane fade" id="tab-models">
        <div class="content-card">
          <h4 class="mb-4">Supervised Classification Benchmark (Held-Out Test Set: 521,348 rows)</h4>
          <div class="table-responsive">
            <table class="table table-hover table-bordered align-middle">
              <thead class="table-dark">
                <tr>
                  <th>Model</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Accuracy</th>
                  <th>ROC-AUC</th>
                  <th>PR-AUC</th>
                  <th>Threshold</th>
                  <th>Train Time (s)</th>
                </tr>
              </thead>
              <tbody>
                {"".join(f'''<tr class="{'table-success fw-bold' if r['model'] == meta['best_model'] else ''}">
                  <td>{r['model']}</td>
                  <td>{r['precision']:.3f}</td>
                  <td>{r['recall']:.3f}</td>
                  <td>{r['f1']:.3f}</td>
                  <td>{r['accuracy']:.3f}</td>
                  <td>{r['roc_auc']:.3f}</td>
                  <td>{r['avg_precision']:.3f}</td>
                  <td>{r['threshold']:.3f}</td>
                  <td>{r['train_time_s']:.1f}</td>
                </tr>''' for _, r in comp_df.iterrows())}
              </tbody>
            </table>
          </div>
          <div class="row mt-4">
            <div class="col-md-6 text-center">
              <h6>Algorithm Comparison Radar</h6>
              <img src="/figures/20_model_comparison.png" class="img-fluid-preview" alt="Model Comparison">
            </div>
            <div class="col-md-6 text-center">
              <h6>Precision-Recall Curves</h6>
              <img src="/figures/21_precision_recall.png" class="img-fluid-preview" alt="Precision Recall">
            </div>
            <div class="col-md-6 text-center">
              <h6>Top Gini Feature Importances</h6>
              <img src="/figures/23_feature_importance.png" class="img-fluid-preview" alt="Feature Importance">
            </div>
            <div class="col-md-6 text-center">
              <h6>Confusion Matrices (Default vs Tuned)</h6>
              <img src="/figures/22_confusion_matrices.png" class="img-fluid-preview" alt="Confusion Matrices">
            </div>
          </div>
        </div>
      </div>

      <!-- 5. PREDICT TAB -->
      <div class="tab-pane fade" id="tab-predict">
        <div class="content-card">
          <h4 class="mb-3 text-success"><i class="fa-solid fa-basket-shopping me-2"></i> Next-Order Repeat Purchase Predictor</h4>
          <p class="text-muted">Select a customer from the held-out test split (never seen during training) to predict which previously purchased items will be repurchased:</p>
          <div class="row g-3 p-3 bg-light rounded mb-4">
            <div class="col-md-4">
              <label class="form-label fw-bold">Customer ID:</label>
              <select id="userSelect" class="form-select" onchange="runPrediction()">
                {user_options}
              </select>
            </div>
            <div class="col-md-4">
              <label class="form-label fw-bold">Model:</label>
              <select id="modelSelect" class="form-select" onchange="runPrediction()">
                {model_options}
              </select>
            </div>
            <div class="col-md-4">
              <label class="form-label fw-bold">Top Products to Show: <span id="topNVal">10</span></label>
              <input type="range" class="form-range" id="topNSlider" min="5" max="25" value="10" oninput="document.getElementById('topNVal').innerText = this.value; runPrediction()">
            </div>
          </div>

          <div id="userInfoBox" class="alert alert-secondary d-none">
            <!-- User summary injected here -->
          </div>

          <div class="table-responsive">
            <table class="table table-bordered table-striped" id="predTable">
              <thead class="table-primary">
                <tr>
                  <th>#</th>
                  <th>Product Name</th>
                  <th>Reorder Probability</th>
                  <th>Model Prediction</th>
                  <th>Actual Reorder Status</th>
                </tr>
              </thead>
              <tbody id="predTbody">
                <tr><td colspan="5" class="text-center">Loading predictions...</td></tr>
              </tbody>
            </table>
          </div>

          <div class="row mt-3 text-center" id="metricsRow">
            <div class="col-md-4"><div class="p-3 bg-light rounded"><strong>Products Recommended:</strong> <span id="recCount" class="fs-5 text-primary">-</span></div></div>
            <div class="col-md-4"><div class="p-3 bg-light rounded"><strong>Hit Precision:</strong> <span id="recPrec" class="fs-5 text-success">-</span></div></div>
            <div class="col-md-4"><div class="p-3 bg-light rounded"><strong>Caught Recall:</strong> <span id="recRecall" class="fs-5 text-info">-</span></div></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <footer class="text-center py-4 text-muted border-top">
    <p class="mb-0">Instacart Customer Segmentation & Reorder Prediction System · Ramdeobaba University, Nagpur</p>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    function updateSim() {{
      const o = document.getElementById('slider_orders').value;
      const g = document.getElementById('slider_gap').value;
      const b = document.getElementById('slider_basket').value;
      const r = document.getElementById('slider_reorder').value / 100;
      const org = document.getElementById('slider_organic').value / 100;
      const prd = document.getElementById('slider_produce').value / 100;

      document.getElementById('val_orders').innerText = o;
      document.getElementById('val_gap').innerText = g;
      document.getElementById('val_basket').innerText = b;
      document.getElementById('val_reorder').innerText = Math.round(r * 100) + '%';
      document.getElementById('val_organic').innerText = Math.round(org * 100) + '%';
      document.getElementById('val_produce').innerText = Math.round(prd * 100) + '%';

      fetch(`/api/predict_segment?n_orders=${{o}}&avg_days_between=${{g}}&basket_size=${{b}}&reorder_ratio=${{r}}&organic_share=${{org}}&produce_share=${{prd}}`)
        .then(res => res.json())
        .then(data => {{
          document.getElementById('simResult').innerHTML = 
            `<strong>Assigned Persona: Segment ${{data.segment}} — ${{data.name}}</strong><br>` +
            `<small class="text-muted">Typical Basket: ${{data.profile.basket_size.toFixed(1)}} items | Orders: ${{data.profile.n_orders.toFixed(1)}} | Organic: ${{(data.profile.organic_share*100).toFixed(0)}}%</small>`;
        }});
    }}

    function runPrediction() {{
      const uid = document.getElementById('userSelect').value;
      const m = document.getElementById('modelSelect').value;
      const topN = document.getElementById('topNSlider').value;

      fetch(`/api/predict_customer?user_id=${{uid}}&model_name=${{encodeURIComponent(m)}}&top_n=${{topN}}`)
        .then(res => res.json())
        .then(data => {{
          const ubox = document.getElementById('userInfoBox');
          ubox.classList.remove('d-none');
          ubox.innerHTML = `<strong>Customer #${{data.user_id}} Profile:</strong> ${{data.segment_name}} (Segment ${{data.segment_id}}) · ${{data.n_orders}} past orders · ${{data.basket_size}} items/order · historical re-buy ratio ${{data.reorder_ratio}}%`;

          const tbody = document.getElementById('predTbody');
          tbody.innerHTML = '';
          data.products.forEach((p, idx) => {{
            const pct = (p.reorder_probability * 100).toFixed(1);
            const badgeClass = p.predicted_reorder === 'Yes' ? 'bg-success' : 'bg-secondary';
            tbody.innerHTML += `<tr>
              <td>${{idx + 1}}</td>
              <td>${{p.product_name}}</td>
              <td>
                <div class="progress" style="height: 20px;">
                  <div class="progress-bar bg-success" style="width: ${{pct}}%;">${{pct}}%</div>
                </div>
              </td>
              <td><span class="badge ${{badgeClass}}">${{p.predicted_reorder}}</span></td>
              <td><strong>${{p.actually_reordered}}</strong></td>
            </tr>`;
          }});

          document.getElementById('recCount').innerText = data.recommended_count;
          document.getElementById('recPrec').innerText = data.precision + '%';
          document.getElementById('recRecall').innerText = data.recall + '%';
        }});
    }}

    // Initial calls
    updateSim();
    runPrediction();
  </script>
</body>
</html>
"""
    return html

def run():
    print("Starting Instacart Dashboard Web Application on http://127.0.0.1:8501 ...")
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="info")

if __name__ == "__main__":
    run()
