const Production = {
  mines: [],
  currentMineId: null,
  chart: null,
  lastPrediction: null,

  async init() {
    this.mines = await Api.get("/api/production/mines");
    const select = document.getElementById("mineSelect");
    select.innerHTML = this.mines
      .map((m) => `<option value="${m.mine_id}">${m.name} (${m.type})</option>`)
      .join("");
    select.addEventListener("change", () => this.loadMine(select.value));

    document.getElementById("runWhatIf").addEventListener("click", () => this.runWhatIf());
    document.getElementById("resetWhatIf").addEventListener("click", () => this.loadMine(this.currentMineId));

    ["rainfall", "equip", "downtime", "labor", "soil", "target"].forEach((id) => {
      const input = document.getElementById(`wf-${id}`);
      input.addEventListener("input", () => this.syncSliderLabel(id));
    });

    await this.loadMine(this.mines[0].mine_id);
  },

  syncSliderLabel(id) {
    const input = document.getElementById(`wf-${id}`);
    const label = document.getElementById(`wf-${id}-val`);
    const units = { rainfall: " mm", equip: "%", downtime: " h", labor: "%", soil: "%", target: " t" };
    label.textContent = Number(input.value).toLocaleString() + (units[id] || "");
  },

  async loadMine(mineId) {
    this.currentMineId = mineId;
    const [history, prediction] = await Promise.all([
      Api.get(`/api/production/history?mine_id=${mineId}`),
      Api.get(`/api/production/predict?mine_id=${mineId}`),
    ]);
    this.lastPrediction = prediction;
    this.renderChart(history.records, prediction);
    this.renderForecast(prediction);
    this.renderDrivers(prediction);
    this.renderRecommendations(prediction);
    this.presetSliders(prediction);
    document.getElementById("whatifResult").innerHTML = "";
  },

  presetSliders(prediction) {
    const inputs = prediction.inputs;
    const map = {
      rainfall: inputs.rainfall_mm,
      equip: inputs.equipment_availability_pct,
      downtime: inputs.downtime_hours,
      labor: inputs.labor_availability_pct,
      soil: inputs.soil_moisture_pct,
      target: inputs.planned_target_tonnes,
    };
    Object.entries(map).forEach(([id, val]) => {
      document.getElementById(`wf-${id}`).value = val;
      this.syncSliderLabel(id);
    });
  },

  renderChart(records, prediction) {
    const labels = records.map((r) => `${r.year}-${String(r.month).padStart(2, "0")}`);
    const target = records.map((r) => r.planned_target_tonnes);
    const actual = records.map((r) => r.actual_production_tonnes);

    labels.push("Forecast");
    target.push(prediction.planned_target_tonnes);
    actual.push(null);
    const predicted = new Array(records.length).fill(null);
    predicted.push(prediction.predicted_production_tonnes);

    const ctx = document.getElementById("productionChart").getContext("2d");
    if (this.chart) this.chart.destroy();
    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Planned Target", data: target, borderColor: "#8695a3", borderDash: [4, 3], pointRadius: 0, tension: 0.15 },
          { label: "Actual Production", data: actual, borderColor: "#2563a8", backgroundColor: "rgba(37,99,168,.08)", fill: true, pointRadius: 2, tension: 0.2 },
          { label: "Predicted (ML)", data: predicted, borderColor: "#b3382c", pointBackgroundColor: "#b3382c", pointRadius: 5, showLine: false },
        ],
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } },
        scales: {
          x: { ticks: { font: { size: 10 } }, grid: { display: false } },
          y: { ticks: { font: { size: 10 } }, grid: { color: "#eef2f6" } },
        },
      },
    });
  },

  renderForecast(p) {
    document.getElementById("f-target").textContent = p.planned_target_tonnes.toLocaleString() + " t";
    document.getElementById("f-predicted").textContent = p.predicted_production_tonnes.toLocaleString() + " t";
    document.getElementById("f-shortfall").textContent = p.predicted_shortfall_tonnes.toLocaleString() + " t";
    document.getElementById("f-prob").textContent = (p.shortfall_probability * 100).toFixed(1) + "%";
    document.getElementById("f-conf").textContent = (p.prediction_confidence * 100).toFixed(0) + "%";

    const badge = document.getElementById("riskBadge");
    badge.textContent = p.risk_level + " Risk";
    badge.className = "risk-badge " + p.risk_level;
  },

  renderDrivers(p) {
    const el = document.getElementById("riskDrivers");
    el.innerHTML = "";
    p.risk_drivers.forEach((d) => {
      const pct = Math.min(100, Math.abs(d.contribution) * 60 + 10);
      const row = document.createElement("div");
      row.className = "explain-row";
      row.innerHTML = `
        <div class="explain-row-top"><span>${d.label}</span><span class="val">${d.value} &middot; ${d.direction}</span></div>
        <div class="explain-bar-track"><div class="explain-bar-fill ${d.direction === "increases risk" ? "neg" : "pos"}" style="width:${pct}%"></div></div>
      `;
      el.appendChild(row);
    });
  },

  renderRecommendations(p) {
    const el = document.getElementById("recommendationList");
    el.innerHTML = p.recommendations.map((r) => `<li>${r}</li>`).join("");
  },

  async runWhatIf() {
    const payload = {
      mine_id: this.currentMineId,
      rainfall_mm: Number(document.getElementById("wf-rainfall").value),
      equipment_availability_pct: Number(document.getElementById("wf-equip").value),
      downtime_hours: Number(document.getElementById("wf-downtime").value),
      labor_availability_pct: Number(document.getElementById("wf-labor").value),
      soil_moisture_pct: Number(document.getElementById("wf-soil").value),
      planned_target_tonnes: Number(document.getElementById("wf-target").value),
    };
    const btn = document.getElementById("runWhatIf");
    btn.disabled = true;
    btn.textContent = "Running model…";
    try {
      const result = await Api.post("/api/production/whatif", payload);
      this.renderWhatIfResult(result);
      this.renderDrivers(result);
      this.renderRecommendations(result);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run What-if Scenario";
    }
  },

  renderWhatIfResult(r) {
    const el = document.getElementById("whatifResult");
    el.innerHTML = `
      <h4 class="explain-heading">What-if Result (live model inference)</h4>
      <div class="forecast-grid">
        <div class="forecast-item"><span>Predicted Production</span><b>${r.predicted_production_tonnes.toLocaleString()} t</b></div>
        <div class="forecast-item"><span>Predicted Shortfall</span><b>${r.predicted_shortfall_tonnes.toLocaleString()} t</b></div>
        <div class="forecast-item"><span>Shortfall Probability</span><b>${(r.shortfall_probability * 100).toFixed(1)}%</b></div>
        <div class="forecast-item"><span>Risk Level</span><b class="risk-badge ${r.risk_level}" style="display:inline-block">${r.risk_level}</b></div>
      </div>
    `;
  },
};
