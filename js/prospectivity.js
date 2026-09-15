const Prospectivity = {
  map: null,
  cellLayer: null,
  mineLayer: null,
  summary: null,

  zoneColor(zone) {
    if (zone === "High") return "#b3382c";
    if (zone === "Medium") return "#b8860b";
    return "#9db3c4";
  },

  async init() {
    this.map = L.map("map", {
      preferCanvas: true,
      zoomControl: true,
      attributionControl: false,
    }).setView([21.35, 79.6], 8);

    // Clean, light, professional basemap.
    // We use OpenStreetMap's standard tile server — the most universally
    // reliable free tile source (no key ever required). If for any reason
    // tiles can't load (corporate firewall, offline demo, etc.) we fall back
    // to a plain neutral background rather than showing broken/placeholder
    // tile images, so the dashboard always looks clean.
    const tileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 16,
      subdomains: "abc",
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(this.map);

    let tileErrorCount = 0;
    tileLayer.on("tileerror", () => {
      tileErrorCount += 1;
      if (tileErrorCount > 6) {
        this.map.removeLayer(tileLayer);
        document.getElementById("map").style.background = "#eef2f6";
      }
    });

    this.cellLayer = L.layerGroup().addTo(this.map);
    this.mineLayer = L.layerGroup().addTo(this.map);

    await this.loadSummary();
    await this.loadGrid();
    this.renderOccurrences();
  },

  async loadSummary() {
    this.summary = await Api.get("/api/prospectivity/summary");
    document.getElementById("areaChip").textContent = this.summary.area_name;
    document.getElementById("stat-high").textContent = this.summary.zone_counts.High;
    document.getElementById("stat-medium").textContent = this.summary.zone_counts.Medium;
    document.getElementById("stat-low").textContent = this.summary.zone_counts.Low;
    document.getElementById("metric-auc").textContent = this.summary.metrics.test_roc_auc.toFixed(3);
    document.getElementById("metric-ap").textContent = this.summary.metrics.test_average_precision.toFixed(3);

    this.renderFeatureImportance();
  },

  renderFeatureImportance() {
    const el = document.getElementById("featureImportanceChart");
    el.innerHTML = "";
    const entries = Object.entries(this.summary.feature_importances)
      .sort((a, b) => b[1] - a[1]);
    const max = Math.max(...entries.map((e) => e[1]));
    entries.forEach(([key, val]) => {
      const label = this.summary.feature_labels[key] || key;
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div>${label}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(val / max) * 100}%"></div></div>
        <div>${(val * 100).toFixed(1)}%</div>
      `;
      el.appendChild(row);
    });
  },

  async loadGrid() {
    const data = await Api.get("/api/prospectivity/grid");
    this.cellLayer.clearLayers();
    data.cells.forEach((c) => {
      // Skip most "Low" cells for rendering performance/clarity, keep all Medium/High
      if (c.zone === "Low" && Math.random() > 0.12) return;

      // Invisible larger hit-target underneath the visible dot, so clicking
      // anywhere near a point (not just its exact 2-5px centre) selects it.
      const hitTarget = L.circleMarker([c.lat, c.lon], {
        radius: 10,
        stroke: false,
        fill: true,
        fillOpacity: 0.001, // effectively invisible but still hit-testable
      });
      hitTarget.on("click", () => this.selectPoint(c.lat, c.lon));
      hitTarget.addTo(this.cellLayer);

      const marker = L.circleMarker([c.lat, c.lon], {
        radius: c.zone === "High" ? 5 : c.zone === "Medium" ? 4 : 2.5,
        color: this.zoneColor(c.zone),
        fillColor: this.zoneColor(c.zone),
        fillOpacity: c.zone === "Low" ? 0.35 : 0.75,
        weight: c.zone === "Low" ? 0 : 1,
        opacity: 0.9,
        interactive: false, // clicks are handled by the larger hitTarget above
      });
      marker.addTo(this.cellLayer);
    });
  },

  renderOccurrences() {
    this.summary.known_occurrences.forEach((o) => {
      const icon = L.divIcon({
        className: "",
        html: `<div style="width:14px;height:14px;background:#2563a8;border:2px solid white;border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,.3);"></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });
      const marker = L.marker([o.lat, o.lon], { icon });
      marker.bindPopup(`<b>${o.name}</b><br/>Documented manganese occurrence`);
      marker.on("click", () => this.selectPoint(o.lat, o.lon));
      marker.addTo(this.mineLayer);
    });
  },

  async selectPoint(lat, lon) {
    try {
      const point = await Api.get(`/api/prospectivity/point?lat=${lat}&lon=${lon}`);
      document.getElementById("pointDetailEmpty").classList.add("hidden");
      const content = document.getElementById("pointDetailContent");
      content.classList.remove("hidden");

      document.getElementById("pd-score").textContent = point.prospectivity_score.toFixed(2);
      const zoneBadge = document.getElementById("pd-zone");
      zoneBadge.textContent = point.zone + " Prospectivity";
      zoneBadge.className = "zone-badge " + point.zone;
      document.getElementById("pd-confidence").textContent = (point.confidence * 100).toFixed(0) + "%";
      document.getElementById("pd-coords").textContent = `${point.lat.toFixed(3)}, ${point.lon.toFixed(3)}`;
      document.getElementById("pd-nearest").textContent =
        `${point.nearest_known_occurrence.name} (${point.nearest_known_occurrence.distance_km} km)`;

      const explainEl = document.getElementById("pd-explanation");
      explainEl.innerHTML = "";
      point.explanation.forEach((f) => {
        const pct = Math.min(100, Math.abs(f.contribution) * 40 + 8);
        const row = document.createElement("div");
        row.className = "explain-row";
        row.innerHTML = `
          <div class="explain-row-top"><span>${f.label}</span><span class="val">value: ${f.value.toFixed(2)} (avg ${f.population_mean.toFixed(2)})</span></div>
          <div class="explain-bar-track"><div class="explain-bar-fill ${f.contribution >= 0 ? "pos" : "neg"}" style="width:${pct}%"></div></div>
        `;
        explainEl.appendChild(row);
      });
    } catch (e) {
      console.error(e);
    }
  },
};
