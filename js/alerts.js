const Alerts = {
  async init() {
    const data = await Api.get("/api/alerts");
    const el = document.getElementById("alertsList");
    el.innerHTML = "";
    if (data.alerts.length === 0) {
      el.innerHTML = '<div class="empty-state">No active alerts.</div>';
      return;
    }
    data.alerts.forEach((a) => {
      const item = document.createElement("div");
      item.className = `alert-item ${a.severity}`;
      item.innerHTML = `
        <span class="alert-icon"></span>
        <div>
          <div class="alert-title">${a.title}</div>
          <div class="alert-message">${a.message}</div>
        </div>
      `;
      el.appendChild(item);
    });
  },
};
