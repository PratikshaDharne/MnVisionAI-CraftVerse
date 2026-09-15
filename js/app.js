function setupTabs() {
  const buttons = document.querySelectorAll(".nav-btn");
  const panels = document.querySelectorAll(".tab-panel");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "alerts") Alerts.init();
    });
  });
}

async function bootstrap() {
  setupTabs();
  Reports.init();
  await checkApiHealth();
  try {
    await Prospectivity.init();
    await Production.init();
  } catch (e) {
    console.error("Failed to initialise MnVision dashboard:", e);
    document.getElementById("apiStatus").innerHTML = '<span class="dot"></span> backend offline';
    document.getElementById("apiStatus").classList.add("offline");
  }
}

document.addEventListener("DOMContentLoaded", bootstrap);
