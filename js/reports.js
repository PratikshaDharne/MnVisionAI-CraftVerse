const Reports = {
  lastReportText: "",

  init() {
    document.getElementById("generateReportBtn").addEventListener("click", () => this.generate());
    document.getElementById("downloadReportBtn").addEventListener("click", () => this.download());
  },

  async generate() {
    const btn = document.getElementById("generateReportBtn");
    btn.disabled = true;
    btn.textContent = "Generating…";
    try {
      const text = await Api.getText("/api/report");
      this.lastReportText = text;
      document.getElementById("reportOutput").textContent = text;
      document.getElementById("downloadReportBtn").disabled = false;
    } catch (e) {
      document.getElementById("reportOutput").textContent = "Failed to generate report: " + e.message;
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate Report";
    }
  },

  download() {
    const blob = new Blob([this.lastReportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `MnVision_Report_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  },
};
