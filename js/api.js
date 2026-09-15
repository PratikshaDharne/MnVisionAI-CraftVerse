const Api = {
  async get(path) {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
    return res.json();
  },
  async getText(path) {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
    return res.text();
  },
  async post(path, body) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
    return res.json();
  },
};

async function checkApiHealth() {
  const chip = document.getElementById("apiStatus");
  try {
    await Api.get("/api/health");
    chip.innerHTML = '<span class="dot"></span> backend live';
    chip.classList.remove("offline");
  } catch (e) {
    chip.innerHTML = '<span class="dot"></span> backend offline';
    chip.classList.add("offline");
  }
}
