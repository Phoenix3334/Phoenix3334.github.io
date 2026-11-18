/* Load public GitHub Stars on demand.
   GitHub's public API is unauthenticated and rate-limited; the curated links
   on the page remain usable when the live API is unavailable. */
(function () {
  "use strict";

  const root = document.getElementById("star-explorer");
  if (!root) return;

  const list = document.getElementById("star-results");
  const search = document.getElementById("star-search");
  const filter = document.getElementById("star-filter");
  const status = document.getElementById("star-status");
  const moreButton = document.getElementById("star-more");

  const all = [];
  const seen = new Set();
  const pageSize = 100;
  let page = 1;
  let loading = false;
  let hasMore = true;
  let started = false;

  function isLearningResource(repo) {
    // Convenience keyword filter, not a claim about the user's categories.
    const content = [
      repo.full_name, repo.description,
      Array.isArray(repo.topics) ? repo.topics.join(" ") : ""
    ].join(" ");
    return /infra|roadmap|curriculum|learning|guide|tutorial|course|awesome|book|playbook|lecture|notes|mlsys|systems|educat|handbook|gpu|cuda|triton|llm|training|inference/i.test(content);
  }

  function shown() {
    const q = search.value.trim().toLowerCase();
    const onlyLearning = filter.value === "learning";
    return all.filter(function (repo) {
      const blob = (repo.full_name + " " + (repo.description || "")).toLowerCase();
      return (!onlyLearning || isLearningResource(repo)) && (!q || blob.includes(q));
    });
  }

  function draw() {
    const matches = shown();
    list.replaceChildren();
    const fragment = document.createDocumentFragment();
    for (const repo of matches) {
      const a = document.createElement("a");
      a.className = "star-result";
      a.href = repo.html_url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";

      const heading = document.createElement("span");
      heading.className = "star-result-name";
      heading.textContent = repo.full_name;
      const desc = document.createElement("span");
      desc.className = "star-result-desc";
      desc.textContent = repo.description || "GitHub repository";
      const arrow = document.createElement("span");
      arrow.className = "star-result-arrow";
      arrow.textContent = "↗";
      arrow.setAttribute("aria-hidden", "true");

      a.append(heading, desc, arrow);
      fragment.appendChild(a);
    }
    list.appendChild(fragment);
    if (!loading) {
      status.textContent = "已加载 " + all.length + " 个公开 Stars，当前显示 " + matches.length +
        " 个" + (hasMore ? " · 点击「加载更多」查看下一页" : " · 已到最后一页") +
        "。筛选仅针对已加载的项目。";
    }
  }

  async function loadNext() {
    if (loading || !hasMore) return;
    loading = true;
    moreButton.disabled = true;
    status.textContent = "正在读取第 " + page + " 页 GitHub Stars…";
    try {
      const url = "https://api.github.com/users/Phoenix3334/starred?per_page=" +
        pageSize + "&page=" + page;
      const response = await fetch(url, {
        headers: { "Accept": "application/vnd.github+json" }
      });
      if (!response.ok) throw new Error("GitHub API HTTP " + response.status);
      const rows = await response.json();
      if (!Array.isArray(rows)) throw new Error("Unexpected API response");
      for (const repo of rows) {
        if (!repo || !repo.full_name || !repo.html_url || seen.has(repo.full_name)) continue;
        seen.add(repo.full_name);
        all.push({
          full_name: repo.full_name,
          html_url: repo.html_url,
          description: repo.description || "",
          topics: Array.isArray(repo.topics) ? repo.topics : []
        });
      }
      hasMore = rows.length === pageSize;
      page += 1;
      if (rows.length === 0) hasMore = false;
      draw();
    } catch (error) {
      status.textContent = "读取失败（" + error.message +
        "）。可能遇到 GitHub API 访问限制；可使用下方链接打开完整公开 Stars 页面。";
    } finally {
      loading = false;
      moreButton.disabled = false;
      moreButton.hidden = !hasMore;
    }
  }

  root.addEventListener("toggle", function () {
    if (!root.open || started) return;
    started = true;
    loadNext();
  });
  search.addEventListener("input", draw);
  filter.addEventListener("change", draw);
  moreButton.addEventListener("click", loadNext);
}());
