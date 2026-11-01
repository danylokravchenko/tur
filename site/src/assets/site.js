// Theme toggle, code copying, command tabs, chart tooltips and docs search.
(function () {
  var base = document.body.dataset.base || "/";
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Charts animate in when they first scroll into view.
  if ("IntersectionObserver" in window && !reduced) {
    document.documentElement.classList.add("js");
    var reveal = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-in");
        reveal.unobserve(entry.target);
      });
    }, { threshold: 0.15 });
    document.querySelectorAll(".chart").forEach(function (chart) { reveal.observe(chart); });
  }

  // A legend entry hides or shows its series; the scale stays put so the others don't jump.
  document.querySelectorAll("[data-legend]").forEach(function (button) {
    button.addEventListener("click", function () {
      var on = button.getAttribute("aria-pressed") !== "true";
      button.setAttribute("aria-pressed", on ? "true" : "false");
      button.closest(".chart").querySelectorAll("[data-series]").forEach(function (mark) {
        if (mark.dataset.series === button.dataset.legend) mark.classList.toggle("is-off", !on);
      });
    });
  });

  // Docs tables sort by clicking a header. Numbers sort as numbers, whatever units follow them.
  function cellValue(row, index) {
    var text = row.children[index] ? row.children[index].textContent.trim() : "";
    var number = parseFloat(text.replace(/,/g, ""));
    return isNaN(number) ? text.toLowerCase() : number;
  }
  document.querySelectorAll(".prose table").forEach(function (table) {
    var body = table.tBodies[0];
    if (!body || body.rows.length < 3) return;
    // A blank first cell continues the row above, so these rows only make sense in their order.
    var grouped = Array.prototype.some.call(body.rows, function (row) { return row.cells[0].textContent.trim() === ""; });
    if (grouped) return;
    table.querySelectorAll("th").forEach(function (th, index) {
      th.dataset.sort = "none";
      th.tabIndex = 0;
      function sort() {
        var direction = th.dataset.sort === "asc" ? "desc" : "asc";
        table.querySelectorAll("th").forEach(function (other) { other.dataset.sort = "none"; });
        th.dataset.sort = direction;
        Array.prototype.slice.call(body.rows).sort(function (a, b) {
          var x = cellValue(a, index);
          var y = cellValue(b, index);
          var order = typeof x === typeof y ? (x > y ? 1 : x < y ? -1 : 0) : typeof x === "number" ? -1 : 1;
          return direction === "asc" ? order : -order;
        }).forEach(function (row) { body.appendChild(row); });
      }
      th.addEventListener("click", sort);
      th.addEventListener("keydown", function (event) {
        if (event.key === "Enter") sort();
      });
    });
  });

  // The on-this-page list follows the section being read.
  var tocLinks = document.querySelectorAll(".docs__toc a");
  if (tocLinks.length && "IntersectionObserver" in window) {
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        tocLinks.forEach(function (link) {
          link.classList.toggle("is-current", link.getAttribute("href") === "#" + entry.target.id);
        });
      });
    }, { rootMargin: "-70px 0px -75% 0px" });
    tocLinks.forEach(function (link) {
      var target = document.getElementById(link.getAttribute("href").slice(1));
      if (target) spy.observe(target);
    });
  }

  document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
    button.addEventListener("click", function () {
      var next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("tur-theme", next); } catch (e) {}
    });
  });

  function copyButton(button, text) {
    button.addEventListener("click", function () {
      if (!navigator.clipboard) return;
      navigator.clipboard.writeText(text).then(function () {
        var label = button.textContent;
        button.textContent = "Copied";
        setTimeout(function () { button.textContent = label; }, 1400);
      });
    });
  }
  document.querySelectorAll("[data-copy]").forEach(function (button) {
    copyButton(button, button.dataset.copy);
  });
  document.querySelectorAll("pre > code").forEach(function (code) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    copyButton(button, code.textContent);
    code.parentNode.appendChild(button);
  });

  document.querySelectorAll("[data-tabs]").forEach(function (group) {
    var tabs = group.querySelectorAll("[data-tab]");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (other) { other.setAttribute("aria-selected", other === tab ? "true" : "false"); });
        group.querySelectorAll("[data-panel]").forEach(function (panel) {
          panel.hidden = panel.dataset.panel !== tab.dataset.tab;
        });
      });
    });
  });

  // Tooltips work for pointer and keyboard, since the marks are focusable.
  var tip = document.querySelector("[data-tip-box]");
  function showTip(target, x, y) {
    tip.textContent = target.dataset.tip;
    tip.hidden = false;
    var width = tip.offsetWidth;
    tip.style.left = Math.max(8, Math.min(x - width / 2, window.innerWidth - width - 8)) + "px";
    tip.style.top = Math.max(8, y - tip.offsetHeight - 12) + "px";
  }
  document.addEventListener("pointerover", function (event) {
    var target = event.target.closest && event.target.closest("[data-tip]");
    if (target) showTip(target, event.clientX, event.clientY); else tip.hidden = true;
  });
  document.addEventListener("focusin", function (event) {
    var target = event.target.closest && event.target.closest("[data-tip]");
    if (!target) return;
    var box = target.getBoundingClientRect();
    showTip(target, box.left + box.width / 2, box.top);
  });
  document.addEventListener("focusout", function () { tip.hidden = true; });
  window.addEventListener("scroll", function () { tip.hidden = true; }, { passive: true });

  var dialog = document.querySelector("[data-search]");
  var input = document.querySelector("[data-search-input]");
  var results = document.querySelector("[data-search-results]");
  var index = null;

  function openSearch() {
    dialog.showModal();
    input.select();
    if (index) return;
    fetch(base + "search-index.json")
      .then(function (response) { return response.json(); })
      .then(function (entries) {
        index = entries.map(function (entry) {
          entry.lowTitle = entry.title.toLowerCase();
          entry.lowText = entry.text.toLowerCase();
          return entry;
        });
        render();
      })
      .catch(function () { results.innerHTML = '<li class="search__empty">The search index could not load.</li>'; });
  }

  function score(entry, terms) {
    var total = 0;
    for (var i = 0; i < terms.length; i++) {
      var inTitle = entry.lowTitle.indexOf(terms[i]) >= 0;
      var inText = entry.lowText.indexOf(terms[i]) >= 0;
      if (!inTitle && !inText) return 0;
      total += (inTitle ? 10 : 0) + (inText ? 1 : 0);
    }
    return total;
  }

  function snippet(entry, term) {
    var at = Math.max(0, entry.lowText.indexOf(term) - 50);
    return (at > 0 ? "..." : "") + entry.text.slice(at, at + 150);
  }

  function render() {
    results.textContent = "";
    var query = input.value.trim().toLowerCase();
    if (!index || !query) return;
    var terms = query.split(/\s+/);
    var hits = index
      .map(function (entry) { return { entry: entry, score: score(entry, terms) }; })
      .filter(function (hit) { return hit.score > 0; })
      .sort(function (a, b) { return b.score - a.score; })
      .slice(0, 12);
    if (!hits.length) {
      var empty = document.createElement("li");
      empty.className = "search__empty";
      empty.textContent = "Nothing in the docs matches that.";
      results.appendChild(empty);
      return;
    }
    hits.forEach(function (hit) {
      var item = document.createElement("li");
      var link = document.createElement("a");
      link.href = base + hit.entry.url.replace(/^\//, "");
      var title = document.createElement("b");
      title.textContent = hit.entry.title;
      var where = document.createElement("small");
      where.textContent = hit.entry.chapter;
      var text = document.createElement("span");
      text.textContent = snippet(hit.entry, terms[0]);
      link.append(title, where, text);
      link.addEventListener("click", function () { dialog.close(); });
      item.appendChild(link);
      results.appendChild(item);
    });
  }

  document.querySelectorAll("[data-search-open]").forEach(function (button) {
    button.addEventListener("click", openSearch);
  });
  input.addEventListener("input", render);
  input.addEventListener("keydown", function (event) {
    // A search input's native Esc only clears the text; here it should leave the dialog too.
    if (event.key === "Escape") {
      event.preventDefault();
      dialog.close();
      return;
    }
    if (event.key !== "Enter") return;
    event.preventDefault();
    var first = results.querySelector("a");
    if (first) first.click();
  });
  document.addEventListener("keydown", function (event) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (event.key === "/" && !typing && !dialog.open) {
      event.preventDefault();
      openSearch();
    }
  });
  dialog.addEventListener("click", function (event) {
    if (event.target === dialog) dialog.close();
  });
})();
