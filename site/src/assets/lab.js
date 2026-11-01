// Interactive tools. Every number comes from the book's tables through the #lab-data JSON.
(function () {
  var data = JSON.parse(document.getElementById("lab-data").textContent);
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function fmt(value) {
    if (value >= 1000) return Math.round(value).toLocaleString("en-US");
    if (value >= 100) return String(Math.round(value));
    return String(Number(value.toPrecision(3)));
  }

  // A row of mutually exclusive buttons. Calls back with the chosen index.
  function segmented(label, options, selected, onChange) {
    var field = el("div", "lab__field");
    field.appendChild(el("span", "", label));
    var group = el("div", "seg");
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", label);
    options.forEach(function (option, index) {
      var button = el("button", "", String(option));
      button.type = "button";
      button.setAttribute("aria-pressed", index === selected ? "true" : "false");
      button.addEventListener("click", function () {
        Array.prototype.forEach.call(group.children, function (other) {
          other.setAttribute("aria-pressed", other === button ? "true" : "false");
        });
        onChange(index);
      });
      group.appendChild(button);
    });
    field.appendChild(group);
    return field;
  }

  function meter(label, value, fraction, color) {
    var box = el("div", "meter");
    var head = el("div", "meter__head");
    head.append(el("span", "", label), el("b", "", value));
    var track = el("div", "meter__track");
    var fill = el("div", "meter__fill");
    fill.style.setProperty("--w", (fraction * 100).toFixed(1) + "%");
    if (color) fill.style.setProperty("--c", "var(--" + color + ")");
    track.appendChild(fill);
    box.append(head, track);
    return box;
  }

  function header(root, title, note) {
    root.append(el("h3", "lab__title", title), el("p", "lab__note", note));
  }

  // Replays measured rates in real time: each request gets its share of the total tok/s.
  function race(root) {
    var tokensPerRequest = 64;
    var scenario = 0;
    var seqIndex = data.race[0].sequences.length - 1;
    var frame = 0;
    header(root, "Token race", "Each request generates " + tokensPerRequest + " tokens at its share of the engine's measured total.");
    var controls = el("div", "lab__controls");
    var lanes = el("div");
    root.append(controls, lanes);

    function lane(name, color, rate, sequences) {
      var box = el("div", "race__lane");
      var head = el("div", "race__head");
      var count = el("span", "race__count");
      head.append(el("b", "", name + ", " + fmt(rate) + " tok/s"), count);
      var rows = el("div", "race__seqs");
      var fills = [];
      for (var i = 0; i < sequences; i++) {
        var row = el("div", "race__seq");
        var fill = el("i");
        fill.style.setProperty("--c", "var(--" + color + ")");
        row.appendChild(fill);
        rows.appendChild(row);
        fills.push(fill);
      }
      box.append(head, rows);
      lanes.appendChild(box);
      return { rate: rate, fills: fills, count: count, total: tokensPerRequest * sequences };
    }

    function paint(item, elapsed) {
      var done = Math.min(item.total, item.rate * elapsed);
      var fraction = done / item.total;
      item.fills.forEach(function (fill) { fill.style.transform = "scaleX(" + fraction + ")"; });
      item.count.textContent = done >= item.total
        ? item.total + " tokens in " + (item.total / item.rate).toFixed(2) + " s"
        : Math.floor(done) + " / " + item.total + " tokens";
      return done >= item.total;
    }

    function run() {
      cancelAnimationFrame(frame);
      lanes.textContent = "";
      var current = data.race[scenario];
      var sequences = current.sequences[seqIndex];
      var items = [
        lane("Tur", "s1", current.tur[seqIndex], sequences),
        lane("llama.cpp", "s2", current.llama[seqIndex], sequences),
      ];
      if (reduced) {
        items.forEach(function (item) { paint(item, Infinity); });
        return;
      }
      var start = performance.now();
      (function tick(now) {
        var elapsed = (now - start) / 1000;
        var finished = items.map(function (item) { return paint(item, elapsed); });
        if (finished.indexOf(false) >= 0) frame = requestAnimationFrame(tick);
      })(start);
    }

    function buildControls() {
      controls.textContent = "";
      var current = data.race[scenario];
      seqIndex = Math.min(seqIndex, current.sequences.length - 1);
      var play = el("button", "lab__play", "Race again");
      play.type = "button";
      play.addEventListener("click", run);
      controls.append(
        segmented("Model and Mac", data.race.map(function (s) { return s.name; }), scenario, function (index) {
          scenario = index;
          buildControls();
          run();
        }),
        segmented("Sequences", current.sequences, seqIndex, function (index) {
          seqIndex = index;
          run();
        }),
        play
      );
    }

    buildControls();
    // Starts when seen, so the race isn't over by the time the reader scrolls to it.
    if ("IntersectionObserver" in window) {
      var seen = new IntersectionObserver(function (entries) {
        if (!entries[0].isIntersecting) return;
        seen.disconnect();
        run();
      }, { threshold: 0.3 });
      seen.observe(root);
    } else {
      run();
    }
  }

  function serving(root) {
    var scenario = 0;
    var index = 0;
    header(root, "Serving explorer", "Drag to change how many requests arrive at once, and watch throughput trade against latency.");
    var controls = el("div", "lab__controls");
    var out = el("div");
    root.append(controls, out);

    function render() {
      var current = data.serving[scenario];
      out.textContent = "";
      var big = el("p", "lab__big", current.requests[index] + (current.requests[index] === "1" ? " request" : " requests"));
      out.appendChild(big);
      var cols = el("div", "lab__cols");
      current.metrics.forEach(function (metric) {
        var max = Math.max.apply(null, metric.values);
        var value = metric.values[index];
        cols.appendChild(meter(metric.label, fmt(value) + " " + metric.unit, value / max, metric.best === "high" ? "s1" : "s2"));
      });
      out.appendChild(cols);
    }

    function build() {
      controls.textContent = "";
      var current = data.serving[scenario];
      index = Math.min(index, current.requests.length - 1);
      var field = el("label", "lab__field");
      field.appendChild(el("span", "", "Requests at once"));
      var slider = el("input");
      slider.type = "range";
      slider.min = 0;
      slider.max = current.requests.length - 1;
      slider.step = 1;
      slider.value = index;
      slider.setAttribute("aria-valuetext", current.requests[index] + " requests");
      slider.addEventListener("input", function () {
        index = Number(slider.value);
        slider.setAttribute("aria-valuetext", current.requests[index] + " requests");
        render();
      });
      field.appendChild(slider);
      controls.append(
        segmented("Model and Mac", data.serving.map(function (s) { return s.name; }), scenario, function (next) {
          scenario = next;
          build();
          render();
        }),
        field
      );
    }

    build();
    render();
  }

  function speculation(root) {
    var modes = Object.keys(data.speculation);
    var mode = 0;
    var prompt = 2;
    var seq = 0;
    var labels = ["Normal", "Prompt lookup", "Lookup + MTP head"];
    var colors = ["s-muted", "s2", "s1"];
    var rows = data.speculation[modes[0]];
    var prompts = rows.map(function (r) { return r.prompt; }).filter(function (p, i, all) { return all.indexOf(p) === i; });
    var counts = rows.map(function (r) { return r.sequences; }).filter(function (p, i, all) { return all.indexOf(p) === i; });

    header(root, "Speculation lab", "Drafts pay off when they're often right. Try repetitive text, then new prose. Qwen3.5-0.8B on a 16 GB M1, with output identical to normal decoding.");
    var controls = el("div", "lab__controls");
    var out = el("div");
    root.append(controls, out);

    function render() {
      var row = data.speculation[modes[mode]].find(function (r) {
        return r.prompt === prompts[prompt] && r.sequences === counts[seq];
      });
      var best = Math.max.apply(null, row.values);
      var max = Math.max.apply(null, data.speculation[modes[mode]].map(function (r) { return Math.max.apply(null, r.values); }));
      out.textContent = "";
      var big = el("p", "lab__big", (best / row.values[0]).toFixed(2) + "×");
      big.appendChild(el("small", "", "best speedup over normal decoding here"));
      out.appendChild(big);
      row.values.forEach(function (value, i) {
        out.appendChild(meter(labels[i], fmt(value) + " tok/s", value / max, colors[i]));
      });
    }

    controls.append(
      segmented("Prompt", prompts, prompt, function (i) { prompt = i; render(); }),
      segmented("Sequences", counts, seq, function (i) { seq = i; render(); }),
      segmented("Sampling", modes, mode, function (i) { mode = i; render(); })
    );
    render();
  }

  function moeSpeculation(root) {
    var rows = data.moeSpeculation;
    var counts = rows.map(function (r) { return r.requests; }).filter(function (c, i, all) { return all.indexOf(c) === i; });
    var max = Math.max.apply(null, rows.map(function (r) { return Math.max(r.normal, r.mtp); }));
    header(root, "Speculation with the model's own head", "Normal decoding against prompt lookup plus the MTP head, sampled, 128 output tokens per request. Output matched normal decoding for every request compared.");
    var controls = el("div", "lab__controls");
    var out = el("div", "lab__cols");
    root.append(controls, out);

    function render(index) {
      out.textContent = "";
      rows.filter(function (r) { return r.requests === counts[index]; }).forEach(function (row) {
        var group = el("div");
        group.appendChild(el("p", "lab__big", row.speedup));
        group.lastChild.appendChild(el("small", "", row.prompt));
        group.append(
          meter("Normal", fmt(row.normal) + " tok/s", row.normal / max, "s-muted"),
          meter("Lookup + MTP head", fmt(row.mtp) + " tok/s", row.mtp / max, "s1")
        );
        out.appendChild(group);
      });
    }

    controls.appendChild(segmented("Requests", counts, 0, render));
    render(0);
  }

  function formats(root) {
    var choice = 0;
    header(root, "Pick a weight format", "Prepacked artifacts for Qwen3.5-0.8B on a 16 GB M1. Fewer bytes load faster and decode one sequence faster, and they lose some agreement with F32 weights.");
    var controls = el("div", "lab__controls");
    var out = el("div");
    root.append(controls, out);

    function tile(value, label) {
      var box = el("div", "lab__tile", label);
      box.prepend(el("b", "", value));
      return box;
    }

    function render() {
      var format = data.formats[choice];
      out.textContent = "";
      var tiles = el("div", "lab__tiles");
      tiles.append(
        tile(format.load + " s", "load time"),
        tile(fmt(format.memory) + " MB", "weight memory"),
        tile(format.peak + " GB", "peak RSS"),
        tile(fmt(format.decode1), "tok/s, 1 sequence"),
        tile(fmt(format.decode8), "tok/s, 8 sequences")
      );
      out.appendChild(tiles);
      out.appendChild(el("p", "lab__note", "How often the most likely token matches F32:"));
      var cols = el("div", "lab__cols");
      format.quality.forEach(function (item) {
        cols.appendChild(meter(item[0], item[1] + "%", item[1] / 100, "s1"));
      });
      out.appendChild(cols);
    }

    controls.appendChild(segmented("Format", data.formats.map(function (f) { return f.name; }), choice, function (i) {
      choice = i;
      render();
    }));
    render();
  }

  function chooser(root) {
    var needs = el("div", "chooser__needs");
    var result = el("p", "lab__big");
    result.setAttribute("aria-live", "polite");
    function pick(index, button) {
      Array.prototype.forEach.call(needs.children, function (other) {
        other.setAttribute("aria-pressed", other === button ? "true" : "false");
      });
      result.textContent = "Use " + data.chooser[index].engine;
    }
    data.chooser.forEach(function (item, index) {
      var button = el("button", "", item.need);
      button.type = "button";
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", function () { pick(index, button); });
      needs.appendChild(button);
    });
    root.append(needs, result);
    pick(0, needs.children[0]);
  }

  var tools = { race: race, serving: serving, speculation: speculation, moeSpeculation: moeSpeculation, formats: formats, chooser: chooser };
  document.querySelectorAll("[data-lab]").forEach(function (root) {
    tools[root.dataset.lab](root);
  });
})();
