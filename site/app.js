const fixtureSelect = document.getElementById('fixture');
const chart = document.getElementById('replay-chart');
const takeaway = document.getElementById('chart-takeaway');
const stageInputs = [...document.querySelectorAll('input[name="stage"]')];

function drawReplay() {
  const fixture = fixtureSelect.value;
  const stage = stageInputs.find(input => input.checked).value;
  const conditions = [
    ['lora_no_replay', 'Without replay', 'without-replay'],
    ['lora_replay', 'With replay', 'with-replay'],
  ];
  chart.replaceChildren();
  for (const [method, label, className] of conditions) {
    const values = window.sequenceResults[fixture][method][stage];
    const panel = document.createElement('div');
    panel.className = `chart-panel ${className}`;
    const title = document.createElement('h4');
    title.textContent = label;
    panel.append(title);
    values.forEach((value, index) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      const info = document.createElement('div');
      info.className = 'bar-info';
      const name = document.createElement('span');
      name.textContent = `Task ${['A', 'B', 'C'][index]}`;
      const count = document.createElement('span');
      count.className = 'bar-value';
      count.textContent = `${value} / 64`;
      info.append(name, count);
      const track = document.createElement('div');
      track.className = 'bar-track';
      track.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('div');
      fill.className = 'bar-fill';
      fill.style.width = `${value / 64 * 100}%`;
      track.append(fill);
      row.append(info, track);
      panel.append(row);
    });
    const total = document.createElement('p');
    total.className = 'chart-total';
    const sum = values.reduce((a, b) => a + b, 0);
    total.textContent = `All three tasks: ${sum}/192 (${(sum / 192 * 100).toFixed(1)}%)`;
    panel.append(total);
    chart.append(panel);
  }
  const explanations = {
    initial: 'Same starting scores within this pair. The model has not been trained on these new tasks yet. Some initial answers are correct by chance or prior ability.',
    after_a: 'Both conditions learned A. They match because there are no old lessons to replay yet. Scores on B and C do not count as learning those tasks.',
    after_b: 'Both conditions learned B. With replay, A stays at 64/64. Without old practice, A has already started to slip.',
    after_c: fixture === '303'
      ? 'Both conditions learned C. Without replay, A and B each fell to 8/64. With replay, 127 of 128 old-task answers survived.'
      : 'The confirmation repeats the pattern. Without replay, only 25 of 128 old-task answers survive. With replay, all 128 survive.',
  };
  takeaway.textContent = explanations[stage];
}

fixtureSelect.addEventListener('change', drawReplay);
stageInputs.forEach(input => input.addEventListener('change', drawReplay));
drawReplay();

const navLinks = [...document.querySelectorAll('nav a')];
const chapters = navLinks.map(link => document.querySelector(link.getAttribute('href')));
function highlightChapter() {
  const threshold = window.innerWidth <= 800 ? 170 : 120;
  let current = chapters[0];
  for (const chapter of chapters) {
    if (chapter.getBoundingClientRect().top <= threshold) current = chapter;
  }
  navLinks.forEach(link => {
    if (link.getAttribute('href') === `#${current.id}`) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
let pendingFrame = false;
window.addEventListener('scroll', () => {
  if (pendingFrame) return;
  pendingFrame = true;
  requestAnimationFrame(() => {
    highlightChapter();
    pendingFrame = false;
  });
}, { passive: true });
highlightChapter();
