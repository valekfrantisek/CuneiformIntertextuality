let ORACC = { projects: [], textsByProject: {} };

    function norm(s) { return String(s ?? '').toLowerCase(); }

    async function loadOraccData() {
    const res = await fetch('texts_map.json', { cache: 'no-store' });
    ORACC = await res.json();

    // volitelná přednormalizace pro rychlejší filtrování
    ORACC.projects.forEach(p => { p._nLabel = norm(p.label); p._nValue = norm(p.value); });
    Object.values(ORACC.textsByProject).forEach(arr => {
        arr.forEach(t => { t._nValue = norm(t.value); }); // { t._nLabel = norm(t.label); t._nValue = norm(t.value); });  (we use only values now)
    });
    }

document.addEventListener('DOMContentLoaded', async () => {
    const SERVER_URL = 'http://127.0.0.1:5000';

    await loadOraccData();

    const output = document.getElementById('basic-output');
    const errors = document.getElementById('error-output')

    const analyzeButtonInput = document.getElementById('analyze');
    const analyzeButtonText = document.getElementById('analyze-by-id');
    
    const downloadButtonCSV = document.getElementById('download-button-csv');
    const downloadButtonXLSX = document.getElementById('download-button-xlsx');

    const modeOptions = document.querySelectorAll('.mode_option');
    const processingOptions = document.querySelectorAll('.processing_option');
    const normaliseSignsOpt = document.getElementById('normalize-signs');

    document.querySelectorAll('.mode-option').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-option').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
        });
    });

    document.querySelectorAll('.processing-option').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.processing-option').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
        });
    });

    const textarea = document.getElementById("input-text")

    let currentUploadId = null;
    let mode = null;
    let processing = null;
    let currentAnalysis = null

    downloadButtonCSV.disabled = true;
    downloadButtonXLSX.disabled = true;


    // Special Characters buttons functionality
    document.querySelectorAll("[data-char]").forEach(button => {
        button.addEventListener("click", () => {
        const char = button.dataset.char;
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const text = textarea.value;

        textarea.value = text.slice(0, start) + char + text.slice(end);
        textarea.selectionStart = textarea.selectionEnd = start + char.length;
        textarea.focus();
        });
    });

    modeOptions.forEach(optionMode => {
        optionMode.addEventListener('click', () => {
            modeOptions.forEach(opt => opt.classList.remove('selected'));
            optionMode.classList.add('selected');
        });
    });

    processingOptions.forEach(optionProcessing => {
        optionProcessing.addEventListener('click', () => {
            processingOptions.forEach(opt => opt.classList.remove('selected'));
            optionProcessing.classList.add('selected');
        });
    });

    // Printing example text
    const exampleButton = document.getElementById('generate-example-button');
    const exampleOutput = document.getElementById('input-text-example');
    const exampleTextId = document.getElementById('example-text-id');
    const inputTextExample = document.getElementById('input-text-example');

    exampleButton.addEventListener('click', () => {
        giveAnExample();
    });

    const exampleButtonFull = document.getElementById('show-full-text-button');

    exampleButtonFull.addEventListener('click', () => {
        giveAnExampleFull();
    });

    function giveAnExample() {
        let inputTextID = exampleTextId.value.trim();
        inputTextID = inputTextID.replaceAll('/', '-'); // sanitize input to be processable by flask
        if (!inputTextID) { inputTextID = 'nere-Q009326'; }

        const mode = getSelectedMode();

        const normalizeSigns = !!normaliseSignsOpt?.checked;

        // rychlá kontrola do konzole
        console.log('POST params:', { mode, normalizeSigns, inputTextID: inputTextID.slice(0,80) });

        fetch(`${SERVER_URL}/get_text_example/${encodeURIComponent(inputTextID)}/${encodeURIComponent(mode)}/${encodeURIComponent(normalizeSigns)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(r => r.json())
        .then(data => {
            inputTextExample.innerHTML = `Text example: <i>${data.text_example}</i><br>(input mode: ${data.mode})`;
            errors.textContent = '';
        })
        .catch(err => {
            console.error('Error:', err);
            errors.textContent = 'An error occurred when processing example text.';
        });
    }

    exampleButton.addEventListener('click', () => {
        const inputText = textarea.value;
        exampleOutput.textContent = inputText;
    });

    function giveAnExampleFull() {
        let inputTextID = exampleTextId.value.trim();
        inputTextID = inputTextID.replaceAll('/', '-'); // sanitize input to be processable by flask
        if (!inputTextID) { inputTextID = 'nere-Q009326'; }

        const mode = getSelectedMode();

        const normalizeSigns = !!normaliseSignsOpt?.checked;

        // rychlá kontrola do konzole
        console.log('POST params:', { mode, normalizeSigns, inputTextID: inputTextID.slice(0,80) });

        fetch(`${SERVER_URL}/get_text_example_full/${encodeURIComponent(inputTextID)}/${encodeURIComponent(mode)}/${encodeURIComponent(normalizeSigns)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(r => r.json())
        .then(data => {
            inputTextID = inputTextID.replaceAll('-', '/')
            output.innerHTML = `<p>(input mode: ${data.mode})<br>See text in <a href="http://oracc.org/${inputTextID}" target="_blank">ORACC</a><br>${data.text_example}</p>`;
        })
        .catch(err => {
            console.error('Error:', err);
            errors.textContent = 'An error occurred when processing example text.';
        });
    }

    exampleButton.addEventListener('click', () => {
        const inputText = textarea.value;
        exampleOutput.textContent = inputText;
    });

    // Project selection for the case of searching by text ID

    function filterProjects(query, limit = 50) {
        const q = norm(query);
        const res = ORACC.projects.filter(p =>
            p._nLabel.includes(q) || p._nValue.includes(q)
        );
        return res.slice(0, limit);
    }

    const input    = document.getElementById('combo-input');
    const list     = document.getElementById('combo-list');
    const selVal   = document.getElementById('selected-value');

    if (!input || !list || !selVal) {
      console.warn('Combo: chybí povinné elementy (zkontroluj ID).');
      return;
    }

    let activeIndex = -1;
    let selectedValue = null;
    let isOpen = false;
    let filtered = [];

    const norm = (s) => String(s ?? '').toLowerCase();

    function renderList() {
      list.innerHTML = '';
      filtered.forEach((it, i) => {
        const li = document.createElement('li');
        li.setAttribute('role', 'option');
        li.dataset.value = it.value;
        li.textContent = `${it.label} (${it.value})`;
        if (i === activeIndex) li.classList.add('active');
        if (it.value === selectedValue) li.classList.add('selected');
        list.appendChild(li);
      });
      const open = isOpen && filtered.length > 0;
      list.classList.toggle('open', open);
      input.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    function choose(item) {
        selectedValue = item.value;
        selVal.value = item.value;

        // pošli informaci o zvoleném projektu
        document.dispatchEvent(new CustomEvent('project-selected', {
            detail: { value: item.value, label: item.label }
        }));

        // zobraz label v inputu (nebo dej '' pokud nechceš nic)
        input.value = item.label;

        // ZAVŘÍT a neodkazovat na `items`
        isOpen = false;
        filtered = [];        // nebo: filtered = ORACC.projects.slice(0, 50);
        activeIndex = -1;
        renderList();
    }

    // Filtrování
    input.addEventListener('input', () => {
        isOpen = true;
        filtered = filterProjects(input.value, 50);
        activeIndex = -1;
        renderList();
    });

    // Klávesnice
    input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        activeIndex = Math.min(activeIndex + 1, filtered.length - 1);
        renderList();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        activeIndex = Math.max(activeIndex - 1, 0);
        renderList();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filtered[activeIndex]) choose(filtered[activeIndex]);
      } else if (e.key === 'Escape') {
        list.classList.remove('open');
      }
    });

    // Klik na položku (delegace, stačí jediný listener)
    list.addEventListener('mousedown', (e) => {
        const el = e.target instanceof Element ? e.target : e.target?.parentElement;
        const li = el && el.closest('li[role="option"]');
        if (!li) return;
        e.preventDefault(); // aby blur nezrušil výběr

        const v = li.dataset.value;
        // hledej primárně v aktuálně zobrazeném seznamu
        const item = (typeof filtered !== 'undefined' && Array.isArray(filtered) ? filtered : [])
                        .find(x => x.value === v)
                    || (typeof items !== 'undefined' && Array.isArray(items) ? items : [])
                        .find(x => x.value === v)
                    || (typeof ORACC !== 'undefined' && ORACC.projects || [])
                        .find(x => x.value === v);

        if (item) choose(item);
    });

    input.addEventListener('focus', () => {
        isOpen = true;
        // předvyplň (třeba všechno/50), ať to hned něco ukáže
        filtered = ORACC.projects.slice(0, 50);
        activeIndex = -1;
        renderList();
    });

    input.addEventListener('blur',  () => { setTimeout(() => list.classList.remove('open'), 100); });

    // první render
    const open = isOpen && filtered.length > 0;
    list.classList.toggle('open', open);
    input.setAttribute('aria-expanded', open ? 'true' : 'false');

    // === Text Combo (závislý na vybraném projektu) ===
    (() => {
    const tInput    = document.getElementById('text-combo-input');
    const tList     = document.getElementById('text-combo-list');
    const tSelVal   = document.getElementById('text-selected-value');

    let currentProjectCode = null;

    function setProject(projectCode) {
        currentProjectCode = projectCode || null;
        tItems = currentProjectCode
            ? (ORACC.textsByProject[currentProjectCode] || [])
            : [];

        tFiltered = tItems.slice();
        tActiveIndex = -1;
        tSelectedValue = null;
        tSelVal.value = '';
        tInput.value = '';

        const hasData = tItems.length > 0;
        tInput.disabled = !hasData;
        tInput.placeholder = hasData ? 'Select text…' : 'Select text… (first, select a project)';

        tIsOpen = false;
        tRender();
    }

    if (!tInput || !tList || !tSelVal) return;

    let tItems = [];            // aktuální položky dle projektu
    let tFiltered = [];
    let tActiveIndex = -1;
    let tSelectedValue = null;
    let tIsOpen = false;

    const norm = (s) => String(s ?? '').toLowerCase();

    function tRender() {
        tList.innerHTML = '';
        tFiltered.forEach((it, i) => {
        const li = document.createElement('li');
        li.setAttribute('role', 'option');
        li.dataset.value = it.value;
        li.textContent = `${it.value}`;
        if (i === tActiveIndex) li.classList.add('active');
        if (it.value === tSelectedValue) li.classList.add('selected');
        tList.appendChild(li);
        });
        const open = tIsOpen && tFiltered.length > 0;
        tList.classList.toggle('open', open);
        tInput.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    function tChoose(item) {
        tSelectedValue = item.value;
        tSelVal.value = item.value;

        tInput.value = item.value;

        tIsOpen = false;
        tFiltered = tItems.slice();
        tActiveIndex = -1;
        tRender();
    }

    // Filtrování/ovládání
    tInput.addEventListener('input', () => {
        const q = norm(tInput.value);
        tFiltered = tItems.filter(it => (it._nValue || '').includes(q));
        tActiveIndex = -1;
        tIsOpen = true;
        tRender();
    });

    tInput.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown') {
        e.preventDefault();
        tActiveIndex = Math.min(tActiveIndex + 1, tFiltered.length - 1);
        tRender();
        } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        tActiveIndex = Math.max(tActiveIndex - 1, 0);
        tRender();
        } else if (e.key === 'Enter') {
        e.preventDefault();
        if (tFiltered[tActiveIndex]) tChoose(tFiltered[tActiveIndex]);
        } else if (e.key === 'Escape') {
        tIsOpen = false;
        tRender();
        }
    });

    tInput.addEventListener('focus', () => { 
        if (tItems.length === 0) return;
        tFiltered = tItems.slice(); 
        tIsOpen = true; 
        tRender(); 
    });

    tInput.addEventListener('blur', () => {
        setTimeout(() => { tIsOpen = false; tList.classList.remove('open'); }, 100);
    });

    tList.addEventListener('mousedown', (e) => {
        const el = e.target instanceof Element ? e.target : e.target?.parentElement;
        const li = el && el.closest('li[role="option"]');
        if (!li) return;
        e.preventDefault();

        const v = li.dataset.value;
        const item = (Array.isArray(tFiltered) ? tFiltered : []).find(x => x.value === v)
                    || (Array.isArray(tItems) ? tItems : []).find(x => x.value === v);

        if (item) tChoose(item);
    });

    // Napojení na události z PROJECT comboboxu:
    document.addEventListener('project-selected', (e) => {
        setProject(e.detail.value);
    });

    // start: bez projektu je disabled
    setProject(null);
    })();

    // parsing settings -----------------------------------------------------------------

    const winLenEl = document.getElementById('window-length');
    const strideEl = document.getElementById('stride');
    const edDistanceEl = document.getElementById('ed_distance');

    function getParsingParams() {
        const windowLength = parseInt(winLenEl.value, 10);
        const stride = parseInt(strideEl.value, 10);
        const edDistance = parseInt(edDistanceEl.value, 10);

        if (!Number.isInteger(windowLength) || windowLength <= 0) {
            return { error: 'Window length must be a positive integer.' };
        }
        if (!Number.isInteger(stride) || stride <= 0) {
            return { error: 'Stride must be a positive integer.' };
        }
        if (stride > windowLength) {
            return { error: 'Stride must not exceed window length.' };
        }
        return { windowLength, stride, edDistance };
    }

    const ignoreSelfOpt = document.getElementById('ignore-self');
    const ignoreCoreProjectOpt = document.getElementById('ignore-core-project');

    // Loading functions ---------------------------------------------------------------
    const loadingEl    = document.getElementById('loading');
    const loadingMsgEl = document.getElementById('loading-msg');
    const loadingDots  = document.getElementById('loading-dots');

    let __dotsTimer = null;

    function showLoading(msg = 'Analyzing') {
        if (loadingMsgEl) loadingMsgEl.textContent = msg;
        if (loadingEl) loadingEl.hidden = false;
        let i = 0;
        clearInterval(__dotsTimer);
        __dotsTimer = setInterval(() => {
            loadingDots.textContent = '.'.repeat((i++ % 3) + 1);
        }, 300);

        // volitelně: zablokuj akční tlačítka během běhu
        analyzeButtonInput  && (analyzeButtonInput.disabled = true);
        analyzeButtonText   && (analyzeButtonText.disabled  = true);
        downloadButtonCSV   && (downloadButtonCSV.disabled  = true);
        downloadButtonXLSX  && (downloadButtonXLSX.disabled = true);
        exampleButton       && (exampleButton.disabled      = true);
    }

    function hideLoading() {
    clearInterval(__dotsTimer);
        __dotsTimer = null;
        if (loadingDots) loadingDots.textContent = '';
        if (loadingEl) loadingEl.hidden = true;

        // re-enable tlačítka
        analyzeButtonInput && (analyzeButtonInput.disabled = false);
        analyzeButtonText  && (analyzeButtonText.disabled  = false);
        exampleButton      && (exampleButton.disabled      = false);
    }

    // ANALYSIS ------------------------------------------------------------------------

    let currentInputID = null;
    let currentDownloadID = null;

    analyzeButtonInput.addEventListener('click', (event) => {
        event.preventDefault();
        analyzeIntertextualityInput();
    });

    analyzeButtonText.addEventListener('click', (event) => {
        event.preventDefault();
        analyzeIntertextualityText();
    });
    
    function getSelectedMode() {
        const el = document.querySelector('.mode-option.selected');
        return el ? (el.dataset.mode || 'normalized') : 'normalized';
    }

    function getSelectedProcessing() {
        const el = document.querySelector('.processing-option.selected');
        return el ? (el.dataset.processing || 'edit_distance_inner') : 'edit_distance_inner';
    }

    function analyzeIntertextualityInput() {
        
        showLoading('Analysis in progress')

        const inputText = textarea.value.trim();
        if (!inputText) { errors.textContent = 'Input text cannot be empty.'; return; }

        const parsingParams = getParsingParams();
        if (parsingParams.error) { errors.textContent = parsingParams.error; return; }

        const mode = getSelectedMode();
        const processing = getSelectedProcessing();

        const query = encodeURIComponent(inputText);
        const maxTotalEd = parsingParams.edDistance;

        const normalizeSigns     = !!normaliseSignsOpt?.checked;

        // rychlá kontrola do konzole
        console.log('POST params:', { mode, processing, maxTotalEd, query: inputText.slice(0,80) });

        fetch(`${SERVER_URL}/analyse_input/${encodeURIComponent(mode)}/${encodeURIComponent(processing)}/${maxTotalEd}/${query}/${normalizeSigns}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(r => r.json())
        .then(data => {
            currentAnalysis = data.results_html;
            currentDownloadID = data.data_for_download;
            console.log('Current Download ID:', currentDownloadID);
            currentInputID = data.input_id;
            output.innerHTML = data.results_html || '';
            errors.textContent = '';
            downloadButtonCSV.disabled = false;
            downloadButtonXLSX.disabled = false;
        })
        .catch(err => {
            console.error('Error:', err);
            errors.textContent = 'An error occurred while analyzing the input.';
        })

        .finally(() => hideLoading());
    }

    function getSelectedTextID() {
        const tSelVal = document.getElementById('text-selected-value');
        const tID = tSelVal?.value.trim();
        console.log('Text ID element:', tID);
        return tID || null;
    }

    function getSelectedTextID() {
        const tSelVal = document.getElementById('text-selected-value');
        const tID = tSelVal?.value.trim();
        console.log('Text ID element:', tID);
        return tID || null;
    }

    function analyzeIntertextualityText() {
        
        showLoading('Analysis in progress')

        const parsingParams = getParsingParams();
        if (parsingParams.error) { errors.textContent = parsingParams.error; return; }

        const mode = getSelectedMode();
        const processing = getSelectedProcessing();

        let queryTextId = getSelectedTextID();
        console.log('Selected Text ID:', queryTextId);
        queryTextId = queryTextId.replaceAll('/', '-'); // sanitize input to be processable by flask

        const maxTotalEd = parsingParams.edDistance;
        const windowLength = parsingParams.windowLength;
        const stride = parsingParams.stride;

        const normalizeSigns = !!normaliseSignsOpt?.checked;

        const ignoreSelf = !!ignoreSelfOpt?.checked;
        const ignoreCoreProject = !!ignoreCoreProjectOpt?.checked;

        // rychlá kontrola do konzole
        console.log('POST params:', { mode, processing, maxTotalEd, queryTextId, parsingParams });

        fetch(`${SERVER_URL}/analyse_text_by_id/${encodeURIComponent(queryTextId)}/${encodeURIComponent(mode)}/${encodeURIComponent(processing)}/${encodeURIComponent(maxTotalEd)}/${encodeURIComponent(normalizeSigns)}/${encodeURIComponent(windowLength)}/${encodeURIComponent(stride)}/${encodeURIComponent(ignoreSelf)}/${encodeURIComponent(ignoreCoreProject)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(r => r.json())
        .then(data => {
            currentAnalysis = data.results_html;
            currentDownloadID = data.data_for_download;
            console.log('Current Download ID:', currentDownloadID);
            currentInputID = data.input_id;
            output.innerHTML = data.results_html || '';
            errors.textContent = '';
            downloadButtonCSV.disabled = false;
            downloadButtonXLSX.disabled = false;
        })
        .catch(err => {
            console.error('Error:', err);
            errors.textContent = 'An error occurred while analyzing the input.';
        })

        .finally(() => hideLoading());
    }

    // DOWNLOADING OPTIONS ---------------------------------------------

    downloadButtonCSV.addEventListener('click', (event) => {
        event.preventDefault();
        downloadFile('csv');
    });

    downloadButtonXLSX.addEventListener('click', (event) => {
        event.preventDefault();
        downloadFile('xlsx');
    });

    function downloadFile(format) {
        let endpoint = `${SERVER_URL}/download_${format}/${currentDownloadID}`;
        window.location.href = endpoint;
    }

});