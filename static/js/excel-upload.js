const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("excel-file");
const dropText = document.getElementById("drop-text");
const fileCaption = document.getElementById("file-caption");

const form = document.querySelector(".needs-validation");
const fileFeedback = document.getElementById("file-invalid-feedback");

fileFeedback.style.display = "none";

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");

    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        dropText.textContent = e.dataTransfer.files[0].name;
        fileCaption.textContent = "";
        dropZone.classList.add("uploaded");
        fileFeedback.style.display = "none";
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        dropText.textContent = fileInput.files[0].name;
        fileCaption.textContent = "";
        dropZone.classList.add("uploaded");
        fileFeedback.style.display = "none";
    }
});

if (form) {
    form.addEventListener("submit", (event) => {
        let fileSelected = fileInput.files && fileInput.files.length > 0;

        const emailInput = document.getElementById("user-email");
        const identifierInput = document.getElementById("signal-name");

        if (emailInput.checkValidity()) {
            emailInput.classList.add("is-valid");
            emailInput.classList.remove("is-invalid");
        } else {
            emailInput.classList.add("is-invalid");
            emailInput.classList.remove("is-valid");
        }

        if (identifierInput.checkValidity()) {
            identifierInput.classList.add("is-valid");
            identifierInput.classList.remove("is-invalid");
        } else {
            identifierInput.classList.add("is-invalid");
            identifierInput.classList.remove("is-valid");
        }

        if (!form.checkValidity() || !fileSelected) {
            event.preventDefault();
            event.stopPropagation();

            if (!fileSelected) {
                fileFeedback.style.display = "block";
            }
        }

        form.classList.add("was-validated");
    });
}

(() => {
    const form = document.querySelector('form.needs-validation');
    const minEl = document.getElementById('mc-min');
    const maxEl = document.getElementById('mc-max');
    const rangeFb = document.getElementById('mc-range-feedback');

    if (!form || !minEl || !maxEl || !rangeFb) return;

    const readNum = (el) => {
        const v = el.value.trim();
        if (v === '') return null; // leer = kein Filter
        const num = Number(v);
        return Number.isFinite(num) ? num : null;
    };

    const clearVisual = (el) => {
        el.setCustomValidity('');
        el.classList.remove('is-valid', 'is-invalid');
    };

    const setInvalidVisual = (el) => {
        el.classList.add('is-invalid');
        el.classList.remove('is-valid');
    };

    const setValidIfFilled = (el) => {
        if (el.value.trim() !== '') {
            el.classList.add('is-valid');
            el.classList.remove('is-invalid');
        } else {
            el.classList.remove('is-valid', 'is-invalid');
        }
    };

    function validateOnSubmit() {
        // Reset
        clearVisual(minEl);
        clearVisual(maxEl);
        rangeFb.textContent = '';
        rangeFb.style.display = '';

        let ok = true;
        const msgs = [];

        const minVal = readNum(minEl);
        const maxVal = readNum(maxEl);

        // Bereichsprüfung (0..100) pro Feld
        if (minVal !== null && (minVal < 0 || minVal > 100)) {
            ok = false;
            setInvalidVisual(minEl);
            msgs.push('Min has to be between 0 and 100.');
        }
        if (maxVal !== null && (maxVal < 0 || maxVal > 100)) {
            ok = false;
            setInvalidVisual(maxEl);
            msgs.push('Max has to be between 0 and 100.');
        }

        // Paarregel
        if (ok && minVal !== null && maxVal !== null && minVal > maxVal) {
            ok = false;
            setInvalidVisual(minEl);
            setInvalidVisual(maxEl);
            msgs.push('Min has to be smaller than Max');
        }

        if (!ok) {
            // Eine gemeinsame Meldung
            rangeFb.textContent = msgs.join(' ');
            // Damit Bootstrap sie zeigt
            rangeFb.style.display = 'block';
            // dem Browser mitteilen, dass das Feldset ungültig ist
            // (wir hängen die "Ungültigkeit" an beide Inputs, reicht aber an einem)
            minEl.setCustomValidity('invalid');
        } else {
            // optisches Grün nur, wenn befüllt
            setValidIfFilled(minEl);
            setValidIfFilled(maxEl);
            minEl.setCustomValidity('');
            maxEl.setCustomValidity('');
            rangeFb.textContent = '';
            rangeFb.style.display = '';
        }

        return ok;
    }

    form.addEventListener('submit', (event) => {
        const ok = validateOnSubmit();
        if (!form.checkValidity() || !ok) {
            event.preventDefault();
            event.stopPropagation();
        }
        form.classList.add('was-validated');
    }, false);
})();
