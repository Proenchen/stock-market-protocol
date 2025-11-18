(function () {
    function initMsDropdown(root) {
        if (!root) return;

        const menu = root.querySelector('.ms-menu');
        const searchInput = menu.querySelector('.ms-search');
        const list = menu.querySelector('.ms-list');
        const checks = Array.from(list.querySelectorAll('input.form-check-input[type="checkbox"]'));
        const labelEl = root.querySelector('.ms-label');
        const selectAllBtn = menu.querySelector('.ms-select-all');
        const clearBtn = menu.querySelector('.ms-clear');

        const col = root.closest('.col-md-6, .col-12, .col-md-12, .col');
        const hiddenContainer = col ? col.querySelector('div[id$="Hidden"]') : null;

        const fieldName = hiddenContainer ? hiddenContainer.id.replace(/Hidden$/, '') : null;

        const norm = (s) => (s || '').toString().toLowerCase().trim();

        function applySearch() {
            const q = norm(searchInput.value);
            checks.forEach((cb) => {
                const text = cb.getAttribute('data-label') || cb.closest('label').textContent;
                const match = norm(text).includes(q);
                cb.closest('.form-check').classList.toggle('d-none', !match);
            });
        }

        function updateHiddenInputs() {
            if (!hiddenContainer || !fieldName) return;
            hiddenContainer.innerHTML = '';
            const selected = checks.filter((c) => c.checked);

            selected.forEach((cb) => {
                const inp = document.createElement('input');
                inp.type = 'hidden';
                inp.name = fieldName;         
                inp.value = cb.getAttribute('data-value');
                hiddenContainer.appendChild(inp);
            });
        }

        function updateButtonLabel() {
            const selected = checks.filter((c) => c.checked);
            if (selected.length === 0) {
                labelEl.textContent = labelEl.dataset.placeholder || 'Select…';
                return;
            }
            const names = selected.map((c) => c.getAttribute('data-label') || c.value);
            if (names.length <= 3) {
                labelEl.textContent = names.join(', ');
            } else {
                labelEl.textContent = `${names.length} selected`;
            }
        }

        function updateAll() {
            updateHiddenInputs();
            updateButtonLabel();
        }

        function selectAll() {
            checks
                .filter((cb) => !cb.closest('.form-check').classList.contains('d-none'))
                .forEach((cb) => (cb.checked = true));
            updateAll();
        }

        function clearAll() {
            checks.forEach((cb) => (cb.checked = false));
            updateAll();
        }

        // --- Events ---
        if (searchInput) {
            if (labelEl && !labelEl.dataset.placeholder) {
                labelEl.dataset.placeholder = labelEl.textContent.trim();
            }
            searchInput.addEventListener('input', applySearch);
        }

        checks.forEach((cb) => {
            cb.addEventListener('change', updateAll);
            cb.style.cursor = 'pointer';
            const lbl = cb.closest('label');
            if (lbl) lbl.style.cursor = 'pointer';
        });

        if (selectAllBtn) {
            selectAllBtn.addEventListener('click', (e) => {
                e.preventDefault();
                selectAll();
            });
        }
        if (clearBtn) {
            clearBtn.addEventListener('click', (e) => {
                e.preventDefault();
                clearAll();
            });
        }

        root.addEventListener('shown.bs.dropdown', () => {
            if (searchInput) {
                searchInput.value = '';
                applySearch();
                setTimeout(() => searchInput.focus(), 0);
            }
        });

        updateAll();
    }

    function initAll() {
        document.querySelectorAll('.ms-dropdown').forEach(initMsDropdown);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAll);
    } else {
        initAll();
    }
})();
