document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.container pre').forEach((pre) => {
        const code = pre.querySelector('code');
        const text = code ? code.innerText : pre.innerText;

        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.type = 'button';
        btn.textContent = 'Copy';

        btn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(text);
                const original = btn.textContent;
                btn.textContent = 'Copied';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = original;
                    btn.classList.remove('copied');
                }, 1200);
            } catch {
                btn.textContent = 'Ctrl/Cmd+C';
                setTimeout(() => {
                    btn.textContent = 'Copy';
                }, 1200);
            }
        });

        pre.appendChild(btn);
    });
});
