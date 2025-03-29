document.addEventListener("DOMContentLoaded", function () {
    highlightActivePage();
    initializeThemeSwitch();
});

function highlightActivePage() {
    let currentPage = window.location.pathname.split("/").pop();

    document.querySelectorAll(".nav-link, .dropdown-item").forEach(link => {
        let hrefAttribute = link.getAttribute("href");
        if (hrefAttribute) {
            let targetPage = hrefAttribute.split("/").pop();
            if (targetPage === currentPage) {
                link.classList.add("active");
                let parentDropdown = link.closest(".dropdown");
                if (parentDropdown) {
                    parentDropdown.querySelector(".nav-link").classList.add("active");
                }
            } else {
                link.classList.remove("active");
            }
        }
    });
}


function initializeThemeSwitch() {
    const themeSwitch = document.getElementById('theme-switch');
    const navbar = document.getElementById('navbar');
    let lightmode = localStorage.getItem('lightmode');

    const enableLightmode = () => {
        document.body.classList.add('lightmode');
        navbar.setAttribute("data-bs-theme", "light");
        localStorage.setItem('lightmode', 'active');
    };

    const disableLightmode = () => {
        document.body.classList.remove('lightmode');
        navbar.setAttribute("data-bs-theme", "dark");
        localStorage.removeItem('lightmode');
    };

    if (lightmode === 'active') enableLightmode();

    themeSwitch.addEventListener('click', () => {
        let lightmode = localStorage.getItem('lightmode');
        if (lightmode !== "active") {
            enableLightmode();
        } else {
            disableLightmode();
        }
    });
}