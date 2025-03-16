document.addEventListener("DOMContentLoaded", function () {
    fetch("/navbar")
        .then(response => {
            if (!response.ok) {
                throw new Error(`Navbar failed to load: ${response.statusText}`);
            }
            return response.text();
        })
        .then(data => {
            document.getElementById("navbar-container").innerHTML = data;
            highlightActivePage();
        })
        .catch(error => console.error(error));

    /*fetch("assets/components/footer.html")
        .then(response => {
            if (!response.ok) {
                throw new Error(`Footer failed to load: ${response.statusText}`);
            }
            return response.text();
        })
        .then(data => {
            document.getElementById("footer-container").innerHTML = data;
        })
        .catch(error => console.error(error));*/
});

function highlightActivePage() {
    let currentPage = window.location.pathname.split("/").pop();
    
    document.querySelectorAll(".nav-link").forEach(link => {
        let targetPage = link.getAttribute("href").split("/").pop(); 
        if (targetPage === currentPage) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }
    });
}

