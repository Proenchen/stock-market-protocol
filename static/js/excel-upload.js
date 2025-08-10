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
        const identifierInput = document.getElementById("user-identifier");

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

