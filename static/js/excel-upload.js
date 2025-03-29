const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("excel-file");
const dropText = document.getElementById("drop-text");
const fileCaption = document.getElementById("file-caption");

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
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        dropText.textContent = fileInput.files[0].name;
        fileCaption.textContent = "";
        dropZone.classList.add("uploaded"); 
    }
});