document.addEventListener("DOMContentLoaded", function () {
    let fileInput = document.getElementById("fileInput");
    let resetButton = document.getElementById("resetButton");
    let uploadForm = document.getElementById("uploadForm");
    let preview = document.getElementById("preview");
    let selectedImage = null;

    // Function to show preview of selected image
    function showPreview(imageSrc) {
        preview.src = imageSrc;
        preview.style.display = "block";
    }

    // File upload preview
    fileInput.addEventListener("change", function () {
        let file = fileInput.files[0];
        if (file) {
            let reader = new FileReader();
            reader.onload = function (e) {
                selectedImage = file;
                showPreview(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    });

    // Reset button
    resetButton.addEventListener("click", function () {
        fileInput.value = "";
        selectedImage = null;
        preview.style.display = "none";
    });

    // Submit form
    uploadForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        let formData = new FormData();

        if (selectedImage) {
            formData.append("file", selectedImage);
        } else if (fileInput.files.length > 0) {
            formData.append("file", fileInput.files[0]);
        } else {
            alert("Please upload an image!");
            return;
        }

        try {
            let response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (response.redirected) {
                window.location.href = response.url;
            } else {
                alert("Prediction failed. Try again.");
            }
        } catch (error) {
            console.error("Error:", error);
            alert("Something went wrong. Check console for details.");
        }
    });
});