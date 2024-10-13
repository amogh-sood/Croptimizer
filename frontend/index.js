let displayArea = document.getElementById("resultant");

document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector("#predictionForm")
    form.addEventListener("submit", function(event) {
        event.preventDefault();
        const formData = new FormData(form);
        const formValues = {};
        
        console.log(formData);

        formData.forEach((value, key) => {
            formValues[key] = value;
        });
        console.log(formValues);
        fetch("http://127.0.0.1:5000/submit", {
            method: "POST",
            body: JSON.stringify(formValues),
            headers: {
                "Content-Type": "application/json"
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            displayArea.innerText = "Predicted Yield in tons per hectare: " + data["prediction"]
        })
        .catch(error => {
            console.error(error);
        })
    });
});