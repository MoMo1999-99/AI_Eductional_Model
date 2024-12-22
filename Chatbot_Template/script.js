document.addEventListener('DOMContentLoaded', function() {
    const modelTypeSelect = document.getElementById('model-type');
    const questionGroup = document.getElementById('question-group');
    const pdfGroup = document.getElementById('pdf-group');
    const urlGroup = document.getElementById('url-group');

    // Function to update the form based on the model type
    function updateForm() {
        const modelType = modelTypeSelect.value;

        if (modelType === 'generative') {
            questionGroup.style.display = 'block';
            pdfGroup.style.display = 'none';
            urlGroup.style.display = 'none';
        } else if (modelType === 'rag') {
            questionGroup.style.display = 'block';
            pdfGroup.style.display = 'block';
            urlGroup.style.display = 'none';
        } else if (modelType === 'image') {
            questionGroup.style.display = 'block';
            pdfGroup.style.display = 'none';
            urlGroup.style.display = 'block';
        }
    }

    // Initial form update
    updateForm();

    // Update form when model type is changed
    modelTypeSelect.addEventListener('change', updateForm);
});
