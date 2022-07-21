function updateProgress (progressUrl) {
    fetch(progressUrl).then(function(response) {
        response.json().then(function(data) {
            // update the appropriate UI components
            setProgress(data.state, data.details);
            setTimeout(updateProgress, 500, progressUrl);
        });
    });
}

var progressUrl = '{% url "task_status" task_id %}';  // django template usage
updateProgress(progressUrl);