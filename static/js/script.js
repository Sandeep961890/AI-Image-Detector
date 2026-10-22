"use strict";

const imageInput = document.querySelector("#image");
const uploadArea = document.querySelector("#upload-area");
const analyzeForm = document.querySelector("#analyze-form");
const analyzeButton = document.querySelector("#analyze-button");
const fileName = document.querySelector("#file-name");
const previewContainer = document.querySelector("#preview-container");
const preview = document.querySelector("#image-preview");
const removeButton = document.querySelector("#remove-image");
const clientError = document.querySelector("#client-error");
const loadingIndicator = document.querySelector("#loading-indicator");
const resultsSection = document.querySelector("#results");
const metadataList = document.querySelector("#metadata dl");
const maxFileSize = 10 * 1024 * 1024;
const acceptedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
const acceptedExtensions = new Set(["jpg", "jpeg", "png", "webp"]);
let selectedFile;
let previewUrl;
let requestInProgress = false;

function isSupportedImage(file) {
  const extension = file.name.split(".").pop().toLowerCase();
  return acceptedTypes.has(file.type)
    || (!file.type && acceptedExtensions.has(extension));
}

function showError(message) {
  clientError.textContent = message;
  clientError.hidden = false;
}

function clearError() {
  clientError.textContent = "";
  clientError.hidden = true;
}

function setMetadata(id, label, value) {
  let valueElement = document.querySelector(`#${id}`);
  if (!valueElement) {
    const row = document.createElement("div");
    const labelElement = document.createElement("dt");
    valueElement = document.createElement("dd");
    labelElement.textContent = label;
    valueElement.id = id;
    row.append(labelElement, valueElement);
    metadataList.append(row);
  }

  valueElement.textContent = value;
}

function clearResults() {
  resultsSection.querySelectorAll(".api-result, .api-error").forEach((element) => {
    element.remove();
  });
  const placeholder = document.querySelector("#results-placeholder");
  if (placeholder) {
    placeholder.hidden = false;
  }
}

function displayError(message) {
  clearResults();
  const error = document.createElement("p");
  error.className = "message message--error api-error";
  error.setAttribute("role", "alert");
  error.textContent = message;
  resultsSection.append(error);
}

function getServerErrorMessage(data, fallback) {
  if (data && typeof data === "object") {
    if (typeof data.message === "string") {
      return data.message;
    }
    if (typeof data.error === "string") {
      return data.error;
    }
    if (data.error && typeof data.error.message === "string") {
      return data.error.message;
    }
  }
  return fallback;
}

function addResultField(container, label, value) {
  if (value === undefined || value === null || value === "") {
    return;
  }

  const row = document.createElement("p");
  const name = document.createElement("strong");
  name.textContent = `${label}: `;
  row.append(name, document.createTextNode(String(value)));
  container.append(row);
}

function getPredictionLabel(data) {
  if (typeof data.prediction_label === "string") {
    return data.prediction_label;
  }
  if (typeof data.prediction === "string") {
    return data.prediction;
  }
  if (data.prediction && typeof data.prediction.label === "string") {
    return data.prediction.label;
  }
  if (typeof data.label === "string") {
    return data.label;
  }
  return undefined;
}

function getScore(data) {
  if (data.score !== undefined) {
    return data.score;
  }
  if (data.prediction && typeof data.prediction === "object") {
    return data.prediction.score;
  }
  return undefined;
}

function getScoreMeaning(data) {
  if (typeof data.score_meaning === "string") {
    return data.score_meaning;
  }
  if (data.prediction && typeof data.prediction === "object") {
    return data.prediction.score_meaning;
  }
  return undefined;
}

function getModelName(data) {
  if (typeof data.model_name === "string") {
    return data.model_name;
  }
  if (typeof data.model === "string") {
    return data.model;
  }
  if (data.model && typeof data.model.name === "string") {
    return data.model.name;
  }
  return undefined;
}

function getDimensions(data) {
  const dimensions = data.image_dimensions || data.dimensions;
  if (dimensions && typeof dimensions === "object") {
    const width = dimensions.width;
    const height = dimensions.height;
    if (width !== undefined && height !== undefined) {
      return `${width} × ${height} px`;
    }
  }
  if (data.width !== undefined && data.height !== undefined) {
    return `${data.width} × ${data.height} px`;
  }
  return undefined;
}

function renderResult(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    displayError("The server returned an invalid JSON response.");
    return;
  }

  const errorObject = data.error;
  const errorCode = typeof data.error_code === "string"
    ? data.error_code
    : errorObject && typeof errorObject.code === "string"
      ? errorObject.code
      : undefined;
  if (errorCode === "model_not_configured" || errorCode === "model_missing") {
    displayError(getServerErrorMessage(data, errorCode));
    return;
  }
  if (errorObject !== undefined || typeof data.message === "string") {
    displayError(getServerErrorMessage(data, "The analysis service reported an error."));
    return;
  }

  clearResults();
  const placeholder = document.querySelector("#results-placeholder");
  if (placeholder) {
    placeholder.hidden = true;
  }

  const result = document.createElement("div");
  result.className = "api-result";

  addResultField(result, "Prediction", getPredictionLabel(data));

  const score = getScore(data);
  const scoreMeaning = getScoreMeaning(data);
  if (score !== undefined && scoreMeaning) {
    addResultField(result, "Score", `${score} (${scoreMeaning})`);
  }

  addResultField(result, "Model", getModelName(data));
  if (typeof data.model_id === "string") {
    addResultField(result, "Model identifier", data.model_id);
  }
  if (typeof data.model_revision === "string") {
    addResultField(result, "Model revision", data.model_revision);
  }
  if (
    data.label_mapping
    && typeof data.label_mapping === "object"
    && !Array.isArray(data.label_mapping)
  ) {
    const mappings = Object.entries(data.label_mapping)
      .filter(([index, label]) => /^\d+$/.test(index) && typeof label === "string")
      .map(([index, label]) => `${index} → ${label}`);
    if (mappings.length) {
      addResultField(result, "Model class mapping", mappings.join("; "));
    }
  }

  if (
    data.class_scores
    && typeof data.class_scores === "object"
    && !Array.isArray(data.class_scores)
  ) {
    const classScores = Object.entries(data.class_scores).filter(
      ([label, value]) => typeof label === "string"
        && typeof value === "number"
        && Number.isFinite(value)
        && value >= 0
        && value <= 1,
    );
    if (classScores.length) {
      const heading = document.createElement("h3");
      heading.textContent = "Class scores";
      result.append(heading);

      classScores.forEach(([label, value]) => {
        const row = document.createElement("p");
        const name = document.createElement("strong");
        const meter = document.createElement("progress");
        name.textContent = `${label}: `;
        meter.className = "score-meter";
        meter.max = 1;
        meter.value = value;
        meter.setAttribute("aria-label", `${label} softmax score`);
        row.append(name, document.createTextNode(`${(value * 100).toFixed(1)}% `), meter);
        result.append(row);
      });
      if (scoreMeaning) {
        addResultField(result, "Score meaning", scoreMeaning);
      }
    }
  }

  const dimensions = getDimensions(data);
  if (dimensions) {
    addResultField(result, "Image dimensions", dimensions);
  }

  if (Array.isArray(data.limitations)) {
    const heading = document.createElement("h3");
    heading.textContent = "Limitations";
    result.append(heading);
    const list = document.createElement("ul");
    data.limitations.forEach((limitation) => {
      if (typeof limitation === "string") {
        const item = document.createElement("li");
        item.textContent = limitation;
        list.append(item);
      }
    });
    if (list.childElementCount) {
      result.append(list);
    } else {
      heading.remove();
    }
  } else if (typeof data.limitations === "string") {
    addResultField(result, "Limitations", data.limitations);
  }

  if (result.childElementCount === 0) {
    displayError("The server returned JSON without any recognized result fields.");
    return;
  }
  resultsSection.append(result);
}

function releasePreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = undefined;
  }
}

function clearSelection() {
  releasePreview();
  selectedFile = undefined;
  imageInput.value = "";
  fileName.textContent = "No image selected";
  preview.removeAttribute("src");
  preview.alt = "Preview of the selected image";
  previewContainer.hidden = true;
  setMetadata("metadata-name", "File name", "Not selected");
  setMetadata("metadata-type", "File type", "Not selected");
  setMetadata("metadata-size", "File size", "Not selected");
  setMetadata("metadata-dimensions", "Image dimensions", "Not selected");
  clearError();
}

function selectFile(file) {
  clearError();
  clearResults();
  if (!file) {
    clearSelection();
    return;
  }

  if (!isSupportedImage(file)) {
    clearSelection();
    showError("Choose a JPEG, PNG, or WebP image.");
    return;
  }
  if (file.size > maxFileSize) {
    clearSelection();
    showError("The image exceeds the 10 MB file size limit.");
    return;
  }

  releasePreview();
  selectedFile = file;
  fileName.textContent = file.name;
  setMetadata("metadata-name", "File name", file.name);
  setMetadata("metadata-type", "File type", file.type || "Unknown");
  setMetadata("metadata-size", "File size", `${(file.size / (1024 * 1024)).toFixed(2)} MB`);
  setMetadata("metadata-dimensions", "Image dimensions", "Reading image dimensions…");

  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.alt = `Preview of ${file.name}`;
  previewContainer.hidden = false;
  preview.onload = () => {
    if (selectedFile === file) {
      setMetadata(
        "metadata-dimensions",
        "Image dimensions",
        `${preview.naturalWidth} × ${preview.naturalHeight} px`,
      );
    }
  };
  preview.onerror = () => {
    if (selectedFile === file) {
      setMetadata("metadata-dimensions", "Image dimensions", "Unavailable");
    }
  };
}

imageInput.addEventListener("change", () => {
  selectFile(imageInput.files[0]);
});

removeButton.addEventListener("click", () => {
  if (requestInProgress) {
    return;
  }
  clearSelection();
  clearResults();
  imageInput.focus();
});

uploadArea.addEventListener("dragenter", (event) => {
  event.preventDefault();
  uploadArea.classList.add("is-dragover");
});

uploadArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadArea.classList.add("is-dragover");
});

uploadArea.addEventListener("dragleave", (event) => {
  if (!(event.relatedTarget instanceof Node) || !uploadArea.contains(event.relatedTarget)) {
    uploadArea.classList.remove("is-dragover");
  }
});

uploadArea.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadArea.classList.remove("is-dragover");
  const file = event.dataTransfer.files[0];
  if (!file || requestInProgress) {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  selectFile(file);
});

analyzeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (requestInProgress) {
    return;
  }

  const file = selectedFile || imageInput.files[0];
  if (!file) {
    showError("Choose an image before analyzing.");
    imageInput.focus();
    return;
  }
  if (!isSupportedImage(file)) {
    showError("Choose a JPEG, PNG, or WebP image.");
    return;
  }
  if (file.size > maxFileSize) {
    showError("The image exceeds the 10 MB file size limit.");
    return;
  }

  requestInProgress = true;
  clearError();
  clearResults();
  loadingIndicator.hidden = false;
  analyzeButton.disabled = true;
  removeButton.disabled = true;
  imageInput.disabled = true;

  try {
    const formData = new FormData();
    formData.append("image", file, file.name);
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { Accept: "application/json" },
      body: formData,
    });
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.toLowerCase().includes("application/json")) {
      throw new Error(
        `The server returned a non-JSON response (HTTP ${response.status}).`,
      );
    }

    const data = await response.json();
    if (!response.ok) {
      displayError(
        getServerErrorMessage(
          data,
          `The server could not analyze this image (HTTP ${response.status}).`,
        ),
      );
      return;
    }

    renderResult(data);
  } catch (error) {
    displayError(
      error instanceof TypeError
        ? "Could not reach the local analysis service. Check that the server is running."
        : error.message,
    );
  } finally {
    requestInProgress = false;
    loadingIndicator.hidden = true;
    analyzeButton.disabled = false;
    removeButton.disabled = false;
    imageInput.disabled = false;
  }
});

window.addEventListener("beforeunload", releasePreview);
