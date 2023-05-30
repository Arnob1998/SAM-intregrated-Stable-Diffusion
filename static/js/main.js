document.addEventListener('DOMContentLoaded', () => {
    const canvas = new fabric.Canvas('canvas');
    const uploadedImage = document.getElementById('uploaded-image');
    const saveButton = document.getElementById('save-button');
    let boundingBox = {};

    // Set canvas dimensions based on the uploaded image
    canvas.setDimensions({
        width: uploadedImage.width,
        height: uploadedImage.height
    });

    let isDrawing = false;
    let startCoords = {};

    // Event listener for mouse down event on the canvas
    canvas.on('mouse:down', (options) => {
        if (options.target) return; // Ignore if an object is already selected
        isDrawing = true;
        const pointer = canvas.getPointer(options.e);
        startCoords = pointer;
        const rect = new fabric.Rect({
            left: pointer.x,
            top: pointer.y,
            width: 1,
            height: 1,
            fill: 'transparent',
            stroke: 'red',
            strokeWidth: 2,
            selectable: false
        });
        canvas.add(rect);
    });

    // Event listener for mouse move event on the canvas
    canvas.on('mouse:move', (options) => {
        if (!isDrawing) return;
        const pointer = canvas.getPointer(options.e);
        const width = pointer.x - startCoords.x;
        const height = pointer.y - startCoords.y;
        const rect = canvas.getObjects()[0];
        rect.set({ width, height });
        canvas.renderAll();
    });

    // Event listener for mouse up event on the canvas
    canvas.on('mouse:up', () => {
        isDrawing = false;
        const rect = canvas.getObjects()[0];
        boundingBox = {
            x: rect.left,
            y: rect.top,
            width: rect.width,
            height: rect.height
        };
    });

    // Event listener for the save button click
    saveButton.addEventListener('click', () => {
        console.log(boundingBox);
        // You can perform further actions with the bounding box coordinates here
    });
});
