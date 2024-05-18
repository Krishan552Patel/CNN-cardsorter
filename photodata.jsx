app.bringToFront();

function main() {
    var inputFolder = Folder.selectDialog("Select the folder with PNG files");
    if (inputFolder == null) {
        alert("No folder selected.");
        return;
    }

    var files = inputFolder.getFiles("*.png"); // Get all PNG files in the folder
    if (files.length == 0) {
        alert("No PNG files found in the selected folder.");
        return;
    }

    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        open(file);
        var doc = app.activeDocument;

        // Create a folder for the images
        var outputFolder = new Folder(doc.path + "/" + doc.name.replace(/\.[^\.]+$/, ''));
        if (!outputFolder.exists) {
            outputFolder.create();
        }

        // Save the original image in the new folder
        saveAsPng(doc, outputFolder, doc.name);

        // Create variations
        createVariations(doc, outputFolder);
        createVariationsf(doc,outputFolder);

        doc.close(SaveOptions.DONOTSAVECHANGES);
    }
}

function saveAsPng(doc, outputFolder, fileName) {
    var pngFile = new File(outputFolder + "/" + fileName);
    doc.saveAs(pngFile, new PNGSaveOptions(), true, Extension.LOWERCASE);
}

function createVariations(originalDoc, outputFolder) {
    for (var i = 0; i <= 100; i += 10) { // Increments of 10 for noise
        var copyDoc = originalDoc.duplicate(originalDoc.name + " Noise " + i);
        applyNoise(copyDoc, i);
        var fileName = originalDoc.name.replace(/\.[^\.]+$/, '') + "_noise_" + i + ".png";
        saveAsPng(copyDoc, outputFolder, fileName);
        copyDoc.close(SaveOptions.DONOTSAVECHANGES);
    }
}
function createVariationsf(originalDoc, outputFolder) {
    for (var i = 0; i <= 100; i += 10) { // Increments of 10 for noise
        var copyDoc = originalDoc.duplicate(originalDoc.name + " Noise " + i);
        copyDoc.rotateCanvas(180);
        applyNoise(copyDoc, i);
        var fileName = originalDoc.name.replace(/\.[^\.]+$/, '') + "_noise_Fliped" + i + ".png";
        saveAsPng(copyDoc, outputFolder, fileName);
        copyDoc.close(SaveOptions.DONOTSAVECHANGES);
    }
}

function applyNoise(doc, noiseLevel) {
    // Add a new layer for the noise
    var noiseLayer = doc.artLayers.add();
    noiseLayer.name = "Noise";
    noiseLayer.blendMode = BlendMode.SOFTLIGHT;

    // Fill the layer with gray
    var fillColor = new SolidColor();
    fillColor.rgb.red = 128;
    fillColor.rgb.green = 128;
    fillColor.rgb.blue = 128;
    doc.selection.selectAll();
    doc.selection.fill(fillColor);
    doc.selection.deselect();

    // Apply the noise
    noiseLayer.applyAddNoise(noiseLevel, NoiseDistribution.UNIFORM, true);
    noiseLayer.opacity = 50;
}

main()