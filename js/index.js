'use strict'

function createLogEntry(entry) {
    document.getElementById('log').innerHTML += '<br>' + entry;
}

var model;
async function load() {
    createLogEntry('Loading Model ...');
    model = await tf.loadModel('js/model.json');
    createLogEntry('Data loaded successfully');
}

document.getElementById('selectTestDataButton').addEventListener('click', openDialog);

function openDialog() {
  document.getElementById('files').click();
}

async function main() {
    
    await load();
//    const batch = "./dog.png";
//    const imageTest = new Image(batch);
//    console.log(imageTest);
//    await predict(batch);
    document.getElementById('selectTestDataButton').disabled = false;
    document.getElementById('selectTestDataButton').innerText = "Upload Image And Predict";
}

async function previewImage(img, prediction_value) {
    tf.tidy(() => {
        //const input_value = Array.from(batch.labels.argMax(1).dataSync());

        const div = document.createElement('div');
        div.className = 'prediction-div';

        //const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

        //const prediction_value = Array.from(output.argMax(1).dataSync());
        const image = img.slice([0, 0], [1, img.shape[1]]);

        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        draw(image.flatten(), canvas);

        const label = document.createElement('div');
        //label.innerHTML = 'Original Value: ' + input_value;
        label.innerHTML += '<br>Prediction Value: ' + prediction_value;

//        if (prediction_value - input_value == 0) {
//            label.innerHTML += '<br>Value recognized successfully';
//        } else {
//            label.innerHTML += '<br>Recognition failed!'
//        }

        div.appendChild(canvas);
        div.appendChild(label);
        document.getElementById('predictionResult').appendChild(div);
    });
}

function draw(image, canvas) {
    const [width, height] = [32, 32];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    console.log(data)
    var count = 0;
    for (let i = 0; i < (height * width); i++) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
      count = count + 3;
    }
    console.log(count);
    console.log(imageData);
    ctx.putImageData(imageData, 0, 0);
}

//document.getElementById('selectTestDataButton').addEventListener('click', async (el,ev) => {
//    // const batch = data.nextTestBatch(1);
//    
//    await predict(batch);
//});

async function handleFileSelect(evt) {
    var files = evt.target.files; // FileList object

    // Loop through the FileList and render image files as thumbnails.
    for (var i = 0, f; f = files[i]; i++) {

      // Only process image files.
      if (!f.type.match('image.*')) {
        continue;
      }

      var reader = new FileReader();

      // Closure to capture the file information.
      reader.onload = (function(theFile) {
        return async function(e) {
          // Render thumbnail.
//          var span = document.createElement('span');
//          span.innerHTML = ['<img id="imgTest" class="thumb" src="', e.target.result,
//                            '" title="', escape(theFile.name), '"/>'].join('');
//          document.getElementById('list').insertBefore(span, null);
//            
//          console.log(e.target.result);
//          var imgTest = new Image(); 
//          imgTest.src = e.target.result;
//          console.log(imgTest);
          // await predict(imgTest);
          var imgTest = e.target.result;
          predictImage(imgTest);
        };
      })(f);

      // Read in the image file as a data URL.
      reader.readAsDataURL(f);
//      console.log(f);
//      prepareImage(f);
       
    }
}



function predictImage(imgTest) {
    const IMAGE_SIZE = 32*32;
    // Make a request for the MNIST sprited image.
    const img = new Image();
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;
          
          if(img.width > 32) {
              img.width = 32
          }
          if(img.height > 32) {
              img.height = 32
          }
          
          console.log(img.width);
        const datasetBytesBuffer =
            new ArrayBuffer(IMAGE_SIZE * 3 * 4);

        console.log(datasetBytesBuffer);
        canvas.width = img.width;
        canvas.height = img.height;

        
      const datasetBytesView = new Float32Array(
          datasetBytesBuffer);
          console.log(datasetBytesView);
      ctx.drawImage(
          img, 0, 0, img.width, img.height);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          var count = 0;
          console.log(imageData.data.length);
      for (let j = 0; j < imageData.data.length / 4; j++) {
        // All channels hold an equal value since the image is grayscale, so
        // just read the red channel.
        datasetBytesView[count] = imageData.data[j * 4] / 255;
          datasetBytesView[count+1] = imageData.data[j * 4 + 1] / 255;
          datasetBytesView[count+2] = imageData.data[j * 4 + 2] / 255;
          count = count + 1;
      }
        
        var finalImgTest = new Float32Array(datasetBytesBuffer);
          console.log(finalImgTest);
          
          
          
          const finalImgTestTensorModified = finalImgTest.slice(0, 32*32*3);
          
          const finalImgTestTensor = tf.tensor4d(finalImgTestTensorModified, [1, 32, 32, 3]);
        console.log(finalImgTestTensor.shape);
          
          var output = model.predict(finalImgTestTensor);
          // y_hat = model.predict(finalImgTestTensor)
        console.log(output);
          const prediction_value = Array.from(output.argMax(1).dataSync());
          console.log(prediction_value);
        var cifar10_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'];
          console.log(cifar10_labels[prediction_value]);
          
          previewImage(finalImgTestTensor, cifar10_labels[prediction_value]);
          
        resolve();
      };
      img.src = imgTest;
    });
}

document.getElementById('files').addEventListener('change', handleFileSelect, false);

main();