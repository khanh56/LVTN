(function ($) {
  var $window = $(window),
    $body = $("body"),
    $wrapper = $("#wrapper"),
    $main = $("#main"),
    $panels = $main.children(".panel"),
    $nav = $("#nav"),
    $nav_links = $nav.children("a");

  // Breakpoints.
  breakpoints({
    xlarge: ["1281px", "1680px"],
    large: ["981px", "1280px"],
    medium: ["737px", "980px"],
    small: ["361px", "736px"],
    xsmall: [null, "360px"],
  });

  // Play initial animations on page load.
  $window.on("load", function () {
    window.setTimeout(function () {
      $body.removeClass("is-preload");
    }, 100);
  });

  // Nav.
  $nav_links.on("click", function (event) {
    var href = $(this).attr("href");

    // Not a panel link? Bail.
    if (href.charAt(0) != "#" || $panels.filter(href).length == 0) return;

    // Prevent default.
    event.preventDefault();
    event.stopPropagation();

    // Change panels.
    if (window.location.hash != href) window.location.hash = href;
  });

  // Panels.

  // Initialize.
  (function () {
    var $panel, $link;

    // Get panel, link.
    if (window.location.hash) {
      $panel = $panels.filter(window.location.hash);
      $link = $nav_links.filter('[href="' + window.location.hash + '"]');
    }

    // No panel/link? Default to first.
    if (!$panel || $panel.length == 0) {
      $panel = $panels.first();
      $link = $nav_links.first();
    }

    // Deactivate all panels except this one.
    $panels.not($panel).addClass("inactive").hide();

    // Activate link.
    $link.addClass("active");

    // Reset scroll.
    $window.scrollTop(0);
  })();

  // Hashchange event.
  $window.on("hashchange", function (event) {
    var $panel, $link;

    // Get panel, link.
    if (window.location.hash) {
      $panel = $panels.filter(window.location.hash);
      $link = $nav_links.filter('[href="' + window.location.hash + '"]');

      // No target panel? Bail.
      if ($panel.length == 0) return;
    }

    // No panel/link? Default to first.
    else {
      $panel = $panels.first();
      $link = $nav_links.first();
    }

    // Deactivate all panels.
    $panels.addClass("inactive");

    // Deactivate all links.
    $nav_links.removeClass("active");

    // Activate target link.
    $link.addClass("active");

    // Set max/min height.
    $main
      .css("max-height", $main.height() + "px")
      .css("min-height", $main.height() + "px");

    // Delay.
    setTimeout(function () {
      // Hide all panels.
      $panels.hide();

      // Show target panel.
      $panel.show();

      // Set new max/min height.
      $main
        .css("max-height", $panel.outerHeight() + "px")
        .css("min-height", $panel.outerHeight() + "px");

      // Reset scroll.
      $window.scrollTop(0);

      // Delay.
      window.setTimeout(
        function () {
          // Activate target panel.
          $panel.removeClass("inactive");

          // Clear max/min height.
          $main.css("max-height", "").css("min-height", "");

          // IE: Refresh.
          $window.triggerHandler("--refresh");

          // Unlock.
          locked = false;
        },
        breakpoints.active("small") ? 0 : 500
      );
    }, 250);
  });

  // IE: Fixes.
  if (browser.name == "ie") {
    // Fix min-height/flexbox.
    $window.on("--refresh", function () {
      $wrapper.css("height", "auto");

      window.setTimeout(function () {
        var h = $wrapper.height(),
          wh = $window.height();

        if (h < wh) $wrapper.css("height", "100vh");
      }, 0);
    });

    $window.on("resize load", function () {
      $window.triggerHandler("--refresh");
    });

    // Fix intro pic.
    $(".panel.intro").each(function () {
      var $pic = $(this).children(".pic"),
        $img = $pic.children("img");

      $pic
        .css("background-image", "url(" + $img.attr("src") + ")")
        .css("background-size", "cover")
        .css("background-position", "center");

      $img.css("visibility", "hidden");
    });
  }
})(jQuery);

// Thêm sự kiện onchange cho input type="file"

var selectedImagePath = null;

document.getElementById("images").onchange = function (e) {
  var files = e.target.files;

  var file = files[0];
  var reader = new FileReader();

  reader.onload = function (e) {
    selectedImagePath = e.target.result;
    // Hiển thị hình ảnh mới trong content-photo và xóa hình ảnh cũ
    displaySelectedImage();
  };
  reader.readAsDataURL(file);
};

function displaySelectedImage() {
  if (selectedImagePath) {
    // Xóa hình ảnh cũ nếu đã tồn tại
    removeSelectedImage();
    var img = document.createElement("img");
    img.onload = function () {
      // Resize ảnh thành kích thước 500x500
      var canvas = document.createElement("canvas");
      var ctx = canvas.getContext("2d");
      canvas.width = 500;
      canvas.height = 500;
      ctx.drawImage(img, 0, 0, 500, 500);

      // Tạo một ảnh mới từ canvas và hiển thị
      var resizedImg = new Image();
      resizedImg.src = canvas.toDataURL();
      resizedImg.className = "selected-image";
      var label = document.getElementById("dropcontainer");
      label.appendChild(resizedImg);

      // Apply the same resizing to the result image
      var resultImg = document.querySelector(".image-2");
      if (resultImg) {
        var resultCanvas = document.createElement("canvas");
        var resultCtx = resultCanvas.getContext("2d");
        resultCanvas.width = 500;
        resultCanvas.height = 500;
        resultCtx.drawImage(resultImg, 0, 0, 500, 500);
        resultImg.src = resultCanvas.toDataURL();
      }

      var contentPhoto = document.querySelector(".content-photo");
      contentPhoto.style.opacity = "0";

      fetch(selectedImagePath)
      .then(response => response.blob())
      .then(blob => {
        selectedImageSourceOrBlob = blob;
      })
      .catch(error => console.error('Error fetching uploaded image:', error));
    };
    img.src = selectedImagePath;
  }
}

function removeSelectedImage() {
  // Select all elements with the class 'selected-image' and remove them
  var selectedImages = document.querySelectorAll('.selected-image');
  selectedImages.forEach(function(image) {
      image.parentNode.removeChild(image);
  });
}

// Assuming you have a global variable to track the selected image source or Blob
var selectedImageSourceOrBlob = null;

function selectImage(element) {
  var imgSrc = element.querySelector("img").src;
  fetch(imgSrc)
      .then(response => response.blob())
      .then(blob => {
          var reader = new FileReader();
          reader.onload = function(e) {
              selectedImagePath = e.target.result;
              displaySelectedImage(); // Sử dụng cùng một hàm với kéo và thả
          };
          reader.readAsDataURL(blob);
      })
      .catch(error => console.error('Error fetching selected sample image:', error));
}


function drawLabel() {
  var formData = new FormData();

  fetch('/draw_label', {
    method: 'POST',
    body: formData
  })
  .then(response => response.blob())
  .then(blob => {
    const imageUrl = URL.createObjectURL(blob);
    var img = new Image();
    img.onload = function () {
        // Resize the image to 500x500
        var canvas = document.createElement("canvas");
        var ctx = canvas.getContext("2d");
        canvas.width = 500;
        canvas.height = 500;
        ctx.drawImage(img, 0, 0, 500, 500);

        var resizedImg = new Image();
        resizedImg.src = canvas.toDataURL();
        document.querySelector('.image-3').src = resizedImg.src;
    };
    img.src = imageUrl;
})
}



document
  .getElementById("dropcontainer")
  .addEventListener("dragover", function (event) {
    event.preventDefault();
  });

document.getElementById("dropcontainer").addEventListener("drop", function (event) {
  event.preventDefault();
  removeSelectedImage();
  var contentPhoto = document.querySelector(".content-photo");
  contentPhoto.style.opacity = "0";

  var files = event.dataTransfer.files;

  for (var i = 0; i < files.length; i++) {
    var file = files[i];
    var reader = new FileReader();

    reader.onload = function (e) {
      selectedImagePath = e.target.result;
      displaySelectedImage();
    };

    reader.readAsDataURL(file);
  }
});


/////////////////// Model //////////////////////

var selectedImagePathModel = null;
var selectedImageSourceOrBlobModel = null;

function removeSelectedImageModel() {
  // Select all elements with the class 'selected-image' in the model page and remove them
  var selectedImagesModel = document.querySelectorAll('.selected-image-model');
  selectedImagesModel.forEach(function(image) {
      image.parentNode.removeChild(image);
  });
}

document.getElementById("imagesModel").onchange = function (e) {
  var files = e.target.files;
  var file = files[0];
  var reader = new FileReader();
  reader.onload = function (e) {
    selectedImagePathModel = e.target.result;
    displaySelectedImageModel(selectedImagePathModel);
  };
  reader.readAsDataURL(file);
};

document.getElementById("dropcontainerModel").addEventListener("dragover", function (event) {
  event.preventDefault();
});
document.getElementById("dropcontainerModel").addEventListener("drop", function (event) {
  event.preventDefault();
  var files = event.dataTransfer.files;
  for (var i = 0; i < files.length; i++) {
    var file = files[i];
    var reader = new FileReader();
    reader.onload = function (e) {
      selectedImagePathModel = e.target.result;
      displaySelectedImageModel();
    };
    reader.readAsDataURL(file);
  }
});

function selectImageModel(element) {
  selectedImagePathModel = element.children[0].src;
  displaySelectedImageModel();
}

function displaySelectedImageModel() {
  if (selectedImagePathModel) {
    removeSelectedImageModel();
    var img = document.createElement("img");
    img.onload = function () {
      var canvas = document.createElement("canvas");
      var ctx = canvas.getContext("2d");
      canvas.width = 640;
      canvas.height = 640;

      //////////
      var scaleFactor = Math.min(canvas.width / img.width, canvas.height / img.height);

      var x = (canvas.width - img.width * scaleFactor) / 2;
      var y = (canvas.height - img.height * scaleFactor) / 2;

      // Fill the canvas with black color
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw the image on the canvas
      // ctx.drawImage(img, x, y, img.width * scaleFactor, img.height * scaleFactor);
      var newWidth = img.width * scaleFactor;
      var newHeight = img.height * scaleFactor;      
      ctx.drawImage(img, 0, 0, newWidth, newHeight);

      
      
      //////////


      var resizedImg = new Image();
      resizedImg.src = canvas.toDataURL();
      resizedImg.className = "selected-image-model";
      var label = document.getElementById("dropcontainerModel"); 
      label.appendChild(resizedImg);

      // Apply the same resizing to the result image
      var resultImg = document.querySelector(".image-4");
      if (resultImg) {
        var resultCanvas = document.createElement("canvas");
        var resultCtx = resultCanvas.getContext("2d");
        resultCanvas.width = 640;
        resultCanvas.height = 640;
        resultCtx.drawImage(resultImg, 0, 0, 640, 640);
        resultImg.src = resultCanvas.toDataURL();
      }

      var contentPhoto = document.querySelector(".content-photo-model");
      contentPhoto.style.opacity = "0";
      contentPhoto.style.display = "none";
      fetch(selectedImagePathModel)
      .then(response => response.blob())
      .then(blob => {
        selectedImageSourceOrBlobModel = blob;
      })
      .catch(error => console.error('Error fetching uploaded image:', error));
    };
    img.src = selectedImagePathModel;
  }
}


function processImageModel() {
  var formData = new FormData();
  const spinner = document.querySelector('.loading')
  spinner.style.display = 'block'


  /// new  code
  const resultImage = document.querySelector('.image-4'); // Select the image to be hidden
  resultImage.style.visibility  = 'hidden'; // Hide the image
  ///
  // Check if selectedImageSourceOrBlob is a Blob (sample image scenario)
  if (selectedImageSourceOrBlobModel instanceof Blob) {
      formData.append('image', selectedImageSourceOrBlobModel);
  } else { // Else, use the file input (local file scenario)
      var imageFile = document.getElementById('imagesModel').files[0];
      formData.append('image', imageFile);
  }

  var modelType = document.getElementById('model_type').value;
  formData.append('model_type', modelType);

  var confidenceThreshold = document.getElementById('confidence-threshold').value;
  formData.append('confidence_threshold', confidenceThreshold);

  // document.getElementById('model_type').form.submit();
  fetch('/model', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const imageUrl = URL.createObjectURL(blob);
    var img = new Image();
    img.onload = function () {
        // Resize the image to 500x500
        var canvas = document.createElement("canvas");
        var ctx = canvas.getContext("2d");
        canvas.width = 640;
        canvas.height = 640;

        ctx.drawImage(img, 0, 0, 640, 640);

        // Create a new image from the canvas and set it as the source for the result image
        var resizedImg = new Image();
        resizedImg.src = canvas.toDataURL();
        resultImage.src = resizedImg.src;
        resultImage.style.visibility = 'visible'; // Show the image again

        // document.querySelector('.image-4').src = resizedImg.src;
        spinner.style.display = 'none'
    };
    img.src = imageUrl;
})
.catch(error => {
    console.error('Error:', error);
    spinner.style.display = 'none'; 
    resultImage.style.visibility = 'visible'; // Show the image even if there is an error

});
return false
}


function openDrawer() {
  document.getElementById("myDrawer").style.width = "500px"; // Set the width of the sidebar
  document.getElementById("overlay").style.display = "block"; // Show the overlay
}

function closeDrawer() {
  document.getElementById("myDrawer").style.width = "0"; // Close the sidebar by setting width to 0
  document.getElementById("overlay").style.display = "none"; // Hide the overlay
}

function openModelDrawer() {
  document.getElementById("myModelDrawer").style.width = "500px";
  document.getElementById("overlay").style.display = "block";
}

function closeModelDrawer() {
  document.getElementById("myModelDrawer").style.width = "0";
  document.getElementById("overlay").style.display = "none";
}

function closeAllDrawers() {
  // Set the width of all drawers to 0
  document.getElementById("myDrawer").style.width = "0";
  document.getElementById("myModelDrawer").style.width = "0";
  // Hide the overlay
  document.getElementById("overlay").style.display = "none";
}

function updateThresholdValue(value) {
  document.getElementById('threshold-value').textContent = value;
}


// document.addEventListener("DOMContentLoaded", function () {
//   var slider = document.getElementById('word_length_slider');

//   noUiSlider.create(slider, {
//     start: [1, 30], // Initial handle positions
//     connect: true, // Display a colored bar between the handles
//     range: {
//         'min': 1,
//         'max': 30
//     },
//     step: 1, // Move in increments of 1
//     tooltips: true, // Show tooltips with values above handles
//     format: {
//       to: function(value) {
//         return value.toFixed(0); // Display values as integers
//       },
//       from: function(value) {
//         return Number(value).toFixed(0);
//       }
//     }
//   });
// });


function hexToRGB(hex) {
  let r = 0, g = 0, b = 0;
  // 3 digits
  if (hex.length == 4) {
      r = parseInt(hex[1] + hex[1], 16);
      g = parseInt(hex[2] + hex[2], 16);
      b = parseInt(hex[3] + hex[3], 16);
  }
  // 6 digits
  else if (hex.length == 7) {
      r = parseInt(hex[1] + hex[2], 16);
      g = parseInt(hex[3] + hex[4], 16);
      b = parseInt(hex[5] + hex[6], 16);
  }
  return r + ',' + g + ',' + b;
}


document.addEventListener('DOMContentLoaded', function() {

  var colorPicker = document.getElementById('colorPicker');
  colorPicker.value = '#' + Math.floor(Math.random() * 16777215).toString(16); 


  // New slider for font size
  var fontSizeSlider = document.getElementById('font_size_slider');
  noUiSlider.create(fontSizeSlider, {
      start: [8, 32], // Default values for font size
      connect: true,
      range: { 'min': 8, 'max': 72 },
      step: 1,
      tooltips: [true, true],
      format: {
        to: function(value) {
          return parseInt(value);
        },
        from: function(value) {
          return parseInt(value);
        }
      }
  });


  var angleSlider = document.getElementById('angle_slider');
    noUiSlider.create(angleSlider, {
        start: [-45, 45], // Default values for angles
        connect: true,
        range: { 'min': -180, 'max': 180 },
        step: 1,
        tooltips: [true, true],
        format: {
          to: function(value) {
            return parseInt(value);
          },
          from: function(value) {
            return parseInt(value);
          }
        }
    });

    var wordCountSlider = document.getElementById('word_count_slider');
    noUiSlider.create(wordCountSlider, {
        start: [5], // Default value for word count
        connect: [true, false], // Connect the lower part of the slider
        range: { 'min': 1, 'max': 100 },
        step: 1,
        tooltips: [true],
        format: {
          to: function(value) {
            return parseInt(value);
          },
          from: function(value) {
            return parseInt(value);
          }
        }
    });




  var slider = document.getElementById('word_length_slider');

  noUiSlider.create(slider, {
      start: [1, 30], // Default range
      connect: true,
      step: 1,
      range: {
          'min': 1,
          'max': 30
      },
      step: 1, // Move in increments of 1
      tooltips: true, // Show tooltips with values above handles
      format: {
        to: function(value) {
          return value.toFixed(0); // Display values as integers
        },
        from: function(value) {
          return Number(value).toFixed(0);
        }
      }
  
  });

  window.processImage = function() {
      var formData = new FormData();
      if (selectedImageSourceOrBlob instanceof Blob) {
        formData.append('image', selectedImageSourceOrBlob);
    } else { // Else, use the file input (local file scenario)
        var imageFile = document.getElementById('images').files[0];
        formData.append('image', imageFile);
    }
  
        formData.append('image', imageFile);

      // Get and append slider values
      var values = slider.noUiSlider.get();
      formData.append('word_length_min', parseInt(values[0]));
      formData.append('word_length_max', parseInt(values[1]));

      var fontSizeValues = fontSizeSlider.noUiSlider.get();
      formData.append('font_size_min', fontSizeValues[0]);
      formData.append('font_size_max', fontSizeValues[1]);


      var angleValues = angleSlider.noUiSlider.get();
        formData.append('angle_min', angleValues[0]);
        formData.append('angle_max', angleValues[1]);


      var wordCount = wordCountSlider.noUiSlider.get();
      formData.append('word_count', wordCount);

      var color = document.getElementById('colorPicker').value;
      formData.append('selected_color', hexToRGB(color));
  
  


      fetch('/convert_to_bw', {
          method: 'POST',
          body: formData
      })
      .then(response => response.blob())
      .then(blob => {
        const imageUrl = URL.createObjectURL(blob);
        var img = new Image();
        img.onload = function () {
            // Resize the image to 500x500
            var canvas = document.createElement("canvas");
            var ctx = canvas.getContext("2d");
            canvas.width = 500;
            canvas.height = 500;
            ctx.drawImage(img, 0, 0, 500, 500);
    
            // Create a new image from the canvas and set it as the source for the result image
            var resizedImg = new Image();
            resizedImg.src = canvas.toDataURL();
            document.querySelector('.image-2').src = resizedImg.src;
        };
        img.src = imageUrl;
    })
      .catch(error => console.error('Error:', error));
  }
});



// function processImage() {
//   var formData = new FormData();
//   var values = slider.noUiSlider.get(); // Get current slider values

//   formData.append('word_length_min', parseInt(values[0]));
//   formData.append('word_length_max', parseInt(values[1]));


//   // Check if selectedImageSourceOrBlob is a Blob (sample image scenario)
//   if (selectedImageSourceOrBlob instanceof Blob) {
//       formData.append('image', selectedImageSourceOrBlob);
//   } else { // Else, use the file input (local file scenario)
//       var imageFile = document.getElementById('images').files[0];
//       formData.append('image', imageFile);
//   }

//   fetch('/convert_to_bw', {
//     method: 'POST',
//     body: formData
// })
// .then(response => response.blob())
// .then(blob => {
//     const imageUrl = URL.createObjectURL(blob);
//     var img = new Image();
//     img.onload = function () {
//         // Resize the image to 500x500
//         var canvas = document.createElement("canvas");
//         var ctx = canvas.getContext("2d");
//         canvas.width = 500;
//         canvas.height = 500;
//         ctx.drawImage(img, 0, 0, 500, 500);

//         // Create a new image from the canvas and set it as the source for the result image
//         var resizedImg = new Image();
//         resizedImg.src = canvas.toDataURL();
//         document.querySelector('.image-2').src = resizedImg.src;
//     };
//     img.src = imageUrl;
// })
// .catch(error => {
//     console.error('Error:', error);
// });
// }
