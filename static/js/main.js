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
  
  // Clear any existing selected images before adding a new one
  removeSelectedImage(); // This function clears the existing image
  
  // For sample images, fetch the image as a Blob
  fetch(imgSrc)
      .then(response => response.blob())
      .then(blob => {
          // Store the Blob globally or in a way that processImage can use it
          selectedImageSourceOrBlob = blob;
          
          // Display the selected image as before
          var img = document.createElement("img");
          img.src = URL.createObjectURL(blob);
          img.className = "selected-image";
          var label = document.getElementById("dropcontainer");
          var contentPhoto = document.querySelector(".content-photo");
          contentPhoto.style.opacity = "0";
          label.appendChild(img);
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

function processImage() {
  var formData = new FormData();
  
  // Check if selectedImageSourceOrBlob is a Blob (sample image scenario)
  if (selectedImageSourceOrBlob instanceof Blob) {
      formData.append('image', selectedImageSourceOrBlob);
  } else { // Else, use the file input (local file scenario)
      var imageFile = document.getElementById('images').files[0];
      formData.append('image', imageFile);
  }

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
.catch(error => {
    console.error('Error:', error);
});
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
  
  // Check if selectedImageSourceOrBlob is a Blob (sample image scenario)
  if (selectedImageSourceOrBlobModel instanceof Blob) {
      formData.append('image', selectedImageSourceOrBlobModel);
  } else { // Else, use the file input (local file scenario)
      var imageFile = document.getElementById('imagesModel').files[0];
      formData.append('image', imageFile);
  }

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
        document.querySelector('.image-4').src = resizedImg.src;
    };
    img.src = imageUrl;
})
.catch(error => {
    console.error('Error:', error);
});
}
