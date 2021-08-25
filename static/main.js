// create new drawing board
var defaultBoard = new DrawingBoard.Board('default-board');

$("#transform-button").on('click', function () {
    // get drawing from drawing board
    var drawing = defaultBoard.getImg();
    // displayImage(drawing);

    fetch("/transform", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(drawing)
    })
    .then(resp => {
        if (resp.ok)
            // displayImage(drawing);
            resp.json().then(data => {
                displayImage(data.result);
            });
    })
    .catch(err => {
        console.log("An error occured", err.message);
        window.alert("Oops! Something went wrong.");
    });
});

function displayImage(src) {
    // remove previous img
    $('#prediction').children('img').first().remove();

    // create new img DOM element
    var img = document.createElement("img");
    // set src to drawing
    img.src = src;
    img.width = 512;
    img.height = 512;
    $('#prediction').append(img);
}
