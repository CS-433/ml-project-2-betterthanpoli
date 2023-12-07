function onSend() {
    message = document.getElementById("message").value;
    document.getElementById("message").value = "";
    var user_message = document.createElement("div");
    user_message.className = "user-message";
    user_message.innerHTML = message;
    // add the newly created element and its content into the DOM
    var container = document.getElementById("chat-messages");
    container.appendChild(user_message);
    // fetch request to server to get response sending the message
    //fetch("/infer").then(onJsonResponse).then(addResponse);
    // add loading dots
    var dots = document.createElement("section");
    dots.className = "dots-container";
    dots.innerHTML = "  <div class='dot'></div> <div class='dot'></div> <div class='dot'></div>"
    container.appendChild(dots);
    fetch("/infer", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify({
          'user_input': message
        })
      }).then(onJsonResponse).then(addResponse);
      
}

function onJsonResponse(response) {
    return response.json();
}

function addResponse(json) {
    answer = json["bot_response"];
    // create a new div element
    var bot_message = document.createElement("div");
    bot_message.className = "bot-message";
    bot_message.innerHTML = answer;
    // add the newly created element and its content into the DOM
    var container = document.getElementById("chat-messages");
    container.removeChild(container.lastChild);
    container.appendChild(bot_message);

}

button = document.getElementById("send-message");
button.addEventListener("click", onSend);



input2 = document.getElementById("chat-input");
input2.addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        onSend();
    }
});