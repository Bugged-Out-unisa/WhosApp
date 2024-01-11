$(function() {
    const userInput = $("#user-input");
    const isThinking = $("#is-thinking");
    let isWaiting = false;

    isThinking.css("opacity", "0");

    const getResponse = (text) => {
        $.ajax({
            url: '/getResponse',
            data: { text: text },
            type: 'POST',
            success: function(response) {
                $('#message-display').append(response);
                isWaiting = false;
                isThinking.css("opacity","0");
            },
            
            error: () =>{
                alert("Errore nella comunicazione con il modello.");
                isWaiting = false;
                isThinking.css("opacity","0");
            }
        });
    }

    const sendMessage = (text) => {
        if(text == "" || text == "\n") return;
        if(isWaiting) return;

        text = text.replace(/\n/g, '\\n');

        userInput.val("");
        
        $.ajax({
            url: '/newUserMessage',
            data: { text: text },
            type: 'POST',
            success: function(response) {
                $('#message-display').append(response);
                isWaiting = true;
                isThinking.css("opacity", "1");
                getResponse(text);
            }
          });
    }

    $("#app").on("click", "#send-btn", (e) => {
        sendMessage(userInput.val())
    })

    $("#app").on("keypress", "#user-input", (e) => {
        var id = e.key || e.which || e.keyCode || 0; 
                
        if ((id == 13  || id == "Enter") && !e.shiftKey) {
            e.preventDefault();
            sendMessage(userInput.val())
        }
    })

})