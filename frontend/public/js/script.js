$(function() {
    const userInput = $("#user-input");
    const isThinking = $("#is-thinking");
    const emojiPicker = $("#emoji-picker");
    const msgSent = new Audio('/assets/sent.mp3');
    const msgArrived = new Audio('/assets/arrived.mp3');
    let isWaiting = false;

    isThinking.css("opacity", "0");

    emojiPicker.on("emoji-click", (event) => {
        userInput.val(userInput.val() + event.detail.unicode);
    });

    $("#emoji-btn").on("click", () => {
        emojiPicker.toggle();
    });

    const getResponse = (text) => {
        $.ajax({
            url: "/getResponse",
            data: { text: text },
            type: "POST",
            success: function(response) {
                $("#message-display").append(response);
            },
            
            error: () =>{
                alert("Errore nella ricezione della risposta.");
            },

            complete: () => {
                isWaiting = false;
                isThinking.css("opacity","0");

                msgArrived.play();
            }
        });
    }

    const sendMessage = (text) => {
        text = text.trim();
        if(text == "" || (/^[\n\r]*$/.test(text))) return;
        if(isWaiting) return;

        text = text.replace(/\n/g, "\\n");

        userInput.val("");
        
        $.ajax({
            url: "/newUserMessage",
            data: { text: text },
            type: "POST",
            success: function(response) {
                $("#message-display").append(response);
                isWaiting = true;
                isThinking.css("opacity", "1");

                msgSent.play();        

                getResponse(text);
            },
            error: () =>{
                alert("Errore nell'invio del messaggio.");
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