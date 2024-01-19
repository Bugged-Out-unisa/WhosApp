$(function() {
    const userInput = $("#user-input");
    const isThinking = $("#is-thinking");
    const emojiPicker = $("emoji-picker");
    
    // Audio per la notifica di invio e ricezione messaggio
    const msgSent = new Audio('/assets/sent.mp3');
    const msgArrived = new Audio('/assets/arrived.mp3');

    // Variabile per evitare l'invio di più messaggi contemporaneamente
    let isWaiting = false;
    let isFirstResponse = true;

    // Listener per l'aggiunzione di un emoji al messaggio
    emojiPicker.on("emoji-click", (event) => {
        userInput.val(userInput.val() + event.detail.unicode);
    });

    // Listener per l'apertura e la chiusura del picker
    $("#emoji-btn").on("click", () => {
        emojiPicker.toggle();
    });

    $("#app").on("click", "#options" , (e) => {
        $("#global-meters").toggle();
    })

    // Funzione per l'invio del messaggio al modello
    const getResponse = (text) => {
        $.ajax({
            url: "/getResponse",
            data: { text: text },
            type: "POST",
            success: function(response) {
                $("#message-display").append(response.template);
                $("#inside-wrapper")[0].scrollTop = $("#inside-wrapper")[0].scrollHeight;

                let maxIndex = response.data.average.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);

                if(isFirstResponse){
                    isFirstResponse = false;

                    response.data.average.forEach((element, index) => {
                        let html = `<tr>\
                          <td>${response.data.mappedUsers[index]}:</td>\
                          <td>\
                              <meter id="meter-${index}" class="${index === maxIndex ? 'highestPrediction' : '' } globals" value="${element}" min="0" max="1"></meter>\
                          </td>\
                          <td id="percent-${index}">${(element * 100).toFixed(2)}%</td>\
                        </tr>`;

                        $("#global-meters>tbody").append(html);
                      })    
                } else {
                    $(".globals").removeClass("highestPrediction");

                    response.data.average.forEach((element, index) => {
                        
                        $(`#meter-${index}`).attr("value", element);
                        $(`#percent-${index}`).text(`${(element * 100).toFixed(2)}%`);

                        if(index === maxIndex) 
                            $(`#meter-${index}`).addClass("highestPrediction");
                    })
                }

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

    // Funzione per la stampa del messaggio dell'utente e invio al backend
    const sendMessage = (text) => {
        text = text.trim();
        if(text == "" || (/^[\n\r]*$/.test(text))) return;
        if(isWaiting) return;

        text = text.replace(/\n/g, "\\n");

        userInput.val("");

        emojiPicker.hide();
        
        $.ajax({
            url: "/newUserMessage",
            data: { text: text },
            type: "POST",
            success: function(response) {
                $("#message-display").append(response);7
                $("#inside-wrapper")[0].scrollTop = $("#inside-wrapper")[0].scrollHeight;
                
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

    // Listener per l'invio del messaggio tramite pulsante
    $("#app").on("click", "#send-btn", (e) => {
        sendMessage(userInput.val())
    })

    // Listener per l'invio del messaggio tramite tasto invio
    $("#app").on("keypress", "#user-input", (e) => {
        var id = e.key || e.which || e.keyCode || 0; 
        
        // 13 = tasto invio
        if ((id == 13  || id == "Enter") && !e.shiftKey) {
            e.preventDefault();
            sendMessage(userInput.val())
        }
    })

})