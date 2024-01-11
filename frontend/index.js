const express = require("express");
const path = require("path");
const bodyParser = require("body-parser");
const app = express();
app.use(bodyParser.urlencoded({ extended: false }));
var jsdom = require("jsdom");
const { JSDOM } = jsdom;
const { window } = new JSDOM();
const { document } = (new JSDOM("")).window;
global.document = document;

var $ = jQuery = require("jquery")(window);

app.set("view engine", "ejs");
app.engine("ejs", require("ejs").__express);
app.use(express.static(path.join(__dirname, "public")));

app.get("/", (request, response) => {
    response.render("index");
});

app.post("/newUserMessage", (req, res) => {
    let text = req.body.text;
    text = text.replace(/\\n/g, "<br>")
    res.render("user-bubble.ejs", {userText: text});
});

app.post("/getResponse", async (req, res) => {
    let userText = req.body.text

    try {
        const result = await axios.get(
          ``
        );
        
        response = result.data;

        res.render("bot-bubble.ejs", {responseText : response});

      } catch (error) {
        console.log(error);
        res.status(400).send("Errore nella comunicazione con il modello");
      }
})

app.listen(port = 3000, () => {
    console.log("Server avviato su porta " + port);
});