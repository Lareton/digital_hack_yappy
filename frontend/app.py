from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        return render_template("index.html", user_input=user_input)
    return render_template("index.html", user_input=None)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
