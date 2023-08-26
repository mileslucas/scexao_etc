from flask import Flask
from flask import render_template

app = Flask("scexao_etc")

@app.route("/")
def index():
    return render_template("index.html")


def main():
    app.run()

if __name__ == "__main__":
    main()
