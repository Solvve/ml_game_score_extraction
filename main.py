from app import app
import view


if __name__ == '__main__':
    app.run(host='0.0.0.0',threaded=False, port=8050)
