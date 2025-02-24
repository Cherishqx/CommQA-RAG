from flask import Flask
from flask_sqlalchemy import SQLAlchemy

def create_app():
    """构建 Flask 核心应用"""


    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 导入并注册 blueprint
    from app.routes import chat1  # 请根据项目实际目录调整导入路径
    app.register_blueprint(chat1)

    return app
