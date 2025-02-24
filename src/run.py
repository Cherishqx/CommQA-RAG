from app import create_app

app = create_app()
# from app import app

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
    #若调试，可考虑开启debug模式，修改代码后服务自动重启，使修改生效，方便调试，生产环境
    #要关闭如：app.run(host='0.0.0.0',port=5000,debug=True)
